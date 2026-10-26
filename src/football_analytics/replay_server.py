from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import mcp_server as intel
from . import commentary
from . import overlay_cues as cues
from . import valuation
from .data_loader import load_tracking_data
from .match_intelligence import match_clock

ROOT = Path(__file__).resolve().parents[2]
TRACKING_FILES = sorted((ROOT / "data").glob("gfootball_results_games_*.csv"))
UI_DIR = Path(os.environ.get("OVERLAY_UI_DIR", ROOT / "overlay" / "ui"))
CACHE_DIR = Path(os.environ.get("ENRICH_CACHE_DIR", ROOT / "build" / "enrich_cache"))
PROJECT_ENDPOINT = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "https://matchintel-dl41i.services.ai.azure.com/api/projects/matchintel")
ORCHESTRATOR = os.environ.get("ORCHESTRATOR_AGENT", "match-orchestrator")
ENRICH = os.environ.get("ENRICH", "1") == "1"
AGENT_CALLS_PER_HOUR = int(os.environ.get("AGENT_CALLS_PER_HOUR", "0"))
ENRICH_MIN_IMPORTANCE = float(os.environ.get("ENRICH_MIN_IMPORTANCE", "0.6"))
TICK_S = 0.1
PROFILE_KEYS = ("persona", "language", "favorite_club_id", "favorite_player_id", "interests", "depth", "tone", "tactical")
MOMENT_ID = re.compile(r"\bM\d+-\d{4}\b")
EVENT_ID = re.compile(r"\b\d+-\d{6}\b")
TEMPLATES = {"goal": "lower_third", "shot": "stat_card", "key_pass": "stat_card", "top_speed": "player_tag", "press_escape": "player_tag", "high_turnover": "stat_card", "momentum_shift": "momentum_bar", "rhythm_change": "momentum_bar", "sustained_build_up": "stat_card"}
TURNING_POINTS = ("goal", "momentum_shift")
RATE_LIMITED = {"state": "rate_limited", "message": "AI analysis is rate-limited this hour; instant cues keep coming."}

log = logging.getLogger("replay")
semaphore = asyncio.Semaphore(int(os.environ.get("ENRICH_CONCURRENCY", "3")))
background: set[asyncio.Task] = set()
inflight: dict[str, list[Any]] = {}


@lru_cache(maxsize=1)
def tracks() -> dict[int, dict[str, Any]]:
    games: dict[int, dict[str, Any]] = {}
    for path in TRACKING_FILES:
        frame = load_tracking_data(path, normalize_coordinates=True)
        for game_id, game in frame.groupby("GAME_ID"):
            game = game.sort_values(["TIMESTEP", "PLAYER_ID"])
            steps = int(game["TIMESTEP"].max()) + 1
            ids = sorted(int(p) for p in game["PLAYER_ID"].unique())
            xy = np.full((steps, len(ids), 2), np.nan)
            column = {pid: i for i, pid in enumerate(ids)}
            xy[game["TIMESTEP"].astype(int).to_numpy(), game["PLAYER_ID"].astype(int).map(column).to_numpy()] = game[["LOCATION_X", "LOCATION_Y"]].to_numpy()
            score = game.groupby("TIMESTEP")[["TEAM_0_SCORE", "TEAM_1_SCORE"]].first().reindex(range(steps)).ffill().fillna(0).astype(int).to_numpy()
            games[int(game_id)] = {"ids": ids, "xy": np.round(xy, 1), "score": score, "steps": steps}
    return games


@lru_cache(maxsize=8)
def running_stats(game_id: int) -> dict[str, np.ndarray]:
    steps = tracks()[game_id]["steps"]
    events = intel._game("events", game_id)
    tags = intel._game("player_tags", game_id)

    def cumulative(rows: pd.DataFrame) -> np.ndarray:
        counts = np.zeros((steps, 2))
        for team in (0, 1):
            at = intel._known(rows[rows["TEAM_ID"] == team]).clip(0, steps - 1).astype(int)
            np.add.at(counts[:, team], at.to_numpy(), 1)
        return np.cumsum(counts, axis=0)

    passes = cumulative(events[events["event_type"] == "pass_attempt"])
    attempts = passes + cumulative(events[events["event_type"] == "incomplete_pass_candidate"])
    rhythm = intel._game("match_rhythm", game_id)
    share = np.full(steps, 0.5)
    for _, window in rhythm.iterrows():
        finished = intel._game("match_rhythm", game_id, int(window["end_timestep"]))
        share[min(int(window["end_timestep"]), steps - 1):] = finished["possession_share_team0"].mean()
    return {
        "possession": np.stack([share, 1 - share], axis=1) * 100,
        "shots": cumulative(events[events["event_type"] == "shot_candidate"]),
        "passes": passes,
        "pass_accuracy": np.divide(passes * 100, attempts, out=np.zeros_like(passes), where=attempts > 0),
        "ball_wins": cumulative(events[events["event_type"] == "interception_candidate"]),
        "sprints": cumulative(tags[tags["tag_type"] == "sprint"]),
    }


def frame_at(game_id: int, timestep: int) -> dict[str, Any]:
    game = tracks()[game_id]
    t = min(timestep, game["steps"] - 1)
    window = intel._game("match_rhythm", game_id, t).tail(1)
    points = [[pid, *game["xy"][t, i]] for i, pid in enumerate(game["ids"]) if not np.isnan(game["xy"][t, i, 0])]
    return {
        "t": t,
        "clock": match_clock(t),
        "score": game["score"][t].tolist(),
        "ball": next((p[1:] for p in points if p[0] == -1), None),
        "players": [p for p in points if p[0] != -1],
        "rhythm": None if window.empty else {k: (v.item() if hasattr(v, "item") else v) for k, v in window.iloc[0][["rhythm", "control_index", "momentum_team0", "possession_share_team0"]].items()},
        "stats": {name: [round(float(v)) for v in values[t]] for name, values in running_stats(game_id).items()},
    }


def _club(team: Any) -> str | None:
    if team is None or pd.isna(team):
        return None
    return intel._roster()["clubs"].get(str(int(team)), {}).get("name")


def instant_overlay(moment: dict[str, Any], profile: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = json.loads(moment["metrics"]) if isinstance(moment.get("metrics"), str) else {}
    player_id = None if moment.get("PLAYER_ID") is None or pd.isna(moment["PLAYER_ID"]) else int(moment["PLAYER_ID"])
    name = intel._name(player_id)
    club = _club(moment.get("TEAM_ID"))
    kind = moment["moment_type"]
    known = int(moment["known_timestep"])
    if kind == "goal":
        headline = f"GOAL {club}, {metrics.get('score_after', '')}"
    elif kind in ("momentum_shift", "rhythm_change"):
        headline = moment["fact_hint"]
    else:
        headline = cues.insight(moment, name or club)
    stats = [{"label": cues.STAT_LABELS[k][0], "value": cues.human(v), "unit": cues.STAT_LABELS[k][1]} for k, v in metrics.items() if k in cues.STAT_LABELS and isinstance(v, (int, float)) and not isinstance(v, bool)]
    overlay = {
        "game_id": int(moment["GAME_ID"]),
        "moment_id": moment["moment_id"],
        "start_timestep": known,
        "display_seconds": 6 if moment["importance"] >= ENRICH_MIN_IMPORTANCE else 4,
        "template": TEMPLATES.get(kind, "stat_card"),
        "language": "en",
        "headline": headline[:60],
        "body": f"{match_clock(known)} · {club or 'Match'}"[:180],
        "player_id": player_id,
        "stats": stats[:4],
    }
    cue = cues.moment_cue(moment, overlay["stats"], profile or {})
    favourite = (profile or {}).get("favorite_player_id")
    category = "turning_point" if kind in TURNING_POINTS else "player" if favourite is not None and player_id == favourite else "supporting"
    return {"marker": {"category": category, "t": known, "clock": match_clock(known)}, "overlay": overlay, "validation": intel.validate_overlay(overlay), "moment_type": kind, "importance": float(moment["importance"]), "team_id": None if pd.isna(moment.get("TEAM_ID")) else int(moment["TEAM_ID"]), "cue": cue.model_dump(), "cue_validation": cue_check(cue)}


def cue_check(cue: cues.OverlayCue) -> dict[str, Any]:
    text = " ".join([cue.primary_insight, cue.why_it_matters] + [f"{intel.shown(m.value)} {m.unit or ''}" for m in cue.supporting_metrics])
    return intel.check_claims(cue.game_id, text, cue.source_event_ids, [cue.moment_id] if cue.moment_id else [], [cue.player_focus.player_id] if cue.player_focus else [], cue.timestep, cue.facts)


def moments(game_id: int) -> list[dict[str, Any]]:
    frame = intel._game("key_moments", game_id)
    frame = frame.assign(known_timestep=intel._known(frame)).sort_values("known_timestep", kind="stable")
    return frame.replace({np.nan: None}).to_dict(orient="records")


def profile_from(params: Any) -> dict[str, Any]:
    def number(value: Any) -> int | None:
        return int(value) if value not in (None, "", "null") else None

    raw = params.get("interests")
    return {"persona": commentary.persona_for(raw) if raw else params.get("persona", "casual"), "language": params.get("language", "en"), "favorite_club_id": number(params.get("club")), "favorite_player_id": number(params.get("player")), "interests": cues.interests(raw), "depth": params.get("depth") if params.get("depth") in commentary.DEPTHS else "balanced", "tone": "emotional" if params.get("tone") == "emotional" else "neutral", "tactical": params.get("tactical") in ("1", "true", True)}


AGENT_PERSONA = {"analyst": "analyst", "producer": "analyst", "fantasy": "analyst"}


def agent_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {**profile, "persona": AGENT_PERSONA.get(profile["persona"], "casual"), "interests": {k: cues.band(v) for k, v in cues.interests(profile.get("interests")).items()}}


@lru_cache(maxsize=1)
def orchestrator() -> Any:
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    return AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential()).get_openai_client(agent_name=ORCHESTRATOR)


class BudgetExceeded(RuntimeError):
    pass


agent_calls: deque[float] = deque()
agent_calls_lock = threading.Lock()


def spend_agent_call() -> None:
    if not AGENT_CALLS_PER_HOUR:
        return
    with agent_calls_lock:
        now = time.monotonic()
        while agent_calls and now - agent_calls[0] > 3600:
            agent_calls.popleft()
        if len(agent_calls) >= AGENT_CALLS_PER_HOUR:
            raise BudgetExceeded("hourly live agent budget reached")
        agent_calls.append(now)


def ask_orchestrator(prompt: str) -> dict[str, Any]:
    spend_agent_call()
    data = orchestrator().responses.with_raw_response.create(input=prompt).http_response.json()
    envelopes, text = [], []
    for item in data.get("output", []):
        if item.get("type") == "function_call_output":
            try:
                envelopes.append(json.loads(item["output"]))
            except (TypeError, json.JSONDecodeError):
                pass
        elif item.get("type") == "message":
            text += [part.get("text", "") for part in item.get("content", []) if part.get("type") == "output_text"]
    return {"text": "\n".join(text).strip(), "envelopes": envelopes}


def enriched_overlay(envelope: dict[str, Any], moment: dict[str, Any]) -> dict[str, Any]:
    narration = envelope.get("narration") or {}
    personalized = envelope.get("personalized") or {}
    profile = envelope.get("profile") or {}
    candidates = []
    base = (narration.get("overlays") or [None])[0]
    if base and personalized.get("overlay_headline"):
        candidates.append({**base, "language": profile.get("language", "en"), "headline": personalized["overlay_headline"][:60], "body": (personalized.get("overlay_body") or "")[:180]})
    if base:
        candidates.append(base)
    for overlay in candidates:
        check = intel.validate_overlay(overlay, int(moment["known_timestep"]))
        if check["valid"] and overlay.get("moment_id") == moment["moment_id"]:
            return {"overlay": overlay, "validation": check}
    return {"overlay": None, "validation": intel.validate_overlay(candidates[0], int(moment["known_timestep"])) if candidates else {"valid": False}}


def verified_text(game_id: int, moment: dict[str, Any], overlay: dict[str, Any] | None, texts: list[tuple[Any, str]]) -> tuple[str | None, str]:
    if not overlay:
        return None, "en"
    events = moment["event_ids"].split(";") if isinstance(moment.get("event_ids"), str) else []
    players = [overlay["player_id"]] if overlay.get("player_id") is not None else []
    for text, language in texts:
        if text and intel.check_claims(game_id, text, events, [moment["moment_id"]], players, int(moment["known_timestep"]))["valid"]:
            return text, language
    return overlay.get("body") or None, overlay.get("language", "en")


async def enrich(game_id: int, moment: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    profile = agent_profile(profile)
    key = json.dumps([game_id, moment["moment_id"], [profile[k] for k in PROFILE_KEYS]])
    path = CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        shown = enriched_overlay({"narration": {"overlays": [cached["overlay"]] if cached.get("overlay") else []}}, moment)
        text, language = verified_text(game_id, moment, shown["overlay"], [(cached.get("text"), cached.get("language", "en"))])
        return {**cached, **shown, "text": text, "language": language, "cached": True}
    return await single_flight(key, lambda: call_agent(path, game_id, moment, profile))


async def single_flight(key: str, make: Callable[[], Awaitable[Any]]) -> Any:
    flight = inflight.get(key)
    if flight is None:
        flight = inflight[key] = [asyncio.create_task(make()), 0]
    flight[1] += 1
    try:
        return await asyncio.shield(flight[0])
    finally:
        flight[1] -= 1
        if not flight[1]:
            if inflight.get(key) is flight:
                del inflight[key]
            flight[0].cancel()


async def call_agent(path: Path, game_id: int, moment: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    async with semaphore:
        prompt = f"Live moment. game_id={game_id} moment_id={moment['moment_id']} as_of={int(moment['known_timestep'])} profile_json={json.dumps(profile)}\nAnswer depth: {profile['depth']}. Tone: {profile['tone']}." + (" Lead with tactics." if profile["tactical"] else "")
        reply = await asyncio.to_thread(ask_orchestrator, prompt)
    envelope = next((e for e in reply["envelopes"] if e.get("kind") == "live_moment"), {})
    personalized = envelope.get("personalized") or {}
    narration = envelope.get("narration") or {}
    shown = enriched_overlay(envelope, moment)
    text, language = verified_text(game_id, moment, shown["overlay"], [(personalized.get("text"), personalized.get("language") or "en"), (narration.get("commentary_line"), "en")])
    result = {
        "moment_id": moment["moment_id"],
        "start_timestep": int(moment["known_timestep"]),
        **shown,
        "text": text,
        "language": language,
        "handoffs": envelope.get("handoffs", []),
        "fallbacks": envelope.get("fallbacks", ["orchestrator_did_not_call_live_moment"] if not envelope else []),
    }
    if result["overlay"] and not result["fallbacks"]:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except OSError as error:
            log.warning("could not cache %s: %s", moment["moment_id"], error)
    return result


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=float)}\n\n"


def replay_plan(game_id: int, position: float, profile: dict[str, Any]) -> tuple[int, list[dict[str, Any]], list[cues.OverlayCue]]:
    steps = tracks()[game_id]["steps"]
    pending = [m for m in moments(game_id) if m["known_timestep"] >= position and cues.wanted(m["moment_type"], profile)]
    extra = sorted((cues.weighted(c, profile) for c in cues.derived_cues(game_id, profile, tracks(), running_stats(game_id)) + valuation.money_cues(game_id, profile, steps) if c.timestep >= position and cues.wanted(c.event_type, profile)), key=lambda c: c.timestep)
    return steps, pending, extra


async def cancel_on_close(chunks: AsyncIterator[str], tasks: set[asyncio.Task]) -> AsyncIterator[str]:
    try:
        async for chunk in chunks:
            yield chunk
    finally:
        for task in list(tasks):
            task.cancel()


async def stream(request: Request) -> StreamingResponse:
    params = request.query_params
    game_id = int(params.get("game", 0))
    speed = max(1.0, min(float(params.get("speed", 10)), 120.0))
    position = float(params.get("start", 0))
    enrich_on = ENRICH and params.get("enrich", "1") == "1"
    profile = profile_from(params)
    tasks: set[asyncio.Task] = set()

    async def events():
        nonlocal position
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def run(moment: dict[str, Any]) -> None:
            await queue.put(("enriching", {"moment_id": moment["moment_id"]}))
            try:
                result = await enrich(game_id, moment, profile)
                await queue.put(("enriched", result) if result["overlay"] else ("enrich_failed", {"moment_id": moment["moment_id"], "error": "validation_failed"}))
            except Exception as error:
                log.warning("enrichment failed for %s: %s", moment["moment_id"], error)
                if isinstance(error, BudgetExceeded):
                    await queue.put(("status", RATE_LIMITED))
                await queue.put(("enrich_failed", {"moment_id": moment["moment_id"], "error": type(error).__name__}))

        yield sse("status", {"state": "loading", "message": "Loading match data…"})
        steps, pending, extra = await asyncio.to_thread(replay_plan, game_id, position, profile)
        yield sse("start", {"game_id": game_id, "speed": speed, "profile": profile, "enrich": enrich_on, "steps": steps, "marker_types": [k for k in cues.THEME_BY_MOMENT if cues.wanted(k, profile)]})
        while position < steps:
            yield sse("frame", frame_at(game_id, int(position)))
            while pending and pending[0]["known_timestep"] <= position:
                moment = pending.pop(0)
                shown = instant_overlay(moment, profile)
                if not (shown["validation"]["valid"] and shown["cue_validation"]["valid"]):
                    log.warning("dropped %s: failed validation", moment["moment_id"])
                    continue
                yield sse("moment", shown)
                if enrich_on and moment["importance"] >= ENRICH_MIN_IMPORTANCE:
                    task = asyncio.create_task(run(moment))
                    tasks.add(task)
                    background.add(task)
                    task.add_done_callback(tasks.discard)
                    task.add_done_callback(background.discard)
            while extra and extra[0].timestep <= position:
                cue = extra.pop(0)
                check = cue_check(cue)
                if check["valid"]:
                    yield sse("cue", {"cue": cue.model_dump(), "cue_validation": check})
                else:
                    log.warning("dropped %s cue %s: failed validation", cue.event_type, cue.event_id)
            while not queue.empty():
                yield sse(*queue.get_nowait())
            await asyncio.sleep(TICK_S)
            position = steps if position >= steps - 1 else min(position + speed * TICK_S, steps - 1)
        yield sse("end", frame_at(game_id, steps - 1))
        while tasks or not queue.empty():
            event = await queue.get()
            yield sse(*event)
        yield sse("done", {})

    return StreamingResponse(cancel_on_close(events(), tasks), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def meta(_: Request) -> JSONResponse:
    return JSONResponse({"games": intel.list_matches(), "roster": intel._roster(), "enrich": ENRICH})


async def ask(request: Request) -> JSONResponse:
    body = await request.json()
    game_id = int(body.get("game", 0))
    timestep = int(body.get("timestep", 0))
    question = str(body.get("question", ""))[:500]
    profile = profile_from(body)
    if not question.strip():
        return JSONResponse({"error": "empty question"}, status_code=400)
    style = f"Viewer type: {profile['persona']}. Answer depth: {profile['depth']}. Tone: {profile['tone']}. {commentary.interest_line(profile['interests'])}" + (" Lead with tactics." if profile["tactical"] else "")
    route = f"Match recap: call match_recap with game_id={game_id} and as_of={timestep}.\n" if body.get("action") == "recap" else ""
    prompt = f"Fan question during game_id={game_id} at timestep={timestep} ({match_clock(timestep)}); pass as_of={timestep} to every tool. profile_json={json.dumps(agent_profile(profile))}\n{style}\n{route}Question: {question}"
    try:
        reply = await asyncio.to_thread(ask_orchestrator, prompt)
    except BudgetExceeded:
        return JSONResponse({"error": "Live agent budget reached for this hour; cached analysis still works."}, status_code=429)
    except Exception as error:
        log.warning("ask failed: %s", error)
        return JSONResponse({"error": type(error).__name__}, status_code=502)
    text = reply["text"]
    claim = text.split("Evidence:")[0].strip()
    check = intel.validate_claims(game_id, claim, EVENT_ID.findall(text), MOMENT_ID.findall(text), [p for p in (profile["favorite_player_id"],) if p is not None], timestep)
    fallbacks = [f for e in reply["envelopes"] for f in e.get("fallbacks", [])]
    return JSONResponse({"answer": claim, "evidence": MOMENT_ID.findall(text) + EVENT_ID.findall(text), "verified": check["valid"], "validation": check, "fallbacks": fallbacks})


feedback_log: deque[dict[str, Any]] = deque(maxlen=500)


def _preferences(body: dict[str, Any]) -> commentary.ViewerPreferences:
    try:
        return commentary.ViewerPreferences.model_validate(body.get("preferences") or {})
    except ValueError:
        return commentary.ViewerPreferences()


async def commentary_route(request: Request) -> JSONResponse:
    body = await request.json()
    game_id = int(body.get("game", 0))
    timestep = max(0, min(int(body.get("timestep", 0)), tracks()[game_id]["steps"] - 1))
    action = body.get("action", "question")
    if action not in commentary.Action.__args__:
        return JSONResponse({"error": f"unknown action {action}"}, status_code=400)
    stats = frame_at(game_id, timestep)["stats"]
    player = body.get("player_id")
    response = commentary.respond(game_id, timestep, _preferences(body), stats, action, (body.get("question") or "")[:500] or None, body.get("moment_id"), int(player) if player not in (None, "") else None)
    return JSONResponse(response.model_dump())


async def feedback_route(request: Request) -> JSONResponse:
    body = await request.json()
    try:
        event = commentary.FeedbackEvent.model_validate(body.get("event") or {})
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    feedback_log.append(event.model_dump())
    return JSONResponse(commentary.apply_feedback(_preferences(body), event).model_dump())


async def value_route(request: Request) -> JSONResponse:
    body = await request.json()
    game_id = int(body.get("game", 0))
    steps = tracks()[game_id]["steps"]
    timestep = max(0, min(int(body.get("timestep", 0)), steps - 1))
    player = body.get("player_id")
    if player not in (None, "") and str(player) not in intel._roster()["players"]:
        return JSONResponse({"error": f"unknown player {player}"}, status_code=400)
    card = valuation.player_value(game_id, int(player), timestep, steps, str(body.get("language") or "en")).model_dump() if player not in (None, "") else None
    return JSONResponse({"player": card, "board": valuation.value_board(game_id, timestep, steps), "actions": {k: {"label": v[0], "unit_gbp": v[2]} for k, v in valuation.ACTIONS.items()}})


routes = [Route("/api/meta", meta), Route("/api/stream", stream), Route("/api/ask", ask, methods=["POST"]), Route("/api/commentary", commentary_route, methods=["POST"]), Route("/api/feedback", feedback_route, methods=["POST"]), Route("/api/value", value_route, methods=["POST"])]
if UI_DIR.exists():
    routes.append(Mount("/", StaticFiles(directory=UI_DIR, html=True)))
app = Starlette(routes=routes, middleware=[Middleware(CORSMiddleware, allow_origins=["http://tauri.localhost", "https://tauri.localhost", "tauri://localhost"], allow_methods=["GET", "POST"], allow_headers=["content-type"], allow_private_network=True)])


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("azure").setLevel(logging.WARNING)
    tracks()
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8765")))


if __name__ == "__main__":
    main()
