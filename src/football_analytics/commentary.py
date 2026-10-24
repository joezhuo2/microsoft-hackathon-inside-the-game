from __future__ import annotations

import json
import re
import uuid
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from . import mcp_server as intel
from .match_intelligence import match_clock
from .overlay_cues import why_it_matters

Action = Literal["explain_more", "show_tactics", "focus_player", "simpler", "numbers", "why", "question"]
Depth = Literal["headlines", "balanced", "deep"]
Tone = Literal["neutral", "emotional"]
FeedbackKind = Literal["too_basic", "too_technical", "more_tactical", "more_emotional", "follow_player"]

DEPTHS: list[Depth] = ["headlines", "balanced", "deep"]
OBSERVED_LIMIT: dict[Depth, int] = {"headlines": 2, "balanced": 3, "deep": 5}
RECENT_MOMENT_WINDOW = 600
TREND_WINDOW = 600
PLAIN = {"casual", "newcomer"}
EVENT_WORDS = {
    "pass_attempt": "completed a pass", "incomplete_pass_candidate": "misplaced a pass", "shot_candidate": "shot",
    "carry": "carried the ball", "dribble_candidate": "dribbled", "interception_candidate": "intercepted",
    "possession_gain_candidate": "won the ball", "possession_loss_candidate": "lost the ball", "reception_candidate": "received the ball",
    "clearance_candidate": "cleared", "goal_candidate": "scored",
}
INTENTS: list[tuple[Action, re.Pattern[str]]] = [
    ("why", re.compile(r"\bwhy\b|matter|important", re.I)),
    ("show_tactics", re.compile(r"tactic|press|shape|control|chaos|formation|possession|territory", re.I)),
    ("numbers", re.compile(r"number|stat|how many|how much|percent|%|speed", re.I)),
    ("simpler", re.compile(r"simple|simpler|eli5|plain", re.I)),
]


class ViewerPreferences(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    persona: str = "casual"
    language: str = "en"
    favorite_club_id: int | None = None
    favorite_player_id: int | None = None
    metric_focus: str | None = None
    depth: Depth = "balanced"
    tone: Tone = "neutral"
    tactical: bool = False


class TrailItem(BaseModel):
    kind: Literal["event", "moment", "metric", "pattern"]
    id: str
    label: str


class CommentaryResponse(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    response_id: str
    game_id: int
    timestep: int
    match_clock: str
    action: Action
    question: str | None = None
    headline: str
    observed: list[str]
    interpretation: str
    projection: str | None = None
    context_trail: list[TrailItem]
    quick_replies: list[str] = Field(max_length=4)
    depth: Depth
    tone: Tone
    moment_id: str | None = None
    player_id: int | None = None
    synthetic: Literal[True] = True


class FeedbackEvent(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: FeedbackKind
    game_id: int
    timestep: int
    response_id: str | None = None
    player_id: int | None = None


class FeedbackResult(BaseModel):
    preferences: ViewerPreferences
    effects: list[str]


def default_depth(persona: str) -> Depth:
    return "deep" if persona in ("analyst", "producer") else "headlines" if persona == "newcomer" else "balanced"


def apply_feedback(prefs: ViewerPreferences, event: FeedbackEvent) -> FeedbackResult:
    updated = prefs.model_copy()
    level = DEPTHS.index(prefs.depth)
    effects: list[str] = []
    if event.kind == "too_basic":
        updated.depth = DEPTHS[min(level + 1, 2)]
        effects.append(f"Depth set to {updated.depth}: more detail and more overlays.")
    elif event.kind == "too_technical":
        updated.depth = DEPTHS[max(level - 1, 0)]
        updated.tactical = False
        effects.append(f"Depth set to {updated.depth}: plainer language and fewer overlays.")
    elif event.kind == "more_tactical":
        updated.tactical = True
        updated.depth = DEPTHS[max(level, 1)]
        effects.append("Tactical context first: control, pressure and territory.")
    elif event.kind == "more_emotional":
        updated.tone = "emotional"
        effects.append("Commentary tone set to emotional.")
    elif event.kind == "follow_player" and event.player_id is not None:
        updated.favorite_player_id = event.player_id
        effects.append(f"Following {intel._name(event.player_id) or event.player_id}: their moments get the spotlight.")
    return FeedbackResult(preferences=updated, effects=effects)


def detect_action(question: str) -> Action:
    for action, pattern in INTENTS:
        if pattern.search(question):
            return action
    return "explain_more"


def _player_in_question(question: str) -> int | None:
    lowered = question.lower()
    for pid, person in intel._roster().get("players", {}).items():
        if person["name"].lower() in lowered or re.search(rf"\b{re.escape(person['short'].lower())}\b", lowered):
            return int(pid)
    return None


def _club(team: Any) -> str:
    return "the match" if team is None or pd.isna(team) else intel._roster()["clubs"].get(str(int(team)), {}).get("name", f"Team {int(team)}")


def _current_moment(game_id: int, timestep: int, moment_id: str | None) -> dict[str, Any] | None:
    moments = intel._game("key_moments", game_id)
    if moments.empty:
        return None
    if moment_id:
        row = moments[moments["moment_id"] == moment_id]
        if len(row):
            return row.replace({np.nan: None}).iloc[0].to_dict()
    recent = moments[(moments["start_timestep"] <= timestep) & (moments["start_timestep"] >= timestep - RECENT_MOMENT_WINDOW)]
    if recent.empty:
        return None
    return recent.sort_values(["importance", "start_timestep"]).replace({np.nan: None}).iloc[-1].to_dict()


def _rhythm(game_id: int, timestep: int) -> dict[str, Any] | None:
    rhythm = intel._game("match_rhythm", game_id)
    if rhythm.empty:
        return None
    window = rhythm[rhythm["start_timestep"] <= timestep].tail(1)
    return None if window.empty else window.iloc[0].to_dict()


def _event_lines(game_id: int, event_ids: list[str], timestep: int) -> list[tuple[str, str]]:
    events = intel._game("events", game_id)
    rows = events[events["event_id"].isin(event_ids) & (events["start_timestep"] <= timestep)].sort_values("start_timestep")
    lines = []
    for _, event in rows.iterrows():
        who = intel._name(event["PLAYER_ID"]) or _club(event["TEAM_ID"])
        detail = EVENT_WORDS.get(event["event_type"], event["event_type"].replace("_candidate", "").replace("_", " "))
        speed = event.get("ball_speed_kmh")
        extra = f" at about {round(float(speed), 1)} km/h" if event["event_type"] == "shot_candidate" and speed is not None and not pd.isna(speed) else ""
        lines.append((event["event_id"], f"{match_clock(event['start_timestep'])} {who} {detail}{extra}"))
    return lines


def _player_so_far(game_id: int, player_id: int, timestep: int) -> dict[str, int]:
    events = intel._game("events", game_id)
    mine = events[(events["PLAYER_ID"] == player_id) & (events["start_timestep"] <= timestep)]
    recent = mine[mine["start_timestep"] > timestep - TREND_WINDOW]
    def count(frame: pd.DataFrame, kind: str) -> int:
        return int((frame["event_type"] == kind).sum())

    return {
        "passes": count(mine, "pass_attempt"), "misplaced": count(mine, "incomplete_pass_candidate"), "shots": count(mine, "shot_candidate"),
        "interceptions": count(mine, "interception_candidate"), "carries": count(mine, "carry"), "touches": len(mine), "recent_touches": len(recent),
        "earlier_rate": round((len(mine) - len(recent)) * TREND_WINDOW / (timestep - TREND_WINDOW)) if timestep > TREND_WINDOW else len(recent),
    }


def _quick_replies(action: Action, prefs: ViewerPreferences, moment: dict[str, Any] | None) -> list[str]:
    options: list[str] = []
    player = intel._roster().get("players", {}).get(str(int(moment["PLAYER_ID"]))) if moment and moment.get("PLAYER_ID") is not None else None
    if player and action != "focus_player":
        options.append(f"Focus on {player['short']}")
    if action != "why":
        options.append("Why did this matter?")
    if action != "show_tactics" and (prefs.tactical or prefs.depth != "headlines"):
        options.append("Show tactics")
    if action != "numbers":
        options.append("Give me the numbers")
    options.append("Make it simpler" if prefs.depth == "deep" and action != "simpler" else "Explain more")
    return list(dict.fromkeys(options))[:4]


def respond(game_id: int, timestep: int, prefs: ViewerPreferences, stats: dict[str, list[int]], action: Action, question: str | None = None, moment_id: str | None = None, player_id: int | None = None) -> CommentaryResponse:
    if action == "question" and question:
        player_id = player_id or _player_in_question(question)
        action = "focus_player" if player_id is not None else detect_action(question)
    moment = _current_moment(game_id, timestep, moment_id)
    rhythm = _rhythm(game_id, timestep)
    plain = prefs.persona in PLAIN or prefs.depth == "headlines"
    limit = OBSERVED_LIMIT[prefs.depth]
    trail: list[TrailItem] = []
    observed: list[str] = []
    projection = None
    clubs = intel._roster()["clubs"]
    short = [clubs.get("0", {}).get("short", "Home"), clubs.get("1", {}).get("short", "Away")]
    metrics = json.loads(moment["metrics"]) if moment and isinstance(moment.get("metrics"), str) else {}
    name = intel._name(moment.get("PLAYER_ID")) if moment else None
    why = why_it_matters(moment["moment_type"], metrics, moment.get("TEAM_ID"), name) if moment else "No standout moment in the last few minutes; the game is in a settled phase."
    if moment:
        trail.append(TrailItem(kind="moment", id=moment["moment_id"], label=moment["fact_hint"]))
        event_ids = moment["event_ids"].split(";") if isinstance(moment.get("event_ids"), str) else []
    else:
        event_ids = []
    rhythm_line = None
    if rhythm:
        label = rhythm["rhythm"]
        trail.append(TrailItem(kind="pattern", id=f"window-{int(rhythm['window'])}", label=f"{label} spell"))
        rhythm_line = (f"Right now it's a {label} spell." if plain else f"Current window is {label}: control index {round(float(rhythm['control_index']), 2)}, momentum {round(float(rhythm['momentum_team0']), 2)} towards {short[0] if rhythm['momentum_team0'] >= 0 else short[1]}.")

    if not moment:
        headline = f"{match_clock(timestep)}: settled phase"
    elif moment["moment_type"] == "goal":
        headline = f"Goal, {_club(moment.get('TEAM_ID'))}"
    else:
        headline = f"{name}: {moment['fact_hint']}" if name else moment["fact_hint"]
    if action in ("why", "explain_more", "simpler"):
        lines = _event_lines(game_id, event_ids, timestep)[-limit:]
        observed = [text for _, text in lines] or ([headline] if moment else [])
        if not observed and stats.get("possession"):
            possession, shots = stats["possession"], stats.get("shots", [0, 0])
            observed = [f"Possession so far {possession[0]}% {short[0]} to {possession[1]}% {short[1]}", f"Shots {shots[0]} to {shots[1]}"][:limit]
        trail += [TrailItem(kind="event", id=eid, label=text) for eid, text in lines]
        interpretation = why if action != "simpler" else why.split(":")[0].split(". ")[0].rstrip(".") + "."
        if action == "explain_more" and rhythm_line:
            interpretation = f"{interpretation} {rhythm_line}"
        if action == "explain_more" and prefs.depth == "deep" and moment and moment["moment_type"] in ("momentum_shift", "high_turnover", "press_escape", "goal"):
            team = moment.get("TEAM_ID")
            projection = f"If this pattern holds, expect {_club(team)} to keep pushing higher for the next few minutes." if team is not None else None
    elif action == "show_tactics":
        possession, wins = stats.get("possession", [50, 50]), stats.get("ball_wins", [0, 0])
        observed = [f"Possession so far {possession[0]}% {short[0]} to {possession[1]}% {short[1]}", f"Ball wins {wins[0]} to {wins[1]}"]
        if rhythm:
            observed.append(f"Territory {round(float(rhythm['territory_team0']) * 100)}% in {short[0]}'s attacking half this spell, {int(rhythm['turnovers'])} turnovers")
        observed = observed[:limit]
        trail += [TrailItem(kind="metric", id="possession", label="possession share"), TrailItem(kind="metric", id="ball_wins", label="ball wins")]
        if rhythm:
            leader = short[0] if rhythm["possession_share_team0"] >= 0.5 else short[1]
            interpretation = (f"{leader} have the ball more right now. " if plain else f"{leader} are dictating this spell. ") + (rhythm_line or "")
        else:
            interpretation = "Not enough play yet to read a tactical pattern."
    elif action == "numbers":
        observed = [f"{label}: {values[0]} {short[0]} – {values[1]} {short[1]}" for label, values in (("Shots", stats.get("shots")), ("Passes", stats.get("passes")), ("Pass accuracy %", stats.get("pass_accuracy")), ("Ball wins", stats.get("ball_wins")), ("Sprints", stats.get("sprints"))) if values][:max(limit, 3)]
        observed += [f"{key.replace('_', ' ')}: {value}" for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)][: 2 if prefs.depth != "headlines" else 0]
        trail += [TrailItem(kind="metric", id=key, label=key.replace("_", " ")) for key in ("shots", "passes", "pass_accuracy")]
        interpretation = why
    else:
        player_id = player_id or prefs.favorite_player_id or (int(moment["PLAYER_ID"]) if moment and moment.get("PLAYER_ID") is not None else None)
        if player_id is None:
            return respond(game_id, timestep, prefs, stats, "explain_more", question, moment_id)
        who = intel._name(player_id) or f"Player {player_id}"
        so_far = _player_so_far(game_id, player_id, timestep)
        headline = f"Player focus: {who}"
        observed = [f"{so_far['touches']} involvements so far, {so_far['recent_touches']} in the last spell", f"{so_far['passes']} completed passes, {so_far['misplaced']} misplaced", f"{so_far['shots']} shots, {so_far['interceptions']} interceptions, {so_far['carries']} carries"][:limit]
        trail.append(TrailItem(kind="metric", id=f"player-{player_id}", label=f"{who} events up to {match_clock(timestep)}"))
        trend = "more involved than earlier" if so_far["recent_touches"] > so_far["earlier_rate"] else "quieter than earlier" if so_far["recent_touches"] < so_far["earlier_rate"] else "as involved as earlier"
        involved = moment is not None and moment.get("PLAYER_ID") == player_id
        interpretation = f"{who} is {trend} in this match." + (f" They were at the centre of the latest moment: {why}" if involved else "")
        action = "focus_player"
    if prefs.tone == "emotional" and interpretation:
        interpretation = ("What a moment! " if moment and moment["moment_type"] in ("goal", "shot", "momentum_shift") else "Here's the story: ") + interpretation
    if prefs.tactical and action not in ("show_tactics", "focus_player") and rhythm_line and rhythm_line not in interpretation:
        interpretation = f"{interpretation} {rhythm_line}"
    return CommentaryResponse(
        response_id=uuid.uuid4().hex[:12], game_id=game_id, timestep=timestep, match_clock=match_clock(timestep), action=action, question=question,
        headline=headline, observed=observed, interpretation=interpretation.strip(), projection=projection if prefs.depth == "deep" else None,
        context_trail=trail[: 3 + limit], quick_replies=_quick_replies(action, prefs, moment), depth=prefs.depth, tone=prefs.tone,
        moment_id=moment["moment_id"] if moment else None, player_id=player_id,
    )
