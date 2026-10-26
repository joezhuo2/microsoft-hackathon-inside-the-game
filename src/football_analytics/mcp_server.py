from __future__ import annotations

import hmac
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ValidationError

from .match_intelligence import known_timestep as _known
from .match_intelligence import load_roster, match_clock

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MATCH_DATA_DIR", ROOT / "output"))
ROSTER_PATH = Path(os.environ.get("MATCH_ROSTER_PATH", ROOT / "data" / "roster.json"))
TABLES = ["events", "key_moments", "match_rhythm", "player_tags", "player_match_intel"]
MAX_ROWS = 60
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
PERCENT = re.compile(r"\s*(%|percent|por ciento|pour cent)", re.I)
CLOCK = re.compile(r"\b\d{1,3}:\d{2}\b")
UNIT = re.compile(r"\s*(m|km/h|s|metres|meters|mètres|metros)\b", re.I)
RATE_FIELD = re.compile(r"share|territory|accuracy|index|momentum|possession|probab|percent", re.I)
ID_FIELDS = {"GAME_ID", "PLAYER_ID", "TEAM_ID", "event_id", "moment_id", "event_ids", "previous_controller", "next_controller", "rule_version", "match_clock"}
RESTRICTED = {
    "penalty": re.compile(r"penalt", re.I),
    "card": re.compile(r"\b(red|yellow) card|\bbooked\b|\bbooking|sent off|tarjeta|carton (rouge|jaune)", re.I),
    "foul": re.compile(r"\bfoul", re.I),
    "offside": re.compile(r"offside|fuera de juego|hors-jeu", re.I),
    "var": re.compile(r"\bVAR\b"),
    "injury": re.compile(r"injur|lesi[oó]n|blessure|bless[ée]", re.I),
    "referee": re.compile(r"referee|[aá]rbitro|arbitre", re.I),
    "substitution": re.compile(r"substitut|subbed|sustituc|remplacement", re.I),
}

mcp = FastMCP(
    "match-intelligence",
    instructions="Read-only football match intelligence over synthetic tracking data. Every number an agent states must come from these tools and cite event_ids or moment_ids; check drafts with validate_claims before publishing.",
    host=os.environ.get("HOST", "127.0.0.1"),
    port=int(os.environ.get("PORT", "8000")),
    stateless_http=True,
    json_response=True,
)


@lru_cache(maxsize=1)
def _data() -> dict[str, pd.DataFrame]:
    tables = {name: [] for name in TABLES}
    for folder in sorted(DATA_DIR.glob("games_*")):
        for name in TABLES:
            path = folder / f"{name}.csv"
            if path.exists():
                tables[name].append(pd.read_csv(path))
    return {name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for name, parts in tables.items()}


@lru_cache(maxsize=1)
def _roster() -> dict[str, Any]:
    return load_roster(ROSTER_PATH)


def _records(frame: pd.DataFrame, limit: int = MAX_ROWS) -> list[dict[str, Any]]:
    clean = frame.head(limit).replace({np.nan: None})
    return json.loads(clean.to_json(orient="records"))


def _match_end(game_id: int) -> int:
    rhythm = _game("match_rhythm", game_id)
    return int(rhythm["end_timestep"].max()) if not rhythm.empty else 10**9


def _game(table: str, game_id: int, as_of: int | None = None) -> pd.DataFrame:
    frame = _data()[table]
    frame = frame[frame["GAME_ID"] == game_id] if not frame.empty else frame
    if as_of is None or frame.empty:
        return frame
    if table == "player_match_intel":
        return frame if as_of >= _match_end(game_id) else frame.iloc[:0]
    return frame[_known(frame) <= as_of]


def _name(player_id: Any) -> str | None:
    if player_id is None or pd.isna(player_id):
        return None
    return _roster().get("players", {}).get(str(int(player_id)), {}).get("name")


def _with_names(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "PLAYER_ID" not in frame.columns:
        return frame
    out = frame.copy()
    out["player_name"] = out["PLAYER_ID"].map(_name)
    if "TEAM_ID" in out.columns:
        clubs = _roster().get("clubs", {})
        out["club_name"] = out["TEAM_ID"].map(lambda team: None if pd.isna(team) else clubs.get(str(int(team)), {}).get("name"))
    return out


def _score_at(game_id: int, timestep: int) -> str:
    goals = _game("key_moments", game_id, timestep)
    goals = goals[goals["moment_type"] == "goal"] if not goals.empty else goals
    team0 = int((goals["TEAM_ID"] == 0).sum()) if not goals.empty else 0
    team1 = int((goals["TEAM_ID"] == 1).sum()) if not goals.empty else 0
    return f"{team0}-{team1}"


@mcp.tool()
def list_matches(as_of: int | None = None) -> list[dict[str, Any]]:
    """List available synthetic matches with clubs and final score, or only the score at as_of when given."""
    moments = _data()["key_moments"]
    clubs = _roster().get("clubs", {})
    games = sorted(int(g) for g in moments["GAME_ID"].unique()) if not moments.empty else []
    return [{"game_id": g, "home": clubs.get("0", {}).get("name"), "away": clubs.get("1", {}).get("name"), **({"final_score": _score_at(g, 10**9)} if as_of is None else {"score": _score_at(g, as_of)})} for g in games]


@mcp.tool()
def get_roster() -> dict[str, Any]:
    """Fictional clubs and players: ids, names, shirt numbers, roles, nationalities."""
    return _roster()


@mcp.tool()
def get_match_state(game_id: int, timestep: int) -> dict[str, Any]:
    """Score, display clock, latest finished rhythm window and the latest key moments known at a timestep (0-3000)."""
    so_far = _game("match_rhythm", game_id, timestep)
    current = so_far.tail(1)
    recent = _game("key_moments", game_id, timestep).tail(5)
    return {
        "game_id": game_id,
        "timestep": timestep,
        "match_clock": match_clock(timestep),
        "score": _score_at(game_id, timestep),
        "score_by_club": _score_detail(game_id, timestep),
        "possession_share_team0_so_far": None if so_far.empty else round(float(so_far["possession_share_team0"].mean()), 3),
        "current_window": _records(current)[0] if len(current) else None,
        "recent_moments": _records(_with_names(recent)),
    }


@mcp.tool()
def get_events(game_id: int, from_timestep: int, to_timestep: int, event_types: list[str] | None = None, as_of: int | None = None) -> list[dict[str, Any]]:
    """Tracking-derived candidate events in a timestep window (max 60 rows). Optional event_types filter, e.g. pass_attempt, shot_candidate, interception_candidate, carry. as_of drops events not finished by then."""
    events = _game("events", game_id, as_of)
    window = events[events["start_timestep"].between(from_timestep, to_timestep)]
    if event_types:
        window = window[window["event_type"].isin(event_types)]
    return _records(_with_names(window))


@mcp.tool()
def get_key_moments(game_id: int, from_timestep: int = 0, to_timestep: int = 3000, min_importance: float = 0.0, moment_types: list[str] | None = None, as_of: int | None = None) -> list[dict[str, Any]]:
    """Rule-detected key moments (goal, shot, key_pass, high_turnover, press_escape, sustained_build_up, momentum_shift, rhythm_change, top_speed) with cited event_ids. as_of drops moments not finished by then."""
    moments = _game("key_moments", game_id, as_of)
    picked = moments[moments["start_timestep"].between(from_timestep, to_timestep) & (moments["importance"] >= min_importance)]
    if moment_types:
        picked = picked[picked["moment_type"].isin(moment_types)]
    return _records(_with_names(picked))


def _score_detail(game_id: int, timestep: int) -> dict[str, Any]:
    clubs = _roster().get("clubs", {})
    team0, team1 = (int(part) for part in _score_at(game_id, timestep).split("-"))
    return {clubs.get("0", {}).get("name", "team_0"): team0, clubs.get("1", {}).get("name", "team_1"): team1}


@mcp.tool()
def get_moment_context(game_id: int, moment_id: str, window: int = 100, as_of: int | None = None) -> dict[str, Any]:
    """Everything needed to explain one moment in a single call: the moment, nearby events, rhythm windows before/during/after, and the score by club before and after. Only data finished by as_of (default: when the moment became known) is returned."""
    moments = _game("key_moments", game_id)
    row = moments[moments["moment_id"] == moment_id]
    if row.empty:
        return {"error": f"unknown moment_id {moment_id} for game {game_id}"}
    moment = row.iloc[0]
    start, end, known = int(moment["start_timestep"]), int(moment["end_timestep"]), int(_known(row).iloc[0])
    as_of = known if as_of is None else as_of
    if as_of < known:
        return {"error": f"moment_id {moment_id} is not finished at as_of {as_of}"}
    events = _game("events", game_id, as_of)
    nearby = events[events["start_timestep"].between(start - window, end + window) & ~events["event_type"].isin(["touch_candidate", "dead_ball_candidate"])]
    rhythm = _game("match_rhythm", game_id, as_of)
    during = rhythm[(rhythm["start_timestep"] <= end) & (rhythm["end_timestep"] >= start)]
    before = rhythm[rhythm["end_timestep"] < start].tail(1)
    after = rhythm[rhythm["start_timestep"] > end].head(1)
    return {
        "moment": _records(_with_names(row))[0],
        "score_before": _score_detail(game_id, start - 1),
        "score_after": _score_detail(game_id, min(end, as_of)),
        "rhythm_before": _records(before)[0] if len(before) else None,
        "rhythm_during": _records(during),
        "rhythm_after": _records(after)[0] if len(after) else None,
        "events": _records(_with_names(nearby), limit=40),
        "note": "Score dictionaries map club name to goals. Events are tracking-derived candidates.",
    }


@mcp.tool()
def get_match_rhythm(game_id: int, as_of: int | None = None) -> list[dict[str, Any]]:
    """Per-window possession share, territory, control_index (control vs chaos) and momentum; only windows finished by as_of when given."""
    return _records(_game("match_rhythm", game_id, as_of), limit=100)


@mcp.tool()
def get_player_profile(game_id: int, player_id: int, as_of: int | None = None) -> dict[str, Any]:
    """One player's whole-match stats (null before full time when as_of is given), sprint/distance tags and key moments finished by as_of."""
    intel = _game("player_match_intel", game_id, as_of)
    tags = _game("player_tags", game_id, as_of)
    moments = _game("key_moments", game_id, as_of)
    return {
        "stats": _records(intel[intel["PLAYER_ID"] == player_id])[:1] or None,
        "tags": _records(tags[tags["PLAYER_ID"] == player_id]),
        "moments": _records(_with_names(moments[moments["PLAYER_ID"] == player_id])),
        "roster": _roster().get("players", {}).get(str(player_id)),
    }


@mcp.tool()
def compare_players(game_id: int, role: str | None = None, sort_by: str = "passes_attempted", as_of: int | None = None) -> list[dict[str, Any]]:
    """Compare players' whole-match stats, optionally filtered by role (e.g. 'central midfield'), sorted by a stat column. Empty before full time when as_of is given."""
    intel = _game("player_match_intel", game_id, as_of)
    if role:
        intel = intel[intel["role"] == role]
    if sort_by in intel.columns:
        intel = intel.sort_values(sort_by, ascending=False)
    return _records(intel)


def _evidence(game_id: int, events: pd.DataFrame, moments: pd.DataFrame, player_ids: list[int], as_of: int | None, facts: dict[str, Any]) -> tuple[list[float], list[float]]:
    values: list[float] = []
    rates: list[float] = []

    def collect(item: Any, key: str = "") -> None:
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, (int, float, np.integer, np.floating)) and np.isfinite(item):
            values.append(float(item))
            if key.endswith("_gbp"):
                values.extend([abs(float(item)) / 1000, abs(float(item)) / 1_000_000])
            if RATE_FIELD.search(key):
                rates.extend([abs(float(item)), abs(float(item)) * 100])
        elif isinstance(item, str):
            for match in NUMBER.findall(item):
                collect(float(match), key)
        elif isinstance(item, dict):
            for nested_key, nested in item.items():
                if nested_key not in ID_FIELDS:
                    collect(json.loads(nested) if nested_key == "metrics" and isinstance(nested, str) else nested, nested_key)
        elif isinstance(item, list):
            for nested in item:
                collect(nested, key)

    intel = _game("player_match_intel", game_id, as_of)
    cited_intel = intel[intel["PLAYER_ID"].isin(player_ids)] if not intel.empty else intel
    collect(_records(events, limit=1000) + _records(moments, limit=1000) + _records(cited_intel, limit=1000))
    clocks = {_clock(match_clock(t)) for frame in (events, moments) for column in ("start_timestep", "end_timestep") for t in frame.get(column, [])} | {_clock(c) for c in moments.get("match_clock", [])}
    for start, end in zip(moments.get("start_timestep", []), moments.get("end_timestep", [])):
        collect([_score_at(game_id, int(start) - 1), _score_at(game_id, int(end) if as_of is None else min(int(end), as_of))])
    for player_id in player_ids:
        collect(_roster().get("players", {}).get(str(player_id), {}).get("number"))
    collect(facts)
    return values, rates, clocks


def _clock(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(":"))


def shown(value: Any) -> str:
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def _supported(token: str, values: list[float]) -> bool:
    tolerance = 0.5 * 10 ** -(len(token.split(".")[1]) if "." in token else 0) + 1e-9
    return any(abs(value - float(token)) <= tolerance for value in values)


def check_claims(game_id: int, text: str, event_ids: list[str], moment_ids: list[str], player_ids: list[int], as_of: int | None = None, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    facts = facts or {}
    every_event, every_moment = set(_game("events", game_id)["event_id"]), set(_game("key_moments", game_id)["moment_id"])
    events, moments = _game("events", game_id, as_of), _game("key_moments", game_id, as_of)
    unknown = [i for i in event_ids if i not in every_event] + [i for i in moment_ids if i not in every_moment]
    events, moments = events[events["event_id"].isin(event_ids)], moments[moments["moment_id"].isin(moment_ids)]
    future = [i for i in event_ids if i in every_event and i not in set(events["event_id"])] + [i for i in moment_ids if i in every_moment and i not in set(moments["moment_id"])]
    values, rates, clocks = _evidence(game_id, events, moments, player_ids, as_of, facts)
    unsupported = [c for c in CLOCK.findall(text) if _clock(c) not in clocks]
    numbers = CLOCK.sub(" ", text)
    for match in NUMBER.finditer(numbers):
        token = match.group()
        rate = bool(PERCENT.match(numbers, match.end())) or ("." in token and abs(float(token)) < 1 and not UNIT.match(numbers, match.end()))
        if not rate and "." not in token and abs(float(token)) <= 1:
            continue
        if not (_supported(token.lstrip("-"), rates) if rate else _supported(token, values)):
            unsupported.append(token)
    cited_players = set(player_ids)
    for rows in (events, moments):
        cited_players |= {int(p) for col in ("PLAYER_ID", "previous_controller", "next_controller") if col in rows.columns for p in rows[col].dropna()}
    lowered = text.lower()
    wrong_names = [person["name"] for pid, person in _roster().get("players", {}).items() if int(pid) not in cited_players and (person["name"].lower() in lowered or re.search(rf"\b{re.escape(person['short'].lower())}\b", lowered))]
    kinds = " ".join(str(v) for rows in (events, moments) for col in ("event_type", "subtype", "moment_type") if col in rows.columns for v in rows[col].dropna())
    invented = [kind for kind, pattern in RESTRICTED.items() if pattern.search(text) and kind not in kinds]
    has_citations = bool(event_ids or moment_ids or facts)
    return {"valid": not unknown and not future and not unsupported and not wrong_names and not invented and has_citations, "unknown_ids": unknown, "future_ids": future, "unsupported_numbers": unsupported, "uncited_players": wrong_names, "unsupported_events": invented, "has_citations": has_citations}


@mcp.tool()
def validate_claims(game_id: int, text: str, event_ids: list[str] | None = None, moment_ids: list[str] | None = None, player_ids: list[int] | None = None, as_of: int | None = None) -> dict[str, Any]:
    """Quality gate: cited ids exist and are finished by as_of; every number in text matches a value in the cited records (percentages and decimals below 1 only a cited share/index/momentum field); no penalty, card, foul, offside, VAR, injury, referee or substitution unless a cited record has that type. Run on every draft and every translation before publishing."""
    return check_claims(game_id, text, event_ids or [], moment_ids or [], player_ids or [], as_of)


class OverlayStat(BaseModel):
    label: str = Field(max_length=24)
    value: float | str
    unit: str | None = Field(default=None, max_length=8)


class Overlay(BaseModel):
    game_id: int
    moment_id: str
    start_timestep: int = Field(ge=0, le=3000)
    display_seconds: float = Field(ge=2, le=15)
    template: Literal["lower_third", "stat_card", "player_tag", "momentum_bar", "recap_card"]
    language: str = Field(min_length=2, max_length=10)
    headline: str = Field(max_length=60)
    body: str = Field(default="", max_length=180)
    player_id: int | None = None
    stats: list[OverlayStat] = Field(default_factory=list, max_length=4)


@mcp.tool()
def validate_overlay(payload: dict[str, Any], as_of: int | None = None) -> dict[str, Any]:
    """Quality gate for broadcast overlay JSON: schema, a real moment_id finished by as_of (default start_timestep), and text backed by that moment under the validate_claims rules."""
    try:
        overlay = Overlay.model_validate(payload)
    except ValidationError as error:
        return {"valid": False, "schema_errors": json.loads(error.json(include_url=False))}
    text = " ".join([overlay.headline, overlay.body] + [f"{shown(stat.value)} {stat.unit or ''}" for stat in overlay.stats])
    moments = _game("key_moments", overlay.game_id)
    row = moments[moments["moment_id"] == overlay.moment_id]
    event_ids = row["event_ids"].iloc[0].split(";") if len(row) and isinstance(row["event_ids"].iloc[0], str) else []
    claims = check_claims(overlay.game_id, text, event_ids, [overlay.moment_id], [overlay.player_id] if overlay.player_id is not None else [], overlay.start_timestep if as_of is None else as_of)
    return {"valid": claims["valid"], "schema_errors": [], **{k: v for k, v in claims.items() if k != "valid"}}


def main() -> None:
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    if transport != "streamable-http":
        mcp.run(transport=transport)
        return
    import uvicorn

    app = mcp.streamable_http_app()
    api_key = os.environ.get("MCP_API_KEY")
    if api_key:
        inner = app

        async def app(scope, receive, send):  # type: ignore[no-redef]
            if scope["type"] == "http" and not hmac.compare_digest(dict(scope["headers"]).get(b"x-api-key", b""), api_key.encode()):
                await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"unauthorized"})
                return
            await inner(scope, receive, send)

    uvicorn.run(app, host=mcp.settings.host, port=mcp.settings.port)


if __name__ == "__main__":
    main()
