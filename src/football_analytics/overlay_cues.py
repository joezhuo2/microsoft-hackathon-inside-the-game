from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from . import mcp_server as intel
from .i18n import t as tr, word
from .match_intelligence import match_clock

Theme = Literal["breakthrough", "pressure_shift", "momentum_swing", "player_spotlight", "goal_threat", "defensive_masterclass", "chaos_index", "control_index", "milestone", "recap_cue", "value_swing"]
Priority = Literal["high", "medium", "low"]
Audience = Literal["casual", "superfan", "analyst", "fantasy", "producer", "newcomer", "money"]

THEME_BY_MOMENT: dict[str, tuple[Theme, Priority]] = {
    "goal": ("goal_threat", "high"),
    "shot": ("goal_threat", "high"),
    "key_pass": ("breakthrough", "medium"),
    "sustained_build_up": ("control_index", "medium"),
    "press_escape": ("pressure_shift", "medium"),
    "high_turnover": ("pressure_shift", "medium"),
    "momentum_shift": ("momentum_swing", "medium"),
    "rhythm_change": ("chaos_index", "low"),
    "top_speed": ("player_spotlight", "low"),
}
DURATION_MS: dict[Priority, int] = {"high": 6000, "medium": 4500, "low": 3500}
CONFIDENCE = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.45, "UNRESOLVED": 0.3}
MEASURED = 0.8
OWN_THIRD_M = 40.0
BUMP: dict[Priority, Priority] = {"low": "medium", "medium": "high", "high": "high"}
INTERESTS = ("speed", "passing", "pressure", "possession", "money")
BANDS = ((75, "Obsessed"), (50, "Into it"), (20, "Light"), (0, "Skip"))
SKIP_BELOW, BUMP_FROM = 20, 75
STAT_BY_EVENT = {"top_speed": "speed", "key_pass": "passing", "sustained_build_up": "passing", "press_escape": "pressure", "high_turnover": "pressure", "interception": "pressure", "momentum_shift": "possession", "rhythm_change": "possession"}


class Metric(BaseModel):
    label: str
    value: float | str
    unit: str | None = None


class PlayerFocus(BaseModel):
    player_id: int
    name: str | None = None
    involvement: str


class OverlayCue(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    event_id: str
    moment_id: str | None = None
    game_id: int
    timestep: int = Field(ge=0)
    match_clock: str
    event_type: str
    primary_insight: str = Field(max_length=90)
    why_it_matters: str = Field(max_length=220)
    confidence: float = Field(ge=0, le=1)
    evidence: Literal["observed", "derived"]
    priority: Priority
    duration_ms: int = Field(ge=1500, le=15000)
    theme: Theme
    audience_mode: Audience = "casual"
    team_id: int | None = None
    player_focus: PlayerFocus | None = None
    supporting_metrics: list[Metric] = Field(default_factory=list, max_length=4)
    source_event_ids: list[str] = Field(default_factory=list)
    synthetic: Literal[True] = True
    facts: SkipJsonSchema[dict[str, Any]] = Field(default_factory=dict, exclude=True)


def interests(raw: Any = None) -> dict[str, int]:
    if isinstance(raw, str):
        raw = dict(part.split(":", 1) for part in raw.split(",") if ":" in part)
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for key in INTERESTS:
        try:
            out[key] = max(0, min(100, int(float(raw.get(key, 50)))))
        except (TypeError, ValueError, OverflowError):
            out[key] = 50
    return out


def band(value: int) -> str:
    return next(name for floor, name in BANDS if value >= floor)


def interest(kind: str, profile: dict[str, Any]) -> int:
    stat = "money" if kind.startswith("value_") else STAT_BY_EVENT.get(kind)
    return interests(profile.get("interests")).get(stat, 50) if stat else 50


def wanted(kind: str, profile: dict[str, Any]) -> bool:
    return interest(kind, profile) >= SKIP_BELOW


def weighted(cue: OverlayCue, profile: dict[str, Any]) -> OverlayCue:
    if interest(cue.event_type, profile) < BUMP_FROM:
        return cue
    priority = BUMP[cue.priority]
    return cue.model_copy(update={"priority": priority, "duration_ms": max(cue.duration_ms, DURATION_MS[priority])})


def _club(team: Any) -> str:
    if team is None or pd.isna(team):
        return "Match"
    return intel._roster()["clubs"].get(str(int(team)), {}).get("name", f"Team {int(team)}")


def _short(team: int) -> str:
    return intel._roster()["clubs"].get(str(team), {}).get("short", f"T{team}")


def _goal_story(team: int, score_after: str) -> str:
    goals = [int(part) for part in score_after.split("-")]
    mine, theirs = goals[team], goals[1 - team]
    if mine > theirs:
        return "goal_lead" if mine - 1 <= theirs else "goal_extend"
    return "goal_level" if mine == theirs else "goal_back"


def why_it_matters(kind: str, metrics: dict[str, Any], team: Any, name: str | None, language: str = "en") -> str:
    club = _club(team)
    who = name or club
    if kind == "goal" and metrics.get("score_after") and team is not None and not pd.isna(team):
        return tr(language, _goal_story(int(team), metrics["score_after"]), club=club, score=metrics["score_after"])
    if kind == "shot":
        speed = metrics.get("shot_speed_kmh")
        return tr(language, "shot_fast" if speed else "shot", speed=speed, distance=metrics.get("distance_to_goal_line_m"), club=club)
    if kind == "key_pass":
        return tr(language, "key_pass", difficulty=metrics.get("pass_difficulty"), gap=metrics.get("lane_pressure_m"), length=metrics.get("ball_travel_m"))
    if kind == "sustained_build_up":
        return tr(language, "build_up", passes=metrics.get("consecutive_passes"), players=metrics.get("players_involved"), club=club)
    if kind == "press_escape":
        return tr(language, "press_escape", who=who, opponents=metrics.get("opponents_within_5m"))
    if kind == "high_turnover":
        return tr(language, "high_turnover", x=metrics.get("x_from_own_goal_m"), club=club)
    if kind == "momentum_shift":
        return tr(language, "momentum", side=_club(0 if (metrics.get("momentum_after") or 0) > 0 else 1))
    if kind == "rhythm_change":
        target = metrics.get("to")
        if target == "chaos":
            return tr(language, "rhythm_chaos", origin=word(language, str(metrics.get("from")), str(metrics.get("from"))), turnovers=metrics.get("turnovers"))
        return tr(language, "rhythm_control" if target == "control" else "rhythm_even")
    if kind == "top_speed":
        return tr(language, "top_speed", who=who, speed=metrics.get("top_speed_kmh"), distance=metrics.get("sprint_distance_m"))
    return tr(language, "moment")


def _confidence(event_ids: list[str], game_id: int) -> tuple[float, Literal["observed", "derived"]]:
    if not event_ids:
        return MEASURED, "derived"
    events = intel._game("events", game_id)
    if "confidence_level" not in events.columns:
        return MEASURED, "observed"
    levels = events.loc[events["event_id"].isin(event_ids), "confidence_level"].map(CONFIDENCE).dropna()
    return (round(float(levels.mean()), 2) if len(levels) else MEASURED), "observed"


def _metrics(stats: list[dict[str, Any]]) -> list[Metric]:
    return [Metric(**stat) for stat in stats[:4]]


def moment_cue(moment: dict[str, Any], stats: list[dict[str, Any]], profile: dict[str, Any]) -> OverlayCue:
    kind = moment["moment_type"]
    game_id = int(moment["GAME_ID"])
    metrics = json.loads(moment["metrics"]) if isinstance(moment.get("metrics"), str) else {}
    theme, priority = THEME_BY_MOMENT.get(kind, ("momentum_swing", "low"))
    if kind == "rhythm_change" and metrics.get("to") == "control":
        theme = "control_index"
    if kind == "momentum_shift" and abs((metrics.get("momentum_after") or 0) - (metrics.get("momentum_before") or 0)) >= 0.5:
        priority = "high"
    player_id = None if moment.get("PLAYER_ID") is None or pd.isna(moment["PLAYER_ID"]) else int(moment["PLAYER_ID"])
    name = intel._name(player_id)
    focus = None
    if player_id is not None:
        focus = PlayerFocus(player_id=player_id, name=name, involvement=kind.replace("_", " "))
        if profile.get("favorite_player_id") == player_id:
            theme, priority = ("player_spotlight" if theme != "goal_threat" else theme), BUMP[priority]
    event_ids = moment["event_ids"].split(";") if isinstance(moment.get("event_ids"), str) else []
    confidence, evidence = _confidence(event_ids, game_id)
    if kind == "goal":
        confidence = max(confidence, 0.95)
    team = moment.get("TEAM_ID")
    return weighted(OverlayCue(
        event_id=event_ids[-1] if event_ids else moment["moment_id"],
        moment_id=moment["moment_id"],
        game_id=game_id,
        timestep=int(moment["known_timestep"]),
        match_clock=match_clock(moment["known_timestep"]),
        event_type=kind,
        primary_insight=tr(profile.get("language"), "goal_insight", club=_club(team)) if kind == "goal" else (f"{name}: {moment['fact_hint']}" if name else moment["fact_hint"]),
        why_it_matters=why_it_matters(kind, metrics, team, name, profile.get("language", "en"))[:220],
        confidence=confidence,
        evidence=evidence,
        priority=priority,
        duration_ms=DURATION_MS[priority],
        theme=theme,
        audience_mode=profile.get("persona", "casual") if profile.get("persona") in Audience.__args__ else "casual",
        team_id=None if team is None or pd.isna(team) else int(team),
        player_focus=focus,
        supporting_metrics=_metrics(stats),
        source_event_ids=event_ids,
    ), profile)


def _own_goal_x(game_id: int, tracks: dict[int, dict[str, Any]]) -> dict[int, float]:
    game = tracks[game_id]
    keepers = {0: game["ids"].index(0) if 0 in game["ids"] else None, 1: game["ids"].index(100) if 100 in game["ids"] else None}
    return {team: (0.0 if index is None or np.nanmean(game["xy"][:, index, 0]) < 60 else 120.0) for team, index in keepers.items()}


def derived_cues(game_id: int, profile: dict[str, Any], tracks: dict[int, dict[str, Any]], running: dict[str, np.ndarray]) -> list[OverlayCue]:
    audience = profile.get("persona") if profile.get("persona") in Audience.__args__ else "casual"
    cues: list[OverlayCue] = []
    events = intel._game("events", game_id)
    own_goal = _own_goal_x(game_id, tracks)
    for _, event in events[events["event_type"] == "interception_candidate"].iterrows():
        team = int(event["TEAM_ID"]) if not pd.isna(event["TEAM_ID"]) else None
        if team is None:
            continue
        depth = round(abs(float(event["start_x"]) - own_goal[team]), 1)
        if depth > OWN_THIRD_M:
            continue
        name = intel._name(event["PLAYER_ID"])
        player_id = None if pd.isna(event["PLAYER_ID"]) else int(event["PLAYER_ID"])
        cues.append(OverlayCue(
            event_id=event["event_id"], game_id=game_id, timestep=int(event["start_timestep"]), match_clock=match_clock(event["start_timestep"]),
            event_type="interception", primary_insight=tr(profile.get("language"), "interception", who=name or _club(team)),
            why_it_matters=tr(profile.get("language"), "interception_why"),
            confidence=CONFIDENCE.get(str(event["confidence_level"]), 0.5), evidence="observed", priority="medium" if profile.get("favorite_player_id") == player_id else "low",
            duration_ms=DURATION_MS["low"], theme="defensive_masterclass", audience_mode=audience, team_id=team,
            player_focus=PlayerFocus(player_id=player_id, name=name, involvement="interception") if player_id is not None else None,
            supporting_metrics=[Metric(label="from own goal", value=depth, unit="m")], source_event_ids=[event["event_id"]], facts={"from_own_goal_m": depth},
        ))
    favourite = profile.get("favorite_player_id")
    if favourite is not None:
        tags = intel._game("player_tags", game_id)
        milestones = tags[(tags["PLAYER_ID"] == favourite) & (tags["tag_type"] == "distance_milestone")].sort_values("start_timestep")
        name = intel._name(favourite)
        for index, (_, tag) in enumerate(milestones.iterrows(), start=1):
            distance = int(tag["distance_m"]) if not pd.isna(tag.get("distance_m")) else int(tag["value"]) * index
            cues.append(OverlayCue(
                event_id=f"T{game_id}-{favourite}-{index}", game_id=game_id, timestep=int(tag["start_timestep"]), match_clock=match_clock(tag["start_timestep"]),
                event_type="distance_milestone", primary_insight=tr(profile.get("language"), "milestone", name=name, distance=distance),
                why_it_matters=tr(profile.get("language"), "milestone_why", name=name, distance=distance),
                confidence=MEASURED, evidence="derived", priority="medium", duration_ms=DURATION_MS["medium"], theme="milestone", audience_mode=audience,
                team_id=int(tag["TEAM_ID"]), player_focus=PlayerFocus(player_id=favourite, name=name, involvement="distance milestone"),
                supporting_metrics=[Metric(label="distance", value=distance, unit="m")], facts={"distance_m": distance},
            ))
    steps = tracks[game_id]["steps"]
    score = tracks[game_id]["score"]
    for t, title in ((steps // 2, "Half time"), (steps - 1, "Full time")):
        possession, shots = running["possession"][t], running["shots"][t]
        cues.append(OverlayCue(
            event_id=f"R{game_id}-{t}", game_id=game_id, timestep=t, match_clock=match_clock(t), event_type=title.lower().replace(" ", "_"),
            primary_insight=f"{title}: {_short(0)} {score[t][0]}–{score[t][1]} {_short(1)}",
            why_it_matters=f"Possession {round(possession[0])}% to {round(possession[1])}%, shots {int(shots[0])} to {int(shots[1])}.",
            confidence=MEASURED, evidence="derived", priority="high", duration_ms=8000, theme="recap_cue", audience_mode=audience,
            supporting_metrics=[Metric(label="possession", value=round(possession[0]), unit="%"), Metric(label="shots", value=f"{int(shots[0])}–{int(shots[1])}")],
            facts={"possession": [float(possession[0]), float(possession[1])], "shots": [int(shots[0]), int(shots[1])], "score": [int(score[t][0]), int(score[t][1])]},
        ))
    return sorted(cues, key=lambda cue: cue.timestep)


def cue_schema() -> dict[str, Any]:
    return OverlayCue.model_json_schema()
