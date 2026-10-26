from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from . import mcp_server as intel
from .match_intelligence import match_clock
from .i18n import t, word
from .overlay_cues import CONFIDENCE, DURATION_MS, Metric, OverlayCue, PlayerFocus

PROGRESSIVE_M = 10.0
HARD_PASS = 50.0
MIN_SHARE = 0.2
SHOT_TO_GOAL = 180
CUE_ACTIONS = ("goal", "shot", "progressive_pass", "interception")

ACTIONS: dict[str, tuple[str, str, int]] = {
    "goal": ("Goals", "goal", 150_000),
    "shot": ("Shots", "shot", 15_000),
    "progressive_pass": ("Progressive passes", "progressive pass", 20_000),
    "hard_pass": ("Hard passes completed", "hard pass", 10_000),
    "pass": ("Completed passes", "completed pass", 4_000),
    "interception": ("Interceptions", "interception", 15_000),
    "tackle": ("Tackles", "tackle", 8_000),
    "pressure": ("Pressing actions", "pressing action", 4_000),
    "dribble": ("Dribbles", "dribble", 10_000),
    "shot_faced": ("Shots faced without conceding", "shot survived", 12_000),
    "loss": ("Possession losses", "possession loss", -6_000),
    "misplaced": ("Misplaced passes", "misplaced pass", -3_000),
}
EVENT_ACTION = {"shot_candidate": "shot", "interception_candidate": "interception", "dribble_candidate": "dribble", "possession_loss_candidate": "loss", "incomplete_pass_candidate": "misplaced"}
SYNTHETIC_ACTION = {"tackle_synthetic": "tackle", "pressure_synthetic": "pressure"}

ROLE_PAY: dict[str, tuple[str, tuple[str, ...]]] = {
    "goalkeeper": ("Keepers are paid to keep the ball out; every opposition shot that does not go in counts for them.", ("shot_faced", "pass")),
    "centre back": ("Centre-backs are paid to win the ball back and start attacks without risk.", ("interception", "tackle", "pass")),
    "left back": ("Full-backs are paid to defend their flank and carry the ball forward.", ("tackle", "pressure", "progressive_pass")),
    "right back": ("Full-backs are paid to defend their flank and carry the ball forward.", ("tackle", "pressure", "progressive_pass")),
    "central midfield": ("Central midfielders are paid to move the ball forward and break lines.", ("progressive_pass", "hard_pass", "interception")),
    "left midfield": ("Wide midfielders are paid to beat players and create shots.", ("dribble", "progressive_pass", "shot")),
    "right midfield": ("Wide midfielders are paid to beat players and create shots.", ("dribble", "progressive_pass", "shot")),
    "central front": ("Strikers are paid for goals and shots; everything else is a bonus.", ("goal", "shot")),
}

Verdict = Literal["bargain", "fair value", "overpaid so far"]


class ActionLine(BaseModel):
    action: str
    label: str
    count: int
    unit_gbp: int
    value_gbp: int
    synthetic: bool
    moments: list[str] = Field(default_factory=list)


class RoleStat(BaseModel):
    action: str
    label: str
    player_per_match: float
    role_per_match: float


class PlayerValue(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    game_id: int
    timestep: int
    match_clock: str
    player_id: int
    name: str
    club: str | None
    role: str
    weekly_wage_gbp: int
    contract_note: str
    wage_so_far_gbp: int
    match_value_gbp: int
    value_ratio: float
    verdict: Verdict
    breakdown: list[ActionLine]
    role_stats: list[RoleStat]
    why_paid: str
    why_value: str
    synthetic: Literal[True] = True


def _synthetic_events() -> pd.DataFrame:
    parts = [pd.read_csv(path) for path in sorted(intel.DATA_DIR.glob("games_*/synthetic_events.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["event_id", "GAME_ID", "start_timestep", "event_type", "PLAYER_ID", "TEAM_ID"])


def _rows(frame: pd.DataFrame, action: str, synthetic: bool = False, id_column: str = "event_id") -> pd.DataFrame:
    out = frame.assign(start_timestep=intel._known(frame))[[id_column, "GAME_ID", "start_timestep", "PLAYER_ID"]].rename(columns={id_column: "event_id"}).dropna(subset=["PLAYER_ID"])
    return out.assign(action=action, synthetic=synthetic)


def _shots_faced(events: pd.DataFrame, moments: pd.DataFrame) -> pd.DataFrame:
    keepers = {int(pid) for pid, person in intel._roster()["players"].items() if person.get("role") == "goalkeeper" and not person.get("bench")}
    keeper_of = {0 if pid < 100 else 1: pid for pid in keepers}
    goals = moments[moments["moment_type"] == "goal"]
    rows = []
    for _, shot in events[(events["event_type"] == "shot_candidate") & events["TEAM_ID"].notna()].iterrows():
        defending = 1 - int(shot["TEAM_ID"])
        scored = goals[(goals["GAME_ID"] == shot["GAME_ID"]) & (goals["end_timestep"] - shot["start_timestep"]).between(0, SHOT_TO_GOAL)]
        if defending in keeper_of and scored.empty:
            rows.append({"event_id": shot["event_id"], "GAME_ID": shot["GAME_ID"], "start_timestep": min(int(shot["start_timestep"]) + SHOT_TO_GOAL, intel._match_end(int(shot["GAME_ID"]))), "PLAYER_ID": keeper_of[defending], "action": "shot_faced", "synthetic": False})
    return pd.DataFrame(rows, columns=["event_id", "GAME_ID", "start_timestep", "PLAYER_ID", "action", "synthetic"])


@lru_cache(maxsize=1)
def actions() -> pd.DataFrame:
    events = intel._data()["events"]
    moments = intel._data()["key_moments"]
    parts = [_rows(events[events["event_type"] == kind], action) for kind, action in EVENT_ACTION.items()]
    passes = events[(events["event_type"] == "pass_attempt") & events["completed"].eq(True)]
    parts += [
        _rows(passes, "pass"),
        _rows(passes[passes["progression_m"] >= PROGRESSIVE_M], "progressive_pass"),
        _rows(passes[passes["pass_difficulty"] >= HARD_PASS], "hard_pass"),
        _rows(moments[moments["moment_type"] == "goal"], "goal", id_column="moment_id"),
    ]
    synthetic = _synthetic_events()
    parts += [_rows(synthetic[synthetic["event_type"] == kind], action, synthetic=True) for kind, action in SYNTHETIC_ACTION.items()]
    parts.append(_shots_faced(events, moments))
    out = pd.concat(parts, ignore_index=True)
    out["PLAYER_ID"] = out["PLAYER_ID"].astype(int)
    out["value_gbp"] = out["action"].map(lambda a: ACTIONS[a][2])
    return out.sort_values(["GAME_ID", "start_timestep"], ignore_index=True)


def _player(player_id: int) -> dict[str, Any]:
    return intel._roster()["players"].get(str(player_id), {})


def _gbp(amount: float) -> str:
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    return f"{sign}£{amount / 1_000_000:.1f}m" if amount >= 1_000_000 else f"{sign}£{round(amount / 1000)}k"


def _verdict(ratio: float) -> Verdict:
    return "bargain" if ratio >= 1.2 else "fair value" if ratio >= 0.7 else "overpaid so far"


def _role_stats(game_id: int, player_id: int, role: str) -> list[RoleStat]:
    others = actions()[actions()["GAME_ID"] != game_id]
    games = max(others["GAME_ID"].nunique(), 1)
    peers = [int(pid) for pid, person in intel._roster()["players"].items() if person.get("role") == role and not person.get("bench")]
    stats = []
    for action in ROLE_PAY.get(role, ("", ()))[1]:
        rows = others[others["action"] == action]
        mine = (rows["PLAYER_ID"] == player_id).sum() / games
        role_avg = rows["PLAYER_ID"].isin(peers).sum() / games / max(len(peers), 1)
        stats.append(RoleStat(action=action, label=ACTIONS[action][0], player_per_match=round(float(mine), 1), role_per_match=round(float(role_avg), 1)))
    return stats


def _label(language: str, action: str, count: float) -> str:
    return word(language, f"n:{action}", ACTIONS[action][0].lower()) if count != 1 else word(language, f"1:{action}", ACTIONS[action][1])


def _why_paid(name: str, person: dict[str, Any], stats: list[RoleStat], language: str = "en") -> str:
    role = person.get("role", "player")
    text = t(language, "paid", name=name, wage=_gbp(person.get("weekly_wage_gbp", 0)), note=person.get("contract_note", "squad player"), role_line=word(language, f"pay:{role}", ROLE_PAY.get(role, ("", ()))[0]))
    if stats:
        best = max(stats, key=lambda s: s.player_per_match - s.role_per_match)
        side = "above" if best.player_per_match > best.role_per_match else "below" if best.player_per_match < best.role_per_match else "level"
        text += t(language, "paid_peer", mine=best.player_per_match, label=_label(language, best.action, 2), side=t(language, side), role=word(language, role, role), avg=best.role_per_match)
    return text


def _why_value(name: str, lines: list[ActionLine], value: int, wage_so_far: int, verdict: Verdict, language: str = "en") -> str:
    gains = sorted((line for line in lines if line.value_gbp > 0), key=lambda line: -line.value_gbp)[:2]
    if not gains:
        return t(language, "value_none", name=name, wage=_gbp(wage_so_far))
    drivers = t(language, "and").join(f"{line.count} {_label(language, line.action, line.count)} ({_gbp(line.value_gbp)})" for line in gains)
    costs = [line for line in lines if line.value_gbp < 0]
    cost = t(language, "cost", amount=_gbp(-sum(line.value_gbp for line in costs))) if costs else ""
    return t(language, "value", value=_gbp(value), wage=_gbp(wage_so_far), drivers=drivers, cost=cost, verdict=word(language, verdict, verdict))


def player_value(game_id: int, player_id: int, timestep: int, steps: int, language: str = "en") -> PlayerValue:
    person = _player(player_id)
    name = person.get("name", f"Player {player_id}")
    rows = actions()
    rows = rows[(rows["GAME_ID"] == game_id) & (rows["PLAYER_ID"] == player_id) & (rows["start_timestep"] <= timestep)]
    lines = []
    for action, (label, _, unit) in ACTIONS.items():
        hits = rows[rows["action"] == action]
        if len(hits):
            lines.append(ActionLine(action=action, label=label, count=len(hits), unit_gbp=unit, value_gbp=int(unit * len(hits)), synthetic=bool(hits["synthetic"].any()), moments=[match_clock(t) for t in hits["start_timestep"]][-5:]))
    value = sum(line.value_gbp for line in lines)
    wage = int(person.get("weekly_wage_gbp", 0))
    wage_so_far = int(wage * max(min(timestep / max(steps - 1, 1), 1.0), MIN_SHARE))
    ratio = round(value / wage_so_far, 2) if wage_so_far else 0.0
    verdict = _verdict(ratio)
    stats = _role_stats(game_id, player_id, person.get("role", ""))
    team = 0 if player_id < 100 else 1
    return PlayerValue(
        game_id=game_id, timestep=timestep, match_clock=match_clock(timestep), player_id=player_id, name=name,
        club=intel._roster()["clubs"].get(str(team), {}).get("name"), role=person.get("role", "player"), weekly_wage_gbp=wage,
        contract_note=person.get("contract_note", "squad player"), wage_so_far_gbp=wage_so_far, match_value_gbp=value, value_ratio=ratio,
        verdict=verdict, breakdown=sorted(lines, key=lambda line: -abs(line.value_gbp)), role_stats=stats,
        why_paid=_why_paid(name, person, stats, language), why_value=_why_value(name, lines, value, wage_so_far, verdict, language),
    )


def value_board(game_id: int, timestep: int, steps: int) -> list[dict[str, Any]]:
    board = []
    for pid, person in intel._roster()["players"].items():
        if person.get("bench"):
            continue
        card = player_value(game_id, int(pid), timestep, steps)
        board.append({"player_id": card.player_id, "name": card.name, "club": card.club, "role": card.role, "weekly_wage_gbp": card.weekly_wage_gbp, "match_value_gbp": card.match_value_gbp, "value_ratio": card.value_ratio, "verdict": card.verdict})
    return sorted(board, key=lambda row: -row["value_ratio"])


def money_cues(game_id: int, profile: dict[str, Any], steps: int) -> list[OverlayCue]:
    rows = actions()
    rows = rows[(rows["GAME_ID"] == game_id) & rows["action"].isin(CUE_ACTIONS)]
    events = intel._game("events", game_id).set_index("event_id")
    favourite = profile.get("favorite_player_id")
    language = profile.get("language", "en")
    cues = []
    for _, row in rows.iterrows():
        player_id = int(row["PLAYER_ID"])
        person = _player(player_id)
        name = person.get("name", f"Player {player_id}")
        label, singular, unit = ACTIONS[row["action"]]
        card = player_value(game_id, player_id, int(row["start_timestep"]), steps)
        key = row["action"] in ROLE_PAY.get(person.get("role", ""), ("", ()))[1]
        role = person.get("role", "player")
        priority = "high" if row["action"] == "goal" or player_id == favourite else "medium"
        observed = row["event_id"] in events.index
        cues.append(OverlayCue(
            event_id=row["event_id"], moment_id=row["event_id"] if row["action"] == "goal" else None, game_id=game_id, timestep=int(row["start_timestep"]),
            match_clock=match_clock(row["start_timestep"]), event_type=f"value_{row['action']}",
            primary_insight=t(language, "adds_value", name=name, action=_label(language, row["action"], 1))[:90],
            why_it_matters=t(language, "cue_core" if key else "cue_bonus", role=word(language, role, role)) + t(language, "cue_verdict", verdict=word(language, card.verdict, card.verdict)),
            confidence=CONFIDENCE.get(str(events.loc[row["event_id"], "confidence_level"]), 0.7) if observed and "confidence_level" in events.columns else (0.95 if row["action"] == "goal" else 0.6),
            evidence="observed" if not row["synthetic"] else "derived", priority=priority, duration_ms=DURATION_MS[priority], theme="value_swing", audience_mode="money",
            team_id=0 if player_id < 100 else 1, player_focus=PlayerFocus(player_id=player_id, name=name, involvement=singular),
            supporting_metrics=[Metric(label="action value", value=_gbp(unit)), Metric(label="match value", value=_gbp(card.match_value_gbp)), Metric(label="weekly wage", value=_gbp(card.weekly_wage_gbp))],
            source_event_ids=[row["event_id"]] if observed else [],
            facts={"action_value_gbp": unit, "match_value_gbp": card.match_value_gbp, "weekly_wage_gbp": card.weekly_wage_gbp},
        ))
    return cues
