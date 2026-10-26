from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PASS_TYPES = {"pass_attempt", "incomplete_pass_candidate"}
RELEASE_TYPES = PASS_TYPES | {"shot_candidate", "clearance_candidate", "ball_release_candidate"}
PRESSURE_TYPES = RELEASE_TYPES | {"carry", "dribble_candidate", "possession_gain_candidate", "interception_candidate"}
SUBTYPE_DIFFICULTY_BONUS = {"through_ball_candidate": 10.0, "switch_of_play_candidate": 5.0, "cross_candidate": 5.0}
KNOWN_AT_START = {"shot_candidate", "shot"}
MOMENT_IMPORTANCE = {
    "goal": 1.0,
    "shot": 0.7,
    "momentum_shift": 0.6,
    "key_pass": 0.55,
    "high_turnover": 0.55,
    "sustained_build_up": 0.5,
    "press_escape": 0.45,
    "rhythm_change": 0.45,
    "top_speed": 0.35,
}


def load_roster(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"clubs": {}, "players": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def known_timestep(frame: pd.DataFrame) -> pd.Series:
    if "end_timestep" not in frame.columns:
        return frame["start_timestep"]
    kind = frame.get("event_type", frame.get("moment_type"))
    return frame["end_timestep"] if kind is None else frame["start_timestep"].where(kind.isin(KNOWN_AT_START), frame["end_timestep"])


def match_clock(timestep: float, timesteps_per_match: int = 3000, match_minutes: float = 90.0) -> str:
    seconds = int(round(float(timestep) / timesteps_per_match * match_minutes * 60))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def pass_difficulty(distance: float, lane_pressure: float, progression: float, subtype: Any = None) -> float:
    lane = 10.0 if not np.isfinite(lane_pressure) else lane_pressure
    score = (
        40.0 * min(distance / 40.0, 1.0)
        + 35.0 * (1.0 - min(lane / 10.0, 1.0))
        + 15.0 * min(max(progression, 0.0) / 30.0, 1.0)
        + SUBTYPE_DIFFICULTY_BONUS.get(subtype, 0.0)
    )
    return round(min(score, 100.0), 1)


def _segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    length_squared = float(segment @ segment)
    if length_squared == 0.0:
        return np.hypot(*(points - start).T)
    t = np.clip(((points - start) @ segment) / length_squared, 0.0, 1.0)
    return np.hypot(*(points - (start + t[:, None] * segment)).T)


def _frame_lookup(features: pd.DataFrame) -> dict[tuple[Any, int], tuple[np.ndarray, np.ndarray]]:
    players = features[features["PLAYER_ID"] != -1]
    return {
        (game_id, int(timestep)): (rows["TEAM_ID"].to_numpy(), rows[["LOCATION_X", "LOCATION_Y"]].to_numpy(dtype=float))
        for (game_id, timestep), rows in players.groupby(["GAME_ID", "TIMESTEP"], sort=False)
    }


def _direction(attacking_directions: Mapping[Any, float], team_id: Any) -> float:
    if pd.isna(team_id):
        return 0.0
    return float(attacking_directions.get(int(team_id), attacking_directions.get(str(int(team_id)), 0.0)))


def enrich_events(events: pd.DataFrame, features: pd.DataFrame, attacking_directions: Mapping[Any, float], seconds_per_timestep: float) -> pd.DataFrame:
    """Add pass quality, ball speed and on-ball pressure to tracking-only events."""
    out = events.copy()
    for column in ["ball_travel_m", "progression_m", "lane_pressure_m", "opponents_within_5m", "ball_speed_kmh", "pass_difficulty"]:
        out[column] = np.nan
    out["completed"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    if out.empty:
        return out
    frames = _frame_lookup(features)
    ball = features[features["PLAYER_ID"] == -1].set_index(["GAME_ID", "TIMESTEP"])[["VELOCITY_X", "VELOCITY_Y"]]
    kmh_per_unit = 3.6 / seconds_per_timestep

    for index, event in out[out["event_type"].isin(PRESSURE_TYPES)].iterrows():
        key = (event["GAME_ID"], int(event["start_timestep"]))
        start = np.array([event["start_x"], event["start_y"]], dtype=float)
        if not np.isfinite(start).all():
            continue
        frame = frames.get(key)
        opponents = frame[1][frame[0] != event["TEAM_ID"]] if frame is not None and pd.notna(event["TEAM_ID"]) else np.empty((0, 2))
        if len(opponents):
            out.at[index, "opponents_within_5m"] = int((np.hypot(*(opponents - start).T) <= 5.0).sum())
        if event["event_type"] not in RELEASE_TYPES:
            continue
        if key in ball.index:
            velocity = ball.loc[key].to_numpy(dtype=float)
            out.at[index, "ball_speed_kmh"] = round(float(np.hypot(*velocity)) * kmh_per_unit, 1)
        end = np.array([event["end_x"], event["end_y"]], dtype=float)
        if not np.isfinite(end).all():
            continue
        distance = float(np.hypot(*(end - start)))
        progression = float((end[0] - start[0]) * _direction(attacking_directions, event["TEAM_ID"]))
        lane = float(_segment_distance(opponents, start, end).min()) if len(opponents) else np.nan
        out.at[index, "ball_travel_m"] = round(distance, 2)
        out.at[index, "progression_m"] = round(progression, 2)
        out.at[index, "lane_pressure_m"] = round(lane, 2) if np.isfinite(lane) else np.nan
        if event["event_type"] in PASS_TYPES:
            out.at[index, "pass_difficulty"] = pass_difficulty(distance, lane, progression, event.get("subtype"))
            out.at[index, "completed"] = event["event_type"] == "pass_attempt"
    return out


def player_speed_tags(features: pd.DataFrame, seconds_per_timestep: float, sprint_kmh: float = 32.0, min_sprint_frames: int = 10, distance_step_m: float = 1000.0) -> pd.DataFrame:
    """Threshold-triggered tags for on-screen player labels: sprints and distance milestones."""
    players = features[features["PLAYER_ID"] != -1].sort_values(["GAME_ID", "PLAYER_ID", "TIMESTEP"], kind="mergesort")
    kmh_per_unit = 3.6 / seconds_per_timestep
    tags: list[dict[str, Any]] = []
    for (game_id, player_id), rows in players.groupby(["GAME_ID", "PLAYER_ID"], sort=False):
        speed = rows["horizontal_speed"].to_numpy(dtype=float) * kmh_per_unit
        timesteps = rows["TIMESTEP"].to_numpy()
        step_distance = rows["distance_travelled"].to_numpy(dtype=float)
        team_id = int(rows["TEAM_ID"].iloc[0])
        edges = np.diff(np.concatenate([[0], (speed >= sprint_kmh).astype(int), [0]]))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if stop - start >= min_sprint_frames:
                tags.append({"GAME_ID": game_id, "PLAYER_ID": player_id, "TEAM_ID": team_id, "tag_type": "sprint", "start_timestep": int(timesteps[start]), "end_timestep": int(timesteps[stop - 1]), "value": round(float(speed[start:stop].max()), 1), "unit": "km/h", "distance_m": round(float(step_distance[start:stop].sum()), 1)})
        cumulative = np.cumsum(step_distance)
        for milestone in np.arange(distance_step_m, cumulative[-1] + 1e-9, distance_step_m) if len(cumulative) else []:
            at = int(np.searchsorted(cumulative, milestone))
            tags.append({"GAME_ID": game_id, "PLAYER_ID": player_id, "TEAM_ID": team_id, "tag_type": "distance_milestone", "start_timestep": int(timesteps[at]), "end_timestep": int(timesteps[at]), "value": float(milestone), "unit": "m", "distance_m": float(milestone)})
    columns = ["GAME_ID", "PLAYER_ID", "TEAM_ID", "tag_type", "start_timestep", "end_timestep", "value", "unit", "distance_m"]
    return pd.DataFrame(tags, columns=columns)


def match_rhythm(possession_frame: pd.DataFrame, events: pd.DataFrame, attacking_directions: Mapping[Any, float], window: int = 150) -> pd.DataFrame:
    """Rolling windows of possession, territory and control-vs-chaos for both teams."""
    ball = possession_frame[possession_frame["PLAYER_ID"] == -1][["GAME_ID", "TIMESTEP", "LOCATION_X", "owner_team_id", "horizontal_speed"]].copy()
    ball["window"] = ball["TIMESTEP"] // window
    team0_sign = _direction(attacking_directions, 0) or 1.0
    rows: list[dict[str, Any]] = []
    for (game_id, window_index), part in ball.groupby(["GAME_ID", "window"], sort=True):
        owned = part["owner_team_id"].dropna()
        share0 = float((owned == 0).mean()) if len(owned) else 0.5
        territory0 = float(((part["LOCATION_X"] / 120.0) if team0_sign > 0 else (1.0 - part["LOCATION_X"] / 120.0)).clip(0, 1).mean())
        lo, hi = int(part["TIMESTEP"].min()), int(part["TIMESTEP"].max())
        in_window = events[(events["GAME_ID"] == game_id) & events["start_timestep"].between(lo, hi)] if not events.empty else events
        counts = {team: {kind: int(((in_window["TEAM_ID"] == team) & (in_window["event_type"] == kind)).sum()) for kind in ("pass_attempt", "shot_candidate", "possession_loss_candidate")} for team in (0, 1)} if not in_window.empty else {0: {}, 1: {}}
        completed = sum(counts[team].get("pass_attempt", 0) for team in (0, 1))
        turnovers = sum(counts[team].get("possession_loss_candidate", 0) for team in (0, 1))
        shots0, shots1 = counts[0].get("shot_candidate", 0), counts[1].get("shot_candidate", 0)
        control = completed / max(completed + turnovers, 1)
        momentum0 = 0.4 * (share0 - 0.5) * 2 + 0.4 * (territory0 - 0.5) * 2 + 0.2 * (shots0 - shots1) / max(shots0 + shots1, 1)
        rows.append({
            "GAME_ID": game_id, "window": int(window_index), "start_timestep": lo, "end_timestep": hi,
            "possession_share_team0": round(share0, 3), "territory_team0": round(territory0, 3),
            "completed_passes": completed, "turnovers": turnovers, "shots_team0": shots0, "shots_team1": shots1,
            "control_index": round(control, 3),
            "rhythm": "control" if control >= 0.75 else "chaos" if control < 0.5 else "balanced",
            "ball_pace": round(float(part["horizontal_speed"].mean()), 3),
            "momentum_team0": round(float(np.clip(momentum0, -1, 1)), 3),
        })
    return pd.DataFrame(rows)


def _moment(game_id: Any, start: int, end: int, moment_type: str, team_id: Any, player_id: Any, event_ids: list[str], metrics: dict[str, Any], hint: str, importance: float | None = None) -> dict[str, Any]:
    return {
        "GAME_ID": game_id, "start_timestep": int(start), "end_timestep": int(end), "moment_type": moment_type,
        "TEAM_ID": None if team_id is None or pd.isna(team_id) else int(team_id),
        "PLAYER_ID": None if player_id is None or pd.isna(player_id) else int(player_id),
        "importance": MOMENT_IMPORTANCE[moment_type] if importance is None else importance,
        "event_ids": ";".join(event_ids), "metrics": json.dumps({key: _json_value(value) for key, value in metrics.items()}, sort_keys=True), "fact_hint": hint,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def detect_key_moments(features: pd.DataFrame, events: pd.DataFrame, rhythm: pd.DataFrame, tags: pd.DataFrame, attacking_directions: Mapping[Any, float], min_build_up_passes: int = 6, key_pass_difficulty: float = 65.0) -> pd.DataFrame:
    """Rule-based candidate moments worth an agent's explanation; every moment cites event ids."""
    moments: list[dict[str, Any]] = []
    ball = features[features["PLAYER_ID"] == -1].sort_values(["GAME_ID", "TIMESTEP"])
    for game_id, game_ball in ball.groupby("GAME_ID", sort=False):
        game_events = events[events["GAME_ID"] == game_id].sort_values("start_timestep", kind="mergesort") if not events.empty else events
        for team in (0, 1):
            score = game_ball[f"TEAM_{team}_SCORE"].to_numpy()
            for at in np.flatnonzero(np.diff(score) > 0) + 1:
                timestep = int(game_ball["TIMESTEP"].iloc[at])
                lead_up = game_events[(game_events["TEAM_ID"] == team) & game_events["start_timestep"].between(timestep - 60, timestep) & (known_timestep(game_events) <= timestep) & game_events["event_type"].isin(RELEASE_TYPES | {"carry", "goal_candidate"})]
                shots = lead_up[lead_up["event_type"] == "shot_candidate"]
                scorer_event = shots.iloc[-1] if len(shots) else lead_up.iloc[-1] if len(lead_up) else None
                score_line = f"{int(game_ball['TEAM_0_SCORE'].iloc[at])}-{int(game_ball['TEAM_1_SCORE'].iloc[at])}"
                metrics = {"score_after": score_line, "shot_speed_kmh": None if scorer_event is None or pd.isna(scorer_event.get("ball_speed_kmh")) else float(scorer_event["ball_speed_kmh"])}
                moments.append(_moment(game_id, timestep - 60, timestep, "goal", team, None if scorer_event is None else scorer_event["PLAYER_ID"], lead_up["event_id"].tolist()[-5:], metrics, f"Goal for team {team}; score {score_line}"))

        goal_windows = [(m["start_timestep"], m["end_timestep"]) for m in moments if m["moment_type"] == "goal" and m["GAME_ID"] == game_id]
        for _, shot in game_events[game_events["event_type"] == "shot_candidate"].iterrows():
            if any(lo <= shot["start_timestep"] <= hi for lo, hi in goal_windows):
                continue
            moments.append(_moment(game_id, shot["start_timestep"], shot["end_timestep"], "shot", shot["TEAM_ID"], shot["PLAYER_ID"], [shot["event_id"]], {"shot_speed_kmh": shot.get("ball_speed_kmh"), "distance_to_goal_line_m": round(abs((120.0 if _direction(attacking_directions, shot["TEAM_ID"]) > 0 else 0.0) - shot["start_x"]), 1)}, "Shot candidate"))

        passes = game_events[game_events["event_type"] == "pass_attempt"]
        for _, completed in passes[passes["pass_difficulty"] >= key_pass_difficulty].iterrows():
            moments.append(_moment(game_id, completed["start_timestep"], completed["end_timestep"], "key_pass", completed["TEAM_ID"], completed["PLAYER_ID"], [completed["event_id"]], {"pass_difficulty": completed["pass_difficulty"], "ball_travel_m": completed["ball_travel_m"], "lane_pressure_m": completed["lane_pressure_m"], "subtype": completed["subtype"]}, "Difficult completed pass"))

        for _, gain in game_events[game_events["event_type"] == "interception_candidate"].iterrows():
            direction = _direction(attacking_directions, gain["TEAM_ID"])
            if direction and np.isfinite(gain["start_x"]) and ((gain["start_x"] if direction > 0 else 120.0 - gain["start_x"]) >= 80.0):
                moments.append(_moment(game_id, gain["start_timestep"], gain["end_timestep"], "high_turnover", gain["TEAM_ID"], gain["PLAYER_ID"], [gain["event_id"]], {"x_from_own_goal_m": round(gain["start_x"] if direction > 0 else 120.0 - gain["start_x"], 1)}, "Ball won in the final third"))

        for _, carry in game_events[game_events["event_type"].isin({"carry", "dribble_candidate"}) & (game_events["opponents_within_5m"] >= 3)].iterrows():
            moments.append(_moment(game_id, carry["start_timestep"], carry["end_timestep"], "press_escape", carry["TEAM_ID"], carry["PLAYER_ID"], [carry["event_id"]], {"opponents_within_5m": int(carry["opponents_within_5m"])}, "Carried the ball out of a press"))

        run: list[pd.Series] = []
        for _, event in pd.concat([passes, game_events[game_events["event_type"].isin({"possession_loss_candidate", "shot_candidate", "boundary_exit_candidate"})]]).sort_values("start_timestep", kind="mergesort").iterrows():
            if event["event_type"] == "pass_attempt" and (not run or run[-1]["TEAM_ID"] == event["TEAM_ID"]):
                run.append(event)
                continue
            if len(run) >= min_build_up_passes:
                moments.append(_moment(game_id, run[0]["start_timestep"], run[-1]["end_timestep"], "sustained_build_up", run[0]["TEAM_ID"], None, [item["event_id"] for item in run], {"consecutive_passes": len(run), "players_involved": len({item["PLAYER_ID"] for item in run})}, f"{len(run)} consecutive completed passes"))
            run = [event] if event["event_type"] == "pass_attempt" else []
        if len(run) >= min_build_up_passes:
            moments.append(_moment(game_id, run[0]["start_timestep"], run[-1]["end_timestep"], "sustained_build_up", run[0]["TEAM_ID"], None, [item["event_id"] for item in run], {"consecutive_passes": len(run), "players_involved": len({item["PLAYER_ID"] for item in run})}, f"{len(run)} consecutive completed passes"))

        game_rhythm = rhythm[rhythm["GAME_ID"] == game_id].sort_values("window") if not rhythm.empty else rhythm
        for (_, before), (_, after) in zip(game_rhythm.iterrows(), game_rhythm.iloc[1:].iterrows()):
            if np.sign(before["momentum_team0"]) != np.sign(after["momentum_team0"]) and abs(after["momentum_team0"]) >= 0.25:
                moments.append(_moment(game_id, after["start_timestep"], after["end_timestep"], "momentum_shift", 0 if after["momentum_team0"] > 0 else 1, None, [], {"momentum_before": before["momentum_team0"], "momentum_after": after["momentum_team0"], "possession_share_team0": after["possession_share_team0"], "territory_team0": after["territory_team0"]}, "Momentum swung"))
            if before["rhythm"] != after["rhythm"] and "chaos" in (before["rhythm"], after["rhythm"]):
                moments.append(_moment(game_id, after["start_timestep"], after["end_timestep"], "rhythm_change", None, None, [], {"from": before["rhythm"], "to": after["rhythm"], "control_index": after["control_index"], "turnovers": after["turnovers"]}, f"Game moved from {before['rhythm']} to {after['rhythm']}"))

        sprints = tags[(tags["GAME_ID"] == game_id) & (tags["tag_type"] == "sprint")] if not tags.empty else pd.DataFrame(columns=["TEAM_ID", "value"])
        for team, team_sprints in sprints.groupby("TEAM_ID"):
            best = team_sprints.loc[team_sprints["value"].idxmax()]
            moments.append(_moment(game_id, best["start_timestep"], best["end_timestep"], "top_speed", team, best["PLAYER_ID"], [], {"top_speed_kmh": best["value"], "sprint_distance_m": best["distance_m"]}, f"Top speed {best['value']} km/h"))

    columns = ["GAME_ID", "start_timestep", "end_timestep", "moment_type", "TEAM_ID", "PLAYER_ID", "importance", "event_ids", "metrics", "fact_hint"]
    result = pd.DataFrame(moments, columns=columns)
    if result.empty:
        return result.assign(moment_id=pd.Series(dtype=str))
    result = result.sort_values(["GAME_ID", "start_timestep", "importance"], ascending=[True, True, False], kind="mergesort").reset_index(drop=True)
    result.insert(0, "moment_id", [f"M{row.GAME_ID}-{index:04d}" for index, row in enumerate(result.itertuples(index=False), start=1)])
    return result


def player_match_intel(events: pd.DataFrame, features: pd.DataFrame, tags: pd.DataFrame, seconds_per_timestep: float, roster: Mapping[str, Any]) -> pd.DataFrame:
    """Per-player per-game broadcast stats built from enriched events and tracking."""
    players = features[features["PLAYER_ID"] != -1]
    base = players.groupby(["GAME_ID", "PLAYER_ID", "TEAM_ID"], as_index=False).agg(distance_m=("distance_travelled", "sum"), top_speed_kmh=("horizontal_speed", "max"))
    base["top_speed_kmh"] = (base["top_speed_kmh"] * 3.6 / seconds_per_timestep).round(1)
    base["distance_m"] = base["distance_m"].round(1)
    keys = ["GAME_ID", "PLAYER_ID"]
    if not events.empty:
        counts = events.pivot_table(index=keys, columns="event_type", values="event_id", aggfunc="count", fill_value=0)
        stats = pd.DataFrame(index=counts.index)
        get = lambda name: counts[name] if name in counts.columns else 0
        stats["passes_completed"] = get("pass_attempt")
        stats["passes_attempted"] = stats["passes_completed"] + get("incomplete_pass_candidate")
        stats["pass_accuracy"] = (stats["passes_completed"] / stats["passes_attempted"].replace(0, np.nan)).round(3)
        stats["shots"] = get("shot_candidate")
        stats["interceptions"] = get("interception_candidate")
        stats["possession_losses"] = get("possession_loss_candidate")
        stats["carries"] = get("carry")
        stats["dribbles"] = get("dribble_candidate")
        pass_rows = events[events["event_type"].isin(PASS_TYPES)]
        stats["avg_pass_difficulty"] = pass_rows.groupby(keys)["pass_difficulty"].mean().round(1)
        stats["hard_passes_completed"] = pass_rows[(pass_rows["event_type"] == "pass_attempt") & (pass_rows["pass_difficulty"] >= 60)].groupby(keys).size()
        stats["progressive_passes"] = pass_rows[(pass_rows["event_type"] == "pass_attempt") & (pass_rows["progression_m"] >= 10)].groupby(keys).size()
        stats["max_shot_speed_kmh"] = events[events["event_type"] == "shot_candidate"].groupby(keys)["ball_speed_kmh"].max()
        stats["progressive_carries"] = events[(events["event_type"] == "carry") & (events["subtype"] == "progressive")].groupby(keys).size()
        base = base.merge(stats.reset_index(), on=keys, how="left")
    if not tags.empty:
        base = base.merge(tags[tags["tag_type"] == "sprint"].groupby(keys).size().rename("sprints").reset_index(), on=keys, how="left")
    count_columns = [c for c in ["passes_completed", "passes_attempted", "shots", "interceptions", "possession_losses", "carries", "dribbles", "hard_passes_completed", "progressive_passes", "progressive_carries", "sprints"] if c in base.columns]
    base[count_columns] = base[count_columns].fillna(0).astype(int)
    people = roster.get("players", {})
    clubs = roster.get("clubs", {})
    base.insert(2, "player_name", base["PLAYER_ID"].map(lambda pid: people.get(str(int(pid)), {}).get("name")))
    base.insert(4, "club_name", base["TEAM_ID"].map(lambda tid: clubs.get(str(int(tid)), {}).get("name")))
    base.insert(5, "role", base["PLAYER_ID"].map(lambda pid: people.get(str(int(pid)), {}).get("role")))
    return base
