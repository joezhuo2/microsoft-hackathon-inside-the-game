"""Auditable candidate event inference from tracking and possession transitions."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "GAME_ID", "event_id", "TIMESTEP", "start_timestep", "end_timestep", "event_type",
    "TEAM_ID", "PLAYER_ID", "RELATED_PLAYER_ID", "RELATED_TEAM_ID", "start_x", "start_y", "end_x", "end_y",
    "event_outcome", "confidence", "evidence_json", "detection_rule", "rule_version",
    "alternative_interpretations",
]


def _row_position(row: pd.Series) -> tuple[float, float]:
    x, y = row.get("BALL_X", np.nan), row.get("BALL_Y", np.nan)
    return (float(x) if pd.notna(x) else np.nan, float(y) if pd.notna(y) else np.nan)


def detect_events(frames: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Detect score changes, owner transitions, pass candidates and carry candidates.

    Labels are deliberately phrased as candidates: this source does not provide
    event labels, body contact, intended receivers, or calibrated pitch geometry.
    """
    if frames.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    event_cfg = config.get("events", {})
    min_pass = float(event_cfg.get("min_pass_distance_native", 2.0))
    min_carry = float(event_cfg.get("min_carry_distance_native", 1.5))
    progressive_threshold = float(event_cfg.get("progressive_distance_native", 5.0))
    direction_map = {str(k): float(v) for k, v in config.get("pitch", {}).get("attacking_direction_by_team", {}).items()}
    events: list[dict[str, Any]] = []

    def add(game_id: Any, timestep: float, start_t: float, end_t: float, event_type: str,
            team_id: Any = np.nan, player_id: Any = np.nan, related_player_id: Any = np.nan,
            related_team_id: Any = np.nan,
            start_xy: tuple[float, float] = (np.nan, np.nan), end_xy: tuple[float, float] = (np.nan, np.nan),
            outcome: str = "unknown", confidence: str = "low", evidence: dict[str, Any] | None = None,
            rule: str = "tracking_candidate_v1", alternatives: str = "") -> None:
        eid = f"{game_id}:event:{len(events) + 1:07d}"
        events.append({
            "GAME_ID": game_id, "event_id": eid, "TIMESTEP": timestep, "start_timestep": start_t,
            "end_timestep": end_t, "event_type": event_type, "TEAM_ID": team_id,
            "PLAYER_ID": player_id, "RELATED_PLAYER_ID": related_player_id,
            "RELATED_TEAM_ID": related_team_id,
            "start_x": start_xy[0], "start_y": start_xy[1], "end_x": end_xy[0], "end_y": end_xy[1],
            "event_outcome": outcome, "confidence": confidence,
            "evidence_json": json.dumps(evidence or {}, sort_keys=True, default=str),
            "detection_rule": rule, "rule_version": "1.0.0",
            "alternative_interpretations": alternatives,
        })

    for game_id, raw_game in frames.groupby("GAME_ID", sort=False):
        game = raw_game.sort_values("TIMESTEP", kind="mergesort").reset_index(drop=True)
        # Score changes are the strongest event evidence in this schema. A score change
        # confirms that the displayed score changed, but does not prove scorer identity.
        score_team_map = _score_team_mapping(game_id, config, game)
        for i in range(1, len(game)):
            prev, cur = game.iloc[i - 1], game.iloc[i]
            if float(cur["TIMESTEP"]) - float(prev["TIMESTEP"]) != 1:
                continue
            for score_col, score_slot in [("TEAM_0_SCORE_FRAME", 0), ("TEAM_1_SCORE_FRAME", 1)]:
                a, b = prev[score_col], cur[score_col]
                if pd.notna(a) and pd.notna(b) and float(b) > float(a):
                    delta = float(b) - float(a)
                    scoring_team = score_team_map.get(score_slot, np.nan)
                    add(game_id, float(cur["TIMESTEP"]), float(prev["TIMESTEP"]), float(cur["TIMESTEP"]),
                        "score_change_goal_candidate", team_id=scoring_team,
                        player_id=np.nan, start_xy=_row_position(prev), end_xy=_row_position(cur),
                        outcome="score_increased", confidence="high",
                        evidence={"score_column": score_col, "score_before": float(a), "score_after": float(b), "score_delta": delta,
                                  "team_attribution_method": "configured-or-inferred TEAM_0/TEAM_1 mapping"},
                        rule="score_transition_v1", alternatives="Exact goal time and scorer may differ from the score-transition frame.")

        # Ownership changes; pass/interception labels are inferred from owner transition + ball travel.
        for i in range(1, len(game)):
            prev, cur = game.iloc[i - 1], game.iloc[i]
            if float(cur["TIMESTEP"]) - float(prev["TIMESTEP"]) != 1:
                continue
            prev_state, cur_state = prev["possession_state"], cur["possession_state"]
            prev_owner, cur_owner = prev["possession_owner_id"], cur["possession_owner_id"]
            if prev_state == "observed" and cur_state != "observed" and pd.notna(prev_owner):
                x0, y0 = _row_position(prev); x1, y1 = _row_position(cur)
                travel = float(np.hypot(x1 - x0, y1 - y0)) if np.isfinite([x0, y0, x1, y1]).all() else np.nan
                if pd.notna(travel) and travel >= min_pass:
                    add(game_id, float(cur["TIMESTEP"]), float(prev["TIMESTEP"]), float(cur["TIMESTEP"]),
                        "ball_release_candidate", team_id=prev["possession_team_id"], player_id=prev_owner,
                        start_xy=(x0, y0), end_xy=(x1, y1), outcome="control_lost_while_ball_moved", confidence="low",
                        evidence={"ball_displacement_native": travel, "previous_owner": int(prev_owner), "current_state": cur_state},
                        rule="ball_release_v1", alternatives="Could be pass, deflection, clearance, shot, or tracking error.")

            if prev_state == "observed" and cur_state == "observed" and pd.notna(prev_owner) and pd.notna(cur_owner) and int(prev_owner) != int(cur_owner):
                x0, y0 = _row_position(prev); x1, y1 = _row_position(cur)
                travel = float(np.hypot(x1 - x0, y1 - y0)) if np.isfinite([x0, y0, x1, y1]).all() else np.nan
                moved = pd.notna(travel) and travel >= min_pass
                same_team = prev["possession_team_id"] == cur["possession_team_id"]
                if same_team and moved:
                    # Primary actor is the passer; counterpart is the teammate receiver.
                    etype, outcome, confidence, rule = "pass_candidate", "completed_pass_candidate", "medium", "owner_change_plus_ball_displacement_v1"
                    event_team, event_player, related_player, related_team = prev["possession_team_id"], prev_owner, cur_owner, cur["possession_team_id"]
                elif not same_team and moved:
                    # Primary actor is the new owner (candidate interceptor); counterpart lost control.
                    etype, outcome, confidence, rule = "interception_candidate", "opponent_control_after_ball_travel", "medium", "opponent_owner_change_plus_ball_displacement_v1"
                    event_team, event_player, related_player, related_team = cur["possession_team_id"], cur_owner, prev_owner, prev["possession_team_id"]
                elif same_team:
                    etype, outcome, confidence, rule = "touch_or_reception_candidate", "owner_changed_without_strong_pass_evidence", "low", "owner_change_v1"
                    event_team, event_player, related_player, related_team = cur["possession_team_id"], cur_owner, prev_owner, prev["possession_team_id"]
                else:
                    etype, outcome, confidence, rule = "possession_change_candidate", "opponent_owner_change_without_strong_interception_evidence", "low", "opponent_owner_change_v1"
                    event_team, event_player, related_player, related_team = cur["possession_team_id"], cur_owner, prev_owner, prev["possession_team_id"]
                add(game_id, float(cur["TIMESTEP"]), float(prev["TIMESTEP"]), float(cur["TIMESTEP"]), etype,
                    team_id=event_team, player_id=event_player, related_player_id=related_player, related_team_id=related_team,
                    start_xy=(x0, y0), end_xy=(x1, y1), outcome=outcome, confidence=confidence,
                    evidence={"previous_owner": int(prev_owner), "new_owner": int(cur_owner),
                              "previous_team": prev["possession_team_id"], "new_team": cur["possession_team_id"],
                              "ball_displacement_native": travel, "minimum_pass_distance_native": min_pass,
                              "possession_evidence": "ON_BALL labels"}, rule=rule,
                    alternatives="Owner changes and ball displacement do not prove intended receiver or pass intent." if moved else "Could be a touch, deflection, or control-label change.")

        # Runs of the same observed owner on contiguous frames can form carry candidates.
        owners = game["possession_owner_id"].to_numpy()
        states = game["possession_state"].to_numpy()
        timesteps = game["TIMESTEP"].to_numpy(dtype=float)
        start = 0
        while start < len(game):
            if states[start] != "observed" or pd.isna(owners[start]):
                start += 1
                continue
            end = start
            while end + 1 < len(game) and states[end + 1] == "observed" and pd.notna(owners[end + 1]) and int(owners[end + 1]) == int(owners[start]) and timesteps[end + 1] - timesteps[end] == 1:
                end += 1
            if end > start:
                first, last = game.iloc[start], game.iloc[end]
                x0, y0 = _row_position(first); x1, y1 = _row_position(last)
                travel = float(np.hypot(x1 - x0, y1 - y0)) if np.isfinite([x0, y0, x1, y1]).all() else np.nan
                if pd.notna(travel) and travel >= min_carry:
                    team = first["possession_team_id"]
                    dx = x1 - x0
                    direction = direction_map.get(str(int(team))) if pd.notna(team) else None
                    progressive = direction is not None and dx * direction >= progressive_threshold
                    add(game_id, float(first["TIMESTEP"]), float(first["TIMESTEP"]), float(last["TIMESTEP"]),
                        "progressive_carry_candidate" if progressive else "carry_candidate", team_id=team, player_id=owners[start],
                        start_xy=(x0, y0), end_xy=(x1, y1), outcome="ball_moved_during_continuous_observed_control",
                        confidence="low", evidence={"ball_displacement_native": travel, "observed_control_frames": int(end - start + 1),
                        "attacking_direction_configured": direction is not None, "progression_threshold_native": progressive_threshold},
                        rule="continuous_owner_ball_displacement_v1",
                        alternatives="Sustained control and ball movement do not by themselves distinguish a carry from stationary touches or a deflection.")
            start = end + 1

    if not events:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    out = pd.DataFrame(events, columns=EVENT_COLUMNS)
    out["TIMESTEP"] = pd.to_numeric(out["TIMESTEP"], errors="coerce")
    return out.sort_values(["GAME_ID", "TIMESTEP", "event_id"], kind="mergesort").reset_index(drop=True)


def _score_team_mapping(game_id: Any, config: dict, game: pd.DataFrame) -> dict[int, Any]:
    configured = config.get("score_team_id_map", {})
    if configured:
        return {int(k): v for k, v in configured.items()}
    # Prefer conventional IDs 0/1; otherwise map score slots to sorted observed team IDs.
    tids = sorted({int(x) for x in game["possession_team_id"].dropna().tolist() if int(x) >= 0})
    if 0 in tids and 1 in tids:
        return {0: 0, 1: 1}
    if len(tids) >= 2:
        return {0: tids[0], 1: tids[1]}
    return {0: 0, 1: 1}
