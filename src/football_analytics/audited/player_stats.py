"""Player-match aggregations with explicit availability semantics."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd


def _count_event(events: pd.DataFrame, game_id, player_id, event_types: set[str], column: str = "PLAYER_ID") -> int:
    if events.empty:
        return 0
    mask = (events["GAME_ID"] == game_id) & events["event_type"].isin(event_types) & pd.to_numeric(events[column], errors="coerce").eq(player_id)
    return int(mask.sum())


def build_player_match_stats(features: pd.DataFrame, frames: pd.DataFrame, events: pd.DataFrame, config: dict) -> pd.DataFrame:
    players = features[features["ENTITY_TYPE"].eq("player")].copy()
    records = []
    seconds = config.get("seconds_per_frame")
    per_game_frame_counts = frames.groupby("GAME_ID")["TIMESTEP"].nunique().to_dict()
    for (game_id, player_id), group in players.groupby(["GAME_ID", "PLAYER_ID"], sort=True):
        roles = group["PLAYER_ROLE"].dropna().astype(str)
        teams = group["TEAM_ID"].dropna().unique().tolist()
        team = teams[0] if len(teams) == 1 else np.nan
        distances = group["DISTANCE_STEP_NATIVE"]
        total_dist = distances.sum(min_count=1)
        speed = group["SPEED_NATIVE_SOURCE"]
        max_speed = speed.max(skipna=True) if speed.notna().any() else np.nan
        owner_frames = int((group["possession_state"].eq("observed") & group["possession_owner_id"].eq(player_id)).sum()) if "possession_state" in group else 0
        total_frames = int(group["TIMESTEP"].nunique())
        completed_pass_candidates = _count_event(events, game_id, player_id, {"pass_candidate"})
        release_candidates = _count_event(events, game_id, player_id, {"ball_release_candidate"})
        reception_candidates = (
            _count_event(events, game_id, player_id, {"pass_candidate"}, column="RELATED_PLAYER_ID")
            + _count_event(events, game_id, player_id, {"touch_or_reception_candidate"}, column="PLAYER_ID")
        )
        event_counts = {
            # A direct same-team owner transition is a candidate completed pass; a ball release
            # into an unknown owner state is only a possible attempt and has uncertain outcome.
            "pass_attempt_candidates": completed_pass_candidates + release_candidates,
            "completed_pass_candidates": completed_pass_candidates,
            "ball_release_candidates": release_candidates,
            "reception_candidates": reception_candidates,
            "carry_candidates": _count_event(events, game_id, player_id, {"carry_candidate", "progressive_carry_candidate"}),
            "progressive_carry_candidates": _count_event(events, game_id, player_id, {"progressive_carry_candidate"}),
            "interception_candidates": _count_event(events, game_id, player_id, {"interception_candidate"}, column="PLAYER_ID"),
            "turnover_candidates": _count_event(events, game_id, player_id, {"interception_candidate", "possession_change_candidate"}, column="RELATED_PLAYER_ID"),
        }
        # The model currently infers only pass candidates that have a subsequent same-team owner.
        # Thus completed_pass_candidates is a subset of candidate passes, not a complete event truth set.
        frame_count = per_game_frame_counts.get(game_id, np.nan)
        record = {
            "GAME_ID": game_id, "PLAYER_ID": player_id, "TEAM_ID": team,
            "PLAYER_ROLE": roles.mode().iloc[0] if not roles.empty and not roles.mode().empty else None,
            "tracked_frames": total_frames,
            "tracking_coverage_pct": (100.0 * total_frames / frame_count) if frame_count else np.nan,
            "distance_travelled_native": float(total_dist) if pd.notna(total_dist) else np.nan,
            "distance_travelled_metres": float(group["DISTANCE_STEP_METRES"].sum(min_count=1)) if group["DISTANCE_STEP_METRES"].notna().any() else np.nan,
            "max_observed_speed_native_source_units": float(max_speed) if pd.notna(max_speed) else np.nan,
            "max_speed_kmh": float(group["SPEED_KMH_FROM_FINITE_DIFF"].max()) if group["SPEED_KMH_FROM_FINITE_DIFF"].notna().any() else np.nan,
            "mean_nearest_opponent_distance_native": float(group["NEAREST_OPPONENT_DISTANCE_NATIVE"].mean()) if group["NEAREST_OPPONENT_DISTANCE_NATIVE"].notna().any() else np.nan,
            "mean_nearest_teammate_distance_native": float(group["NEAREST_TEAMMATE_DISTANCE_NATIVE"].mean()) if group["NEAREST_TEAMMATE_DISTANCE_NATIVE"].notna().any() else np.nan,
            "observed_control_frames": owner_frames,
            "observed_control_seconds": owner_frames * float(seconds) if seconds is not None else np.nan,
            "touches_or_receptions_candidates": event_counts["reception_candidates"],
            "pass_attempt_candidates": event_counts["pass_attempt_candidates"],
            "completed_pass_candidates": event_counts["completed_pass_candidates"],
            # Intentionally unavailable: release-to-reception linking is not yet robust enough
            # to guarantee a complete pass-attempt denominator.
            "pass_completion_pct": np.nan,
            "pass_completion_pct_unavailable_reason": "Candidate events do not yet provide a reliable attempted-pass denominator.",
            "interception_candidates": event_counts["interception_candidates"],
            "turnover_candidates": event_counts["turnover_candidates"],
            "carry_candidates": event_counts["carry_candidates"],
            "progressive_carry_candidates": event_counts["progressive_carry_candidates"],
            # Deliberately null until calibrated geometry / labelled events exist.
            "per_90_distance_native": np.nan,
            "shot_attempts": np.nan, "goals_attributed_to_player": np.nan, "xG": np.nan,
            "take_ons_attempted": np.nan, "take_ons_won": np.nan, "sprint_count": np.nan,
            "high_speed_running_distance_metres": np.nan,
            "metrics_status": "movement measured where records are valid; possession observed; events heuristic; uncalibrated/unsupported metrics null",
        }
        record.update(event_counts)
        records.append(record)
    return pd.DataFrame(records)
