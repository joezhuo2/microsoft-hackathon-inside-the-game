"""Team-match summaries and shape descriptors."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_team_match_stats(features: pd.DataFrame, frames: pd.DataFrame, events: pd.DataFrame, chains: pd.DataFrame, config: dict) -> pd.DataFrame:
    records = []
    team_rows = features[(features["ENTITY_TYPE"].eq("player")) & features["TEAM_ID"].notna() & features["TEAM_ID"].ne(-1)]
    known_frames = frames[frames["possession_state"].eq("observed")]
    seconds = config.get("seconds_per_frame")
    for game_id, game_players in team_rows.groupby("GAME_ID", sort=True):
        all_frames = frames[frames["GAME_ID"].eq(game_id)]
        game_events = events[events["GAME_ID"].eq(game_id)] if not events.empty else events
        game_chains = chains[chains["GAME_ID"].eq(game_id)] if not chains.empty else chains
        known = known_frames[known_frames["GAME_ID"].eq(game_id)]
        known_count = len(known)
        unknown_count = int((all_frames["possession_state"] != "observed").sum())
        teams = sorted(game_players["TEAM_ID"].dropna().unique().tolist())
        for team in teams:
            subset = game_players[game_players["TEAM_ID"].eq(team)]
            own_known = known[known["possession_team_id"].eq(team)]
            ev = game_events[game_events["TEAM_ID"].eq(team)] if not game_events.empty else game_events
            chains_team = game_chains[game_chains["TEAM_ID"].eq(team)] if not game_chains.empty else game_chains
            type_counts = ev["event_type"].value_counts().to_dict() if not ev.empty else {}
            turnovers_against_team = int(game_events[
                game_events["event_type"].isin(["interception_candidate", "possession_change_candidate"])
                & game_events["RELATED_TEAM_ID"].eq(team)
            ].shape[0]) if not game_events.empty and "RELATED_TEAM_ID" in game_events else 0
            # Team shape per frame; horizontal depth and lateral width are geometry-only summaries.
            per_frame = subset.groupby("TIMESTEP", sort=False).agg(
                min_x=("LOCATION_X", "min"), max_x=("LOCATION_X", "max"),
                min_y=("LOCATION_Y", "min"), max_y=("LOCATION_Y", "max"),
            )
            depth = per_frame["max_x"] - per_frame["min_x"]
            width = per_frame["max_y"] - per_frame["min_y"]
            records.append({
                "GAME_ID": game_id, "TEAM_ID": team,
                "observed_possession_frames": int(len(own_known)),
                "known_possession_frames_all_teams": int(known_count),
                "unknown_or_contested_frames": unknown_count,
                "possession_pct_of_known_owner_frames": (100.0 * len(own_known) / known_count) if known_count else np.nan,
                "possession_time_seconds": len(own_known) * float(seconds) if seconds is not None else np.nan,
                "possession_chains": int(len(chains_team)),
                "mean_chain_duration_seconds": float(chains_team["duration_seconds"].mean()) if not chains_team.empty and chains_team["duration_seconds"].notna().any() else np.nan,
                "pass_candidates": int(type_counts.get("pass_candidate", 0)),
                "interception_candidates": int(type_counts.get("interception_candidate", 0)),
                "turnover_candidates": turnovers_against_team,
                "ball_release_candidates": int(type_counts.get("ball_release_candidate", 0)),
                "carry_candidates": int(type_counts.get("carry_candidate", 0) + type_counts.get("progressive_carry_candidate", 0)),
                "progressive_carry_candidates": int(type_counts.get("progressive_carry_candidate", 0)),
                "score_change_goal_candidates": int(type_counts.get("score_change_goal_candidate", 0)),
                "mean_team_width_native": float(width.mean()) if width.notna().any() else np.nan,
                "median_team_width_native": float(width.median()) if width.notna().any() else np.nan,
                "mean_team_depth_native": float(depth.mean()) if depth.notna().any() else np.nan,
                "median_team_depth_native": float(depth.median()) if depth.notna().any() else np.nan,
                "ppda_proxy": np.nan, "field_tilt_pct": np.nan, "xG": np.nan,
                "entries_final_third": np.nan, "entries_penalty_area": np.nan,
                "metrics_status": "possession based on unique ON_BALL labels; event counts are candidates; pitch-zone and model metrics unavailable without calibration/labels",
            })
    out = pd.DataFrame(records)
    if not out.empty:
        # Invariant for the chosen denominator: team shares sum to 100% over known-owner frames.
        totals = out.groupby("GAME_ID")["possession_pct_of_known_owner_frames"].sum(min_count=1)
        out["possession_share_invariant_passed"] = out["GAME_ID"].map(totals.between(99.999, 100.001))
    return out
