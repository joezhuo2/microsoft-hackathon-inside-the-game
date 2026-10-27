"""Observed possession states and possession-chain segmentation."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def build_possession_frame_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create one row per game-frame using ON_BALL as the primary evidence.

    Exactly one positive player label gives an observed owner. Multiple positives
    are ``contested``; no positive label is ``unknown`` (not automatically a loose
    ball or a team without possession).
    """
    rows: list[dict[str, Any]] = []
    scores = []
    for (game_id, timestep), group in df.groupby(["GAME_ID", "TIMESTEP"], sort=True, dropna=False):
        positive = group[(group["ENTITY_TYPE"] == "player") & group["ON_BALL_NORMALIZED"].eq(1)]
        scores0 = group["TEAM_0_SCORE"].dropna().unique()
        scores1 = group["TEAM_1_SCORE"].dropna().unique()
        score0 = float(scores0[0]) if len(scores0) == 1 and group["TEAM_0_SCORE"].notna().all() else np.nan
        score1 = float(scores1[0]) if len(scores1) == 1 and group["TEAM_1_SCORE"].notna().all() else np.nan
        balls = group[group["ENTITY_TYPE"] == "ball"]
        ball_ok = len(balls) == 1
        ball = balls.iloc[0] if ball_ok else None
        if len(positive) == 1:
            owner = positive.iloc[0]
            owner_id, owner_team, state, source = owner["PLAYER_ID"], owner["TEAM_ID"], "observed", "ON_BALL"
        elif len(positive) > 1:
            owner_id, owner_team, state, source = np.nan, np.nan, "contested", "multiple_ON_BALL_labels"
        else:
            owner_id, owner_team, state, source = np.nan, np.nan, "unknown", "no_positive_ON_BALL_label"
        rows.append({
            "GAME_ID": game_id, "TIMESTEP": timestep,
            "possession_owner_id": owner_id, "possession_team_id": owner_team,
            "possession_state": state, "possession_evidence": source,
            "TEAM_0_SCORE_FRAME": score0, "TEAM_1_SCORE_FRAME": score1,
            "BALL_X": float(ball["LOCATION_X"]) if ball_ok and pd.notna(ball["LOCATION_X"]) else np.nan,
            "BALL_Y": float(ball["LOCATION_Y"]) if ball_ok and pd.notna(ball["LOCATION_Y"]) else np.nan,
            "BALL_VX": float(ball["VELOCITY_X"]) if ball_ok and pd.notna(ball["VELOCITY_X"]) else np.nan,
            "BALL_VY": float(ball["VELOCITY_Y"]) if ball_ok and pd.notna(ball["VELOCITY_Y"]) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["GAME_ID", "TIMESTEP"], kind="mergesort").reset_index(drop=True)


def attach_possession_to_entities(features: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Attach frame-level possession state to all entity rows."""
    columns = ["GAME_ID", "TIMESTEP", "possession_owner_id", "possession_team_id", "possession_state", "possession_evidence"]
    return features.merge(frames[columns], on=["GAME_ID", "TIMESTEP"], how="left", validate="many_to_one")


def detect_possession_chains(frames: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Group consecutive, observed same-team control into possession chains."""
    if frames.empty:
        return pd.DataFrame(columns=[
            "GAME_ID", "possession_chain_id", "TEAM_ID", "start_timestep", "end_timestep",
            "observed_frame_count", "duration_seconds", "participants_json", "owner_changes",
            "terminal_state", "reliability_status",
        ])
    seconds = config.get("seconds_per_frame")
    records: list[dict[str, Any]] = []
    for game_id, game in frames.groupby("GAME_ID", sort=False):
        game = game.sort_values("TIMESTEP").reset_index(drop=True)
        run: list[pd.Series] = []

        def flush(items: list[pd.Series], terminal_state: str) -> None:
            if not items:
                return
            team = items[0]["possession_team_id"]
            owners = [int(x) for x in pd.Series([r["possession_owner_id"] for r in items]).dropna().unique().tolist()]
            start, end = float(items[0]["TIMESTEP"]), float(items[-1]["TIMESTEP"])
            observed_count = len(items)
            chain_id = f"{game_id}:chain:{len([r for r in records if r['GAME_ID'] == game_id]) + 1:05d}"
            records.append({
                "GAME_ID": game_id, "possession_chain_id": chain_id, "TEAM_ID": team,
                "start_timestep": start, "end_timestep": end,
                "observed_frame_count": observed_count,
                "duration_seconds": (observed_count * float(seconds)) if seconds is not None else np.nan,
                "participants_json": json.dumps(owners), "owner_changes": max(0, len(owners) - 1),
                "terminal_state": terminal_state,
                "reliability_status": "observed_ON_BALL_team_control; chain boundaries heuristic",
            })

        previous_team: Any = None
        previous_timestep: float | None = None
        for _, row in game.iterrows():
            known = row["possession_state"] == "observed" and pd.notna(row["possession_team_id"])
            timestep = float(row["TIMESTEP"])
            contiguous = previous_timestep is not None and timestep - previous_timestep == 1
            team = row["possession_team_id"] if known else None
            if not known:
                flush(run, str(row["possession_state"]))
                run = []
            elif run and (not contiguous or team != previous_team):
                flush(run, "team_change" if contiguous and team != previous_team else "tracking_gap")
                run = [row]
            else:
                run.append(row)
            previous_team = team
            previous_timestep = timestep
        flush(run, "end_of_observed_data")
    return pd.DataFrame(records)
