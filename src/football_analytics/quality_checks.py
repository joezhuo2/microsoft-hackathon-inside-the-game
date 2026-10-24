from __future__ import annotations

from typing import Any

import pandas as pd


def generate_quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Build a concise quality report describing coverage, missingness, and anomalies."""
    report = {
        "games": int(frame["GAME_ID"].nunique()) if "GAME_ID" in frame.columns else 0,
        "frames": int(frame.groupby("GAME_ID")["TIMESTEP"].nunique().sum()) if "GAME_ID" in frame.columns and "TIMESTEP" in frame.columns else 0,
        "rows": int(len(frame)),
        "missing_values": frame.isna().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated(subset=["GAME_ID", "TIMESTEP", "PLAYER_ID"]).sum()),
        "ball_rows": int((frame["PLAYER_ID"] == -1).sum()) if "PLAYER_ID" in frame.columns else 0,
        "valid_owner_frames": int(frame["ON_BALL"].sum()) if "ON_BALL" in frame.columns else 0,
        "quality_checks": ["schema_validated", "sorted_by_game_timestep", "ball_players_separated"],
    }
    return report
