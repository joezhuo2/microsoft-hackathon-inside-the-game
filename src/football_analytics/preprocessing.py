from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_loader import REQUIRED_COLUMNS


def validate_tracking_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean, sort, and validate a tracking dataset."""
    cleaned = frame.copy()
    if not set(REQUIRED_COLUMNS).issubset(cleaned.columns):
        missing = [column for column in REQUIRED_COLUMNS if column not in cleaned.columns]
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = cleaned.sort_values(["GAME_ID", "TIMESTEP", "PLAYER_ID"], kind="mergesort").reset_index(drop=True)

    cleaned["PLAYER_ROLE"] = cleaned["PLAYER_ROLE"].replace({"nan": np.nan, "None": np.nan})

    for column in ["GAME_ID", "TIMESTEP", "TEAM_0_SCORE", "TEAM_1_SCORE", "PLAYER_ID", "TEAM_ID", "LOCATION_X", "LOCATION_Y", "LOCATION_Z", "VELOCITY_X", "VELOCITY_Y", "VELOCITY_Z"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "ON_BALL" in cleaned.columns:
        cleaned["ON_BALL"] = cleaned["ON_BALL"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes", "y", "on"}
            if pd.notna(value) else False
        )

    duplicate_mask = cleaned.duplicated(subset=["GAME_ID", "TIMESTEP", "PLAYER_ID"], keep=False)
    duplicates = cleaned[duplicate_mask].copy()

    ball_mask = cleaned["PLAYER_ID"] == -1
    cleaned.loc[ball_mask & (cleaned["TEAM_ID"] == -1), "PLAYER_ROLE"] = "ball"

    player_mask = cleaned["PLAYER_ID"] != -1
    sentinel = -1.0
    cleaned.loc[player_mask & (cleaned["LOCATION_Z"] == sentinel), "LOCATION_Z"] = np.nan
    cleaned.loc[player_mask & (cleaned["VELOCITY_Z"] == sentinel), "VELOCITY_Z"] = np.nan

    report = {
        "quality_checks": {
            "duplicate_rows": int(len(duplicates)),
            "missing_values": cleaned.isna().sum().to_dict(),
            "players": int((cleaned["PLAYER_ID"] != -1).sum()),
            "ball_rows": int((cleaned["PLAYER_ID"] == -1).sum()),
            "games": int(cleaned["GAME_ID"].nunique()),
            "timesteps": int(cleaned["TIMESTEP"].nunique()),
        },
        "frame_counts_by_game": cleaned.groupby("GAME_ID")["TIMESTEP"].nunique().to_dict(),
        "entity_counts_by_frame": cleaned.groupby(["GAME_ID", "TIMESTEP"]).size().describe().to_dict(),
    }
    return cleaned, report
