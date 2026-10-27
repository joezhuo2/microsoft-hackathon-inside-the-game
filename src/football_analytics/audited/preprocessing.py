"""Canonicalize entities and documented sentinel values."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_on_ball(value: object, true_values: list, false_values: list) -> float:
    if pd.isna(value):
        return np.nan
    # Type-insensitive comparisons cover CSVs with values parsed as int or string.
    text = str(value).strip()
    if any(value == v or text == str(v).strip() for v in true_values):
        return 1.0
    if any(value == v or text == str(v).strip() for v in false_values):
        return 0.0
    return np.nan


def preprocess_tracking_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return stable, sorted data with entity type and explicit missing sentinels."""
    out = df.copy()
    out["ENTITY_TYPE"] = np.where(out["PLAYER_ID"].eq(-1) & out["TEAM_ID"].eq(-1), "ball", "player")
    possession_cfg = config.get("possession", {})
    true_values = possession_cfg.get("on_ball_true_values", [1, "1", True, "true"])
    false_values = possession_cfg.get("on_ball_false_values", [0, "0", False, "false"])
    out["ON_BALL_NORMALIZED"] = out["ON_BALL"].map(lambda x: _normalize_on_ball(x, true_values, false_values)).astype(float)

    players = out["ENTITY_TYPE"].eq("player")
    # The supplied dataset documents -1 as a sentinel, not a real player height or vertical velocity.
    out.loc[players & out["LOCATION_Z"].eq(-1), "LOCATION_Z"] = np.nan
    out.loc[players & out["VELOCITY_Z"].eq(-1), "VELOCITY_Z"] = np.nan
    out.loc[out["ENTITY_TYPE"].eq("ball"), "PLAYER_ROLE"] = out.loc[out["ENTITY_TYPE"].eq("ball"), "PLAYER_ROLE"].fillna("ball")
    out = out.sort_values(["GAME_ID", "TIMESTEP", "PLAYER_ID"], kind="mergesort").reset_index(drop=True)
    return out
