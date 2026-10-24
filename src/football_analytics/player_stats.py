from __future__ import annotations

import pandas as pd


def aggregate_player_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate common player-level match stats from frame-level tracking features."""
    player_rows = frame[frame["PLAYER_ID"] != -1].copy()
    if player_rows.empty:
        return pd.DataFrame(columns=["GAME_ID", "PLAYER_ID", "TEAM_ID", "touches", "total_distance", "max_speed", "avg_speed"])

    grouped = player_rows.groupby(["GAME_ID", "PLAYER_ID", "TEAM_ID"], as_index=False)
    stats = grouped.agg(
        touches=("ON_BALL", "sum"),
        total_distance=("distance_travelled", "sum"),
        max_speed=("horizontal_speed", "max"),
        avg_speed=("horizontal_speed", "mean"),
    )
    return stats
