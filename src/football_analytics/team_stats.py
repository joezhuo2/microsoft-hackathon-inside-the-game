from __future__ import annotations

import pandas as pd


def aggregate_team_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate simple team-level statistics for each game."""
    rows = frame[frame["PLAYER_ID"] != -1].copy()
    if rows.empty:
        return pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "touches", "distance_total", "player_count"]) 

    stats = rows.groupby(["GAME_ID", "TEAM_ID"], as_index=False).agg(
        touches=("ON_BALL", "sum"),
        distance_total=("distance_travelled", "sum"),
        player_count=("PLAYER_ID", "nunique"),
    )
    return stats
