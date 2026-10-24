from __future__ import annotations

import numpy as np
import pandas as pd


def compute_tracking_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute basic movement and spatial features for each entity-frame."""
    work = frame.copy().sort_values(["GAME_ID", "TIMESTEP", "PLAYER_ID"], kind="mergesort").reset_index(drop=True)
    if "PLAYER_ID" not in work.columns:
        raise ValueError("Expected PLAYER_ID in tracking frame")

    work["is_player"] = work["PLAYER_ID"] != -1
    vertical_velocity = work["VELOCITY_Z"].where(~(work["is_player"] & work["VELOCITY_Z"].eq(-1.0)))
    vertical_location = work["LOCATION_Z"].where(~(work["is_player"] & work["LOCATION_Z"].eq(-1.0)))
    work["horizontal_velocity"] = np.hypot(work["VELOCITY_X"], work["VELOCITY_Y"])
    work["speed_3d"] = np.hypot(work["horizontal_velocity"], vertical_velocity.fillna(0.0))
    work["movement_direction_deg"] = np.degrees(np.arctan2(work["VELOCITY_Y"].fillna(0.0), work["VELOCITY_X"].fillna(0.0)))
    work["movement_direction_rad"] = np.arctan2(work["VELOCITY_Y"].fillna(0.0), work["VELOCITY_X"].fillna(0.0))

    groupby = work.groupby(["GAME_ID", "PLAYER_ID"], sort=False)
    work["dt_seconds"] = groupby["TIMESTEP"].diff()
    work["timestep_gap"] = work["dt_seconds"].ne(1.0)
    work["dx"] = groupby["LOCATION_X"].diff().where(~work["timestep_gap"], 0.0).fillna(0.0)
    work["dy"] = groupby["LOCATION_Y"].diff().where(~work["timestep_gap"], 0.0).fillna(0.0)
    work["dz"] = vertical_location.groupby([work["GAME_ID"], work["PLAYER_ID"]], sort=False).diff().where(~work["timestep_gap"], 0.0).fillna(0.0)
    work["dt_seconds"] = work["dt_seconds"].where(~work["timestep_gap"], 1.0).fillna(1.0)
    work["distance_travelled"] = np.hypot(work["dx"], work["dy"]).fillna(0.0)
    work["finite_difference_speed"] = np.where(work["dt_seconds"] > 0, work["distance_travelled"] / work["dt_seconds"], 0.0)
    work["finite_difference_velocity_x"] = work["dx"] / work["dt_seconds"]
    work["finite_difference_velocity_y"] = work["dy"] / work["dt_seconds"]
    player_reset = work["is_player"] & work["distance_travelled"].gt(3.0)
    ball_reset = ~work["is_player"] & work["distance_travelled"].gt(10.0)
    work["tracking_discontinuity"] = player_reset | ball_reset
    diagnostic_columns = ["dx", "dy", "dz", "distance_travelled", "finite_difference_speed", "finite_difference_velocity_x", "finite_difference_velocity_y"]
    work.loc[work["tracking_discontinuity"], diagnostic_columns] = 0.0
    valid_horizontal_velocity = np.isfinite(work["VELOCITY_X"]) & np.isfinite(work["VELOCITY_Y"])
    work["horizontal_speed"] = work["horizontal_velocity"].where(valid_horizontal_velocity, work["finite_difference_speed"]).fillna(0.0)

    ball_rows = work[work["PLAYER_ID"] == -1][["GAME_ID", "TIMESTEP", "LOCATION_X", "LOCATION_Y", "LOCATION_Z"]].rename(
        columns={"LOCATION_X": "ball_x", "LOCATION_Y": "ball_y", "LOCATION_Z": "ball_z"}
    )
    work = work.merge(ball_rows, on=["GAME_ID", "TIMESTEP"], how="left", suffixes=("", "_ball"))
    work["distance_to_ball"] = np.hypot(work["LOCATION_X"] - work["ball_x"], work["LOCATION_Y"] - work["ball_y"])
    player_vertical = work["LOCATION_Z"].where(~(work["is_player"] & work["LOCATION_Z"].eq(-1.0)), work["ball_z"])
    work["distance_to_ball_3d"] = np.sqrt(
        (work["LOCATION_X"] - work["ball_x"]) ** 2
        + (work["LOCATION_Y"] - work["ball_y"]) ** 2
        + (player_vertical.fillna(0.0) - work["ball_z"].fillna(0.0)) ** 2
    )

    work["delta_velocity_x"] = groupby["VELOCITY_X"].diff().where(~work["timestep_gap"] & ~work["tracking_discontinuity"], 0.0).fillna(0.0)
    work["delta_velocity_y"] = groupby["VELOCITY_Y"].diff().where(~work["timestep_gap"] & ~work["tracking_discontinuity"], 0.0).fillna(0.0)
    work["acceleration_magnitude"] = np.hypot(work["delta_velocity_x"], work["delta_velocity_y"]) / work["dt_seconds"].replace(0, 1.0)
    teammate, opponent = _nearest_player_distances(work[work["is_player"]])
    work["nearest_teammate_distance"] = teammate.reindex(work.index)
    work["nearest_opponent_distance"] = opponent.reindex(work.index)
    return work


def _nearest_player_distances(players: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    teammate = pd.Series(np.nan, index=players.index)
    opponent = pd.Series(np.nan, index=players.index)
    sizes = players.groupby(["GAME_ID", "TIMESTEP"], sort=False).size()
    if sizes.empty:
        return teammate, opponent
    groups = [players] if sizes.nunique() == 1 else [rows for _, rows in players.groupby(["GAME_ID", "TIMESTEP"], sort=False)]
    width = int(sizes.iloc[0])
    for block in groups:
        n = width if len(groups) == 1 else len(block)
        ordered = block.sort_values(["GAME_ID", "TIMESTEP"], kind="mergesort")
        xy = ordered[["LOCATION_X", "LOCATION_Y"]].to_numpy(dtype=float).reshape(-1, n, 2)
        teams = ordered["TEAM_ID"].to_numpy().reshape(-1, n)
        distances = np.hypot(xy[:, :, None, 0] - xy[:, None, :, 0], xy[:, :, None, 1] - xy[:, None, :, 1])
        distances[:, np.arange(n), np.arange(n)] = np.inf
        same_team = teams[:, :, None] == teams[:, None, :]
        nearest_mate = np.where(same_team, distances, np.inf).min(axis=2).ravel()
        nearest_opp = np.where(~same_team, distances, np.inf).min(axis=2).ravel()
        teammate.loc[ordered.index] = np.where(np.isinf(nearest_mate), np.nan, nearest_mate)
        opponent.loc[ordered.index] = np.where(np.isinf(nearest_opp), np.nan, nearest_opp)
    return teammate, opponent
