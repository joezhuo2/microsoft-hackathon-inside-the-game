"""Frame-level movement and spatial features in native source units."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def _time_delta(frames: pd.Series, config: dict) -> pd.Series:
    seconds = config.get("seconds_per_frame")
    return frames * float(seconds) if seconds is not None else frames.astype(float)


def derive_movement_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Derive entity trajectories without differentiating across tracking gaps.

    Distances are native horizontal coordinate units. Time-dependent derivatives
    use seconds only when ``seconds_per_frame`` is configured; otherwise the
    output is explicitly expressed per frame / per frame squared.
    """
    out = df.copy()
    by_entity = out.groupby(["GAME_ID", "PLAYER_ID"], sort=False, dropna=False)
    out["VELOCITY_X_SOURCE"] = pd.to_numeric(out["VELOCITY_X"], errors="coerce")
    out["VELOCITY_Y_SOURCE"] = pd.to_numeric(out["VELOCITY_Y"], errors="coerce")
    out["VELOCITY_Z_SOURCE"] = pd.to_numeric(out["VELOCITY_Z"], errors="coerce")
    vx, vy, vz = out["VELOCITY_X_SOURCE"], out["VELOCITY_Y_SOURCE"], out["VELOCITY_Z_SOURCE"]
    out["SPEED_NATIVE_SOURCE"] = np.sqrt(vx.pow(2) + vy.pow(2))
    out.loc[vx.isna() | vy.isna(), "SPEED_NATIVE_SOURCE"] = np.nan
    out["SPEED_3D_NATIVE_SOURCE"] = np.sqrt(vx.pow(2) + vy.pow(2) + vz.pow(2))
    out.loc[vx.isna() | vy.isna() | vz.isna(), "SPEED_3D_NATIVE_SOURCE"] = np.nan
    epsilon = float(config.get("movement", {}).get("stationary_epsilon", 1e-6))
    out["MOVEMENT_HEADING_RAD"] = np.arctan2(vy, vx)
    out.loc[out["SPEED_NATIVE_SOURCE"] <= epsilon, "MOVEMENT_HEADING_RAD"] = np.nan

    previous_t = by_entity["TIMESTEP"].shift(1)
    dt_frames = out["TIMESTEP"] - previous_t
    max_gap = float(config.get("max_contiguous_gap_frames", 1))
    valid_interval = dt_frames.gt(0) & dt_frames.le(max_gap)
    dt = _time_delta(dt_frames, config)
    dx = out["LOCATION_X"] - by_entity["LOCATION_X"].shift(1)
    dy = out["LOCATION_Y"] - by_entity["LOCATION_Y"].shift(1)
    horizontal_dist = np.sqrt(dx.pow(2) + dy.pow(2))
    out["DISTANCE_STEP_NATIVE"] = horizontal_dist.where(valid_interval)
    out["FINITE_DIFF_VX_NATIVE_PER_TIME_UNIT"] = (dx / dt).where(valid_interval & dt.gt(0))
    out["FINITE_DIFF_VY_NATIVE_PER_TIME_UNIT"] = (dy / dt).where(valid_interval & dt.gt(0))
    out["FINITE_DIFF_SPEED_NATIVE_PER_TIME_UNIT"] = np.sqrt(
        out["FINITE_DIFF_VX_NATIVE_PER_TIME_UNIT"].pow(2) + out["FINITE_DIFF_VY_NATIVE_PER_TIME_UNIT"].pow(2)
    )

    # Finite differences of supplied velocity are a diagnostic; their physical time unit
    # is not declared in the source schema, so they are never labelled m/s^2.
    dt_source = dt_frames.astype(float)
    for source_col, out_col in [("VELOCITY_X_SOURCE", "SUPPLIED_VELOCITY_DERIVATIVE_X_NATIVE_PER_FRAME"), ("VELOCITY_Y_SOURCE", "SUPPLIED_VELOCITY_DERIVATIVE_Y_NATIVE_PER_FRAME")]:
        delta_v = out[source_col] - by_entity[source_col].shift(1)
        out[out_col] = (delta_v / dt_source).where(valid_interval & dt_source.gt(0))
    out["SUPPLIED_VELOCITY_DERIVATIVE_NATIVE_PER_FRAME"] = np.sqrt(
        out["SUPPLIED_VELOCITY_DERIVATIVE_X_NATIVE_PER_FRAME"].pow(2) + out["SUPPLIED_VELOCITY_DERIVATIVE_Y_NATIVE_PER_FRAME"].pow(2)
    )
    # Position-derived acceleration can be physically calibrated when both coordinate scale
    # and frame duration are known. It needs two consecutive valid position-difference steps.
    prev_fd_x = out.groupby(["GAME_ID", "PLAYER_ID"], sort=False)["FINITE_DIFF_VX_NATIVE_PER_TIME_UNIT"].shift(1)
    prev_fd_y = out.groupby(["GAME_ID", "PLAYER_ID"], sort=False)["FINITE_DIFF_VY_NATIVE_PER_TIME_UNIT"].shift(1)
    fd_x_acc = (out["FINITE_DIFF_VX_NATIVE_PER_TIME_UNIT"] - prev_fd_x) / dt
    fd_y_acc = (out["FINITE_DIFF_VY_NATIVE_PER_TIME_UNIT"] - prev_fd_y) / dt
    out["ACCELERATION_NATIVE_PER_TIME2_FROM_POSITION_FD"] = np.sqrt(fd_x_acc.pow(2) + fd_y_acc.pow(2)).where(
        valid_interval & out["FINITE_DIFF_VX_NATIVE_PER_TIME_UNIT"].notna() & prev_fd_x.notna()
        & out["FINITE_DIFF_VY_NATIVE_PER_TIME_UNIT"].notna() & prev_fd_y.notna() & dt.gt(0)
    )
    out["CUMULATIVE_DISTANCE_NATIVE"] = out.groupby(["GAME_ID", "PLAYER_ID"], sort=False)["DISTANCE_STEP_NATIVE"].transform(lambda s: s.fillna(0).cumsum())
    out["GAP_FROM_PREVIOUS_ENTITY_FRAME"] = dt_frames.where(dt_frames.notna())
    out["IS_CONTIGUOUS_ENTITY_INTERVAL"] = valid_interval.fillna(False)

    scale = config.get("coordinate_scale_to_metre")
    seconds = config.get("seconds_per_frame")
    if scale is not None:
        out["DISTANCE_STEP_METRES"] = out["DISTANCE_STEP_NATIVE"] * float(scale)
        out["CUMULATIVE_DISTANCE_METRES"] = out["CUMULATIVE_DISTANCE_NATIVE"] * float(scale)
    else:
        out["DISTANCE_STEP_METRES"] = np.nan
        out["CUMULATIVE_DISTANCE_METRES"] = np.nan
    if scale is not None and seconds is not None:
        out["FINITE_DIFF_SPEED_MPS"] = out["FINITE_DIFF_SPEED_NATIVE_PER_TIME_UNIT"] * float(scale)
        # Since the finite-difference derivative uses seconds when calibrated, scale produces m/s.
        out["ACCELERATION_MPS2_FROM_POSITION_FD"] = out["ACCELERATION_NATIVE_PER_TIME2_FROM_POSITION_FD"] * float(scale)
        out["SPEED_KMH_FROM_FINITE_DIFF"] = out["FINITE_DIFF_SPEED_MPS"] * 3.6
    else:
        out["FINITE_DIFF_SPEED_MPS"] = np.nan
        out["ACCELERATION_MPS2_FROM_POSITION_FD"] = np.nan
        out["SPEED_KMH_FROM_FINITE_DIFF"] = np.nan

    # Smoothing is opt-in and split at every tracking gap; default is raw observations.
    window = int(config.get("movement", {}).get("smoothing_window_frames", 1))
    if window > 1:
        out["_SMOOTH_SEGMENT"] = out.groupby(["GAME_ID", "PLAYER_ID"], sort=False)["TIMESTEP"].diff().fillna(1).ne(1).groupby([out["GAME_ID"], out["PLAYER_ID"]], sort=False).cumsum()
        out["SMOOTHED_SPEED_NATIVE_SOURCE"] = out.groupby(["GAME_ID", "PLAYER_ID", "_SMOOTH_SEGMENT"], sort=False)["SPEED_NATIVE_SOURCE"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        out = out.drop(columns=["_SMOOTH_SEGMENT"])
    else:
        out["SMOOTHED_SPEED_NATIVE_SOURCE"] = out["SPEED_NATIVE_SOURCE"]
    return out


def derive_spatial_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add player-ball, nearest-teammate and nearest-opponent distances.

    Frame-level loops operate on small NumPy arrays; results are written back in
    bulk to avoid millions of individual pandas cell assignments on long matches.
    """
    out = df.copy()
    n_rows = len(out)
    distance_to_ball = np.full(n_rows, np.nan, dtype=float)
    nearest_teammate = np.full(n_rows, np.nan, dtype=float)
    nearest_opponent = np.full(n_rows, np.nan, dtype=float)
    teammates_within = np.full(n_rows, np.nan, dtype=float)
    opponents_within = np.full(n_rows, np.nan, dtype=float)
    radius = config.get("spatial", {}).get("local_density_radius_native")

    xy = out[["LOCATION_X", "LOCATION_Y"]].to_numpy(dtype=float, na_value=np.nan)
    entities = out["ENTITY_TYPE"].to_numpy()
    team_values = pd.to_numeric(out["TEAM_ID"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    grouped_indices = out.groupby(["GAME_ID", "TIMESTEP"], sort=False).indices
    for idx in grouped_indices.values():
        idx = np.asarray(idx, dtype=int)
        frame_entities = entities[idx]
        frame_xy = xy[idx]
        frame_valid = np.isfinite(frame_xy).all(axis=1)
        ball_local = np.flatnonzero(frame_entities == "ball")
        ball_idx = None
        if len(ball_local) == 1 and frame_valid[ball_local[0]]:
            ball_idx = int(idx[ball_local[0]])
            distance_to_ball[ball_idx] = 0.0

        player_local = np.flatnonzero((frame_entities == "player") & frame_valid)
        player_idx = idx[player_local]
        if ball_idx is not None and len(player_idx):
            delta = xy[player_idx] - xy[ball_idx]
            distance_to_ball[player_idx] = np.sqrt((delta * delta).sum(axis=1))
        if len(player_idx) < 2:
            continue

        # Team-relative metrics are left unknown for entities without a usable team ID.
        known_team_mask = np.isfinite(team_values[player_idx]) & (team_values[player_idx] != -1)
        usable_idx = player_idx[known_team_mask]
        if len(usable_idx) < 2:
            continue
        positions = xy[usable_idx]
        teams = team_values[usable_idx]
        pairwise = cdist(positions, positions)
        same_team = teams[:, None] == teams[None, :]
        np.fill_diagonal(same_team, False)
        opponent = teams[:, None] != teams[None, :]
        teammate_dist = np.where(same_team, pairwise, np.inf).min(axis=1)
        opponent_dist = np.where(opponent, pairwise, np.inf).min(axis=1)
        teammate_dist[~np.isfinite(teammate_dist)] = np.nan
        opponent_dist[~np.isfinite(opponent_dist)] = np.nan
        nearest_teammate[usable_idx] = teammate_dist
        nearest_opponent[usable_idx] = opponent_dist
        if radius is not None:
            teammates_within[usable_idx] = (same_team & (pairwise <= float(radius))).sum(axis=1).astype(float)
            opponents_within[usable_idx] = (opponent & (pairwise <= float(radius))).sum(axis=1).astype(float)

    out["DISTANCE_TO_BALL_NATIVE"] = distance_to_ball
    out["NEAREST_TEAMMATE_DISTANCE_NATIVE"] = nearest_teammate
    out["NEAREST_OPPONENT_DISTANCE_NATIVE"] = nearest_opponent
    out["TEAMMATES_WITHIN_CONFIGURED_RADIUS"] = teammates_within
    out["OPPONENTS_WITHIN_CONFIGURED_RADIUS"] = opponents_within

    team_numeric = pd.to_numeric(out["TEAM_ID"], errors="coerce")
    attacking_direction = config.get("pitch", {}).get("attacking_direction_by_team", {})
    if attacking_direction:
        direction_map = {float(k): float(v) for k, v in attacking_direction.items()}
        out["ATTACKING_DIRECTION_X"] = team_numeric.map(direction_map).where(team_numeric.ne(-1), np.nan)
        out["DIRECTIONAL_X_NATIVE"] = out["LOCATION_X"] * out["ATTACKING_DIRECTION_X"]
    else:
        out["ATTACKING_DIRECTION_X"] = np.nan
        out["DIRECTIONAL_X_NATIVE"] = np.nan

    goal_map = config.get("pitch", {}).get("goal_centres_native_by_team", {})
    out["DISTANCE_TO_ATTACKING_GOAL_NATIVE"] = np.nan
    for team_key, centre in goal_map.items():
        gx, gy = map(float, centre)
        mask = team_numeric.eq(float(team_key)) & out["LOCATION_X"].notna() & out["LOCATION_Y"].notna()
        out.loc[mask, "DISTANCE_TO_ATTACKING_GOAL_NATIVE"] = np.hypot(out.loc[mask, "LOCATION_X"] - gx, out.loc[mask, "LOCATION_Y"] - gy)

    posts_map = config.get("pitch", {}).get("goal_posts_native_by_team", {})
    out["DISTANCE_TO_GOAL_POST_1_NATIVE"] = np.nan
    out["DISTANCE_TO_GOAL_POST_2_NATIVE"] = np.nan
    out["ANGLE_TO_ATTACKING_GOAL_RAD"] = np.nan
    for team_key, posts in posts_map.items():
        (x1, y1), (x2, y2) = posts
        mask = team_numeric.eq(float(team_key)) & out["LOCATION_X"].notna() & out["LOCATION_Y"].notna()
        px, py = out.loc[mask, "LOCATION_X"], out.loc[mask, "LOCATION_Y"]
        vx1, vy1 = float(x1) - px, float(y1) - py
        vx2, vy2 = float(x2) - px, float(y2) - py
        d1, d2 = np.hypot(vx1, vy1), np.hypot(vx2, vy2)
        out.loc[mask, "DISTANCE_TO_GOAL_POST_1_NATIVE"] = d1
        out.loc[mask, "DISTANCE_TO_GOAL_POST_2_NATIVE"] = d2
        denom = d1 * d2
        cosine = ((vx1 * vx2 + vy1 * vy2) / denom.replace(0, np.nan)).clip(-1.0, 1.0)
        out.loc[mask, "ANGLE_TO_ATTACKING_GOAL_RAD"] = np.arccos(cosine)
    return out


def derive_tracking_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Convenience function that derives movement and spatial metrics."""
    return derive_spatial_features(derive_movement_features(df, config), config)
