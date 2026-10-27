"""Transparent, configurable native-coordinate pitch-control baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_pitch_control_grid(frame: pd.DataFrame, config: dict, bounds_native: list[float] | None = None) -> pd.DataFrame:
    """Estimate which team reaches each grid point first under a constant-speed model.

    Parameters are in native units. The function refuses to guess speed or pitch
    dimensions; set ``pitch_control.max_speed_native_per_frame`` and provide pitch
    bounds (argument or ``pitch.bounds_native``) to enable it. This is a reachability
    baseline, not a calibrated professional pitch-control model.
    """
    pitch_cfg = config.get("pitch_control", {})
    speed = pitch_cfg.get("max_speed_native_per_frame")
    bounds = bounds_native or config.get("pitch", {}).get("bounds_native")
    if speed is None or bounds is None:
        raise ValueError("Pitch control requires max_speed_native_per_frame and bounds_native; no physical assumptions are inferred.")
    if len(bounds) != 4:
        raise ValueError("bounds_native must be [xmin, xmax, ymin, ymax]")
    xmin, xmax, ymin, ymax = map(float, bounds)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("Pitch bounds must have increasing min/max values")
    players = frame[frame["ENTITY_TYPE"].eq("player") & frame["TEAM_ID"].ne(-1)].copy()
    players = players.dropna(subset=["LOCATION_X", "LOCATION_Y"])
    team_ids = sorted(players["TEAM_ID"].dropna().unique().tolist())
    if len(team_ids) != 2:
        raise ValueError("Pitch-control baseline requires exactly two teams in the selected frame")
    nx, ny = int(pitch_cfg.get("grid_x", 25)), int(pitch_cfg.get("grid_y", 17))
    xs, ys = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    reaction = float(pitch_cfg.get("reaction_frames", 2.0))
    temperature = max(float(pitch_cfg.get("softmax_temperature_frames", 1.0)), 1e-6)
    times: dict[Any, np.ndarray] = {}
    for team in team_ids:
        positions = players.loc[players["TEAM_ID"].eq(team), ["LOCATION_X", "LOCATION_Y"]].to_numpy(dtype=float)
        reach_times = reaction + np.sqrt(((points[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2)) / float(speed)
        times[team] = reach_times.min(axis=1)
    # Logistic pairwise transform gives a smooth proxy probability, not calibrated truth.
    margin = (times[team_ids[0]] - times[team_ids[1]]) / temperature
    p0 = 1.0 / (1.0 + np.exp(np.clip(margin, -60, 60)))
    return pd.DataFrame({
        "x_native": points[:, 0], "y_native": points[:, 1],
        "team_0_id": team_ids[0], "team_0_control_probability_proxy": p0,
        "team_1_id": team_ids[1], "team_1_control_probability_proxy": 1.0 - p0,
        "team_0_min_time_to_reach_frames": times[team_ids[0]],
        "team_1_min_time_to_reach_frames": times[team_ids[1]],
        "model_status": "uncalibrated_constant_speed_reachability_proxy",
    })
