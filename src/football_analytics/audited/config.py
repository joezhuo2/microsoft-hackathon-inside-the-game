"""Configuration loading and validation."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "input_file": None,
    "output_dir": "output",
    # Do not guess physical calibration from the CSV.
    "seconds_per_frame": None,
    "coordinate_scale_to_metre": None,
    "max_contiguous_gap_frames": 1,
    "expected_players_per_frame": 22,
    "movement": {
        "stationary_epsilon": 1e-6,
        "smoothing_window_frames": 1,
    },
    "possession": {
        "on_ball_true_values": [1, "1", True, "true", "True", "TRUE", "yes", "YES"],
        "on_ball_false_values": [0, "0", False, "false", "False", "FALSE", "no", "NO"],
    },
    "events": {
        "min_pass_distance_native": 2.0,
        "min_carry_distance_native": 1.5,
        "progressive_distance_native": 5.0,
        # Event geometry is deliberately opt-in because pitch units are unspecified.
    },
    "pitch": {
        "attacking_direction_by_team": {},  # e.g. {"0": 1, "1": -1}; +1 means +X
        "goal_centres_native_by_team": {},
        "goal_posts_native_by_team": {},
        "bounds_native": None,  # [xmin, xmax, ymin, ymax]
    },
    "pitch_control": {
        "max_speed_native_per_frame": None,
        "reaction_frames": 2.0,
        "softmax_temperature_frames": 1.0,
        "grid_x": 25,
        "grid_y": 17,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: dict[str, Any]) -> None:
    """Fail early for invalid calibration and threshold values."""
    seconds = config.get("seconds_per_frame")
    scale = config.get("coordinate_scale_to_metre")
    if seconds is not None and float(seconds) <= 0:
        raise ValueError("seconds_per_frame must be positive or null")
    if scale is not None and float(scale) <= 0:
        raise ValueError("coordinate_scale_to_metre must be positive or null")
    if int(config.get("max_contiguous_gap_frames", 1)) < 1:
        raise ValueError("max_contiguous_gap_frames must be at least 1")
    for group, key in [("events", "min_pass_distance_native"), ("events", "min_carry_distance_native"), ("events", "progressive_distance_native")]:
        if float(config[group][key]) < 0:
            raise ValueError(f"{group}.{key} must be non-negative")
    max_speed = config.get("pitch_control", {}).get("max_speed_native_per_frame")
    if max_speed is not None and float(max_speed) <= 0:
        raise ValueError("pitch_control.max_speed_native_per_frame must be positive or null")
    for team, direction in config.get("pitch", {}).get("attacking_direction_by_team", {}).items():
        if float(direction) not in (-1.0, 1.0):
            raise ValueError(f"pitch.attacking_direction_by_team[{team!r}] must be -1 or +1")
    bounds = config.get("pitch", {}).get("bounds_native")
    if bounds is not None and (len(bounds) != 4 or not float(bounds[0]) < float(bounds[1]) or not float(bounds[2]) < float(bounds[3])):
        raise ValueError("pitch.bounds_native must be [xmin, xmax, ymin, ymax] with increasing bounds")
    for team, centre in config.get("pitch", {}).get("goal_centres_native_by_team", {}).items():
        if not isinstance(centre, (list, tuple)) or len(centre) != 2:
            raise ValueError(f"pitch.goal_centres_native_by_team[{team!r}] must contain [x, y]")
    for team, posts in config.get("pitch", {}).get("goal_posts_native_by_team", {}).items():
        if not isinstance(posts, (list, tuple)) or len(posts) != 2 or any(not isinstance(post, (list, tuple)) or len(post) != 2 for post in posts):
            raise ValueError(f"pitch.goal_posts_native_by_team[{team!r}] must contain two [x, y] coordinate pairs")
    for key in ("grid_x", "grid_y"):
        if int(config.get("pitch_control", {}).get(key, 1)) < 2:
            raise ValueError(f"pitch_control.{key} must be at least 2")
    if float(config.get("pitch_control", {}).get("reaction_frames", 0)) < 0:
        raise ValueError("pitch_control.reaction_frames must be non-negative")
    if float(config.get("pitch_control", {}).get("softmax_temperature_frames", 1)) <= 0:
        raise ValueError("pitch_control.softmax_temperature_frames must be positive")


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load YAML, merging it over conservative defaults."""
    user: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Configuration root must be a YAML mapping")
        user = loaded
    config = _deep_merge(DEFAULT_CONFIG, user)
    if overrides:
        config = _deep_merge(config, overrides)
    validate_config(config)
    return config
