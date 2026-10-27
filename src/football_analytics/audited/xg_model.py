"""Trainable, auditable expected-goals model using XGBoost.

The model is trained on one row per labelled shot. It builds pre-shot features from
tracking snapshots and the event timeline; post-shot outcome/trajectory fields are
never used as predictors. GAME_ID is used only for grouped evaluation, not as a feature.

Coordinates are expected in a 120 x 80 pitch plane, or in the source normalized ranges
x=[-1, 1], y=[-0.42, 0.42], which can be normalized automatically. Horizontal velocity
is expressed per timestep, not per second. Vertical Z features are opt-in because the
source scale in this project is not calibrated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

XG_MODEL_VERSION = "xgboost-xg-1.0.0"

NUMERIC_FEATURES = [
    "goal_distance_x_m",
    "goal_lateral_offset_m",
    "goal_distance_m",
    "goal_angle_deg",
    "angle_to_goal_center_deg",
    "distance_to_penalty_spot_m",
    "is_inside_penalty_area",
    "is_inside_six_yard_box",
    "ball_speed_per_timestep",
    "ball_velocity_toward_goal_per_timestep",
    "ball_velocity_lateral_per_timestep",
    "ball_goalward_angle_error_deg",
    "shooter_speed_per_timestep",
    "shooter_velocity_toward_goal_per_timestep",
    "shooter_velocity_lateral_per_timestep",
    "ball_shooter_distance_m",
    "nearest_defender_distance_m",
    "defenders_within_2m",
    "defenders_within_5m",
    "defenders_within_10m",
    "nearest_defender_lane_distance_m",
    "defenders_in_shooting_lane_1p5m",
    "goalkeeper_distance_to_shot_m",
    "goalkeeper_distance_to_goal_center_m",
    "goalkeeper_lateral_offset_m",
    "goalkeeper_goal_line_distance_m",
    "goalkeeper_angle_to_shot_deg",
    "shot_distance_to_six_yard_area_m",
    "shot_distance_to_penalty_area_m",
    "is_penalty",
    "is_rebound",
    "is_one_on_one",
    "is_first_time_shot",
    "is_set_piece",
    "ball_height_m",
]

CATEGORICAL_FEATURES = [
    "action_before_shot",
    "shot_body_part",
    "shot_type",
    "play_pattern",
    "assist_type",
    "set_piece_type",
    "shooter_role",
    "goalkeeper_feature_source",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SHOT_EVENT_TYPES = {
    "shot", "shot_attempt", "shot_on_target", "shot_off_target", "blocked_shot",
    "penalty_shot", "direct_free_kick_shot", "shot_saved", "shot_missed",
}
HEURISTIC_SHOT_EVENT_TYPES = {"shot_candidate", "planar_on_target_candidate", "goal_candidate"}


@dataclass(frozen=True)
class XGTrainingSummary:
    model_version: str
    model_path: str
    n_shots: int
    n_goals: int
    goal_rate: float
    n_games: int
    calibration_status: str
    evaluation: dict[str, Any]
    feature_importance_path: str | None = None


def _xg_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve nested xG defaults without changing the caller's config."""
    root = dict(config or {})
    xg = dict(root.get("xg", {}))
    defaults: dict[str, Any] = {
        "pitch_length_m": 120.0,
        "pitch_width_m": 80.0,
        "goal_width_m": 7.32,
        "goal_center_y_m": 40.0,
        "penalty_area_depth_m": 16.5,
        "penalty_area_width_m": 40.32,
        "goal_area_depth_m": 5.5,
        "goal_area_width_m": 18.32,
        "penalty_spot_distance_m": 11.0,
        "coordinate_mode": "auto",
        "event_coordinate_mode": "auto",
        "attacking_direction_by_team": root.get("pitch", {}).get("attacking_direction_by_team", {}),
        "attacking_direction_by_game_team": {},
        "segment_column": "period",
        "attacking_direction_by_game_team_segment": {},
        "allow_velocity_based_direction_inference": False,
        "allow_heuristic_shot_candidates": False,
        "max_pre_shot_action_gap_frames": 100,
        "velocity_lookback_frames": 1,
        "lane_half_width_m": 1.5,
        "use_ball_height": False,
        "z_scale_to_m": None,
        "test_size": 0.2,
        "random_state": 42,
        "calibration_method": "sigmoid",
        "calibration_min_games": 5,
        "xgb_params": {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.035,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 5.0,
            "reg_alpha": 0.05,
        },
    }
    for key, value in defaults.items():
        xg.setdefault(key, value)
    # Existing project config stores this map below pitch; preserve it if present.
    if not xg.get("attacking_direction_by_team"):
        xg["attacking_direction_by_team"] = root.get("pitch", {}).get("attacking_direction_by_team", {})
    if xg["coordinate_mode"] not in {"auto", "source_normalized", "pitch_120x80"}:
        raise ValueError("xg.coordinate_mode must be auto, source_normalized, or pitch_120x80")
    if xg["event_coordinate_mode"] not in {"auto", "source_normalized", "pitch_120x80"}:
        raise ValueError("xg.event_coordinate_mode must be auto, source_normalized, or pitch_120x80")
    return xg


def _canon(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()



def _mapping_get(mapping: Any, key: Any) -> Any:
    """Lookup YAML mappings robustly when keys are numeric in one file and strings in another."""
    if not isinstance(mapping, dict):
        return None
    wanted = _canon(key)
    for candidate_key, value in mapping.items():
        if _canon(candidate_key) == wanted:
            return value
    return None


def _numeric(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _looks_source_normalized(df: pd.DataFrame, x_col: str = "LOCATION_X", y_col: str = "LOCATION_Y") -> bool:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    if x.dropna().empty or y.dropna().empty:
        return False
    return (x.abs().quantile(0.99) <= 1.25 and y.abs().quantile(0.99) <= 0.60)


def _normalize_xy(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    mode: str,
    velocity_x_col: str | None = None,
    velocity_y_col: str | None = None,
) -> pd.DataFrame:
    """Return coordinates in a 120 x 80 pitch plane, preserving source values."""
    out = df.copy()
    should_transform = mode == "source_normalized" or (mode == "auto" and _looks_source_normalized(out, x_col, y_col))
    if not should_transform:
        return out

    source_x = f"SOURCE_{x_col}"
    source_y = f"SOURCE_{y_col}"
    if source_x not in out:
        out[source_x] = pd.to_numeric(out[x_col], errors="coerce")
    if source_y not in out:
        out[source_y] = pd.to_numeric(out[y_col], errors="coerce")
    out[x_col] = (pd.to_numeric(out[source_x], errors="coerce") + 1.0) * 60.0
    out[y_col] = (pd.to_numeric(out[source_y], errors="coerce") + 0.42) * (80.0 / 0.84)

    if velocity_x_col and velocity_y_col and velocity_x_col in out and velocity_y_col in out:
        source_vx = f"SOURCE_{velocity_x_col}"
        source_vy = f"SOURCE_{velocity_y_col}"
        if source_vx not in out:
            out[source_vx] = pd.to_numeric(out[velocity_x_col], errors="coerce")
        if source_vy not in out:
            out[source_vy] = pd.to_numeric(out[velocity_y_col], errors="coerce")
        out[velocity_x_col] = pd.to_numeric(out[source_vx], errors="coerce") * 60.0
        out[velocity_y_col] = pd.to_numeric(out[source_vy], errors="coerce") * (80.0 / 0.84)
    return out


def prepare_tracking_for_xg(tracking: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Normalize source XY automatically, or respect already-normalized pitch data."""
    required = {"GAME_ID", "TIMESTEP", "PLAYER_ID", "TEAM_ID", "LOCATION_X", "LOCATION_Y"}
    missing = sorted(required - set(tracking.columns))
    if missing:
        raise ValueError(f"Tracking data missing required xG columns: {missing}")
    xg = _xg_config(config)
    out = tracking.copy()
    for col in ["GAME_ID", "TIMESTEP", "PLAYER_ID", "TEAM_ID", "LOCATION_X", "LOCATION_Y"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[["GAME_ID", "TIMESTEP", "PLAYER_ID", "LOCATION_X", "LOCATION_Y"]].isna().any().any():
        raise ValueError("Tracking identifiers, timesteps, and XY positions must be numeric and non-null")
    out = _normalize_xy(
        out, "LOCATION_X", "LOCATION_Y", xg["coordinate_mode"], "VELOCITY_X", "VELOCITY_Y"
    )
    out["VELOCITY_X"] = pd.to_numeric(out.get("VELOCITY_X", np.nan), errors="coerce")
    out["VELOCITY_Y"] = pd.to_numeric(out.get("VELOCITY_Y", np.nan), errors="coerce")
    if "PLAYER_ROLE" not in out:
        out["PLAYER_ROLE"] = np.nan
    return out.sort_values(["GAME_ID", "TIMESTEP", "PLAYER_ID"], kind="mergesort").reset_index(drop=True)


def _event_time_col(events: pd.DataFrame) -> str:
    for col in ("start_timestep", "TIMESTEP", "timestep", "frame"):
        if col in events.columns:
            return col
    raise ValueError("Events need one of start_timestep, TIMESTEP, timestep, or frame")


def _event_shot_mask(events: pd.DataFrame, xg: dict[str, Any]) -> pd.Series:
    if "event_type" not in events.columns:
        # A shot-only table does not need an event_type column.
        return pd.Series(True, index=events.index)
    event_types = events["event_type"].fillna("").astype(str).str.strip().str.lower()
    shot_types = set(SHOT_EVENT_TYPES)
    shot_types.update(str(x).strip().lower() for x in xg.get("additional_shot_event_types", []))
    mask = event_types.isin(shot_types)
    if xg.get("allow_heuristic_shot_candidates", False):
        mask |= event_types.isin(HEURISTIC_SHOT_EVENT_TYPES)
    return mask


def _binary_label(value: Any) -> float:
    if pd.isna(value):
        return math.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value) in (0.0, 1.0):
            return float(value)
        return math.nan
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if text in {"1", "true", "yes", "y", "goal", "scored", "converted", "successful_goal"}:
        return 1.0
    if text in {
        "0", "false", "no", "n", "saved", "blocked", "off_target", "missed", "woodwork",
        "post", "crossbar", "no_goal", "not_goal", "not_scored", "failed", "unsuccessful",
    }:
        return 0.0
    # In particular, candidate/provisional goal labels are not accepted as ground truth.
    return math.nan


def _find_label(events: pd.DataFrame, configured_label: str | None) -> tuple[pd.Series | None, str | None]:
    candidates = [configured_label] if configured_label else []
    candidates += ["is_goal", "goal_scored", "scored", "goal", "shot_outcome"]
    for col in candidates:
        if col and col in events.columns:
            label = events[col].map(_binary_label)
            return label, col
    if "outcome" in events.columns:
        label = events["outcome"].map(_binary_label)
        return label, "outcome"
    return None, None


def _text(value: Any, default: str = "unknown") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text.lower() if text else default


def _action_category(value: Any) -> str:
    raw = _text(value, "unknown")
    s = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    if s == "unknown":
        return s
    if "rebound" in s:
        return "rebound"
    if "cross" in s:
        return "cross"
    if any(token in s for token in ("corner", "free_kick", "throw_in", "goal_kick", "penalty", "kickoff", "kick_off", "restart", "set_piece")):
        return "set_piece"
    if any(token in s for token in ("carry", "dribble", "take_on", "run_with_ball")):
        return "carry_dribble"
    if any(token in s for token in ("pass", "through_ball", "switch_of_play")):
        return "pass"
    if any(token in s for token in ("interception", "turnover", "recovery", "tackle", "ball_win")):
        return "turnover_recovery"
    if any(token in s for token in ("clearance", "clear")):
        return "clearance"
    if any(token in s for token in ("touch", "reception", "control")):
        return "touch_control"
    return "other"


def _find_prior_action(
    shot: pd.Series,
    events_sorted: pd.DataFrame,
    time_col: str,
    xg: dict[str, Any],
) -> str:
    for col in ("action_before_shot", "pre_shot_action", "previous_event_type"):
        if col in shot and pd.notna(shot[col]):
            return _action_category(shot[col])

    game = _canon(shot.get("GAME_ID"))
    t = _numeric(shot.get(time_col))
    team = _canon(shot.get("TEAM_ID"))
    if not math.isfinite(t) or game == "":
        return "unknown"
    same_game = events_sorted[events_sorted["_game_key"] == game]
    if same_game.empty:
        return "unknown"
    times = pd.to_numeric(same_game[time_col], errors="coerce")
    prior = same_game.loc[times.lt(t)].copy()
    prior = prior.loc[~prior.index.isin([shot.name])]
    if "event_type" in prior:
        event_types = prior["event_type"].fillna("").astype(str).str.lower()
        excluded = set(SHOT_EVENT_TYPES) | HEURISTIC_SHOT_EVENT_TYPES | {"goal", "own_goal"}
        prior = prior.loc[~event_types.isin(excluded)]
    max_gap = float(xg.get("max_pre_shot_action_gap_frames", 100))
    prior = prior.loc[pd.to_numeric(prior[time_col], errors="coerce").ge(t - max_gap)]
    if "TEAM_ID" in prior and team:
        same_team = prior.loc[prior["TEAM_ID"].map(_canon).eq(team)]
        if not same_team.empty:
            prior = same_team
    if prior.empty:
        return "unknown"
    previous = prior.loc[pd.to_numeric(prior[time_col], errors="coerce").idxmax()]
    return _action_category(previous.get("event_type", previous.get("subtype", "unknown")))


def _direction_for_shot(
    shot: pd.Series,
    tracking_frame: pd.DataFrame,
    xg: dict[str, Any],
    ball_row: pd.Series | None,
) -> tuple[int, float]:
    """Return (+1/-1 attack direction, target goal X)."""
    length = float(xg["pitch_length_m"])
    for col in ("target_goal_x", "goal_x"):
        value = _numeric(shot.get(col))
        if math.isfinite(value) and value in (0.0, length):
            return (1 if value > length / 2 else -1), value
        # Some raw-coordinate event feeds encode the goal line at x=+1 or x=-1.
        if math.isfinite(value) and value in (-1.0, 1.0) and xg.get("event_coordinate_mode", "auto") != "pitch_120x80":
            goal_x = length if value > 0 else 0.0
            return (1 if goal_x > length / 2 else -1), goal_x
    target = _text(shot.get("target_goal", ""), "")
    if target in {"left", "left_goal", "goal_left", "negative_x"}:
        return -1, 0.0
    if target in {"right", "right_goal", "goal_right", "positive_x"}:
        return 1, length

    for col in ("attacking_direction", "attack_direction", "direction"):
        value = _numeric(shot.get(col))
        if value in (-1.0, 1.0):
            return int(value), length if value > 0 else 0.0

    game = _canon(shot.get("GAME_ID"))
    team = _canon(shot.get("TEAM_ID"))
    segment_col = str(xg.get("segment_column", "period"))
    segment = _canon(shot.get(segment_col))
    segmented = xg.get("attacking_direction_by_game_team_segment", {})
    direction = None
    game_map = _mapping_get(segmented, game)
    team_map = _mapping_get(game_map, team)
    if segment:
        direction = _mapping_get(team_map, segment)
    if direction is None:
        by_game = xg.get("attacking_direction_by_game_team", {})
        game_map = _mapping_get(by_game, game)
        direction = _mapping_get(game_map, team)
    if direction is None:
        direction = _mapping_get(xg.get("attacking_direction_by_team", {}), team)
    direction_num = _numeric(direction)
    if direction_num in (-1.0, 1.0):
        d = int(direction_num)
        return d, length if d > 0 else 0.0

    if xg.get("allow_velocity_based_direction_inference", False) and ball_row is not None:
        vx = _numeric(ball_row.get("VELOCITY_X"))
        if math.isfinite(vx) and abs(vx) > 0.2:
            d = 1 if vx > 0 else -1
            return d, length if d > 0 else 0.0
    raise ValueError(
        f"Cannot determine target goal for shot game={shot.get('GAME_ID')!r}, "
        f"team={shot.get('TEAM_ID')!r}, timestep={shot.get('start_timestep', shot.get('TIMESTEP'))!r}. "
        "Provide target_goal_x/attacking_direction on the event or configure "
        "xg.attacking_direction_by_team / xg.attacking_direction_by_game_team_segment."
    )


def _position_from_event(shot: pd.Series, frame: pd.DataFrame) -> tuple[float, float]:
    for xcol, ycol in (("start_x", "start_y"), ("shot_x", "shot_y"), ("LOCATION_X", "LOCATION_Y")):
        x, y = _numeric(shot.get(xcol)), _numeric(shot.get(ycol))
        if math.isfinite(x) and math.isfinite(y):
            return x, y
    ball = frame.loc[pd.to_numeric(frame["PLAYER_ID"], errors="coerce").eq(-1)]
    if len(ball) == 1:
        return _numeric(ball.iloc[0]["LOCATION_X"]), _numeric(ball.iloc[0]["LOCATION_Y"])
    return math.nan, math.nan


def _categorical_event_value(shot: pd.Series, names: Iterable[str]) -> str:
    for col in names:
        if col in shot and pd.notna(shot[col]) and str(shot[col]).strip():
            return _text(shot[col])
    return "unknown"


def _bool_feature(shot: pd.Series, names: Iterable[str], text_hints: Iterable[str] = ()) -> float:
    for col in names:
        if col in shot and pd.notna(shot[col]):
            value = _binary_label(shot[col])
            if math.isfinite(value):
                return value
            text = str(shot[col]).lower()
            if any(hint in text for hint in text_hints):
                return 1.0
    return math.nan


def _point_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vector = end - start
    squared = float(np.dot(vector, vector))
    if squared <= 1e-12:
        return np.linalg.norm(points - start, axis=1), np.zeros(len(points))
    t = np.clip(((points - start) @ vector) / squared, 0.0, 1.0)
    projection = start + t[:, None] * vector
    return np.linalg.norm(points - projection, axis=1), t


def _shot_row_features(
    shot: pd.Series,
    frame: pd.DataFrame,
    motion_frame: pd.DataFrame,
    action_before: str,
    xg: dict[str, Any],
    time_col: str,
    include_labels: bool,
) -> dict[str, Any]:
    length = float(xg["pitch_length_m"])
    width = float(xg["pitch_width_m"])
    goal_y = float(xg["goal_center_y_m"])
    goal_width = float(xg["goal_width_m"])

    if "PLAYER_ID" not in frame:
        frame = frame.copy()
    players = frame.loc[pd.to_numeric(frame["PLAYER_ID"], errors="coerce").ne(-1)].copy()
    balls = frame.loc[pd.to_numeric(frame["PLAYER_ID"], errors="coerce").eq(-1)]
    ball = balls.iloc[0] if len(balls) == 1 else None
    motion_players = motion_frame.loc[pd.to_numeric(motion_frame["PLAYER_ID"], errors="coerce").ne(-1)].copy() if len(motion_frame) else motion_frame
    motion_balls = motion_frame.loc[pd.to_numeric(motion_frame["PLAYER_ID"], errors="coerce").eq(-1)] if len(motion_frame) else motion_frame
    motion_ball = motion_balls.iloc[0] if len(motion_balls) == 1 else None
    shot_x, shot_y = _position_from_event(shot, frame)

    shooter_id = _numeric(shot.get("PLAYER_ID"))
    shooter = players.loc[pd.to_numeric(players["PLAYER_ID"], errors="coerce").eq(shooter_id)] if math.isfinite(shooter_id) else players.iloc[0:0]
    shooter_row = shooter.iloc[0] if len(shooter) == 1 else None
    shooter_motion = motion_players.loc[pd.to_numeric(motion_players["PLAYER_ID"], errors="coerce").eq(shooter_id)] if math.isfinite(shooter_id) and len(motion_players) else motion_players.iloc[0:0]
    shooter_motion_row = shooter_motion.iloc[0] if len(shooter_motion) == 1 else None
    team_value = shot.get("TEAM_ID")
    if pd.isna(team_value) and shooter_row is not None:
        team_value = shooter_row.get("TEAM_ID")
    shot_for_direction = shot.copy()
    shot_for_direction["TEAM_ID"] = team_value
    direction, goal_x = _direction_for_shot(shot_for_direction, frame, xg, motion_ball)

    dx_goal = abs(goal_x - shot_x) if math.isfinite(shot_x) else math.nan
    dy_goal_signed = shot_y - goal_y if math.isfinite(shot_y) else math.nan
    dy_goal = abs(dy_goal_signed) if math.isfinite(dy_goal_signed) else math.nan
    dist_goal = math.hypot(dx_goal, dy_goal) if math.isfinite(dx_goal) and math.isfinite(dy_goal) else math.nan
    angle_center = math.degrees(math.atan2(dy_goal, max(dx_goal, 1e-9))) if math.isfinite(dx_goal) and math.isfinite(dy_goal) else math.nan
    post_a = np.array([goal_x, goal_y - goal_width / 2.0], dtype=float)
    post_b = np.array([goal_x, goal_y + goal_width / 2.0], dtype=float)
    shot_point = np.array([shot_x, shot_y], dtype=float)
    if math.isfinite(shot_x) and math.isfinite(shot_y):
        va, vb = post_a - shot_point, post_b - shot_point
        den = float(np.linalg.norm(va) * np.linalg.norm(vb))
        goal_angle = math.degrees(math.acos(float(np.clip(np.dot(va, vb) / den, -1.0, 1.0)))) if den > 1e-9 else math.nan
    else:
        goal_angle = math.nan

    # Motion predictors are sampled before the shot event by default to avoid using
    # the outgoing shot trajectory (post-shot information) in a pre-shot xG model.
    ball_vx = _numeric(motion_ball.get("VELOCITY_X")) if motion_ball is not None else math.nan
    ball_vy = _numeric(motion_ball.get("VELOCITY_Y")) if motion_ball is not None else math.nan
    ball_speed = math.hypot(ball_vx, ball_vy) if math.isfinite(ball_vx) and math.isfinite(ball_vy) else math.nan
    ball_toward_goal = ball_vx * direction if math.isfinite(ball_vx) else math.nan
    ball_goalward_error = math.nan
    if math.isfinite(ball_vx) and math.isfinite(ball_vy) and math.isfinite(dx_goal) and math.isfinite(dy_goal_signed):
        to_goal = np.array([goal_x - shot_x, goal_y - shot_y])
        velocity = np.array([ball_vx, ball_vy])
        den = float(np.linalg.norm(to_goal) * np.linalg.norm(velocity))
        if den > 1e-9:
            ball_goalward_error = math.degrees(math.acos(float(np.clip(np.dot(to_goal, velocity) / den, -1.0, 1.0))))

    shooter_vx = _numeric(shooter_motion_row.get("VELOCITY_X")) if shooter_motion_row is not None else math.nan
    shooter_vy = _numeric(shooter_motion_row.get("VELOCITY_Y")) if shooter_motion_row is not None else math.nan
    shooter_speed = math.hypot(shooter_vx, shooter_vy) if math.isfinite(shooter_vx) and math.isfinite(shooter_vy) else math.nan
    shooter_toward_goal = shooter_vx * direction if math.isfinite(shooter_vx) else math.nan
    shooter_lateral_v = shooter_vy if math.isfinite(shooter_vy) else math.nan
    ball_shooter_dist = math.nan
    if shooter_row is not None and math.isfinite(shot_x) and math.isfinite(shot_y):
        ball_shooter_dist = math.hypot(shot_x - _numeric(shooter_row.get("LOCATION_X")), shot_y - _numeric(shooter_row.get("LOCATION_Y")))

    defenders = players.iloc[0:0]
    if pd.notna(team_value):
        defenders = players.loc[pd.to_numeric(players["TEAM_ID"], errors="coerce").ne(_numeric(team_value))].copy()
        defenders = defenders.loc[pd.to_numeric(defenders["TEAM_ID"], errors="coerce").notna()]
    defender_dist = np.array([], dtype=float)
    lane_dist = np.array([], dtype=float)
    lane_t = np.array([], dtype=float)
    if len(defenders) and math.isfinite(shot_x) and math.isfinite(shot_y):
        defender_xy = defenders[["LOCATION_X", "LOCATION_Y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(defender_xy).all(axis=1)
        defenders = defenders.iloc[np.flatnonzero(valid)].copy()
        defender_xy = defender_xy[valid]
        if len(defender_xy):
            defender_dist = np.linalg.norm(defender_xy - shot_point, axis=1)
            lane_dist, lane_t = _point_segment_distances(defender_xy, shot_point, np.array([goal_x, goal_y]))
    nearest_def = float(np.min(defender_dist)) if len(defender_dist) else math.nan
    lane_mask = (lane_dist <= float(xg["lane_half_width_m"])) & (lane_t > 0.02) & (lane_t < 0.98) if len(lane_dist) else np.array([], dtype=bool)

    # Use the explicit role when present; otherwise expose a clearly named geometric proxy.
    keeper_source = "unknown"
    keeper_row = None
    if len(defenders) and "PLAYER_ROLE" in defenders.columns:
        roles = defenders["PLAYER_ROLE"].fillna("").astype(str).str.lower()
        role_mask = roles.str.contains(r"goal.?keeper|\bgk\b|keeper", regex=True)
        role_keepers = defenders.loc[role_mask]
        if not role_keepers.empty:
            d = np.hypot(
                pd.to_numeric(role_keepers["LOCATION_X"], errors="coerce") - goal_x,
                pd.to_numeric(role_keepers["LOCATION_Y"], errors="coerce") - goal_y,
            )
            keeper_row = role_keepers.loc[d.idxmin()]
            keeper_source = "role_label"
    if keeper_row is None and len(defenders):
        d = np.hypot(
            pd.to_numeric(defenders["LOCATION_X"], errors="coerce") - goal_x,
            pd.to_numeric(defenders["LOCATION_Y"], errors="coerce") - goal_y,
        )
        if d.notna().any():
            keeper_row = defenders.loc[d.idxmin()]
            keeper_source = "geometric_proxy"

    keeper_distance_shot = keeper_distance_goal = keeper_lat = keeper_goal_line = keeper_angle = math.nan
    if keeper_row is not None:
        kx, ky = _numeric(keeper_row.get("LOCATION_X")), _numeric(keeper_row.get("LOCATION_Y"))
        if math.isfinite(kx) and math.isfinite(ky) and math.isfinite(shot_x) and math.isfinite(shot_y):
            keeper_distance_shot = math.hypot(kx - shot_x, ky - shot_y)
            keeper_angle = math.degrees(math.atan2(abs(ky - shot_y), abs(kx - shot_x)))
        if math.isfinite(kx) and math.isfinite(ky):
            keeper_distance_goal = math.hypot(kx - goal_x, ky - goal_y)
            keeper_lat = abs(ky - goal_y)
            keeper_goal_line = abs(goal_x - kx)

    box_depth = float(xg["penalty_area_depth_m"])
    box_half_width = float(xg["penalty_area_width_m"]) / 2.0
    six_depth = float(xg["goal_area_depth_m"])
    six_half_width = float(xg["goal_area_width_m"]) / 2.0
    dist_penalty_spot = math.hypot(dx_goal - float(xg["penalty_spot_distance_m"]), dy_goal) if math.isfinite(dx_goal) and math.isfinite(dy_goal) else math.nan
    inside_penalty = float(dx_goal <= box_depth and dy_goal <= box_half_width) if math.isfinite(dx_goal) and math.isfinite(dy_goal) else math.nan
    inside_six = float(dx_goal <= six_depth and dy_goal <= six_half_width) if math.isfinite(dx_goal) and math.isfinite(dy_goal) else math.nan
    def distance_to_target_area(depth: float, area_width: float) -> float:
        if not (math.isfinite(shot_x) and math.isfinite(shot_y)):
            return math.nan
        if direction > 0:
            x_min, x_max = length - depth, length
        else:
            x_min, x_max = 0.0, depth
        y_min, y_max = goal_y - area_width / 2.0, goal_y + area_width / 2.0
        nearest_x = float(np.clip(shot_x, x_min, x_max))
        nearest_y = float(np.clip(shot_y, y_min, y_max))
        return math.hypot(shot_x - nearest_x, shot_y - nearest_y)

    dist_six_area = distance_to_target_area(six_depth, float(xg["goal_area_width_m"]))
    dist_pen_area = distance_to_target_area(box_depth, float(xg["penalty_area_width_m"]))

    out: dict[str, Any] = {
        "event_id": shot.get("event_id", shot.get("EVENT_ID", "")),
        "GAME_ID": shot.get("GAME_ID"),
        "start_timestep": _numeric(shot.get(time_col)),
        "PLAYER_ID": shooter_id if math.isfinite(shooter_id) else np.nan,
        "TEAM_ID": _numeric(team_value),
        "shot_x": shot_x,
        "shot_y": shot_y,
        "target_goal_x": goal_x,
        "attacking_direction": direction,
        "goal_distance_x_m": dx_goal,
        "goal_lateral_offset_m": dy_goal,
        "goal_distance_m": dist_goal,
        "goal_angle_deg": goal_angle,
        "angle_to_goal_center_deg": angle_center,
        "distance_to_penalty_spot_m": dist_penalty_spot,
        "is_inside_penalty_area": inside_penalty,
        "is_inside_six_yard_box": inside_six,
        "ball_speed_per_timestep": ball_speed,
        "ball_velocity_toward_goal_per_timestep": ball_toward_goal,
        "ball_velocity_lateral_per_timestep": ball_vy,
        "ball_goalward_angle_error_deg": ball_goalward_error,
        "shooter_speed_per_timestep": shooter_speed,
        "shooter_velocity_toward_goal_per_timestep": shooter_toward_goal,
        "shooter_velocity_lateral_per_timestep": shooter_lateral_v,
        "ball_shooter_distance_m": ball_shooter_dist,
        "nearest_defender_distance_m": nearest_def,
        "defenders_within_2m": float(np.sum(defender_dist <= 2.0)) if len(defender_dist) else math.nan,
        "defenders_within_5m": float(np.sum(defender_dist <= 5.0)) if len(defender_dist) else math.nan,
        "defenders_within_10m": float(np.sum(defender_dist <= 10.0)) if len(defender_dist) else math.nan,
        "nearest_defender_lane_distance_m": float(np.min(lane_dist)) if len(lane_dist) else math.nan,
        "defenders_in_shooting_lane_1p5m": float(np.sum(lane_mask)) if len(lane_mask) else math.nan,
        "goalkeeper_distance_to_shot_m": keeper_distance_shot,
        "goalkeeper_distance_to_goal_center_m": keeper_distance_goal,
        "goalkeeper_lateral_offset_m": keeper_lat,
        "goalkeeper_goal_line_distance_m": keeper_goal_line,
        "goalkeeper_angle_to_shot_deg": keeper_angle,
        "shot_distance_to_six_yard_area_m": dist_six_area,
        "shot_distance_to_penalty_area_m": dist_pen_area,
        "ball_height_m": math.nan,
        "action_before_shot": action_before,
        "shot_body_part": _categorical_event_value(shot, ("body_part", "shot_body_part", "bodypart")),
        "shot_type": _categorical_event_value(shot, ("shot_type", "technique", "shot_subtype")),
        "play_pattern": _categorical_event_value(shot, ("play_pattern", "attack_pattern", "phase")),
        "assist_type": _categorical_event_value(shot, ("assist_type", "assist_pattern")),
        "set_piece_type": _categorical_event_value(shot, ("set_piece_type", "restart_type")),
        "shooter_role": _categorical_event_value(shooter_row if shooter_row is not None else shot, ("PLAYER_ROLE", "player_role", "shooter_role")),
        "goalkeeper_feature_source": keeper_source,
    }
    if xg.get("use_ball_height", False) and xg.get("z_scale_to_m") is not None and ball is not None:
        z = _numeric(ball.get("LOCATION_Z"))
        if math.isfinite(z):
            out["ball_height_m"] = z * float(xg["z_scale_to_m"])

    out["is_penalty"] = _bool_feature(shot, ("is_penalty", "penalty"), ("penalty",))
    out["is_rebound"] = _bool_feature(shot, ("is_rebound",), ("rebound",))
    out["is_one_on_one"] = _bool_feature(shot, ("is_one_on_one", "one_on_one"))
    out["is_first_time_shot"] = _bool_feature(shot, ("is_first_time_shot", "first_time"))
    # These are context flags, not identifiers or outcome-derived variables.
    out["is_set_piece"] = _bool_feature(shot, ("is_set_piece",), ("set_piece", "corner", "free_kick", "penalty"))

    if include_labels:
        for col in ("is_goal", "goal_scored", "scored", "goal"):
            if col in shot.index:
                out["is_goal"] = _binary_label(shot.get(col))
                break
        else:
            out["is_goal"] = _binary_label(shot.get("outcome")) if "outcome" in shot.index else math.nan
    return out


def build_shot_feature_table(
    tracking: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any] | None = None,
    label_column: str | None = None,
    include_labels: bool = True,
) -> pd.DataFrame:
    """Build shot-level pre-shot features from tracking and a full event log.

    Goal labels are read only from an explicit binary label or resolved ``outcome``.
    The event type itself is never used as the goal label, preventing direct leakage.
    """
    xg = _xg_config(config)
    if events.empty:
        raise ValueError("Event table is empty")
    time_col = _event_time_col(events)
    if "GAME_ID" not in events:
        raise ValueError("Events must contain GAME_ID")
    event_rows = events.copy()
    event_rows[time_col] = pd.to_numeric(event_rows[time_col], errors="coerce")
    event_rows = event_rows.loc[event_rows[time_col].notna()].copy()
    if "event_type" not in event_rows:
        # Caller may pass a shot-only event table.
        shot_rows = event_rows.copy()
    else:
        mask = _event_shot_mask(event_rows, xg)
        shot_rows = event_rows.loc[mask].copy()
    if shot_rows.empty:
        raise ValueError(
            "No eligible shot events found. Use a labelled shot event type (e.g. 'shot' or 'shot_attempt'). "
            "Heuristic 'shot_candidate' rows are excluded by default; enable xg.allow_heuristic_shot_candidates "
            "only after independent review."
        )

    prepared = prepare_tracking_for_xg(tracking, config)
    xg_mode = xg["event_coordinate_mode"]
    # Normalize event coordinates only if they are visibly in the source normalized range.
    candidate_xy = [pair for pair in (("start_x", "start_y"), ("shot_x", "shot_y")) if pair[0] in shot_rows and pair[1] in shot_rows]
    if candidate_xy:
        # A copy keeps the caller's event frame immutable. Normalize each available
        # coordinate pair so missing start_x/start_y rows can fall back to shot_x/shot_y.
        shot_rows = shot_rows.copy()
        for xcol, ycol in candidate_xy:
            shot_rows = _normalize_xy(shot_rows, xcol, ycol, xg_mode)

    # For historical-action context, prepare and normalize the complete event log in the
    # same manner as the shot coordinates. It is strictly filtered to prior timesteps later.
    history = event_rows.copy()
    history["_game_key"] = history["GAME_ID"].map(_canon)
    tracking_frames: dict[tuple[str, str], pd.DataFrame] = {
        (_canon(game), _canon(timestep)): frame
        for (game, timestep), frame in prepared.groupby(["GAME_ID", "TIMESTEP"], sort=False)
    }
    labels, resolved_label_col = _find_label(shot_rows, label_column) if include_labels else (None, None)
    if include_labels and labels is not None:
        shot_rows = shot_rows.copy()
        shot_rows["_xg_label"] = labels

    records: list[dict[str, Any]] = []
    for idx, shot in shot_rows.iterrows():
        game_key = _canon(shot.get("GAME_ID"))
        timestep = _numeric(shot.get(time_col))
        frame = tracking_frames.get((game_key, _canon(timestep)))
        if frame is None:
            # Missing tracking frame is represented by an empty frame. Event XY can still
            # provide basic geometry, while defender/velocity features remain null.
            frame = prepared.iloc[0:0]
        lookback = max(0, int(xg.get("velocity_lookback_frames", 1)))
        motion_timestep = timestep - lookback if math.isfinite(timestep) else math.nan
        motion_frame = tracking_frames.get((game_key, _canon(motion_timestep))) if lookback > 0 else frame
        if motion_frame is None:
            motion_frame = prepared.iloc[0:0]
        action = _find_prior_action(shot, history, time_col, xg)
        row = _shot_row_features(shot, frame, motion_frame, action, xg, time_col, include_labels=False)
        if include_labels:
            row["is_goal"] = _binary_label(shot.get("_xg_label")) if "_xg_label" in shot else (
                _binary_label(shot.get(label_column)) if label_column and label_column in shot else math.nan
            )
            if not math.isfinite(row["is_goal"]) and "outcome" in shot.index:
                row["is_goal"] = _binary_label(shot.get("outcome"))
        row["shot_event_type"] = _text(shot.get("event_type", "shot"))
        row["goal_label_source"] = resolved_label_col or ("outcome" if "outcome" in shot_rows.columns else "unlabelled")
        records.append(row)

    features = pd.DataFrame.from_records(records)
    # Keep additional explicit context features if available in source event data.
    for col in ("is_penalty", "is_rebound", "is_one_on_one", "is_first_time_shot", "is_set_piece"):
        if col not in features:
            features[col] = np.nan
    features = features.reset_index(drop=True)
    return features


def _feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Select only permitted pre-shot predictors, excluding labels and identifiers."""
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    X = features.reindex(columns=FEATURE_COLUMNS).copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("unknown").astype(str).replace({"nan": "unknown", "None": "unknown"})
    return X


def _make_pipeline(config: dict[str, Any]):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "XGBoost xG functionality requires the optional model dependencies. Install with "
            "`pip install -e '.[xg]'` or `pip install -r requirements-xg.txt`."
        ) from exc

    xg = _xg_config(config)
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=True)
    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.035,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "reg_alpha": 0.05,
    }
    params.update(xg.get("xgb_params", {}))
    classifier = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=int(xg.get("random_state", 42)),
        n_jobs=int(xg.get("n_jobs", -1)),
    )
    return Pipeline([("preprocess", preprocessor), ("xgb", classifier)])


def _fit_with_optional_calibration(X: pd.DataFrame, y: pd.Series, groups: pd.Series, config: dict[str, Any]):
    """Fit game-grouped calibration CV when the data supports it; otherwise fit XGB raw."""
    from sklearn.model_selection import GroupKFold

    xg = _xg_config(config)
    method = xg.get("calibration_method", "sigmoid")
    min_games = int(xg.get("calibration_min_games", 5))
    unique_games = pd.Series(groups).nunique()
    if method in (None, "none", "off") or unique_games < min_games:
        estimator = _make_pipeline(config)
        estimator.fit(X, y)
        return estimator, "uncalibrated_xgboost_logistic_output"

    n_splits = min(5, int(unique_games))
    splitter = GroupKFold(n_splits=n_splits)
    cv_splits = list(splitter.split(X, y, groups))
    # Calibrator folds need both labels in the held-out group set and in each fit set.
    if any(pd.Series(y.iloc[test]).nunique() < 2 or pd.Series(y.iloc[train]).nunique() < 2 for train, test in cv_splits):
        estimator = _make_pipeline(config)
        estimator.fit(X, y)
        return estimator, "uncalibrated_fallback_insufficient_class_coverage_in_group_folds"

    try:
        from sklearn.calibration import CalibratedClassifierCV
        estimator = CalibratedClassifierCV(
            estimator=_make_pipeline(config), method=str(method), cv=cv_splits, ensemble=True
        )
        estimator.fit(X, y)
        return estimator, f"group_cross_validated_{method}_calibration"
    except (ValueError, TypeError):
        # Avoid making the whole fit fail solely because a sklearn release rejects a
        # calibration fold layout; retain an explicit uncalibrated status instead.
        estimator = _make_pipeline(config)
        estimator.fit(X, y)
        return estimator, "uncalibrated_fallback_calibration_failed"


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    result: dict[str, Any] = {
        "n_shots": int(len(y)),
        "n_goals": int(y.sum()),
        "observed_goal_rate": float(y.mean()) if len(y) else None,
        "mean_predicted_xg": float(p.mean()) if len(p) else None,
        "brier_score": float(brier_score_loss(y, p)) if len(y) else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "calibration_gap_mean_prediction_minus_goal_rate": float(p.mean() - y.mean()) if len(y) else None,
    }
    return result


def _reliability_table(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"is_goal": np.asarray(y_true, dtype=int), "predicted_xg": np.asarray(probabilities, dtype=float)})
    if frame.empty:
        return []
    if frame["predicted_xg"].nunique() <= 1:
        return [{
            "bin": "all_predictions_equal", "n_shots": int(len(frame)),
            "mean_predicted_xg": float(frame["predicted_xg"].mean()),
            "observed_goal_rate": float(frame["is_goal"].mean()),
        }]
    frame["bin"] = pd.qcut(frame["predicted_xg"], q=min(bins, frame["predicted_xg"].nunique()), duplicates="drop")
    return [
        {
            "bin": str(name), "n_shots": int(len(group)),
            "mean_predicted_xg": float(group["predicted_xg"].mean()),
            "observed_goal_rate": float(group["is_goal"].mean()),
        }
        for name, group in frame.groupby("bin", observed=True, sort=True)
    ]


def _feature_importance(estimator: Any) -> pd.DataFrame:
    """Return gain-like XGB feature importance in transformed one-hot feature space."""
    candidates = []
    if hasattr(estimator, "named_steps") and "xgb" in estimator.named_steps:
        candidates = [estimator]
    elif hasattr(estimator, "calibrated_classifiers_"):
        for calibrated in estimator.calibrated_classifiers_:
            candidate = getattr(calibrated, "estimator", None)
            if candidate is None:
                candidate = getattr(calibrated, "base_estimator", None)
            if candidate is not None and hasattr(candidate, "named_steps"):
                candidates.append(candidate)
    if not candidates:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_by_name: dict[str, list[float]] = {}
    for candidate in candidates:
        pre = candidate.named_steps["preprocess"]
        xgb = candidate.named_steps["xgb"]
        names = list(pre.get_feature_names_out())
        values = np.asarray(xgb.feature_importances_, dtype=float)
        for name, value in zip(names, values):
            importance_by_name.setdefault(str(name), []).append(float(value))
    rows = [{"feature": name, "importance": float(np.mean(values))} for name, values in importance_by_name.items()]
    return pd.DataFrame(rows).sort_values("importance", ascending=False, kind="mergesort").reset_index(drop=True)


def train_xgboost_xg_model(
    shot_features: pd.DataFrame,
    model_path: str | Path,
    config: dict[str, Any] | None = None,
    report_path: str | Path | None = None,
    importance_path: str | Path | None = None,
    label_col: str = "is_goal",
) -> dict[str, Any]:
    """Train an XGBoost xG model with a held-out-game evaluation and optional calibration.

    A fitted model is created only from supplied binary goal labels. Events marked merely
    as candidates should be reviewed/labelled independently before training.
    """
    try:
        import joblib
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install model dependencies using `pip install -e '.[xg]'`.") from exc
    xg = _xg_config(config)
    if label_col not in shot_features:
        raise ValueError(f"Shot feature table has no {label_col!r} labels; provide a shot-level goal outcome column.")
    valid = shot_features[label_col].map(_binary_label)
    usable = shot_features.loc[valid.notna()].copy().reset_index(drop=True)
    usable[label_col] = valid.loc[valid.notna()].astype(int).to_numpy()
    if usable.empty:
        raise ValueError("No labelled shots remain after excluding unknown/provisional outcomes.")
    if usable[label_col].nunique() < 2:
        raise ValueError("Training requires both goal (1) and non-goal (0) shot outcomes.")
    if "GAME_ID" not in usable:
        raise ValueError("GAME_ID is required so evaluation can hold out complete games.")

    X = _feature_frame(usable)
    y = usable[label_col].astype(int)
    groups = usable["GAME_ID"].reset_index(drop=True)
    n_games = int(groups.nunique())
    evaluation: dict[str, Any] = {"status": "not_evaluated", "reason": None}
    if n_games >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=float(xg.get("test_size", 0.2)), random_state=int(xg.get("random_state", 42))
        )
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if y_train.nunique() >= 2 and len(y_test) > 0:
            group_train = groups.iloc[train_idx]
            eval_model, eval_calibration_status = _fit_with_optional_calibration(X_train, y_train, group_train, config or {})
            probabilities = eval_model.predict_proba(X_test)[:, 1]
            evaluation = _metrics(y_test.to_numpy(), probabilities)
            evaluation.update({
                "status": "held_out_games",
                "holdout_games": sorted({_canon(x) for x in groups.iloc[test_idx].unique()}),
                "training_games": sorted({_canon(x) for x in groups.iloc[train_idx].unique()}),
                "calibration_status": eval_calibration_status,
                "reliability_by_quantile": _reliability_table(y_test.to_numpy(), probabilities),
            })
        else:
            evaluation["reason"] = "The chosen game holdout left training with one goal class; model fitted without held-out evaluation."
    else:
        evaluation["reason"] = "At least two distinct GAME_ID values are required for leakage-resistant evaluation."

    final_model, calibration_status = _fit_with_optional_calibration(X, y, groups, config or {})
    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final_model,
        "model_version": XG_MODEL_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "xg_config": xg,
        "metadata": {
            "training_shots": int(len(usable)),
            "training_goals": int(y.sum()),
            "training_goal_rate": float(y.mean()),
            "training_games": n_games,
            "calibration_status": calibration_status,
            "target_definition": "1 = shot resulted in a goal for the shooting team; 0 = shot did not score",
            "feature_timing": "pre-shot only; no shot outcome, post-shot placement, or post-shot event is a predictor",
        },
    }
    joblib.dump(bundle, target)

    importance = _feature_importance(final_model)
    if importance_path is not None:
        importance_target = Path(importance_path)
        importance_target.parent.mkdir(parents=True, exist_ok=True)
        importance.to_csv(importance_target, index=False)
    summary = asdict(XGTrainingSummary(
        model_version=XG_MODEL_VERSION,
        model_path=str(target.resolve()),
        n_shots=int(len(usable)),
        n_goals=int(y.sum()),
        goal_rate=float(y.mean()),
        n_games=n_games,
        calibration_status=calibration_status,
        evaluation=evaluation,
        feature_importance_path=str(Path(importance_path).resolve()) if importance_path else None,
    ))
    if report_path is not None:
        report_target = Path(report_path)
        report_target.parent.mkdir(parents=True, exist_ok=True)
        report_target.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return {"summary": summary, "feature_importance": importance, "bundle": bundle}


def predict_xg(
    shot_features: pd.DataFrame,
    model_path: str | Path,
) -> pd.DataFrame:
    """Predict pre-shot xG probabilities from a previously trained model bundle."""
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install model dependencies using `pip install -e '.[xg]'`.") from exc
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "model_version" not in bundle:
        raise ValueError("The model file does not contain a recognized football xG model bundle")
    X = _feature_frame(shot_features)
    prediction = shot_features[[c for c in ("event_id", "GAME_ID", "start_timestep", "PLAYER_ID", "TEAM_ID", "shot_x", "shot_y") if c in shot_features]].copy()
    prediction["predicted_xg"] = np.clip(bundle["model"].predict_proba(X)[:, 1], 0.0, 1.0)
    prediction["model_version"] = bundle["model_version"]
    prediction["calibration_status"] = bundle.get("metadata", {}).get("calibration_status", "unknown")
    return prediction


def make_shot_feature_table_from_files(
    tracking_paths: list[str] | tuple[str, ...],
    events_path: str | Path,
    config: dict[str, Any],
    label_column: str | None = None,
    include_labels: bool = True,
) -> pd.DataFrame:
    """Convenience file API for a list of tracking CSVs and a consolidated event CSV."""
    from .data_loader import load_tracking_data_sources

    tracking, _ = load_tracking_data_sources(list(tracking_paths), config)
    events = pd.read_csv(events_path, low_memory=False)
    return build_shot_feature_table(
        tracking=tracking,
        events=events,
        config=config,
        label_column=label_column,
        include_labels=include_labels,
    )
