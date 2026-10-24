from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = [
    "GAME_ID",
    "TIMESTEP",
    "TEAM_0_SCORE",
    "TEAM_1_SCORE",
    "PLAYER_ID",
    "TEAM_ID",
    "PLAYER_ROLE",
    "LOCATION_X",
    "LOCATION_Y",
    "LOCATION_Z",
    "VELOCITY_X",
    "VELOCITY_Y",
    "VELOCITY_Z",
    "ON_BALL",
]


def _coerce_numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def normalize_pitch_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """Map centered source coordinates to metres on a 120 x 80 pitch."""
    out = frame.copy()
    columns = {
        "LOCATION_X": "SOURCE_LOCATION_X",
        "LOCATION_Y": "SOURCE_LOCATION_Y",
        "VELOCITY_X": "SOURCE_VELOCITY_X",
        "VELOCITY_Y": "SOURCE_VELOCITY_Y",
    }
    if all(source in out.columns for source in columns.values()):
        return out

    for column, source in columns.items():
        if column not in out.columns:
            raise ValueError(f"Missing coordinate column: {column}")
        out[source] = out[column]

    out["LOCATION_X"] = (out["SOURCE_LOCATION_X"] + 1.0) * 60.0
    out["LOCATION_Y"] = (out["SOURCE_LOCATION_Y"] + 0.42) * (80.0 / 0.84)
    out["VELOCITY_X"] = out["SOURCE_VELOCITY_X"] * 60.0
    out["VELOCITY_Y"] = out["SOURCE_VELOCITY_Y"] * (80.0 / 0.84)
    return out


def load_tracking_data(path: str | Path, normalize_coordinates: bool = False) -> pd.DataFrame:
    """Load and minimally validate a football-tracking CSV file."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {data_path}")

    frame = pd.read_csv(data_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = _coerce_numeric_columns(frame, [
        "GAME_ID", "TIMESTEP", "TEAM_0_SCORE", "TEAM_1_SCORE", "PLAYER_ID",
        "TEAM_ID", "LOCATION_X", "LOCATION_Y", "LOCATION_Z", "VELOCITY_X",
        "VELOCITY_Y", "VELOCITY_Z",
    ])

    if "ON_BALL" in frame.columns:
        frame["ON_BALL"] = frame["ON_BALL"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes", "y", "on"}
            if pd.notna(value) else False
        )

    frame["PLAYER_ROLE"] = frame["PLAYER_ROLE"].astype(object)
    if normalize_coordinates:
        frame = normalize_pitch_coordinates(frame)
    return frame
