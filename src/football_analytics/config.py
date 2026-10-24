from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AnalyticsConfig:
    input_path: str = ""
    output_dir: str = "output"
    frame_rate: float = 25.0
    pitch_length: float = 120.0
    pitch_width: float = 80.0
    seconds_per_frame: float = 1.0 / 25.0
    attacking_directions: dict[int, float] = field(default_factory=lambda: {0: 1.0, 1: -1.0})
    possession_distance_threshold: float = 1.2
    pass_distance_threshold: float = 0.5
    ball_release_velocity_threshold: float = 0.2
    progressive_distance_threshold: float = 5.0
    sprint_threshold_mps: float = 6.0
    speed_band_thresholds: list[float] = field(default_factory=lambda: [0.0, 2.0, 4.0, 6.0, 8.0])
    event_confidence_thresholds: dict[str, float] = field(default_factory=lambda: {"low": 0.2, "medium": 0.5, "high": 0.8})
    use_physical_calibration: bool = False
    normalize_source_coordinates: bool = False
    event_detection_parameters_path: str = "football_event_detection_parameters_pitch120x80.yaml"
    seconds_per_timestep: float = 0.1
    match_minutes: float = 90.0
    timesteps_per_match: int = 3000
    roster_path: str = "data/roster.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "frame_rate": self.frame_rate,
            "pitch_length": self.pitch_length,
            "pitch_width": self.pitch_width,
            "seconds_per_frame": self.seconds_per_frame,
            "attacking_directions": self.attacking_directions,
            "possession_distance_threshold": self.possession_distance_threshold,
            "pass_distance_threshold": self.pass_distance_threshold,
            "ball_release_velocity_threshold": self.ball_release_velocity_threshold,
            "progressive_distance_threshold": self.progressive_distance_threshold,
            "sprint_threshold_mps": self.sprint_threshold_mps,
            "speed_band_thresholds": self.speed_band_thresholds,
            "event_confidence_thresholds": self.event_confidence_thresholds,
            "use_physical_calibration": self.use_physical_calibration,
            "normalize_source_coordinates": self.normalize_source_coordinates,
            "event_detection_parameters_path": self.event_detection_parameters_path,
            "seconds_per_timestep": self.seconds_per_timestep,
            "match_minutes": self.match_minutes,
            "timesteps_per_match": self.timesteps_per_match,
            "roster_path": self.roster_path,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnalyticsConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls(**data)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
