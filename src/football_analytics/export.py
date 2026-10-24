from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def export_outputs(frame: pd.DataFrame, events: pd.DataFrame, possession_chains: pd.DataFrame, player_stats: pd.DataFrame, team_stats: pd.DataFrame, quality_report: dict[str, Any], output_dir: str | Path, extra_tables: dict[str, pd.DataFrame] | None = None) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame_path = output_path / "frame_metrics.csv"
    events_path = output_path / "events.csv"
    chains_path = output_path / "possession_chains.csv"
    players_path = output_path / "player_match_stats.csv"
    teams_path = output_path / "team_match_stats.csv"
    quality_path = output_path / "quality_report.json"
    metadata_path = output_path / "metric_metadata.json"

    frame.to_csv(frame_path, index=False)
    events.to_csv(events_path, index=False)
    possession_chains.to_csv(chains_path, index=False)
    player_stats.to_csv(players_path, index=False)
    team_stats.to_csv(teams_path, index=False)

    with quality_path.open("w", encoding="utf-8") as handle:
        json.dump(quality_report, handle, indent=2, sort_keys=True)

    metadata = {
        "frame_metrics": {"description": "One row per game-frame-entity with movement and spatial features; SOURCE_* fields preserve source horizontal values.", "units": "LOCATION_X/Y: metres on 120x80 pitch when configured; horizontal velocity: metres per timestep; Z and timing remain source units"},
        "events": {"description": "Candidate event list with evidence and confidence.", "units": "frame index and team/player identifiers"},
        "possession_chains": {"description": "Ownership segments for team possession.", "units": "frames and team ids"},
        "player_match_stats": {"description": "Aggregated per-player per-game match stats", "units": "distance in configured horizontal coordinates; speed in metres per timestep when normalized"},
        "team_match_stats": {"description": "Aggregated per-team per-game stats", "units": "distance in configured horizontal coordinates; speed in metres per timestep when normalized"},
        "key_moments": {"description": "Rule-based candidate moments for agent explanation; each cites event_ids where applicable. importance is a fixed per-type weight, not a model score.", "units": "timesteps; match_clock maps timesteps_per_match onto match_minutes for display only"},
        "match_rhythm": {"description": "150-timestep windows: possession share, territory (0-1 toward team 0's attacking goal), control_index = completed passes / (completed passes + turnovers), momentum_team0 in [-1, 1].", "units": "fractions and counts"},
        "player_tags": {"description": "Threshold-triggered on-screen tags: sprints (>= 32 km/h for >= 10 timesteps) and 1000 m distance milestones.", "units": "km/h and metres, using the configured seconds_per_timestep"},
        "player_match_intel": {"description": "Per-player per-game broadcast stats with synthetic roster names. pass_difficulty 0-100 = 40*distance/40m + 35*(1 - lane_pressure/10m) + 15*progression/30m + subtype bonus (through ball 10, switch/cross 5).", "units": "metres, km/h using the configured seconds_per_timestep, counts"},
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    extra_paths = {}
    for name, table in (extra_tables or {}).items():
        extra_paths[name] = str(output_path / f"{name}.csv")
        table.to_csv(extra_paths[name], index=False)

    return extra_paths | {
        "frame_metrics": str(frame_path),
        "events": str(events_path),
        "possession_chains": str(chains_path),
        "player_match_stats": str(players_path),
        "team_match_stats": str(teams_path),
        "quality_report": str(quality_path),
        "metric_metadata": str(metadata_path),
    }
