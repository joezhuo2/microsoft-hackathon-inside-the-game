"""End-to-end analysis orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .data_loader import load_tracking_data_sources
from .preprocessing import preprocess_tracking_data
from .tracking_features import derive_tracking_features
from .possession import build_possession_frame_table, attach_possession_to_entities, detect_possession_chains
from .event_detection import detect_events
from .player_stats import build_player_match_stats
from .team_stats import build_team_match_stats
from .value_models import build_model_status_table
from .quality_checks import build_quality_report
from .export import export_outputs


def run_analysis(input_path: str | Path | Sequence[str | Path], output_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    inputs = [input_path] if isinstance(input_path, (str, Path)) else list(input_path)
    raw, source_report = load_tracking_data_sources(inputs, config)
    canonical = preprocess_tracking_data(raw, config)
    feature_base = derive_tracking_features(canonical, config)
    possession_frames = build_possession_frame_table(canonical, config)
    features = attach_possession_to_entities(feature_base, possession_frames)
    chains = detect_possession_chains(possession_frames, config)
    events = detect_events(possession_frames, config)
    player_stats = build_player_match_stats(features, possession_frames, events, config)
    team_stats = build_team_match_stats(features, possession_frames, events, chains, config)
    model_status = build_model_status_table(config)
    report = build_quality_report(source_report, raw, features, possession_frames, events, chains,
                                  player_stats, team_stats, config, output_dir)
    paths = export_outputs(output_dir, features, events, chains, player_stats, team_stats, model_status, report)
    report["analysis"]["written_files"] = [p.name for p in paths]
    # Rewrite report after adding the exact written file inventory.
    from .quality_checks import write_json
    write_json(Path(output_dir) / "quality_report.json", report)
    return report
