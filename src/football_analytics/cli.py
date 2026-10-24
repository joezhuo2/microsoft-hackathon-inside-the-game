from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import AnalyticsConfig, DEFAULT_CONFIG_PATH
from .data_loader import load_tracking_data
from .event_detection import detect_events
from .export import export_outputs
from .match_intelligence import detect_key_moments, enrich_events, load_roster, match_clock, match_rhythm, player_match_intel, player_speed_tags
from .player_stats import aggregate_player_stats
from .possession import build_possession_chains, infer_possession
from .preprocessing import validate_tracking_frame
from .quality_checks import generate_quality_report
from .team_stats import aggregate_team_stats
from .tracking_features import compute_tracking_features


def run_analysis(input_path: str, output_dir: str = "output", config_path: str | None = None) -> dict[str, str]:
    settings_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = AnalyticsConfig.from_yaml(settings_path)
    tracking = load_tracking_data(input_path, normalize_coordinates=config.normalize_source_coordinates)
    cleaned, validation_report = validate_tracking_frame(tracking)
    features = compute_tracking_features(cleaned)
    event_parameters_path = Path(config.event_detection_parameters_path)
    if not event_parameters_path.is_absolute():
        event_parameters_path = settings_path.parent / event_parameters_path
    if event_parameters_path.exists():
        with event_parameters_path.open("r", encoding="utf-8") as handle:
            event_parameters = yaml.safe_load(handle) or {}
    else:
        event_parameters = None
    events = enrich_events(detect_events(cleaned, event_parameters), features, config.attacking_directions, config.seconds_per_timestep)
    possession = infer_possession(features)
    chains = build_possession_chains(possession)
    players = aggregate_player_stats(possession)
    teams = aggregate_team_stats(possession)
    quality = generate_quality_report(possession)
    calibration_status = (
        "xy_normalized_to_120x80m; horizontal_velocity_metres_per_timestep; time_and_z_unverified"
        if config.normalize_source_coordinates
        else "native_horizontal_units; time_and_z_unverified"
    )
    quality.update({"validation": validation_report, "calibration_status": calibration_status})
    roster_path = Path(config.roster_path)
    roster = load_roster(roster_path if roster_path.is_absolute() else settings_path.parent / roster_path)
    tags = player_speed_tags(features, config.seconds_per_timestep)
    rhythm = match_rhythm(possession, events, config.attacking_directions)
    moments = detect_key_moments(features, events, rhythm, tags, config.attacking_directions)
    moments.insert(4, "match_clock", moments["start_timestep"].map(lambda t: match_clock(t, config.timesteps_per_match, config.match_minutes)))
    intel = player_match_intel(events, features, tags, config.seconds_per_timestep, roster)
    paths = export_outputs(possession, events, chains, players, teams, quality, output_dir, {"key_moments": moments, "match_rhythm": rhythm, "player_tags": tags, "player_match_intel": intel})
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Football tracking analytics pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a tracking CSV")
    analyze.add_argument("--input", required=True, help="Path to input tracking CSV")
    analyze.add_argument("--output", default="output", help="Output directory")
    analyze.add_argument("--config", help="Path to analysis configuration YAML")

    validate = subparsers.add_parser("validate", help="Validate input CSV")
    validate.add_argument("--input", required=True, help="Path to input tracking CSV")

    summary = subparsers.add_parser("summary", help="Print a short data summary")
    summary.add_argument("--input", required=True, help="Path to input tracking CSV")

    args = parser.parse_args()

    if args.command == "validate":
        df = load_tracking_data(args.input)
        _, report = validate_tracking_frame(df)
        print(f"Games: {report['quality_checks']['games']}")
        print(f"Frames: {report['quality_checks']['timesteps']}")
        print(f"Rows: {len(df)}")
        return 0

    if args.command == "summary":
        df = load_tracking_data(args.input)
        print(f"Rows: {len(df)}")
        print(f"Games: {df['GAME_ID'].nunique()}")
        print(f"Unique players: {df['PLAYER_ID'].nunique()}")
        print(f"Ball rows: {(df['PLAYER_ID'] == -1).sum()}")
        return 0

    if args.command == "analyze":
        output_dir = Path(args.output)
        result = run_analysis(args.input, str(output_dir), args.config)
        print("Analysis completed")
        for name, path in result.items():
            print(f"{name}: {path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
