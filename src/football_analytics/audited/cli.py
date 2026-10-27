"""Command-line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .data_loader import load_tracking_data_sources, resolve_input_paths, DataValidationError
from .pipeline import run_analysis
from .xg_model import build_shot_feature_table, train_xgboost_xg_model, predict_xg


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="football-analytics", description="Auditable football tracking analytics")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="Run the end-to-end pipeline")
    analyze.add_argument("--input", required=True, nargs="+", help="Input tracking CSV(s), glob pattern(s), or batch directory")
    analyze.add_argument("--output", required=True, help="Output directory")
    analyze.add_argument("--config", help="Optional YAML configuration")
    validate = sub.add_parser("validate", help="Validate a CSV and print source-data diagnostics")
    validate.add_argument("--input", required=True, nargs="+", help="Input tracking CSV(s), glob pattern(s), or batch directory")
    validate.add_argument("--config", help="Optional YAML configuration")
    summary = sub.add_parser("summary", help="Print a concise source-data summary")
    summary.add_argument("--input", required=True, nargs="+", help="Input tracking CSV(s), glob pattern(s), or batch directory")
    summary.add_argument("--config", help="Optional YAML configuration")

    xg_train = sub.add_parser("xg-train", help="Train an XGBoost expected-goals model from labelled shots")
    xg_train.add_argument("--tracking", required=True, nargs="+", help="Tracking CSV(s), glob pattern(s), or batch directory")
    xg_train.add_argument("--events", required=True, nargs="+", help="Event CSV(s) containing shot events and goal outcomes")
    xg_train.add_argument("--model", default="models/xgboost_xg.joblib", help="Output fitted model bundle")
    xg_train.add_argument("--report", default="output/xg_training_report.json", help="Output evaluation report JSON")
    xg_train.add_argument("--importance", default="output/xg_feature_importance.csv", help="Output feature-importance CSV")
    xg_train.add_argument("--features-output", help="Optional output shot-level feature table CSV")
    xg_train.add_argument("--label-column", help="Optional explicit binary goal-label column in the event data")
    xg_train.add_argument("--config", help="Optional YAML configuration")

    xg_predict = sub.add_parser("xg-predict", help="Score shot events with a saved XGBoost xG model")
    xg_predict.add_argument("--tracking", required=True, nargs="+", help="Tracking CSV(s), glob pattern(s), or batch directory")
    xg_predict.add_argument("--events", required=True, nargs="+", help="Event CSV(s); may be a full event log or shot-only table")
    xg_predict.add_argument("--model", required=True, help="Fitted model bundle created by xg-train")
    xg_predict.add_argument("--output", required=True, help="Output shot-level xG predictions CSV")
    xg_predict.add_argument("--config", help="Optional YAML configuration; pitch/direction must match training")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command in {"xg-train", "xg-predict"}:
            tracking, tracking_report = load_tracking_data_sources(args.tracking, config)
            event_paths = resolve_input_paths(args.events)
            event_frames = [pd.read_csv(path, low_memory=False) for path in event_paths]
            reference_columns = list(event_frames[0].columns)
            for path, frame in zip(event_paths[1:], event_frames[1:]):
                if set(frame.columns) != set(reference_columns):
                    raise DataValidationError(f"Event schema mismatch in {path}; event batches must have matching columns")
            events = pd.concat(event_frames, ignore_index=True, copy=False)
            if args.command == "xg-train":
                features = build_shot_feature_table(
                    tracking, events, config=config, label_column=args.label_column, include_labels=True
                )
                if args.features_output:
                    feature_target = Path(args.features_output)
                    feature_target.parent.mkdir(parents=True, exist_ok=True)
                    features.to_csv(feature_target, index=False)
                result = train_xgboost_xg_model(
                    features, model_path=args.model, config=config, report_path=args.report,
                    importance_path=args.importance, label_col="is_goal",
                )
                print(json.dumps(result["summary"], indent=2, ensure_ascii=False, default=str))
                print(f"Model: {Path(args.model).resolve()}")
                print(f"Report: {Path(args.report).resolve()}")
                print(f"Tracking QA: {tracking_report['games']} game(s), {tracking_report['frames']} frame(s)")
                return 0
            features = build_shot_feature_table(
                tracking, events, config=config, include_labels=False
            )
            predictions = predict_xg(features, args.model)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(output_path, index=False)
            print(f"Scored {len(predictions)} shot(s). Output: {output_path.resolve()}")
            return 0

        if args.command == "analyze":
            report = run_analysis(args.input, args.output, config)
            analysis = report["analysis"]
            print(f"Analysis complete: {analysis['games_processed']} game(s), {analysis['frames_processed']} frame(s), {analysis['events_detected']} candidate event(s).")
            print(f"Validation checks: {analysis['validation_checks_passed']}/{analysis['validation_checks_total']} passed.")
            print(f"Outputs: {Path(args.output).resolve()}")
            print(f"Quality report: {(Path(args.output) / 'quality_report.json').resolve()}")
            return 0 if analysis["all_validation_checks_passed"] else 2
        input_paths = resolve_input_paths(args.input)
        raw, report = load_tracking_data_sources(input_paths, config)
        if args.command == "validate":
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            print("Source validation complete.")
            return 0
        if args.command == "summary":
            print(f"File(s): {', '.join(str(path) for path in input_paths)}")
            print(f"Rows: {len(raw):,} | Games: {raw['GAME_ID'].nunique():,} | Frames: {raw[['GAME_ID', 'TIMESTEP']].drop_duplicates().shape[0]:,}")
            print(f"Entities: {raw['PLAYER_ID'].nunique():,} unique PLAYER_ID values (ball included)")
            print(f"Timeline gaps: {report['timeline_gaps']['count']}")
            print(f"Warnings: {len(report.get('warnings', []))}")
            return 0
    except (FileNotFoundError, DataValidationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # ensure useful command-line diagnostics for unexpected malformed input
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
