"""Stable CSV/JSON exports."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .quality_checks import build_metric_metadata, write_json


def export_outputs(output_dir: str | Path, features: pd.DataFrame, events: pd.DataFrame,
                   chains: pd.DataFrame, player_stats: pd.DataFrame, team_stats: pd.DataFrame,
                   model_status: pd.DataFrame, quality_report: dict) -> list[Path]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    tables = {
        "frame_metrics.csv": features,
        "events.csv": events,
        "possession_chains.csv": chains,
        "player_match_stats.csv": player_stats,
        "team_match_stats.csv": team_stats,
        "model_outputs.csv": model_status,
    }
    paths: list[Path] = []
    for name, table in tables.items():
        dest = outdir / name
        table.to_csv(dest, index=False, na_rep="")
        paths.append(dest)
    write_json(outdir / "quality_report.json", quality_report)
    write_json(outdir / "metric_metadata.json", build_metric_metadata(tables))
    paths.extend([outdir / "quality_report.json", outdir / "metric_metadata.json"])
    return paths
