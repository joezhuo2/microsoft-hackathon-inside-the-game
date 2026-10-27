"""CSV ingestion and source-data quality diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import glob
import re

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "GAME_ID", "TIMESTEP", "TEAM_0_SCORE", "TEAM_1_SCORE", "PLAYER_ID", "TEAM_ID",
    "PLAYER_ROLE", "LOCATION_X", "LOCATION_Y", "LOCATION_Z", "VELOCITY_X",
    "VELOCITY_Y", "VELOCITY_Z", "ON_BALL",
]
NUMERIC_COLUMNS = [
    "TIMESTEP", "TEAM_0_SCORE", "TEAM_1_SCORE", "PLAYER_ID", "TEAM_ID",
    "LOCATION_X", "LOCATION_Y", "LOCATION_Z", "VELOCITY_X", "VELOCITY_Y", "VELOCITY_Z",
]
KEY_COLUMNS = ["GAME_ID", "TIMESTEP", "PLAYER_ID"]


class DataValidationError(ValueError):
    """Raised when data ambiguity prevents safe processing."""


def _frame_score_diagnostics(df: pd.DataFrame) -> tuple[int, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    for (game_id, timestep), group in df.groupby(["GAME_ID", "TIMESTEP"], dropna=False, sort=False):
        for score_col in ("TEAM_0_SCORE", "TEAM_1_SCORE"):
            n = group[score_col].nunique(dropna=False)
            if n > 1:
                errors.append({
                    "game_id": _json_scalar(game_id), "timestep": _json_scalar(timestep),
                    "column": score_col, "distinct_values": int(n),
                    "severity": "warning", "message": "Repeated score values disagree within a game-frame; score is treated as unknown for that frame.",
                })
    return len(errors), errors


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_tracking_data(path: str | Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a tracking CSV, validate the schema and report anomalies.

    Exact duplicate entity-frame rows are removed and counted. Conflicting duplicate
    rows are rejected because choosing one would fabricate a tracking observation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pragma: no cover - pandas exception types vary
        raise DataValidationError(f"Could not read CSV {path}: {exc}") from exc
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if df.empty:
        raise DataValidationError("Input CSV contains no rows")

    # Preserve source columns/order, while coercing fields that are required to be numeric.
    type_coercions: dict[str, int] = {}
    for col in NUMERIC_COLUMNS:
        before_non_null = int(df[col].notna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        type_coercions[col] = max(0, before_non_null - int(df[col].notna().sum()))

    report: dict[str, Any] = {
        "input_file": str(path.resolve()),
        "input_rows": int(len(df)),
        "input_columns": list(df.columns),
        "required_columns_present": True,
        "type_coercion_failures_by_column": {k: v for k, v in type_coercions.items() if v},
        "warnings": [],
        "errors": [],
        "duplicates": {"exact_rows_removed": 0, "conflicting_entity_frame_keys": 0},
    }
    missing_required_values = {col: int(df[col].isna().sum()) for col in ["GAME_ID", *NUMERIC_COLUMNS] if col in df}
    report["missing_required_values_by_column"] = {k: v for k, v in missing_required_values.items() if v}
    if df["GAME_ID"].isna().any() or df["TIMESTEP"].isna().any() or df["PLAYER_ID"].isna().any():
        raise DataValidationError("GAME_ID, TIMESTEP and PLAYER_ID must not be null")
    # Duplicates should be handled explicitly, never silently resolved by first/last row.
    duplicated = df.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        dupes = df.loc[duplicated]
        conflicting_keys = []
        exact_drop_indices: list[int] = []
        for key, group in dupes.groupby(KEY_COLUMNS, dropna=False, sort=False):
            # A key is safe to collapse only if every source field agrees (including NA).
            if len(group.drop_duplicates()) == 1:
                exact_drop_indices.extend(group.index[1:].tolist())
            else:
                conflicting_keys.append(tuple(_json_scalar(x) for x in (key if isinstance(key, tuple) else (key,))))
        report["duplicates"]["exact_rows_removed"] = len(exact_drop_indices)
        report["duplicates"]["conflicting_entity_frame_keys"] = len(conflicting_keys)
        if conflicting_keys:
            report["errors"].append({
                "severity": "error", "message": "Conflicting observations share (GAME_ID, TIMESTEP, PLAYER_ID); refusing to choose one.",
                "examples": [list(k) for k in conflicting_keys[:10]],
            })
            raise DataValidationError(report["errors"][-1]["message"] + f" Examples: {conflicting_keys[:5]}")
        if exact_drop_indices:
            df = df.drop(index=exact_drop_indices)
            report["warnings"].append(f"Removed {len(exact_drop_indices)} exact duplicate entity-frame rows; see duplicates report.")

    score_n, score_diags = _frame_score_diagnostics(df)
    if score_n:
        report["warnings"].append(f"{score_n} game-frame score inconsistencies detected; affected score values will be null.")
        report["score_inconsistencies"] = score_diags[:100]

    df = df.sort_values(["GAME_ID", "TIMESTEP", "PLAYER_ID"], kind="mergesort").reset_index(drop=True)
    frames = df[["GAME_ID", "TIMESTEP"]].drop_duplicates().sort_values(["GAME_ID", "TIMESTEP"])
    gap_records: list[dict[str, Any]] = []
    for game_id, group in frames.groupby("GAME_ID", sort=False):
        ts = group["TIMESTEP"].to_numpy(dtype=float)
        diffs = np.diff(ts)
        for idx in np.flatnonzero(diffs > 1):
            gap_records.append({"game_id": _json_scalar(game_id), "from_timestep": float(ts[idx]), "to_timestep": float(ts[idx + 1]), "missing_timestep_count": int(max(0, round(diffs[idx]) - 1))})
    report["timeline_gaps"] = {"count": len(gap_records), "examples": gap_records[:100]}
    report["games"] = int(df["GAME_ID"].nunique())
    report["frames"] = int(len(frames))
    report["rows_after_exact_deduplication"] = int(len(df))
    report["timesteps_by_game"] = {
        str(_json_scalar(game_id)): int(group["TIMESTEP"].nunique()) for game_id, group in df.groupby("GAME_ID", sort=False)
    }

    # Entity coverage and persistent team-assignment diagnostics.
    frame_counts = df.groupby(["GAME_ID", "TIMESTEP"], sort=False).size()
    players_per_frame = df[df["PLAYER_ID"] != -1].groupby(["GAME_ID", "TIMESTEP"], sort=False)["PLAYER_ID"].nunique()
    ball_per_frame = df[df["PLAYER_ID"] == -1].groupby(["GAME_ID", "TIMESTEP"], sort=False).size()
    expected_players = config.get("expected_players_per_frame", 22)
    report["frame_entity_coverage"] = {
        "entity_rows_min": int(frame_counts.min()), "entity_rows_median": float(frame_counts.median()), "entity_rows_max": int(frame_counts.max()),
        "player_count_min": int(players_per_frame.min()) if len(players_per_frame) else 0,
        "player_count_median": float(players_per_frame.median()) if len(players_per_frame) else 0,
        "player_count_max": int(players_per_frame.max()) if len(players_per_frame) else 0,
        "frames_without_ball": int((ball_per_frame.reindex(frame_counts.index, fill_value=0) == 0).sum()),
        "frames_with_multiple_ball_rows": int((ball_per_frame.reindex(frame_counts.index, fill_value=0) > 1).sum()),
        "configured_expected_player_count": expected_players,
        "frames_not_matching_expected_player_count": int((players_per_frame.reindex(frame_counts.index, fill_value=0) != expected_players).sum()) if expected_players is not None else None,
    }
    if report["frame_entity_coverage"]["frames_without_ball"]:
        report["warnings"].append("Some game-frames have no ball row; ball-relative metrics are unavailable for those frames.")
    if report["frame_entity_coverage"]["frames_with_multiple_ball_rows"]:
        report["warnings"].append("Some game-frames have multiple rows marked as the ball; ball coordinates will be treated as ambiguous.")
    if expected_players is not None and report["frame_entity_coverage"]["frames_not_matching_expected_player_count"]:
        report["warnings"].append("Some frames differ from the configured expected player count; this is a diagnostic, not a hard-coded sample constraint.")

    player_rows = df[df["PLAYER_ID"] != -1]
    assignment_changes = []
    for (game_id, player_id), group in player_rows.groupby(["GAME_ID", "PLAYER_ID"], sort=False):
        team_ids = group["TEAM_ID"].dropna().unique().tolist()
        if len(team_ids) > 1:
            assignment_changes.append({"game_id": _json_scalar(game_id), "player_id": _json_scalar(player_id), "team_ids": [_json_scalar(x) for x in team_ids]})
    report["inconsistent_player_team_assignments"] = assignment_changes[:100]
    if assignment_changes:
        report["warnings"].append(f"{len(assignment_changes)} player-game identities change TEAM_ID across frames.")

    report["missing_values_by_column"] = {c: int(n) for c, n in df.isna().sum().items() if n}
    report["sentinel_handling"] = {
        "player_LOCATION_Z_equals_minus_1": "treated as unavailable during preprocessing",
        "player_VELOCITY_Z_equals_minus_1": "treated as unavailable during preprocessing",
        "ball_vertical_values": "preserved if valid",
    }
    return df, report



def resolve_input_paths(inputs: Sequence[str | Path]) -> list[Path]:
    """Expand explicit CSV paths, quoted glob patterns, or directories of batch CSVs.

    Directory inputs prefer ``gfootball_results_games_*.csv`` files to avoid
    accidentally ingesting analysis-output CSVs. If no matching batch files are
    present, all top-level CSVs in that directory are considered.
    """
    resolved: list[Path] = []
    for item in inputs:
        value = str(item)
        path = Path(value)
        if path.is_dir():
            matches = sorted(path.glob("gfootball_results_games_*.csv"))
            if not matches:
                matches = sorted(path.glob("*.csv"))
            if not matches:
                raise FileNotFoundError(f"No CSV files found in input directory: {path}")
        elif any(char in value for char in "*?["):
            matches = [Path(match) for match in sorted(glob.glob(value))]
            if not matches:
                raise FileNotFoundError(f"Input pattern matched no files: {value}")
        else:
            if not path.exists():
                raise FileNotFoundError(f"Input CSV not found: {path}")
            if not path.is_file():
                raise DataValidationError(f"Input is not a file or directory: {path}")
            matches = [path]
        for match in matches:
            if not match.is_file():
                raise DataValidationError(f"Input is not a regular file: {match}")
            candidate = match.resolve()
            if candidate.suffix.lower() != ".csv":
                raise DataValidationError(f"Input must be a CSV file: {candidate}")
            if candidate not in resolved:
                resolved.append(candidate)

    if not resolved:
        raise FileNotFoundError("No input CSV files were supplied")

    # Numeric batch ranges sort naturally (0-2, 3-5, ..., 12-14).
    def batch_sort_key(path: Path) -> tuple[int, int, str]:
        match = re.search(r"games[_-](\d+)[-_](\d+)", path.name, flags=re.IGNORECASE)
        return (0, int(match.group(1)), path.name.lower()) if match else (1, 0, path.name.lower())

    return sorted(resolved, key=batch_sort_key)


def _coverage_summary(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Calculate entity coverage metrics for one or multiple source files."""
    frame_counts = df.groupby(["GAME_ID", "TIMESTEP"], sort=False).size()
    players_per_frame = df[df["PLAYER_ID"] != -1].groupby(["GAME_ID", "TIMESTEP"], sort=False)["PLAYER_ID"].nunique()
    ball_per_frame = df[df["PLAYER_ID"] == -1].groupby(["GAME_ID", "TIMESTEP"], sort=False).size()
    expected_players = config.get("expected_players_per_frame", 22)
    players_per_frame = players_per_frame.reindex(frame_counts.index, fill_value=0)
    ball_per_frame = ball_per_frame.reindex(frame_counts.index, fill_value=0)
    return {
        "entity_rows_min": int(frame_counts.min()),
        "entity_rows_median": float(frame_counts.median()),
        "entity_rows_max": int(frame_counts.max()),
        "player_count_min": int(players_per_frame.min()) if len(players_per_frame) else 0,
        "player_count_median": float(players_per_frame.median()) if len(players_per_frame) else 0,
        "player_count_max": int(players_per_frame.max()) if len(players_per_frame) else 0,
        "frames_without_ball": int((ball_per_frame == 0).sum()),
        "frames_with_multiple_ball_rows": int((ball_per_frame > 1).sum()),
        "configured_expected_player_count": expected_players,
        "frames_not_matching_expected_player_count": int((players_per_frame != expected_players).sum()) if expected_players is not None else None,
    }


def load_tracking_data_sources(inputs: Sequence[str | Path], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one or more non-overlapping match CSVs and combine their diagnostics.

    This deliberately rejects repeated GAME_ID values across files. Automatically
    renumbering matches can corrupt identifiers and mask overlaps; users should
    fix or explicitly remap those IDs upstream if their batches restart numbering.
    """
    paths = resolve_input_paths(inputs)
    if len(paths) == 1:
        return load_tracking_data(paths[0], config)

    loaded: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    games_to_source: dict[Any, str] = {}
    reference_columns: list[str] | None = None
    for path in paths:
        frame, report = load_tracking_data(path, config)
        columns = list(frame.columns)
        if reference_columns is None:
            reference_columns = columns
        elif set(columns) != set(reference_columns):
            missing = sorted(set(reference_columns) - set(columns))
            extra = sorted(set(columns) - set(reference_columns))
            raise DataValidationError(
                f"CSV batch schemas differ for {path.name}; missing columns={missing}, extra columns={extra}. "
                "Use batches with a consistent schema before combining them."
            )
        game_ids = frame["GAME_ID"].drop_duplicates().tolist()
        overlaps = [game_id for game_id in game_ids if game_id in games_to_source]
        if overlaps:
            details = [f"GAME_ID {game_id!r} appears in {games_to_source[game_id]} and {path.name}" for game_id in overlaps[:10]]
            raise DataValidationError(
                "Overlapping GAME_ID values across input files are not combined automatically: "
                + "; ".join(details)
                + ". If each batch restarts GAME_ID numbering, remap IDs to globally unique values before analysis."
            )
        for game_id in game_ids:
            games_to_source[game_id] = path.name
        loaded.append(frame[reference_columns])
        reports.append(report)

    combined = pd.concat(loaded, ignore_index=True, copy=False)
    combined = combined.sort_values(["GAME_ID", "TIMESTEP", "PLAYER_ID"], kind="mergesort").reset_index(drop=True)

    all_warnings: list[str] = []
    for path, report in zip(paths, reports):
        all_warnings.extend(f"[{path.name}] {warning}" for warning in report.get("warnings", []))

    def sum_dicts(key: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for report in reports:
            for name, value in report.get(key, {}).items():
                result[name] = result.get(name, 0) + int(value)
        return {name: value for name, value in result.items() if value}

    gap_examples: list[dict[str, Any]] = []
    score_examples: list[dict[str, Any]] = []
    assignment_examples: list[dict[str, Any]] = []
    source_file_reports = []
    gaps_count = 0
    for path, report in zip(paths, reports):
        gaps_count += int(report.get("timeline_gaps", {}).get("count", 0))
        gap_examples.extend(report.get("timeline_gaps", {}).get("examples", []))
        score_examples.extend(report.get("score_inconsistencies", []))
        assignment_examples.extend(report.get("inconsistent_player_team_assignments", []))
        source_file_reports.append({
            "input_file": str(path),
            "input_rows": int(report["input_rows"]),
            "games": int(report["games"]),
            "frames": int(report["frames"]),
            "game_ids": sorted(report["timesteps_by_game"].keys(), key=lambda x: (not x.lstrip("-").isdigit(), int(x) if x.lstrip("-").isdigit() else x)),
            "warnings": report.get("warnings", []),
        })

    per_file_columns = reference_columns or []
    missing_values: dict[str, int] = {}
    for report in reports:
        for column, count in report.get("missing_values_by_column", {}).items():
            missing_values[column] = missing_values.get(column, 0) + int(count)
    source_report: dict[str, Any] = {
        "input_file": ", ".join(str(path) for path in paths),
        "input_files": [str(path) for path in paths],
        "source_file_reports": source_file_reports,
        "input_rows": int(sum(report["input_rows"] for report in reports)),
        "input_columns": per_file_columns,
        "required_columns_present": True,
        "type_coercion_failures_by_column": sum_dicts("type_coercion_failures_by_column"),
        "warnings": all_warnings,
        "errors": [],
        "duplicates": {
            "exact_rows_removed": int(sum(r["duplicates"]["exact_rows_removed"] for r in reports)),
            "conflicting_entity_frame_keys": int(sum(r["duplicates"]["conflicting_entity_frame_keys"] for r in reports)),
        },
        "missing_required_values_by_column": sum_dicts("missing_required_values_by_column"),
        "timeline_gaps": {"count": gaps_count, "examples": gap_examples[:100]},
        "games": int(combined["GAME_ID"].nunique()),
        "frames": int(combined[["GAME_ID", "TIMESTEP"]].drop_duplicates().shape[0]),
        "rows_after_exact_deduplication": int(len(combined)),
        "timesteps_by_game": {
            str(_json_scalar(game_id)): int(group["TIMESTEP"].nunique())
            for game_id, group in combined.groupby("GAME_ID", sort=False)
        },
        "frame_entity_coverage": _coverage_summary(combined, config),
        "inconsistent_player_team_assignments": assignment_examples[:100],
        "missing_values_by_column": missing_values,
        "sentinel_handling": reports[0].get("sentinel_handling", {}),
    }
    if score_examples:
        source_report["score_inconsistencies"] = score_examples[:100]
    if source_report["frame_entity_coverage"]["frames_without_ball"]:
        source_report["warnings"].append("Some game-frames have no ball row; ball-relative metrics are unavailable for those frames.")
    if source_report["frame_entity_coverage"]["frames_with_multiple_ball_rows"]:
        source_report["warnings"].append("Some game-frames have multiple rows marked as the ball; ball coordinates will be treated as ambiguous.")
    expected_players = config.get("expected_players_per_frame", 22)
    if expected_players is not None and source_report["frame_entity_coverage"]["frames_not_matching_expected_player_count"]:
        source_report["warnings"].append("Some frames differ from the configured expected player count; this is a diagnostic, not a hard-coded sample constraint.")
    return combined, source_report
