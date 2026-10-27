from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_analytics.audited.config import load_config
from football_analytics.audited.data_loader import DataValidationError, load_tracking_data
from football_analytics.audited.preprocessing import preprocess_tracking_data
from football_analytics.audited.tracking_features import derive_tracking_features
from football_analytics.audited.possession import build_possession_frame_table, attach_possession_to_entities, detect_possession_chains
from football_analytics.audited.event_detection import detect_events
from football_analytics.audited.pitch_control import compute_pitch_control_grid
from football_analytics.audited.pipeline import run_analysis
from football_analytics.audited.cli import main

ROOT = Path(__file__).resolve().parent / "fixtures" / "audited"


def synthetic(config=None):
    raw = pd.read_csv(ROOT / "examples" / "synthetic_tracking.csv")
    cfg = config or load_config(ROOT / "examples" / "config.yaml")
    return raw, cfg


def test_ingestion_schema_and_coverage_report():
    raw, cfg = synthetic()
    loaded, report = load_tracking_data(ROOT / "examples" / "synthetic_tracking.csv", cfg)
    assert len(loaded) == len(raw) == 2 * 8 * 23
    assert report["games"] == 2
    assert report["frames"] == 16
    assert report["frame_entity_coverage"]["frames_without_ball"] == 0


def test_ball_identification_and_player_vertical_sentinels():
    raw, cfg = synthetic()
    canon = preprocess_tracking_data(raw, cfg)
    assert (canon[canon["PLAYER_ID"].eq(-1)]["ENTITY_TYPE"] == "ball").all()
    players = canon[canon["PLAYER_ID"].ne(-1)]
    assert players["LOCATION_Z"].isna().all()
    assert players["VELOCITY_Z"].isna().all()
    assert canon[canon["PLAYER_ID"].eq(-1)]["LOCATION_Z"].notna().all()


def test_movement_derivatives_do_not_cross_games_and_distances_are_stable():
    raw, cfg = synthetic()
    features = derive_tracking_features(preprocess_tracking_data(raw, cfg), cfg)
    player = features[(features["PLAYER_ID"] == 2)].sort_values(["GAME_ID", "TIMESTEP"])
    first_each_game = player.groupby("GAME_ID").head(1)
    assert first_each_game["DISTANCE_STEP_NATIVE"].isna().all()
    assert player["CUMULATIVE_DISTANCE_NATIVE"].ge(0).all()
    stationary = features[(features["PLAYER_ID"] == 0) & (features["TIMESTEP"] > 0)]
    assert (stationary["DISTANCE_STEP_NATIVE"].fillna(0) == 0).all()


def test_tracking_gap_is_not_bridged():
    raw, cfg = synthetic()
    raw = raw[(raw["GAME_ID"] == 100) & raw["TIMESTEP"].isin([0, 2])].copy()
    # Move a single player materially between t=0 and t=2.
    mask = (raw["PLAYER_ID"] == 0) & (raw["TIMESTEP"] == 2)
    raw.loc[mask, "LOCATION_X"] += 100
    canon = preprocess_tracking_data(raw, cfg)
    features = derive_tracking_features(canon, cfg)
    player = features[features["PLAYER_ID"] == 0].sort_values("TIMESTEP")
    assert len(player) == 2
    assert player.iloc[1]["GAP_FROM_PREVIOUS_ENTITY_FRAME"] == 2
    assert pd.isna(player.iloc[1]["DISTANCE_STEP_NATIVE"])


def test_possession_unknown_and_contested_states():
    raw, cfg = synthetic()
    # Introduce two on-ball positives at t=1 and remove the label at t=3.
    raw.loc[(raw["GAME_ID"] == 100) & (raw["TIMESTEP"] == 1) & (raw["PLAYER_ID"] == 1), "ON_BALL"] = 1
    raw.loc[(raw["GAME_ID"] == 100) & (raw["TIMESTEP"] == 3), "ON_BALL"] = 0
    frames = build_possession_frame_table(preprocess_tracking_data(raw, cfg), cfg)
    states = frames[frames["GAME_ID"].eq(100)].set_index("TIMESTEP")["possession_state"]
    assert states.loc[1] == "contested"
    assert states.loc[3] == "unknown"
    assert states.loc[0] == "observed"


def test_event_detection_finds_candidate_pass_interception_carry_and_score_change():
    raw, cfg = synthetic()
    canonical = preprocess_tracking_data(raw, cfg)
    frames = build_possession_frame_table(canonical, cfg)
    events = detect_events(frames, cfg)
    types = set(events["event_type"])
    assert "pass_candidate" in types
    assert "interception_candidate" in types
    assert "carry_candidate" in types
    assert "score_change_goal_candidate" in types
    score_event = events[events["event_type"].eq("score_change_goal_candidate")]
    assert (score_event["confidence"] == "high").all()
    assert score_event["PLAYER_ID"].isna().all()
    assert events["event_id"].is_unique
    assert events["evidence_json"].map(lambda x: isinstance(json.loads(x), dict)).all()


def test_possession_chain_team_share_invariants_and_pipeline_exports(tmp_path):
    raw, cfg = synthetic()
    output = tmp_path / "out"
    report = run_analysis(ROOT / "examples" / "synthetic_tracking.csv", output, cfg)
    assert report["analysis"]["validation_checks_total"] >= 6
    assert report["analysis"]["all_validation_checks_passed"]
    expected = {
        "frame_metrics.csv", "events.csv", "possession_chains.csv", "player_match_stats.csv",
        "team_match_stats.csv", "model_outputs.csv", "quality_report.json", "metric_metadata.json",
    }
    assert expected.issubset({p.name for p in output.iterdir()})
    frame = pd.read_csv(output / "frame_metrics.csv")
    events = pd.read_csv(output / "events.csv")
    chains = pd.read_csv(output / "possession_chains.csv")
    players = pd.read_csv(output / "player_match_stats.csv")
    teams = pd.read_csv(output / "team_match_stats.csv")
    assert len(frame) == len(raw)
    assert frame.groupby(["GAME_ID", "TIMESTEP", "PLAYER_ID"]).size().max() == 1
    if not events.empty:
        assert events["event_id"].is_unique
    if not chains.empty:
        assert chains["possession_chain_id"].is_unique
    assert (players["completed_pass_candidates"] <= players["pass_attempt_candidates"]).all()
    assert teams.groupby("GAME_ID")["possession_pct_of_known_owner_frames"].sum().between(99.999, 100.001).all()
    loaded_report = json.loads((output / "quality_report.json").read_text())
    assert loaded_report["analysis"]["all_validation_checks_passed"] is True


def test_physical_units_only_enabled_with_explicit_calibration():
    raw, cfg = synthetic()
    cfg["seconds_per_frame"] = 0.04
    cfg["coordinate_scale_to_metre"] = 0.5
    features = derive_tracking_features(preprocess_tracking_data(raw, cfg), cfg)
    assert features["DISTANCE_STEP_METRES"].notna().any()
    assert features["FINITE_DIFF_SPEED_MPS"].notna().any()
    assert features["SPEED_KMH_FROM_FINITE_DIFF"].notna().any()


def test_pitch_control_requires_explicit_native_calibration_and_returns_grid():
    raw, cfg = synthetic()
    canonical = preprocess_tracking_data(raw, cfg)
    frame = canonical[(canonical["GAME_ID"] == 100) & (canonical["TIMESTEP"] == 0)]
    with pytest.raises(ValueError, match="requires max_speed_native_per_frame"):
        compute_pitch_control_grid(frame, cfg)
    cfg["pitch_control"]["max_speed_native_per_frame"] = 2.0
    cfg["pitch"]["bounds_native"] = [0, 100, 0, 60]
    cfg["pitch_control"]["grid_x"] = 5
    cfg["pitch_control"]["grid_y"] = 3
    grid = compute_pitch_control_grid(frame, cfg)
    assert len(grid) == 15
    assert np.allclose(grid["team_0_control_probability_proxy"] + grid["team_1_control_probability_proxy"], 1.0)
    assert set(grid["model_status"]) == {"uncalibrated_constant_speed_reachability_proxy"}


def test_conflicting_duplicate_entity_frame_fails_loudly(tmp_path):
    raw, cfg = synthetic()
    dup = raw.iloc[[0]].copy()
    dup.loc[:, "LOCATION_X"] = dup["LOCATION_X"] + 10
    malformed = pd.concat([raw, dup], ignore_index=True)
    path = tmp_path / "conflict.csv"
    malformed.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="Conflicting observations"):
        load_tracking_data(path, cfg)


def test_cli_analyze_returns_success_for_fixture(tmp_path):
    code = main(["analyze", "--input", str(ROOT / "examples" / "synthetic_tracking.csv"), "--output", str(tmp_path / "cli_out"), "--config", str(ROOT / "examples" / "config.yaml")])
    assert code == 0
    assert (tmp_path / "cli_out" / "quality_report.json").exists()


def test_position_derived_velocity_on_known_linear_trajectory():
    raw, cfg = synthetic()
    raw = raw[(raw["GAME_ID"] == 100) & (raw["PLAYER_ID"] == 0) & raw["TIMESTEP"].isin([0, 1, 2])].copy()
    raw["LOCATION_X"] = raw["TIMESTEP"] * 2.0
    raw["LOCATION_Y"] = 10.0
    raw["VELOCITY_X"] = 0.0  # finite-difference result is independent of supplied velocity
    raw["VELOCITY_Y"] = 0.0
    features = derive_tracking_features(preprocess_tracking_data(raw, cfg), cfg).sort_values("TIMESTEP")
    assert features.iloc[1]["FINITE_DIFF_SPEED_NATIVE_PER_TIME_UNIT"] == pytest.approx(2.0)
    assert features.iloc[2]["FINITE_DIFF_SPEED_NATIVE_PER_TIME_UNIT"] == pytest.approx(2.0)
    assert features.iloc[2]["ACCELERATION_NATIVE_PER_TIME2_FROM_POSITION_FD"] == pytest.approx(0.0)


def test_smoothing_does_not_bridge_tracking_gaps():
    raw, cfg = synthetic()
    raw = raw[(raw["GAME_ID"] == 100) & (raw["PLAYER_ID"] == 2) & raw["TIMESTEP"].isin([0, 2, 3])].copy()
    raw["VELOCITY_X"] = raw["TIMESTEP"].map({0: 1.0, 2: 3.0, 3: 5.0})
    raw["VELOCITY_Y"] = 0.0
    cfg["movement"]["smoothing_window_frames"] = 3
    features = derive_tracking_features(preprocess_tracking_data(raw, cfg), cfg).sort_values("TIMESTEP")
    t2 = features[features["TIMESTEP"].eq(2)].iloc[0]
    assert t2["SMOOTHED_SPEED_NATIVE_SOURCE"] == pytest.approx(3.0)


def test_progressive_carry_requires_configured_attacking_direction():
    raw, cfg = synthetic()
    canonical = preprocess_tracking_data(raw, cfg)
    frames = build_possession_frame_table(canonical, cfg)
    without_direction = detect_events(frames, cfg)
    assert "progressive_carry_candidate" not in set(without_direction["event_type"])
    cfg["pitch"]["attacking_direction_by_team"] = {"0": 1, "1": -1}
    cfg["events"]["progressive_distance_native"] = 3.0
    with_direction = detect_events(frames, cfg)
    assert "progressive_carry_candidate" in set(with_direction["event_type"])


def test_inconsistent_repeated_score_values_do_not_create_goal_event_for_that_frame(tmp_path):
    raw, cfg = synthetic()
    raw.loc[(raw["GAME_ID"] == 100) & (raw["TIMESTEP"] == 6) & (raw["PLAYER_ID"] == 4), "TEAM_0_SCORE"] = 4
    path = tmp_path / "inconsistent_score.csv"
    raw.to_csv(path, index=False)
    loaded, report = load_tracking_data(path, cfg)
    assert report.get("score_inconsistencies")
    frames = build_possession_frame_table(preprocess_tracking_data(loaded, cfg), cfg)
    events = detect_events(frames, cfg)
    assert not ((events["GAME_ID"] == 100) & (events["TIMESTEP"] == 6) & events["event_type"].eq("score_change_goal_candidate")).any()
    assert ((events["GAME_ID"] == 101) & (events["TIMESTEP"] == 6) & events["event_type"].eq("score_change_goal_candidate")).any()


def test_exact_duplicate_entity_frame_rows_are_counted_and_removed(tmp_path):
    raw, cfg = synthetic()
    path = tmp_path / "exact_duplicate.csv"
    pd.concat([raw, raw.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    loaded, report = load_tracking_data(path, cfg)
    assert len(loaded) == len(raw)
    assert report["duplicates"]["exact_rows_removed"] == 1


def test_missing_schema_column_fails_with_clear_error(tmp_path):
    raw, cfg = synthetic()
    path = tmp_path / "missing_column.csv"
    raw.drop(columns=["PLAYER_ID"]).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="Missing required columns"):
        load_tracking_data(path, cfg)


def test_possession_chains_split_around_unknown_owner_frames():
    raw, cfg = synthetic()
    frames = build_possession_frame_table(preprocess_tracking_data(raw, cfg), cfg)
    chains = detect_possession_chains(frames, cfg)
    game = chains[chains["GAME_ID"].eq(100)]
    assert not ((game["start_timestep"] <= 4) & (game["end_timestep"] >= 6)).any()
    assert game["start_timestep"].eq(6).any()


def test_event_detection_is_reproducible_and_confidence_is_explicit():
    raw, cfg = synthetic()
    frames = build_possession_frame_table(preprocess_tracking_data(raw, cfg), cfg)
    a = detect_events(frames, cfg)
    b = detect_events(frames, cfg)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["confidence"]).issubset({"high", "medium", "low", "unresolved"})
    pass_events = a[a["event_type"].eq("pass_candidate")]
    assert not pass_events.empty and pass_events["confidence"].eq("medium").all()


def test_model_registry_does_not_claim_uncalibrated_xg():
    from football_analytics.audited.value_models import build_model_status_table
    _, cfg = synthetic()
    models = build_model_status_table(cfg).set_index("model_name")
    assert models.loc["xG", "status"] == "implementation_available_requires_labelled_shots"
    assert not bool(models.loc["xG", "output_available"])
    assert "not_calibrated" == models.loc["xG", "calibration_status"]


def test_metric_metadata_documents_exact_export_schemas(tmp_path):
    _, cfg = synthetic()
    output = tmp_path / "metadata_out"
    run_analysis(ROOT / "examples" / "synthetic_tracking.csv", output, cfg)
    metadata = json.loads((output / "metric_metadata.json").read_text())
    assert "DISTANCE_TO_BALL_NATIVE" in metadata["metrics"]
    assert "pass_completion_pct" in metadata["metrics"]
    assert "frame_metrics.csv" in metadata["output_schemas"]
    assert "possession_state" in metadata["output_schemas"]["frame_metrics.csv"]["columns"]
    assert metadata["output_schemas"]["frame_metrics.csv"]["row_count"] == 368


def test_goal_post_distance_and_angle_only_appear_with_configured_geometry():
    raw, cfg = synthetic()
    raw = raw[(raw["GAME_ID"] == 100) & (raw["TIMESTEP"] == 0)].copy()
    raw.loc[raw["PLAYER_ID"] == 0, ["LOCATION_X", "LOCATION_Y"]] = [0.0, 0.0]
    cfg["pitch"]["goal_centres_native_by_team"] = {"0": [10.0, 0.0]}
    cfg["pitch"]["goal_posts_native_by_team"] = {"0": [[10.0, -1.0], [10.0, 1.0]]}
    features = derive_tracking_features(preprocess_tracking_data(raw, cfg), cfg)
    row = features[features["PLAYER_ID"].eq(0)].iloc[0]
    assert row["DISTANCE_TO_ATTACKING_GOAL_NATIVE"] == pytest.approx(10.0)
    assert row["DISTANCE_TO_GOAL_POST_1_NATIVE"] == pytest.approx(np.sqrt(101))
    assert row["DISTANCE_TO_GOAL_POST_2_NATIVE"] == pytest.approx(np.sqrt(101))
    assert row["ANGLE_TO_ATTACKING_GOAL_RAD"] == pytest.approx(2 * np.arctan(1 / 10))


def test_interception_and_turnover_are_attributed_to_opposite_teams_correctly(tmp_path):
    _, cfg = synthetic()
    output = tmp_path / "attribution_out"
    run_analysis(ROOT / "examples" / "synthetic_tracking.csv", output, cfg)
    teams = pd.read_csv(output / "team_match_stats.csv")
    game = teams[teams["GAME_ID"].eq(100)].set_index("TEAM_ID")
    assert game.loc[1, "interception_candidates"] == 1
    assert game.loc[0, "turnover_candidates"] == 1


def test_multiple_batch_csvs_combine_with_provenance_and_no_cross_game_mix(tmp_path):
    from football_analytics.audited.data_loader import load_tracking_data_sources

    raw, cfg = synthetic()
    first = tmp_path / "gfootball_results_games_0-0.csv"
    second = tmp_path / "gfootball_results_games_1-1.csv"
    raw[raw["GAME_ID"].eq(100)].to_csv(first, index=False)
    raw[raw["GAME_ID"].eq(101)].to_csv(second, index=False)

    combined, report = load_tracking_data_sources([first, second], cfg)
    assert set(combined["GAME_ID"].unique()) == {100, 101}
    assert report["games"] == 2
    assert report["frames"] == 16
    assert len(report["source_file_reports"]) == 2
    assert report["frame_entity_coverage"]["frames_without_ball"] == 0


def test_multiple_batch_csvs_reject_overlapping_game_ids(tmp_path):
    from football_analytics.audited.data_loader import load_tracking_data_sources

    raw, cfg = synthetic()
    first = tmp_path / "batch_a.csv"
    second = tmp_path / "batch_b.csv"
    raw.to_csv(first, index=False)
    raw.to_csv(second, index=False)
    with pytest.raises(DataValidationError, match="Overlapping GAME_ID values"):
        load_tracking_data_sources([first, second], cfg)


def test_glob_resolves_sequential_game_batch_files(tmp_path):
    from football_analytics.audited.data_loader import resolve_input_paths

    raw, _ = synthetic()
    raw[raw["GAME_ID"].eq(100)].to_csv(tmp_path / "gfootball_results_games_0-2.csv", index=False)
    raw[raw["GAME_ID"].eq(101)].to_csv(tmp_path / "gfootball_results_games_3-5.csv", index=False)
    paths = resolve_input_paths([str(tmp_path / "gfootball_results_games_*.csv")])
    assert [path.name for path in paths] == ["gfootball_results_games_0-2.csv", "gfootball_results_games_3-5.csv"]


def test_cli_analyze_accepts_multiple_input_csvs(tmp_path):
    raw, cfg = synthetic()
    first = tmp_path / "gfootball_results_games_0-0.csv"
    second = tmp_path / "gfootball_results_games_1-1.csv"
    raw[raw["GAME_ID"].eq(100)].to_csv(first, index=False)
    raw[raw["GAME_ID"].eq(101)].to_csv(second, index=False)
    output = tmp_path / "batch_out"
    code = main(["analyze", "--input", str(first), str(second), "--output", str(output), "--config", str(ROOT / "examples" / "config.yaml")])
    assert code == 0
    report = json.loads((output / "quality_report.json").read_text())
    assert report["analysis"]["games_processed"] == 2
    assert len(report["source_file_reports"]) == 2
