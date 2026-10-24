import json

import numpy as np
import pandas as pd
from pathlib import Path

from football_analytics.data_loader import load_tracking_data, normalize_pitch_coordinates
from football_analytics.preprocessing import validate_tracking_frame
from football_analytics.tracking_features import compute_tracking_features
from football_analytics.possession import infer_possession
from football_analytics.event_detection import detect_events
from football_analytics.match_intelligence import detect_key_moments, enrich_events, match_clock, pass_difficulty, player_speed_tags

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_load_tracking_csv():
    df = load_tracking_data(DATA_DIR / 'gfootball_results_games_0-2.csv')
    assert {'GAME_ID', 'TIMESTEP', 'PLAYER_ID', 'TEAM_ID', 'LOCATION_X', 'LOCATION_Y', 'LOCATION_Z', 'VELOCITY_X', 'VELOCITY_Y', 'VELOCITY_Z', 'ON_BALL'}.issubset(df.columns)
    assert df['GAME_ID'].nunique() == 3
    assert len(df) > 0


def test_normalize_pitch_coordinates_preserves_sources_and_does_not_clip():
    source = pd.DataFrame([{
        'LOCATION_X': -1.0, 'LOCATION_Y': -0.42,
        'VELOCITY_X': 1.0, 'VELOCITY_Y': 0.42,
        'LOCATION_Z': 0.7, 'VELOCITY_Z': 0.2,
    }, {
        'LOCATION_X': 1.01, 'LOCATION_Y': 0.43,
        'VELOCITY_X': -1.0, 'VELOCITY_Y': -0.42,
        'LOCATION_Z': -1.0, 'VELOCITY_Z': -1.0,
    }])

    normalized = normalize_pitch_coordinates(source)

    assert normalized.loc[0, ['LOCATION_X', 'LOCATION_Y']].tolist() == [0.0, 0.0]
    upper = normalize_pitch_coordinates(pd.DataFrame([{
        'LOCATION_X': 1.0, 'LOCATION_Y': 0.42,
        'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0,
    }]))
    assert upper.loc[0, ['LOCATION_X', 'LOCATION_Y']].tolist() == [120.0, 80.0]
    assert normalized.loc[1, 'LOCATION_X'] > 120.0
    assert normalized.loc[1, 'LOCATION_Y'] > 80.0
    assert normalized.loc[0, 'VELOCITY_X'] == 60.0
    assert normalized.loc[0, 'VELOCITY_Y'] == 40.0
    assert normalized.loc[0, 'LOCATION_Z'] == source.loc[0, 'LOCATION_Z']
    assert normalized.loc[0, 'SOURCE_LOCATION_X'] == -1.0


def test_preprocessing_detects_ball_and_players():
    df = load_tracking_data(DATA_DIR / 'gfootball_results_games_0-2.csv')
    cleaned, report = validate_tracking_frame(df)
    ball = cleaned[cleaned['PLAYER_ID'] == -1]
    players = cleaned[cleaned['PLAYER_ID'] != -1]
    assert not ball.empty
    assert not players.empty
    assert 'quality_checks' in report


def test_both_supplied_csvs_load_all_six_games():
    frames = [
        load_tracking_data(DATA_DIR / 'gfootball_results_games_0-2.csv'),
        load_tracking_data(DATA_DIR / 'gfootball_results_games_3-5.csv'),
    ]
    game_ids = set().union(*(set(frame['GAME_ID'].unique()) for frame in frames))
    assert game_ids == {0, 1, 2, 3, 4, 5}


def test_stationary_entities_have_zero_speed():
    df = pd.DataFrame([
        {'GAME_ID': 0, 'TIMESTEP': 0, 'TEAM_0_SCORE': 0, 'TEAM_1_SCORE': 0, 'PLAYER_ID': 0, 'TEAM_ID': 0, 'PLAYER_ROLE': 'goalkeeper', 'LOCATION_X': 0.0, 'LOCATION_Y': 0.0, 'LOCATION_Z': -1.0, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0, 'VELOCITY_Z': -1.0, 'ON_BALL': False},
        {'GAME_ID': 0, 'TIMESTEP': 0, 'TEAM_0_SCORE': 0, 'TEAM_1_SCORE': 0, 'PLAYER_ID': -1, 'TEAM_ID': -1, 'PLAYER_ROLE': -1, 'LOCATION_X': 1.0, 'LOCATION_Y': 2.0, 'LOCATION_Z': 0.1, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0, 'VELOCITY_Z': 0.0, 'ON_BALL': False},
    ])
    feat = compute_tracking_features(df)
    assert feat['horizontal_speed'].abs().max() >= 0.0
    assert feat.loc[feat['PLAYER_ID'] == 0, 'horizontal_speed'].iloc[0] == 0.0
    assert feat.loc[feat['PLAYER_ID'] == 0, 'speed_3d'].iloc[0] == 0.0


def test_tracking_diagnostics_do_not_bridge_gaps_or_resets():
    rows = []
    for timestep, player_x, ball_x in [(0, 10.0, 10.0), (1, 10.5, 10.5), (3, 30.0, 25.0)]:
        rows.extend([
            {'GAME_ID': 0, 'TIMESTEP': timestep, 'PLAYER_ID': 0, 'TEAM_ID': 0,
             'LOCATION_X': player_x, 'LOCATION_Y': 10.0, 'LOCATION_Z': -1.0,
             'VELOCITY_X': 0.5, 'VELOCITY_Y': 0.0, 'VELOCITY_Z': -1.0},
            {'GAME_ID': 0, 'TIMESTEP': timestep, 'PLAYER_ID': -1, 'TEAM_ID': -1,
             'LOCATION_X': ball_x, 'LOCATION_Y': 10.0, 'LOCATION_Z': 0.1,
             'VELOCITY_X': 0.5, 'VELOCITY_Y': 0.0, 'VELOCITY_Z': 0.0},
        ])

    features = compute_tracking_features(pd.DataFrame(rows))
    final_player = features[(features['PLAYER_ID'] == 0) & (features['TIMESTEP'] == 3)].iloc[0]
    final_ball = features[(features['PLAYER_ID'] == -1) & (features['TIMESTEP'] == 3)].iloc[0]

    assert final_player['timestep_gap']
    assert final_player['distance_travelled'] == 0.0
    assert final_player['delta_velocity_x'] == 0.0
    assert final_ball['distance_travelled'] == 0.0


def test_possession_and_event_creation():
    df = pd.DataFrame([
        {'GAME_ID': 0, 'TIMESTEP': 0, 'TEAM_0_SCORE': 0, 'TEAM_1_SCORE': 0, 'PLAYER_ID': -1, 'TEAM_ID': -1, 'PLAYER_ROLE': -1, 'LOCATION_X': 0.0, 'LOCATION_Y': 0.0, 'LOCATION_Z': 0.1, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0, 'VELOCITY_Z': 0.0, 'ON_BALL': False},
        {'GAME_ID': 0, 'TIMESTEP': 0, 'TEAM_0_SCORE': 0, 'TEAM_1_SCORE': 0, 'PLAYER_ID': 0, 'TEAM_ID': 0, 'PLAYER_ROLE': 'left back', 'LOCATION_X': 1.0, 'LOCATION_Y': 0.0, 'LOCATION_Z': -1.0, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0, 'VELOCITY_Z': -1.0, 'ON_BALL': True},
        {'GAME_ID': 0, 'TIMESTEP': 0, 'TEAM_0_SCORE': 0, 'TEAM_1_SCORE': 0, 'PLAYER_ID': 1, 'TEAM_ID': 0, 'PLAYER_ROLE': 'central front', 'LOCATION_X': 3.0, 'LOCATION_Y': 0.0, 'LOCATION_Z': -1.0, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0, 'VELOCITY_Z': -1.0, 'ON_BALL': False},
        {'GAME_ID': 0, 'TIMESTEP': 0, 'TEAM_0_SCORE': 0, 'TEAM_1_SCORE': 0, 'PLAYER_ID': 2, 'TEAM_ID': 1, 'PLAYER_ROLE': 'centre back', 'LOCATION_X': 8.0, 'LOCATION_Y': 0.0, 'LOCATION_Z': -1.0, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0, 'VELOCITY_Z': -1.0, 'ON_BALL': False},
    ])
    poss = infer_possession(df)
    assert 'owner_team_id' in poss.columns
    assert 'owner_player_id' in poss.columns
    events = detect_events(poss)
    assert isinstance(events, pd.DataFrame)


def test_tracking_only_detector_links_pass_and_ignores_labels():
    rows = []
    ball_positions = [30.0, 30.0, 30.0, 32.0, 34.0, 36.0, 39.0, 40.0, 40.0]
    for timestep, ball_x in enumerate(ball_positions):
        rows.extend([
            {'GAME_ID': 7, 'TIMESTEP': timestep, 'PLAYER_ID': -1, 'TEAM_ID': -1,
             'LOCATION_X': ball_x, 'LOCATION_Y': 30.0, 'VELOCITY_X': 2.0 if timestep in (3, 4, 5, 6) else 0.0,
             'VELOCITY_Y': 0.0, 'ON_BALL': False, 'TEAM_0_SCORE': 0, 'TEAM_1_SCORE': 0, 'PLAYER_ROLE': 'ball'},
            {'GAME_ID': 7, 'TIMESTEP': timestep, 'PLAYER_ID': 10, 'TEAM_ID': 0,
             'LOCATION_X': 30.0, 'LOCATION_Y': 30.0, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0,
             'ON_BALL': timestep == 0, 'TEAM_0_SCORE': 0, 'TEAM_1_SCORE': 0, 'PLAYER_ROLE': 'forward'},
            {'GAME_ID': 7, 'TIMESTEP': timestep, 'PLAYER_ID': 11, 'TEAM_ID': 0,
             'LOCATION_X': 40.0, 'LOCATION_Y': 30.0, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0,
             'ON_BALL': False, 'TEAM_0_SCORE': 0, 'TEAM_1_SCORE': 0, 'PLAYER_ROLE': 'defender'},
            {'GAME_ID': 7, 'TIMESTEP': timestep, 'PLAYER_ID': 20, 'TEAM_ID': 1,
             'LOCATION_X': 90.0, 'LOCATION_Y': 40.0, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0,
             'ON_BALL': False, 'TEAM_0_SCORE': 0, 'TEAM_1_SCORE': 0, 'PLAYER_ROLE': 'goalkeeper'},
        ])
    tracking = pd.DataFrame(rows)
    relabeled = tracking.copy()
    relabeled['ON_BALL'] = ~relabeled['ON_BALL']
    relabeled['TEAM_0_SCORE'] = 4
    relabeled['TEAM_1_SCORE'] = 3
    relabeled['PLAYER_ROLE'] = 'changed'

    events = detect_events(tracking)
    changed_events = detect_events(relabeled)

    assert 'pass_attempt' in events['event_type'].tolist()
    pass_row = events[events['event_type'] == 'pass_attempt'].iloc[0]
    assert pass_row['outcome'] == 'completed_pass_candidate'
    assert pass_row['PLAYER_ID'] == 10
    pd.testing.assert_frame_equal(events, changed_events)


def test_uncontrolled_distant_ball_remains_unknown():
    rows = []
    for timestep in range(2):
        rows.extend([
            {'GAME_ID': 0, 'TIMESTEP': timestep, 'TEAM_0_SCORE': 0, 'TEAM_1_SCORE': 0, 'PLAYER_ID': -1, 'TEAM_ID': -1, 'PLAYER_ROLE': 'ball', 'LOCATION_X': 0.0, 'LOCATION_Y': 0.0, 'LOCATION_Z': 0.1, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0, 'VELOCITY_Z': 0.0, 'ON_BALL': False},
            {'GAME_ID': 0, 'TIMESTEP': timestep, 'TEAM_0_SCORE': 0, 'TEAM_1_SCORE': 0, 'PLAYER_ID': 0, 'TEAM_ID': 0, 'PLAYER_ROLE': 'forward', 'LOCATION_X': 8.0, 'LOCATION_Y': 0.0, 'LOCATION_Z': -1.0, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0, 'VELOCITY_Z': -1.0, 'ON_BALL': False},
            {'GAME_ID': 0, 'TIMESTEP': timestep, 'TEAM_0_SCORE': 0, 'TEAM_1_SCORE': 0, 'PLAYER_ID': 1, 'TEAM_ID': 1, 'PLAYER_ROLE': 'defender', 'LOCATION_X': -8.0, 'LOCATION_Y': 0.0, 'LOCATION_Z': -1.0, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0, 'VELOCITY_Z': -1.0, 'ON_BALL': False},
        ])
    possession = infer_possession(pd.DataFrame(rows))
    ball = possession[possession['PLAYER_ID'] == -1]
    assert ball['owner_player_id'].isna().all()
    assert (ball['ownership_confidence'] == 'unresolved').all()


def _pass_sequence(goal_at=None):
    rows = []
    ball_positions = [30.0, 30.0, 30.0, 32.0, 34.0, 36.0, 39.0, 40.0, 40.0]
    for timestep, ball_x in enumerate(ball_positions):
        score = int(goal_at is not None and timestep >= goal_at)
        common = {'GAME_ID': 7, 'TIMESTEP': timestep, 'TEAM_0_SCORE': score, 'TEAM_1_SCORE': 0, 'LOCATION_Z': -1.0, 'VELOCITY_Z': -1.0, 'ON_BALL': False}
        rows.extend([
            {**common, 'PLAYER_ID': -1, 'TEAM_ID': -1, 'PLAYER_ROLE': 'ball', 'LOCATION_X': ball_x, 'LOCATION_Y': 30.0, 'LOCATION_Z': 0.1, 'VELOCITY_X': 2.0 if timestep in (3, 4, 5, 6) else 0.0, 'VELOCITY_Y': 0.0, 'VELOCITY_Z': 0.0},
            {**common, 'PLAYER_ID': 10, 'TEAM_ID': 0, 'PLAYER_ROLE': 'forward', 'LOCATION_X': 30.0, 'LOCATION_Y': 30.0, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0},
            {**common, 'PLAYER_ID': 11, 'TEAM_ID': 0, 'PLAYER_ROLE': 'defender', 'LOCATION_X': 40.0, 'LOCATION_Y': 30.0, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0},
            {**common, 'PLAYER_ID': 20, 'TEAM_ID': 1, 'PLAYER_ROLE': 'goalkeeper', 'LOCATION_X': 35.0, 'LOCATION_Y': 34.0, 'VELOCITY_X': 0.0, 'VELOCITY_Y': 0.0},
        ])
    return pd.DataFrame(rows)


def test_nearest_player_distances_split_by_team():
    features = compute_tracking_features(_pass_sequence())
    first = features[(features['TIMESTEP'] == 0) & features['is_player']].set_index('PLAYER_ID')
    assert first.loc[10, 'nearest_teammate_distance'] == 10.0
    assert abs(first.loc[10, 'nearest_opponent_distance'] - np.hypot(5.0, 4.0)) < 1e-9
    assert np.isnan(first.loc[20, 'nearest_teammate_distance'])
    assert features.loc[~features['is_player'], 'nearest_opponent_distance'].isna().all()


def test_pass_quality_and_ball_speed():
    tracking = _pass_sequence()
    events = enrich_events(detect_events(tracking), compute_tracking_features(tracking), {0: 1.0, 1: -1.0}, seconds_per_timestep=0.1)
    pass_row = events[events['event_type'] == 'pass_attempt'].iloc[0]
    assert bool(pass_row['completed'])
    assert pass_row['ball_speed_kmh'] == 72.0
    assert pass_row['progression_m'] > 0
    assert pass_row['lane_pressure_m'] == round(float(np.hypot(max(pass_row['start_x'] - 35.0, 0.0), 4.0)), 2)
    assert 0 <= pass_row['pass_difficulty'] <= 100


def test_pass_difficulty_rewards_long_tight_progressive_passes():
    assert pass_difficulty(5.0, 10.0, 0.0) < pass_difficulty(35.0, 1.0, 25.0, 'through_ball_candidate') <= 100.0
    assert pass_difficulty(500.0, 0.0, 500.0, 'through_ball_candidate') == 100.0


def test_score_change_becomes_goal_moment_with_clock():
    tracking = _pass_sequence(goal_at=8)
    features = compute_tracking_features(tracking)
    events = enrich_events(detect_events(tracking), features, {0: 1.0, 1: -1.0}, 0.1)
    moments = detect_key_moments(features, events, pd.DataFrame(), pd.DataFrame(), {0: 1.0, 1: -1.0})
    goal = moments[moments['moment_type'] == 'goal'].iloc[0]
    assert goal['TEAM_ID'] == 0 and goal['end_timestep'] == 8
    assert json.loads(goal['metrics'])['score_after'] == '1-0'
    assert match_clock(1500) == '45:00'


def test_sprint_tags_need_sustained_speed():
    rows = [{'GAME_ID': 0, 'TIMESTEP': t, 'PLAYER_ID': 3, 'TEAM_ID': 0, 'horizontal_speed': 0.9 if 2 <= t < 14 else 0.1, 'distance_travelled': 0.5} for t in range(20)]
    tags = player_speed_tags(pd.DataFrame(rows), seconds_per_timestep=0.1, distance_step_m=5.0)
    sprint = tags[tags['tag_type'] == 'sprint']
    assert len(sprint) == 1 and sprint.iloc[0]['value'] == 32.4
    assert tags[tags['tag_type'] == 'distance_milestone']['value'].tolist() == [5.0, 10.0]
