"""Validation invariants and metric dictionary."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METRIC_METADATA: dict[str, dict[str, Any]] = {
    "SPEED_NATIVE_SOURCE": {"meaning": "Horizontal speed from supplied velocity components", "units": "source velocity units (uncalibrated)", "method": "sqrt(VELOCITY_X^2 + VELOCITY_Y^2)", "status": "measured_from_source_velocity"},
    "SPEED_3D_NATIVE_SOURCE": {"meaning": "3D speed when all velocity components are available", "units": "source velocity units (uncalibrated)", "method": "Euclidean norm of supplied velocity components", "status": "measured_from_source_velocity; null if component unavailable"},
    "MOVEMENT_HEADING_RAD": {"meaning": "Direction of movement derived from velocity; not body orientation", "units": "radians", "method": "atan2(VELOCITY_Y, VELOCITY_X); stationary values null", "status": "derived"},
    "DISTANCE_STEP_NATIVE": {"meaning": "Horizontal displacement between adjacent valid entity frames", "units": "native coordinate units", "method": "Euclidean position difference only across permitted contiguous intervals", "status": "measured_position_difference"},
    "DISTANCE_STEP_METRES": {"meaning": "Distance step after configured coordinate calibration", "units": "metres", "method": "DISTANCE_STEP_NATIVE multiplied by coordinate_scale_to_metre", "status": "available_only_with_coordinate_calibration"},
    "FINITE_DIFF_SPEED_NATIVE_PER_TIME_UNIT": {"meaning": "Position-derived horizontal speed diagnostic", "units": "native units/frame unless seconds_per_frame configured, then native units/second", "method": "Position difference divided by actual timestep interval; gaps are not bridged", "status": "derived_diagnostic"},
    "FINITE_DIFF_SPEED_MPS": {"meaning": "Calibrated position-derived speed", "units": "m/s", "method": "Finite-difference native speed scaled by coordinate calibration when frame time is configured", "status": "available_only_with_coordinate_and_time_calibration"},
    "SUPPLIED_VELOCITY_DERIVATIVE_NATIVE_PER_FRAME": {"meaning": "Magnitude of change in the supplied horizontal velocity vector", "units": "source velocity units per frame; source velocity time unit unknown", "method": "Finite difference of supplied velocity components over contiguous timestep intervals", "status": "derived_diagnostic; not reported as physical acceleration"},
    "ACCELERATION_NATIVE_PER_TIME2_FROM_POSITION_FD": {"meaning": "Acceleration magnitude derived from successive position-difference velocity estimates", "units": "native coordinate units per frame squared unless seconds_per_frame is configured, then native units/s^2", "method": "Finite difference of position-derived velocity over contiguous intervals", "status": "derived_diagnostic"},
    "ACCELERATION_MPS2_FROM_POSITION_FD": {"meaning": "Physically scaled position-derived horizontal acceleration", "units": "m/s^2", "method": "Position-derived acceleration multiplied by coordinate scale when both coordinate and time calibration are supplied", "status": "available_only_with_coordinate_and_time_calibration"},
    "DISTANCE_TO_BALL_NATIVE": {"meaning": "Horizontal distance from player to unique ball position in the same frame", "units": "native coordinate units", "method": "Euclidean distance", "status": "derived; null when ball location ambiguous/missing"},
    "NEAREST_TEAMMATE_DISTANCE_NATIVE": {"meaning": "Distance to the closest other player on the same TEAM_ID", "units": "native coordinate units", "method": "Pairwise Euclidean distance per game-frame", "status": "derived"},
    "NEAREST_OPPONENT_DISTANCE_NATIVE": {"meaning": "Distance to the closest player on another TEAM_ID", "units": "native coordinate units", "method": "Pairwise Euclidean distance per game-frame", "status": "derived"},
    "possession_owner_id": {"meaning": "Unique player with ON_BALL normalized true", "units": "player identifier", "method": "Exactly one positive ON_BALL label per frame; otherwise null", "status": "observed-label-based"},
    "possession_team_id": {"meaning": "TEAM_ID for a unique observed on-ball owner", "units": "team identifier", "method": "Inherited from unique ON_BALL owner", "status": "observed-label-based"},
    "possession_pct_of_known_owner_frames": {"meaning": "Share of frames with a unique owner assigned to the team", "units": "percent", "method": "Team observed-owner frames divided by all unique-owner frames", "status": "observed-label-based; unknown/contested frames excluded"},
    "pass_candidate": {"meaning": "Owner changed to a teammate and ball displacement crossed a configured native-unit threshold", "units": "event count", "method": "Rule-based transition heuristic", "status": "inferred_candidate_not_verified_pass"},
    "interception_candidate": {"meaning": "Owner changed to an opponent and ball displacement crossed a configured threshold", "units": "event count", "method": "Rule-based transition heuristic", "status": "inferred_candidate_not_verified_interception"},
    "carry_candidate": {"meaning": "Same player retained observed ON_BALL across consecutive frames while ball moved", "units": "event count", "method": "Continuous ownership run and displacement threshold", "status": "inferred_candidate_not_verified_carry"},
    "score_change_goal_candidate": {"meaning": "A score field increased between adjacent frames", "units": "event count", "method": "TEAM_0_SCORE / TEAM_1_SCORE transition", "status": "high-confidence score transition; exact goal frame/scorer not asserted"},
}


SOURCE_FIELD_METADATA: dict[str, dict[str, Any]] = {
    "GAME_ID": {"meaning": "Source match identifier", "units": "identifier", "status": "source_field"},
    "TIMESTEP": {"meaning": "Frame/timestep index within a match; not a match clock", "units": "frames", "status": "source_field"},
    "TEAM_0_SCORE": {"meaning": "Score for score slot 0, repeated on entity rows", "units": "goals/score units", "status": "source_field; frame consistency checked"},
    "TEAM_1_SCORE": {"meaning": "Score for score slot 1, repeated on entity rows", "units": "goals/score units", "status": "source_field; frame consistency checked"},
    "PLAYER_ID": {"meaning": "Player/entity identifier; -1 identifies ball in this schema", "units": "identifier", "status": "source_field"},
    "TEAM_ID": {"meaning": "Team identifier; -1 identifies ball in this schema", "units": "identifier", "status": "source_field"},
    "PLAYER_ROLE": {"meaning": "Provided role/position label", "units": "categorical label", "status": "source_field"},
    "LOCATION_X": {"meaning": "Horizontal coordinate axis X", "units": "native coordinate units; unspecified until calibrated", "status": "source_field"},
    "LOCATION_Y": {"meaning": "Horizontal coordinate axis Y", "units": "native coordinate units; unspecified until calibrated", "status": "source_field"},
    "LOCATION_Z": {"meaning": "Vertical coordinate; player value -1 is treated as unavailable per supplied-data description", "units": "native coordinate units", "status": "source_field_with_documented_player_sentinel"},
    "VELOCITY_X": {"meaning": "Supplied X velocity component", "units": "source velocity units; time basis unspecified", "status": "source_field"},
    "VELOCITY_Y": {"meaning": "Supplied Y velocity component", "units": "source velocity units; time basis unspecified", "status": "source_field"},
    "VELOCITY_Z": {"meaning": "Supplied vertical velocity; player value -1 is treated as unavailable per supplied-data description", "units": "source velocity units; time basis unspecified", "status": "source_field_with_documented_player_sentinel"},
    "ON_BALL": {"meaning": "Source possession/control label", "units": "boolean-like categorical", "status": "source_field; normalized separately"},
}

# Additional derived fields are listed here so the metadata sidecar documents every
# derived feature rather than only a small subset of the output columns.
METRIC_METADATA.update({
    "ENTITY_TYPE": {"meaning": "Row entity classification", "units": "category", "method": "player unless PLAYER_ID=-1 and TEAM_ID=-1, then ball", "status": "derived_schema_field"},
    "ON_BALL_NORMALIZED": {"meaning": "Normalized source on-ball state", "units": "1/0/null", "method": "Configured true/false value mapping; unknown tokens null", "status": "normalized_source_label"},
    "VELOCITY_X_SOURCE": {"meaning": "Numeric view of source X velocity", "units": "source velocity units", "method": "Numeric coercion of VELOCITY_X", "status": "source_alias"},
    "VELOCITY_Y_SOURCE": {"meaning": "Numeric view of source Y velocity", "units": "source velocity units", "method": "Numeric coercion of VELOCITY_Y", "status": "source_alias"},
    "VELOCITY_Z_SOURCE": {"meaning": "Numeric view of source vertical velocity after player sentinel cleaning", "units": "source velocity units", "method": "Numeric coercion of VELOCITY_Z", "status": "source_alias_with_sentinel_handling"},
    "FINITE_DIFF_VX_NATIVE_PER_TIME_UNIT": {"meaning": "X velocity estimated from positions", "units": "native units/frame or native units/second when configured", "method": "Contiguous-frame position difference divided by elapsed time", "status": "diagnostic_estimate"},
    "FINITE_DIFF_VY_NATIVE_PER_TIME_UNIT": {"meaning": "Y velocity estimated from positions", "units": "native units/frame or native units/second when configured", "method": "Contiguous-frame position difference divided by elapsed time", "status": "diagnostic_estimate"},
    "FINITE_DIFF_SPEED_NATIVE_PER_TIME_UNIT": {"meaning": "Horizontal speed estimated from position changes", "units": "native units/frame or native units/second when configured", "method": "Euclidean norm of finite-difference X/Y velocity", "status": "diagnostic_estimate"},
    "SUPPLIED_VELOCITY_DERIVATIVE_X_NATIVE_PER_FRAME": {"meaning": "Change in source X velocity per frame", "units": "source velocity units/frame", "method": "Finite difference of supplied X velocity", "status": "diagnostic_only; source velocity time basis unspecified"},
    "SUPPLIED_VELOCITY_DERIVATIVE_Y_NATIVE_PER_FRAME": {"meaning": "Change in source Y velocity per frame", "units": "source velocity units/frame", "method": "Finite difference of supplied Y velocity", "status": "diagnostic_only; source velocity time basis unspecified"},
    "CUMULATIVE_DISTANCE_NATIVE": {"meaning": "Cumulative sum of observed contiguous horizontal displacement", "units": "native coordinate units", "method": "Cumulative sum of DISTANCE_STEP_NATIVE, ignoring unavailable gap intervals; may be a lower bound when gaps exist", "status": "observed_displacement_sum_not_gap_corrected"},
    "CUMULATIVE_DISTANCE_METRES": {"meaning": "Cumulative observed displacement after coordinate scaling", "units": "metres", "method": "CUMULATIVE_DISTANCE_NATIVE multiplied by coordinate_scale_to_metre", "status": "available_only_with_coordinate_calibration; missing gap distances remain missing from total"},
    "GAP_FROM_PREVIOUS_ENTITY_FRAME": {"meaning": "Timestep distance from prior row for the same game and entity", "units": "frames", "method": "Difference of TIMESTEP within (GAME_ID, PLAYER_ID)", "status": "diagnostic"},
    "IS_CONTIGUOUS_ENTITY_INTERVAL": {"meaning": "Whether this entity row follows a valid contiguous interval", "units": "boolean", "method": "Positive timestep difference no greater than max_contiguous_gap_frames", "status": "diagnostic"},
    "SPEED_KMH_FROM_FINITE_DIFF": {"meaning": "Position-derived speed in kilometres per hour", "units": "km/h", "method": "Calibrated m/s multiplied by 3.6", "status": "available_only_with_coordinate_and_time_calibration"},
    "SMOOTHED_SPEED_NATIVE_SOURCE": {"meaning": "Trailing mean of supplied horizontal speed", "units": "source velocity units", "method": "Configured rolling window, split at timestep gaps", "status": "derived; opt-in smoothing"},
    "TEAMMATES_WITHIN_CONFIGURED_RADIUS": {"meaning": "Number of teammates within configured radius", "units": "player count", "method": "Pairwise horizontal distance <= spatial.local_density_radius_native; null if radius not configured", "status": "optional_native_unit_metric"},
    "OPPONENTS_WITHIN_CONFIGURED_RADIUS": {"meaning": "Number of opponents within configured radius", "units": "player count", "method": "Pairwise horizontal distance <= spatial.local_density_radius_native; null if radius not configured", "status": "optional_native_unit_metric"},
    "ATTACKING_DIRECTION_X": {"meaning": "Configured attack direction along X for the entity team", "units": "+1/-1", "method": "Lookup in pitch.attacking_direction_by_team", "status": "configuration_only; null unless supplied"},
    "DIRECTIONAL_X_NATIVE": {"meaning": "X coordinate signed by configured team attacking direction", "units": "native coordinate units", "method": "LOCATION_X * ATTACKING_DIRECTION_X", "status": "configuration_only; not half-aware unless segmented configuration is supplied"},
    "DISTANCE_TO_ATTACKING_GOAL_NATIVE": {"meaning": "Distance to configured attacking goal centre", "units": "native coordinate units", "method": "Euclidean distance to pitch.goal_centres_native_by_team", "status": "configuration_only"},
    "DISTANCE_TO_GOAL_POST_1_NATIVE": {"meaning": "Distance to first configured goal post", "units": "native coordinate units", "method": "Euclidean distance to configured pitch.goal_posts_native_by_team post 1", "status": "configuration_only"},
    "DISTANCE_TO_GOAL_POST_2_NATIVE": {"meaning": "Distance to second configured goal post", "units": "native coordinate units", "method": "Euclidean distance to configured pitch.goal_posts_native_by_team post 2", "status": "configuration_only"},
    "ANGLE_TO_ATTACKING_GOAL_RAD": {"meaning": "Angle subtended by configured goal posts from entity location", "units": "radians", "method": "arccos of clamped dot product between vectors to the two configured goal posts", "status": "configuration_only; depends on correct pitch coordinate calibration"},
    "possession_state": {"meaning": "Frame possession classification", "units": "observed/unknown/contested", "method": "Unique positive ON_BALL => observed; multiple => contested; none => unknown", "status": "observed_label_based"},
    "possession_evidence": {"meaning": "Reason for possession frame classification", "units": "categorical label", "method": "Records unique, multiple, or absent positive ON_BALL evidence", "status": "audit_field"},
    "tracked_frames": {"meaning": "Number of distinct frames with a player row", "units": "frames", "denominator": "all unique game frames for tracking_coverage_pct", "method": "Count distinct TIMESTEP per player-game", "status": "direct_coverage_aggregate"},
    "tracking_coverage_pct": {"meaning": "Share of match frames containing a player row", "units": "percent", "denominator": "unique frames in the game", "method": "tracked_frames / game frame count * 100", "status": "coverage_metric"},
    "distance_travelled_native": {"meaning": "Sum of observed contiguous horizontal player displacements", "units": "native coordinate units", "denominator": None, "method": "Sum of valid DISTANCE_STEP_NATIVE values by player-game", "status": "observed_displacement_sum_not_gap_corrected"},
    "distance_travelled_metres": {"meaning": "Sum of calibrated observed horizontal displacements", "units": "metres", "denominator": None, "method": "Sum of DISTANCE_STEP_METRES where available", "status": "available_only_with_coordinate_calibration"},
    "max_observed_speed_native_source_units": {"meaning": "Maximum horizontal speed magnitude from source velocity", "units": "source velocity units", "denominator": None, "method": "Maximum SPEED_NATIVE_SOURCE by player-game", "status": "measured_from_source_velocity; source units unspecified"},
    "max_speed_kmh": {"meaning": "Maximum position-derived speed in km/h", "units": "km/h", "denominator": None, "method": "Maximum calibrated SPEED_KMH_FROM_FINITE_DIFF", "status": "available_only_with_coordinate_and_time_calibration"},
    "mean_nearest_opponent_distance_native": {"meaning": "Mean nearest-opponent distance across the player's valid frames", "units": "native coordinate units", "denominator": "frames where nearest opponent distance is available", "method": "Mean NEAREST_OPPONENT_DISTANCE_NATIVE", "status": "derived_spatial_aggregate"},
    "mean_nearest_teammate_distance_native": {"meaning": "Mean nearest-teammate distance across the player's valid frames", "units": "native coordinate units", "denominator": "frames where nearest teammate distance is available", "method": "Mean NEAREST_TEAMMATE_DISTANCE_NATIVE", "status": "derived_spatial_aggregate"},
    "observed_control_frames": {"meaning": "Frames where this player is the unique observed ON_BALL owner", "units": "frames", "denominator": None, "method": "Count frames with possession_state=observed and matching possession_owner_id", "status": "observed_label_based"},
    "observed_control_seconds": {"meaning": "Observed control frames scaled by configured seconds per frame", "units": "seconds", "denominator": None, "method": "observed_control_frames * seconds_per_frame", "status": "available_only_with_time_calibration; not a clock or playing-time measure"},
    "touches_or_receptions_candidates": {"meaning": "Candidate reception/owner-change event count involving the player as receiver", "units": "event count", "denominator": None, "method": "Pass-candidate counterpart count plus touch_or_reception_candidate events where PLAYER_ID is the new owner", "status": "heuristic_candidate_count_not_verified_touches"},
    "pass_attempt_candidates": {"meaning": "Potential pass attempts from direct candidate pass transitions plus possible ball releases", "units": "event count", "denominator": None, "method": "Count pass_candidate and ball_release_candidate events attributed to player", "status": "heuristic_and_incomplete_event_denominator"},
    "completed_pass_candidates": {"meaning": "Direct same-team owner transitions that pass the ball-displacement threshold", "units": "event count", "denominator": None, "method": "Count pass_candidate events attributed to player", "status": "inferred_candidate_not_verified_pass"},
    "pass_completion_pct": {"meaning": "Official-style pass completion rate", "units": "percent", "denominator": "reliable attempted-pass count not available", "method": "Not calculated; left null until release-to-reception linking supplies a defensible denominator", "status": "unavailable_by_design"},
    "pass_completion_pct_unavailable_reason": {"meaning": "Explanation for the unavailable pass completion percentage", "units": "text", "method": "Static warning about the candidate-event denominator", "status": "availability_explanation"},
    "interception_candidates": {"meaning": "Candidate interceptions credited to this player as the new owner", "units": "event count", "denominator": None, "method": "Opponent-owner-change candidates where PLAYER_ID matches the new owner / candidate interceptor", "status": "heuristic_candidate_count"},
    "interception_candidates_against_opponents": {"meaning": "Alias of candidate interceptions credited to this player", "units": "event count", "denominator": None, "method": "Same event filter as interception_candidates", "status": "heuristic_candidate_count_alias"},
    "turnover_candidates": {"meaning": "Candidate owner transitions ending with opponent control, attributed to prior owner", "units": "event count", "denominator": None, "method": "Count interception_candidate events where PLAYER_ID matches former owner", "status": "heuristic_candidate_count"},
    "carry_candidates": {"meaning": "Continuous-control runs with observed ball movement", "units": "event count", "denominator": None, "method": "Count carry_candidate and progressive_carry_candidate events", "status": "heuristic_candidate_count"},
    "progressive_carry_candidates": {"meaning": "Carry candidate crossing the configured directional X progression threshold", "units": "event count", "denominator": None, "method": "Carry candidate plus configured attacking direction and progression threshold", "status": "configuration-dependent_heuristic"},
    "per_90_distance_native": {"meaning": "Distance normalized to 90 minutes", "units": "native coordinate units per 90 minutes", "denominator": "player on-pitch minutes; not provided by current schema", "method": "Not calculated in this MVP", "status": "unavailable_by_design"},
    "shot_attempts": {"meaning": "Shot attempt count", "units": "event count", "denominator": None, "method": "Not inferred without reliable shot labels or configured geometry/model", "status": "unavailable_not_zero"},
    "goals_attributed_to_player": {"meaning": "Goals attributed to an individual player", "units": "goals", "denominator": None, "method": "Not attributed from score transitions alone", "status": "unavailable_not_zero"},
    "xG": {"meaning": "Expected goals estimate", "units": "probability per shot / aggregate depending model", "denominator": "shot events required", "method": "No calibrated shot model or labelled shots supplied", "status": "unavailable_not_zero"},
    "take_ons_attempted": {"meaning": "Take-on attempts", "units": "event count", "denominator": None, "method": "Requires defender-engagement/action labels unavailable in input", "status": "unavailable_not_zero"},
    "take_ons_won": {"meaning": "Take-ons won", "units": "event count", "denominator": None, "method": "Requires defender-engagement/action labels unavailable in input", "status": "unavailable_not_zero"},
    "sprint_count": {"meaning": "Sprint count", "units": "event count", "denominator": None, "method": "Requires calibrated physical speeds and sprint threshold; not computed", "status": "unavailable_not_zero"},
    "high_speed_running_distance_metres": {"meaning": "Distance above a configured high-speed running threshold", "units": "metres", "denominator": None, "method": "Requires physical coordinate/time calibration and threshold; not computed", "status": "unavailable_not_zero"},
    "observed_possession_frames": {"meaning": "Frames with unique observed possession credited to team", "units": "frames", "denominator": "known_possession_frames_all_teams for share", "method": "Count unique-owner frames assigned to team", "status": "observed_label_based"},
    "known_possession_frames_all_teams": {"meaning": "All frames with a unique observed owner in the match", "units": "frames", "denominator": None, "method": "Count frames with possession_state=observed", "status": "observed_label_based"},
    "unknown_or_contested_frames": {"meaning": "Frames without a unique observed owner", "units": "frames", "denominator": "all game frames for rate interpretation", "method": "Count frames where possession_state is not observed", "status": "observed_label_based"},
    "possession_time_seconds": {"meaning": "Observed team possession frame count scaled by seconds per frame", "units": "seconds", "denominator": None, "method": "observed_possession_frames * seconds_per_frame", "status": "available_only_with_time_calibration; not ball-in-play duration"},
    "possession_chains": {"meaning": "Number of segmented observed team-control chains", "units": "chain count", "denominator": None, "method": "Count team possession-chain records", "status": "observed_label_based_chain_segmentation"},
    "mean_chain_duration_seconds": {"meaning": "Mean duration proxy for chains", "units": "seconds", "denominator": "chains with configured time scale", "method": "Mean observed_frame_count * seconds_per_frame", "status": "available_only_with_time_calibration"},
    "pass_candidates": {"meaning": "Candidate pass transition count for the team", "units": "event count", "denominator": None, "method": "Count pass_candidate events attributed to team", "status": "heuristic_candidate_count"},
    "score_change_goal_candidates": {"meaning": "Score-increase transitions credited to score slot/team", "units": "event count", "denominator": None, "method": "Count score_change_goal_candidate events", "status": "high-confidence_score_transition_not_scorer_attribution"},
    "mean_team_width_native": {"meaning": "Mean lateral coordinate spread of tracked team players", "units": "native coordinate units", "denominator": "team frames with valid player coordinates", "method": "Mean of per-frame max LOCATION_Y minus min LOCATION_Y", "status": "coordinate_spread_proxy_not_tactical_width"},
    "median_team_width_native": {"meaning": "Median lateral coordinate spread of tracked team players", "units": "native coordinate units", "denominator": "team frames with valid player coordinates", "method": "Median of per-frame max LOCATION_Y minus min LOCATION_Y", "status": "coordinate_spread_proxy_not_tactical_width"},
    "mean_team_depth_native": {"meaning": "Mean longitudinal coordinate spread of tracked team players", "units": "native coordinate units", "denominator": "team frames with valid player coordinates", "method": "Mean of per-frame max LOCATION_X minus min LOCATION_X", "status": "coordinate_spread_proxy_not_tactical_depth"},
    "median_team_depth_native": {"meaning": "Median longitudinal coordinate spread of tracked team players", "units": "native coordinate units", "denominator": "team frames with valid player coordinates", "method": "Median of per-frame max LOCATION_X minus min LOCATION_X", "status": "coordinate_spread_proxy_not_tactical_depth"},
    "ppda_proxy": {"meaning": "Passes per defensive action proxy", "units": "ratio", "denominator": "requires defensible opponent pass and defensive-action counts", "method": "Not computed because input does not contain confirmed defensive actions", "status": "unavailable_not_zero"},
    "field_tilt_pct": {"meaning": "Share of attacking-third/territory actions by team", "units": "percent", "denominator": "requires configured pitch geometry and qualifying actions", "method": "Not computed in current MVP", "status": "unavailable_not_zero"},
    "entries_final_third": {"meaning": "Entries into final third", "units": "event count", "denominator": None, "method": "Requires calibrated/defined pitch thirds and entry-event logic", "status": "unavailable_not_zero"},
    "entries_penalty_area": {"meaning": "Entries into penalty area", "units": "event count", "denominator": None, "method": "Requires configured penalty-area geometry and event detection", "status": "unavailable_not_zero"},
    "mean_nearest_opponent_distance_native": {"meaning": "Player-game mean nearest-opponent distance", "units": "native coordinate units", "denominator": "frames with valid nearest-opponent distance", "method": "Mean per-frame nearest-opponent distance", "status": "derived_spatial_aggregate"},
    "mean_nearest_teammate_distance_native": {"meaning": "Player-game mean nearest-teammate distance", "units": "native coordinate units", "denominator": "frames with valid nearest-teammate distance", "method": "Mean per-frame nearest-teammate distance", "status": "derived_spatial_aggregate"},
})


METRIC_METADATA.update({
    "event_id": {"meaning": "Stable event row identifier within a single analysis run", "units": "identifier", "method": "Game ID plus monotonically generated event counter", "status": "structural_identifier"},
    "start_timestep": {"meaning": "Start frame of an event interval", "units": "source timestep", "method": "Rule-specific; event start frame for transitions and first frame for carries", "status": "event_time_not_match_clock"},
    "end_timestep": {"meaning": "End frame of an event interval", "units": "source timestep", "method": "Rule-specific event end frame", "status": "event_time_not_match_clock"},
    "event_type": {"meaning": "Candidate event classification", "units": "categorical label", "method": "Rule-based tracking/possession/score transition logic", "status": "candidate_label_not_official_event"},
    "RELATED_PLAYER_ID": {"meaning": "Counterpart player in a candidate event, where applicable", "units": "player identifier", "method": "For pass candidate, receiver; for interception/possession change, prior owner; meanings follow event type", "status": "event_context_identifier"},
    "RELATED_TEAM_ID": {"meaning": "Counterpart team for the event's related player", "units": "team identifier", "method": "For owner transitions, the team on the other side of the transition", "status": "event_context_identifier"},
    "start_x": {"meaning": "Ball X coordinate at candidate event start", "units": "native coordinate units", "method": "Ball position from frame-level tracking summary", "status": "observed_position_where_available"},
    "start_y": {"meaning": "Ball Y coordinate at candidate event start", "units": "native coordinate units", "method": "Ball position from frame-level tracking summary", "status": "observed_position_where_available"},
    "end_x": {"meaning": "Ball X coordinate at candidate event end", "units": "native coordinate units", "method": "Ball position from frame-level tracking summary", "status": "observed_position_where_available"},
    "end_y": {"meaning": "Ball Y coordinate at candidate event end", "units": "native coordinate units", "method": "Ball position from frame-level tracking summary", "status": "observed_position_where_available"},
    "event_outcome": {"meaning": "Rule-level outcome label", "units": "categorical label", "method": "Assigned by the detection rule; does not replace verified event labels", "status": "heuristic_or_score_transition_outcome"},
    "confidence": {"meaning": "Qualitative reliability tier for evidence available to the rule", "units": "high/medium/low/unresolved", "method": "Rule-defined confidence based on label/trajectory evidence", "status": "qualitative_not_calibrated_probability"},
    "evidence_json": {"meaning": "Machine-readable values and observations supporting the event", "units": "JSON object encoded as text", "method": "Rule-specific evidence payload", "status": "audit_trail"},
    "detection_rule": {"meaning": "Identifier for the event detection heuristic", "units": "rule name", "method": "Assigned by candidate event rule", "status": "audit_trail"},
    "rule_version": {"meaning": "Version of event detection logic", "units": "semantic version string", "method": "Static version for reproducibility", "status": "audit_trail"},
    "alternative_interpretations": {"meaning": "Known plausible explanations other than the selected event candidate", "units": "text", "method": "Rule-specific caution", "status": "uncertainty_documentation"},
    "possession_chain_id": {"meaning": "Unique identifier for a segmented observed team-control chain", "units": "identifier", "method": "Game ID plus chain counter", "status": "chain_identifier"},
    "observed_frame_count": {"meaning": "Number of frames included in a possession chain", "units": "frames", "method": "Count of contiguous unique-owner frames", "status": "observed_label_based"},
    "duration_seconds": {"meaning": "Observed possession chain duration proxy", "units": "seconds", "denominator": None, "method": "observed_frame_count multiplied by configured seconds_per_frame", "status": "null_without_time_calibration; interval duration approximation"},
    "participants_json": {"meaning": "Unique player IDs observed as owners during a chain", "units": "JSON list encoded as text", "method": "Unique possession_owner_id values ordered by first occurrence", "status": "observed_label_based"},
    "owner_changes": {"meaning": "Count of distinct owners after the first owner within a team chain", "units": "change count", "denominator": None, "method": "Unique participant count minus one", "status": "owner-transition proxy_not_pass_count"},
    "terminal_state": {"meaning": "Reason a possession chain ended", "units": "categorical label", "method": "Next-frame state, team change, gap, or end of observed data", "status": "segmentation_reason"},
    "reliability_status": {"meaning": "Textual reliability qualification for the possession chain", "units": "text", "method": "Static note that team-control chains are based on ON_BALL labels and heuristic boundaries", "status": "uncertainty_documentation"},
    "known_possession_frames_all_teams": {"meaning": "Unique-owner frame denominator across all teams", "units": "frames", "denominator": None, "method": "Count frame rows with possession_state=observed", "status": "observed_label_based"},
    "possession_share_invariant_passed": {"meaning": "Whether team possession shares sum to approximately 100 percent for the game", "units": "boolean", "method": "Sum of possession_pct_of_known_owner_frames across team rows in game within tolerance", "status": "quality_check"},
    "possession_pct_of_known_owner_frames": {"meaning": "Team's observed possession share over frames with unique owner", "units": "percent", "denominator": "known unique-owner frames in game; unknown/contested excluded", "method": "observed_possession_frames / known_possession_frames_all_teams * 100", "status": "observed_label_based"},
    "mean_chain_duration_seconds": {"meaning": "Mean configured-time estimate of possession chain duration", "units": "seconds", "denominator": "chains with non-null duration estimate", "method": "Mean possession chain duration_seconds", "status": "null_without_time_calibration"},
    "turnover_candidates": {"meaning": "Candidate turnovers / owner losses to the opponent", "units": "event count", "denominator": None, "method": "Count opponent owner-change candidates linked to the prior player's/team", "status": "heuristic_candidate_count_not_verified_turnover"},
    "ball_release_candidates": {"meaning": "Frames where a prior owner label disappears while ball displacement exceeds configured threshold", "units": "event count", "denominator": None, "method": "Rule-based owner-loss plus ball-displacement heuristic", "status": "ambiguous_release_not_confirmed_pass"},
    "score_change_goal_candidates": {"meaning": "Score increments detected between adjacent frames", "units": "event count", "denominator": None, "method": "Frame-level consistent TEAM_0_SCORE / TEAM_1_SCORE increase", "status": "high-confidence_score_transition_not_scorer_attribution"},
    "model_name": {"meaning": "Name of the advanced model / metric interface", "units": "label", "method": "Registry entry", "status": "model_metadata"},
    "model_version": {"meaning": "Version label for model interface", "units": "version string", "method": "Registry metadata", "status": "model_metadata"},
    "status": {"meaning": "Implementation/readiness status for the model", "units": "categorical label", "method": "Registry status based on available configuration and training data", "status": "model_readiness_not_prediction"},
    "output_available": {"meaning": "Whether row-level model predictions are present in this output", "units": "boolean", "method": "False in current registry-only export; the pitch-control function can be called separately when configured", "status": "current_output_availability"},
    "calibration_status": {"meaning": "Whether the registered model has been empirically calibrated", "units": "categorical label", "method": "Current baseline registry; no supplied labelled training set", "status": "model_metadata"},
    "required_training_data": {"meaning": "Required labelled data or parameters for a usable model", "units": "text", "method": "Registry metadata", "status": "model_metadata"},
    "assumptions": {"meaning": "Model assumptions and limitations", "units": "text", "method": "Registry metadata", "status": "model_metadata"},
    "uncertainty": {"meaning": "Uncertainty-estimation status", "units": "text", "method": "Registry metadata", "status": "model_metadata"},
    "distinction_from_observed_fact": {"meaning": "Warning distinguishing predictions from source observations", "units": "text", "method": "Registry metadata", "status": "model_metadata"},
})


METRIC_METADATA.update({
    "metrics_status": {"meaning": "Textual qualification of metric availability and reliability in the aggregate row", "units": "text", "method": "Static explanation that event metrics are candidates and calibration-dependent metrics may be null", "status": "uncertainty_documentation"},
    "reception_candidates": {"meaning": "Candidate reception or owner-change count for a player", "units": "event count", "denominator": None, "method": "Pass-candidate receiver role plus low-distance owner-change candidates where player is current owner", "status": "heuristic_candidate_count"},
})


def build_metric_metadata(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Build metadata sidecar including definitions and exact output schemas."""
    schemas: dict[str, Any] = {}
    for filename, table in tables.items():
        schemas[filename] = {
            "row_count": int(len(table)),
            "columns": [str(c) for c in table.columns],
            "column_dtypes": {str(c): str(table[c].dtype) for c in table.columns},
        }
    return {
        "metadata_version": "1.0.0",
        "units_policy": "Native units unless calibration is explicitly configured; empty CSV cells represent unavailable/unknown values, not zero.",
        "source_fields": SOURCE_FIELD_METADATA,
        "metrics": METRIC_METADATA,
        "output_schemas": schemas,
        "interpretation_note": "Metrics marked heuristic, proxy, unavailable, or configuration-only should not be interpreted as verified event truth or calibrated probabilities.",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, np.ndarray)) else False:
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def build_quality_report(source_report: dict, raw_df: pd.DataFrame, features: pd.DataFrame,
                         frames: pd.DataFrame, events: pd.DataFrame, chains: pd.DataFrame,
                         player_stats: pd.DataFrame, team_stats: pd.DataFrame, config: dict,
                         output_dir: str | Path) -> dict[str, Any]:
    report = dict(source_report)
    event_types = events["event_type"].value_counts().to_dict() if not events.empty else {}
    unknown_share = float((frames["possession_state"] != "observed").mean()) if len(frames) else None
    checks: dict[str, dict[str, Any]] = {}
    checks["one_frame_metrics_row_per_entity_frame"] = {
        "passed": len(features) == len(raw_df) and not features.duplicated(["GAME_ID", "TIMESTEP", "PLAYER_ID"]).any(),
        "detail": "frame_metrics retains one row per validated input entity-frame key",
    }
    checks["no_duplicate_event_ids"] = {"passed": events.empty or events["event_id"].is_unique, "detail": "event IDs are unique within this analysis run"}
    checks["event_intervals_are_ordered_and_include_event_frame"] = {
        "passed": events.empty or bool((events["start_timestep"] <= events["end_timestep"]).all() and (events["TIMESTEP"].between(events["start_timestep"], events["end_timestep"])).all()),
        "detail": "every candidate event has an ordered frame interval containing its event timestep",
    }
    if not team_stats.empty:
        checks["team_known-possession-shares_sum_to_100_pct"] = {
            "passed": bool(team_stats.groupby("GAME_ID")["possession_pct_of_known_owner_frames"].sum(min_count=1).dropna().between(99.999, 100.001).all()),
            "detail": "team shares use the same unique-owner frame denominator; unknown and contested frames are excluded",
        }
    else:
        checks["team_known-possession-shares_sum_to_100_pct"] = {"passed": True, "detail": "no team rows to check"}
    checks["per90_disabled_without_frame_rate"] = {
        "passed": config.get("seconds_per_frame") is not None or player_stats.empty or player_stats["per_90_distance_native"].isna().all(),
        "detail": "per-90 metrics remain null without a configured time scale and playing-time definition",
    }
    checks["physical_units_disabled_without_calibration"] = {
        "passed": (config.get("coordinate_scale_to_metre") is not None and config.get("seconds_per_frame") is not None)
                 or features.empty or features["FINITE_DIFF_SPEED_MPS"].isna().all(),
        "detail": "metres and m/s columns remain null unless physical coordinate/time calibration is supplied",
    }
    checks["cross_game_motion_differencing_prevented"] = {
        "passed": bool(features.loc[features.groupby(["GAME_ID", "PLAYER_ID"], sort=False).cumcount().eq(0), "DISTANCE_STEP_NATIVE"].isna().all()),
        "detail": "the first record of each entity within each game has no previous-frame displacement",
    }
    checks["completed_passes_do_not_exceed_pass_attempt_candidates"] = {
        "passed": player_stats.empty or (player_stats["completed_pass_candidates"].fillna(0) <= player_stats["pass_attempt_candidates"].fillna(0)).all(),
        "detail": "event-derived player counts are bounded by the candidate attempt definition",
    }
    report["analysis"] = {
        "rows_processed": int(len(features)), "games_processed": int(features["GAME_ID"].nunique()),
        "frames_processed": int(len(frames)), "events_detected": int(len(events)), "event_type_counts": {str(k): int(v) for k, v in event_types.items()},
        "possession_chains": int(len(chains)), "player_match_rows": int(len(player_stats)), "team_match_rows": int(len(team_stats)),
        "fraction_unknown_or_contested_possession_frames": unknown_share,
        "calibration": {
            "seconds_per_frame": config.get("seconds_per_frame"),
            "coordinate_scale_to_metre": config.get("coordinate_scale_to_metre"),
            "pitch_bounds_native": config.get("pitch", {}).get("bounds_native"),
            "attacking_direction_by_team": config.get("pitch", {}).get("attacking_direction_by_team", {}),
        },
        "model_readiness": "xG/xPass/EPV/VAEP unavailable without labelled data or calibrated parameters; see model_outputs.csv",
        "metric_availability": {
            "native_tracking_features": "available where raw fields are valid",
            "physical_units_and_per90": "unavailable unless calibration is supplied",
            "pass_carry_interception": "heuristic candidates, not verified events",
            "shot_events_and_xG": "unavailable without shot labels or configured calibrated shot geometry/model",
        },
        "validation_checks": checks,
        "validation_checks_passed": int(sum(bool(v["passed"]) for v in checks.values())),
        "validation_checks_total": int(len(checks)),
        "all_validation_checks_passed": bool(all(v["passed"] for v in checks.values())),
        "output_directory": str(Path(output_dir).resolve()),
        "output_files": ["frame_metrics.csv", "events.csv", "possession_chains.csv", "player_match_stats.csv", "team_match_stats.csv", "model_outputs.csv", "quality_report.json", "metric_metadata.json"],
        "warnings": list(report.get("warnings", [])),
    }
    return _json_safe(report)


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
