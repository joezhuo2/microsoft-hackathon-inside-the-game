# Data Dictionary

This document describes the supplied tracking CSV fields and the files produced by the analysis pipeline. The exact columns in `frame_metrics.csv` depend on the input columns and enabled coordinate normalization; the sections below describe the default supplied-data configuration.

## Conventions and Units

- `GAME_ID` identifies a match; `TIMESTEP` is that match's integer frame index. Timesteps are not converted to seconds.
- The ball uses `PLAYER_ID = -1` and normally `TEAM_ID = -1`.
- With `normalize_source_coordinates: true`, X/Y locations map from the nominal source bounds `[-1, 1] × [-0.42, 0.42]` to a 120 × 80 coordinate plane. Original horizontal location and velocity values are retained in `SOURCE_*` fields.
- Normalized horizontal locations are nominally metres on the configured pitch. They are not clipped at the boundary. Horizontal velocities are metres per timestep, not metres per second.
- Source Z and vertical velocity scales are unverified. Player Z and vertical-velocity values of `-1` are missing-value sentinels.
- Event confidence labels are qualitative heuristic levels, not calibrated probabilities. Candidate events have not been measured against a manually annotated reference set.

## Input Tracking CSV

The loader requires the following columns. Extra source columns are retained.

| Field | Meaning | Type / units |
| --- | --- | --- |
| `GAME_ID` | Match identifier. | Numeric identifier |
| `TIMESTEP` | Frame index within a match. | Integer frame; interval duration is not established |
| `TEAM_0_SCORE` | Score recorded for team 0. Retained for compatibility; not used by event detection. | Numeric count |
| `TEAM_1_SCORE` | Score recorded for team 1. Retained for compatibility; not used by event detection. | Numeric count |
| `PLAYER_ID` | Entity identifier; `-1` identifies the ball. | Numeric identifier |
| `TEAM_ID` | Team identifier; the ball commonly uses `-1`. Used to associate players with teams. | Numeric identifier |
| `PLAYER_ROLE` | Source role label. Retained but not used by event detection. | Source categorical value |
| `LOCATION_X` | Horizontal X position; normalized when configured. | Source units or metres on nominal 120 × 80 pitch |
| `LOCATION_Y` | Horizontal Y position; normalized when configured. | Source units or metres on nominal 120 × 80 pitch |
| `LOCATION_Z` | Vertical position. Player value `-1` is a sentinel. | Unverified source units |
| `VELOCITY_X` | Horizontal X velocity component; scaled when coordinates are normalized. | Source units or metres per timestep |
| `VELOCITY_Y` | Horizontal Y velocity component; scaled when coordinates are normalized. | Source units or metres per timestep |
| `VELOCITY_Z` | Vertical velocity component. Player value `-1` is a sentinel. | Unverified source units per timestep |
| `ON_BALL` | Source ownership/ball-contact indicator. Ignored by event detection, but used by the separate legacy possession inference and `touches` aggregates. | Boolean-like source value |

## `frame_metrics.csv`

One row per game, timestep, and entity. Source columns are preserved, normalized coordinates may be present, tracking features are appended, and inferred possession columns are added by the separate possession pipeline.

### Source and calibrated-coordinate fields

All required input fields above are retained. When normalization is enabled, these additional fields preserve the original horizontal source values:

| Field | Meaning | Type / units |
| --- | --- | --- |
| `SOURCE_LOCATION_X` | Original X location before normalization. | Source units |
| `SOURCE_LOCATION_Y` | Original Y location before normalization. | Source units |
| `SOURCE_VELOCITY_X` | Original X velocity before normalization. | Source units per timestep |
| `SOURCE_VELOCITY_Y` | Original Y velocity before normalization. | Source units per timestep |

### Tracking features

| Field | Meaning | Type / units |
| --- | --- | --- |
| `is_player` | True for player rows; false for the ball row. | Boolean |
| `horizontal_velocity` | Magnitude of supplied horizontal velocity components. | Coordinate units per timestep |
| `speed_3d` | Horizontal speed combined with vertical velocity when available; vertical scale is unverified. | Unverified mixed 3D units per timestep |
| `movement_direction_deg` | Direction from `VELOCITY_X/Y`, in degrees. | Degrees |
| `movement_direction_rad` | Same direction in radians. | Radians |
| `dt_seconds` | Legacy column name for the timestep difference used in the diagnostic. It is not converted to seconds; non-consecutive gaps are reset to 1 for calculations. | Timesteps (not seconds) |
| `timestep_gap` | True if the prior row for this entity is not exactly one timestep earlier, including its first row. | Boolean |
| `dx`, `dy` | Position difference from the preceding entity row; zeroed across gaps and configured discontinuities. | Coordinate units per observed step |
| `dz` | Vertical position difference; sentinel-masked and zeroed across gaps. | Unverified source units |
| `distance_travelled` | Horizontal finite-difference displacement for this row; zeroed across gaps and large jumps. | Coordinate units |
| `finite_difference_speed` | `distance_travelled` divided by the diagnostic timestep interval. | Coordinate units per timestep |
| `finite_difference_velocity_x` | X finite-difference velocity diagnostic. | Coordinate units per timestep |
| `finite_difference_velocity_y` | Y finite-difference velocity diagnostic. | Coordinate units per timestep |
| `tracking_discontinuity` | Flags a player displacement above 3 coordinate units or ball displacement above 10 between adjacent frames. | Boolean |
| `horizontal_speed` | Supplied horizontal velocity magnitude when valid; otherwise finite-difference speed. | Coordinate units per timestep |
| `ball_x`, `ball_y`, `ball_z` | Ball position joined onto every entity row in the same game and timestep. | Same units as matching location fields |
| `distance_to_ball` | Horizontal distance from the row's entity to the ball. | Coordinate units |
| `distance_to_ball_3d` | 3D distance diagnostic; depends on source Z and is not physically calibrated. | Unverified mixed units |
| `delta_velocity_x`, `delta_velocity_y` | Change in supplied horizontal velocity from the preceding entity row; zeroed across gaps and discontinuities. | Coordinate units per timestep |
| `acceleration_magnitude` | Magnitude of the supplied-velocity change divided by the diagnostic interval. Not physical acceleration unless time calibration is added. | Coordinate units per timestep squared |
| `nearest_teammate_distance` | Reserved feature column; currently missing (`NaN`). | Coordinate units |
| `nearest_opponent_distance` | Reserved feature column; currently missing (`NaN`). | Coordinate units |

### Legacy inferred-possession fields

These fields are created by `infer_possession`, separately from tracking-only event detection. They may use `ON_BALL` first and nearest-player fallback second.

| Field | Meaning | Type / values |
| --- | --- | --- |
| `owner_player_id` | Inferred player associated with possession at the frame, repeated on the ball row. | Player ID or missing |
| `owner_team_id` | Team of the inferred player. | Team ID or missing |
| `ownership_source` | Basis for the inference: `on_ball`, `nearest_to_ball`, or `unknown`. | Categorical string |
| `ownership_confidence` | Legacy qualitative label: `high`, `low`, or `unresolved`. | Categorical string |
| `ownership_team_name` | Display label mapped from team ID 0/1. | `team_0`, `team_1`, or missing |

## `events.csv`

Each row is one tracking-derived event candidate or linked outcome. `PLAYER_ID` and `TEAM_ID` are populated only where the rule supports attribution.

| Field | Meaning | Type / units |
| --- | --- | --- |
| `event_id` | Generated identifier, currently formatted from game ID and row sequence. | String |
| `GAME_ID` | Match identifier. | Same as input |
| `start_timestep` | First frame of the event/candidate. | Frame index |
| `end_timestep` | Last linked frame or outcome frame; may equal start. | Frame index |
| `event_type` | Primary candidate label. Examples include `touch_candidate`, `possession_gain_candidate`, `possession_loss_candidate`, `carry`, `dribble_candidate`, `ball_release_candidate`, `pass_attempt`, `incomplete_pass_candidate`, `reception_candidate`, `interception_candidate`, `shot_candidate`, `goal_candidate`, `clearance_candidate`, `boundary_exit_candidate`, `dead_ball_candidate`, and `restart_candidate`. | String |
| `subtype` | Optional candidate subtype, such as `progressive`, `planar_on_target_candidate`, `cross_candidate`, `switch_of_play_candidate`, `through_ball_candidate`, or a restart subtype. | String or empty |
| `PLAYER_ID` | Primary participant, generally the inferred controller or event player. | Player ID or empty |
| `TEAM_ID` | Participant team. | Team ID or empty |
| `previous_controller` | Inferred controller before a linked transition/release. | Player ID or empty |
| `next_controller` | Inferred controller after a linked transition/release. | Player ID or empty |
| `start_x`, `start_y` | Ball location associated with event start. | Configured horizontal coordinate units |
| `end_x`, `end_y` | Ball location associated with event end/outcome. | Configured horizontal coordinate units |
| `confidence_level` | Qualitative rule confidence: `HIGH`, `MEDIUM`, `LOW`, or `UNRESOLVED`. | Category, not probability |
| `evidence_flags` | Semicolon-separated evidence used by the rule. | String |
| `alternative_hypotheses` | Semicolon-separated plausible competing explanations. | String |
| `rule_version` | Detector rule-set identifier. | String |
| `outcome` | Optional release result, e.g. `completed_pass_candidate`, `interception_candidate`, or `unresolved`. | String or empty |

The event detector uses player/ball positions, horizontal velocities, entity IDs, and team IDs. It does not use score, `ON_BALL`, `PLAYER_ROLE`, or existing event labels. Candidate labels are not ground-truth annotations.

## `possession_chains.csv`

One row per consecutive team segment from the legacy inferred-possession path.

| Field | Meaning | Type / units |
| --- | --- | --- |
| `GAME_ID` | Match identifier. | Same as input |
| `team_id` | Team assigned to the chain. | Team ID |
| `start_timestep` | First frame in the chain. | Frame index |
| `end_timestep` | Last frame in the chain. | Frame index |
| `duration_frames` | Chain duration in frames, not seconds. | Frames |
| `participants` | Serialized list of player IDs observed as inferred owners during the chain. | List representation |

## `player_match_stats.csv`

One row per game/player/team group.

| Field | Meaning | Type / units |
| --- | --- | --- |
| `GAME_ID` | Match identifier. | Same as input |
| `PLAYER_ID` | Player identifier. | Same as input |
| `TEAM_ID` | Team identifier. | Same as input |
| `touches` | Sum of the source `ON_BALL` indicator for that player. This is not a count of tracking-only `touch_candidate` events. | Count-like sum of flags |
| `total_distance` | Sum of `distance_travelled` over the player's rows. | Configured horizontal coordinate units |
| `max_speed` | Maximum `horizontal_speed`. | Coordinate units per timestep |
| `avg_speed` | Mean `horizontal_speed`. | Coordinate units per timestep |

## `team_match_stats.csv`

One row per game/team group.

| Field | Meaning | Type / units |
| --- | --- | --- |
| `GAME_ID` | Match identifier. | Same as input |
| `TEAM_ID` | Team identifier. | Same as input |
| `touches` | Sum of source `ON_BALL` flags for the team's player rows, not event-detector touches. | Count-like sum of flags |
| `distance_total` | Sum of player-row `distance_travelled`; it sums player movement, not ball distance. | Configured horizontal coordinate units |
| `player_count` | Number of distinct player IDs observed for the team. | Count |

## `quality_report.json`

The top-level report contains aggregate quality fields, a nested validation report, and calibration status.

| Field | Meaning |
| --- | --- |
| `games`, `frames`, `rows` | Match count, total distinct game-timestep count, and number of entity rows. |
| `missing_values` | Per-column null counts from the report's input frame. |
| `duplicate_rows` | Number of repeated `(GAME_ID, TIMESTEP, PLAYER_ID)` keys. |
| `ball_rows` | Number of rows where `PLAYER_ID == -1`. |
| `valid_owner_frames` | Legacy field name: sum of source `ON_BALL` flags across rows, not a count of unique frames. |
| `quality_checks` | Summary check labels from the quality report. The nested `validation.quality_checks` object contains duplicate, missingness, player/ball, game, and timestep counts from preprocessing. |
| `validation.frame_counts_by_game` | Distinct timestep count for each game. |
| `validation.entity_counts_by_frame` | Descriptive statistics for rows per game-frame. |
| `calibration_status` | Declares whether XY normalization is configured and notes that timing/Z remain unverified. |

## `metric_metadata.json`

Maps output file groups to short descriptions and unit notes. It is descriptive metadata, not a per-column schema; use this document for field-level definitions.
## Synthetic extension files



Written by `scripts/extend_synthetic.py` into each `output/games_*` folder. Every row and field is synthetic demo data, not observed from tracking, and must never be presented as observed. Timesteps use the same 3000-per-90-minute scale as `match_clock()`.



### `synthetic_events.csv`



Invented tackle and pressure events seeded from real candidate events. Synthetic flag: every row has `data_source = synthetic` and `synthetic = True`.



| Field | Meaning |

| --- | --- |

| `event_id` | Identifier with `SYN-` prefix, unique across files. |

| `GAME_ID` | Match identifier. |

| `start_timestep`, `end_timestep` | Timesteps of the real event the synthetic row was sampled from. |

| `event_type` | `tackle_synthetic` (sampled from possession gain or interception candidates, attributed to the winning player) or `pressure_synthetic` (attributed to an opposing-team player). |

| `PLAYER_ID`, `TEAM_ID` | Player and team from `data/roster.json`. |

| `data_source` | Always `synthetic`. |

| `synthetic` | Always `True`. |

| `label` | Short display text, for example `Synthetic tackle`. |



### `synthetic_substitutions.csv`



One to three invented substitutions per team per game, between minute 60 and 75 (timesteps 2000 to 2500). Synthetic flag: every row has `data_source = synthetic` and `synthetic = True`.



| Field | Meaning |

| --- | --- |

| `GAME_ID`, `TEAM_ID` | Match and team. |

| `start_timestep` | Substitution timestep. |

| `player_off_id` | Starter leaving the pitch. |

| `player_on_id` | Bench player (roster entries with `bench` and `synthetic` true) entering. |

| `data_source`, `synthetic` | Always `synthetic` and `True`. |



### `synthetic_match_meta.json`



Fictional match context per game: `referee`, `venue`, `attendance`, `weather`, `kickoff_local_time`, each shaped `{"value": ..., "synthetic": true}`, under `games.<GAME_ID>`. Top-level `note` reads "All fields synthetic; not from a real match." and `synthetic` is true.
