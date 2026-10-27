# XGBoost expected-goals (xG) model

## What it predicts

The model returns the pre-shot probability that a shot results in a goal for the shooting team. It is a supervised binary classifier, not an xG equation fitted to the tracking-only `games 0–2` file. Training requires a shot-level event feed with **independent goal/non-goal outcome labels**. The supplied tracking-only example has neither a full shot event log nor shot outcomes, so it cannot legitimately train or validate this model.

## Install the optional dependencies

```bash
pip install -e '.[xg]'
# or: pip install -r requirements-xg.txt
```

## Input contract

Tracking data must include `GAME_ID`, `TIMESTEP`, `PLAYER_ID`, `TEAM_ID`, `LOCATION_X`, `LOCATION_Y`, `VELOCITY_X`, and `VELOCITY_Y`. The full pipeline format also contains the other source columns. The feature builder automatically converts source-normalized coordinates (`x ∈ [-1,1]`, `y ∈ [-0.42,0.42]`) to the configured 120 × 80 plane unless `xg.coordinate_mode` overrides it. Already normalized files are not transformed again. Values beyond pitch boundaries remain untouched.

Events must include `GAME_ID` and one frame field (`start_timestep`, `TIMESTEP`, `timestep`, or `frame`). A shot table without `event_type` is treated as shot-only; a full event log is filtered to configured shot event types (`shot`, `shot_attempt`, `shot_on_target`, etc.). Heuristic `shot_candidate` rows are excluded by default. Add a candidate event type only after manual review.

Goal labels should be supplied in a binary column such as `is_goal` (1/0) or a consistently defined `outcome` field. Accepted outcome labels include `goal`/`scored` as positive and `saved`, `blocked`, `off_target`, `missed`, or `woodwork` as non-goals. Unknown/provisional outcomes are dropped from fitting, not treated as misses. The event type itself is never used as the goal label, so a separate `goal` event row cannot leak the target into the features.

Shot coordinates are read from event `start_x/start_y`, `shot_x/shot_y`, or the tracking ball position at that frame. Explicit `target_goal_x`, `target_goal`, or `attacking_direction` on an event take precedence over configuration. Otherwise supply the verified attack direction using `pitch.attacking_direction_by_team`, `xg.attacking_direction_by_game_team`, or `xg.attacking_direction_by_game_team_segment`. The example configuration assumes team 0 attacks +X and team 1 attacks −X only for the sample's observed initial formation; it is not a universal football rule. If directions change by half, provide segment metadata and mappings or split/configure the data accordingly.

## Features

The current model uses the following pre-shot features:

- **Goal geometry:** horizontal distance to goal line, lateral offset from goal centre, Euclidean distance, visible angle between goalposts, angle offset from the goal-centre line, distance to penalty spot, penalty-area and six-yard-box membership.
- **Defensive pressure:** closest defender distance, number of defenders within 2/5/10 m, closest defender distance to the shot-to-goal-centre segment, and count of players within a configurable shooting lane.
- **Goalkeeper/goal coverage:** goalkeeper distance from shot and goal centre, lateral offset, distance to goal line, and angle to the shot. The keeper is identified from an available keeper role label; if unavailable, the nearest defender to goal centre is an explicitly tagged geometric proxy.
- **Motion:** ball speed, lateral and goalward velocity, ball-velocity angle error relative to the goal-centre vector, shooter speed and goalward/lateral velocity, and shooter-ball separation at the shot frame. Horizontal velocity stays in metres per timestep, not m/s; all training/inference datasets should use consistent sampling intervals.
- **Action/context:** prior action (from explicit `action_before_shot` or the latest prior event for the same game and preferably same team), body part, shot type, play pattern, assist type, set-piece type, shooter role, penalty/rebound/one-on-one/first-time/set-piece flags.
- **Optional height:** ball height is excluded unless `use_ball_height: true` and a defensible `z_scale_to_m` are configured. Source Z is unverified in the supplied data.

`GAME_ID`, event IDs, shooter/player IDs, team IDs, event type, outcome, score, final result, post-shot trajectory/placement, and end coordinates are not predictors. `GAME_ID` is used only to keep entire matches together during evaluation.

## Train and predict

Use the example config after verifying its attack directions:

```bash
python -m football_analytics.cli xg-train \
  --tracking "data/gfootball_results_games_*.csv" \
  --events data/events.csv \
  --config config-xg.yaml \
  --model models/xgboost_xg.joblib \
  --report output/xg_training_report.json \
  --importance output/xg_feature_importance.csv \
  --features-output output/shot_features.csv

python -m football_analytics.cli xg-predict \
  --tracking "data/gfootball_results_games_*.csv" \
  --events data/events_to_score.csv \
  --config config-xg.yaml \
  --model models/xgboost_xg.joblib \
  --output output/xg_predictions.csv
```

Training uses a whole-game holdout for evaluation, reporting ROC AUC when the holdout has both classes, log loss, Brier score, observed goal rate, mean predicted xG, calibration gap, and reliability bins. The final model is then refit on all labelled shots. When enough distinct games and outcome support exist, probability calibration is fitted with group-aware cross-validation; otherwise the report explicitly marks the output as uncalibrated. The class prevalence is not reweighted by default because this can distort absolute goal probabilities.

## Interpretation and validation

- The output is `predicted_xg ∈ [0,1]` for each shot. A value of 0.18 means the model estimates an 18% goal probability under the training distribution; it is not a guarantee for that shot.
- Calibration and discrimination are different. Examine Brier/log loss and reliability bins alongside ROC AUC. Mean predicted xG should be compared with the actual goal rate on an independent held-out set.
- Split by game, not by shot row, to avoid nearly identical tracking snapshots from the same match leaking across train and evaluation.
- Use enough games, seasons, and competition contexts; check calibration drift by distance, angle, body part, open play/set piece, and game source.
- The geometrically inferred keeper and prior-action reconstruction are proxies when the event feed is incomplete. Review samples manually before operational use.
- There is no fitted model artifact bundled by default, because the available `games 0–2` tracking CSV does not supply the necessary shot events and goal outcomes. Fitting a model on invented/synthetic labels would create misleading xG values.
