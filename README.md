# Football Tracking Analytics and Match Intelligence Engine

This project implements a minimal but production-oriented football tracking analytics pipeline for the CSV schema used in the provided dataset.

## Setup

```bash
cd football_analytics
python -m pip install -r requirements.txt
```

## Run the pipeline

```bash
python -m football_analytics.cli analyze --input data/gfootball_results_games_0-2.csv --output output/games_0-2
python -m football_analytics.cli analyze --input data/gfootball_results_games_3-5.csv --output output/games_3-5
```

The analysis command uses `config.yaml` by default. To use another configuration:

```bash
python -m football_analytics.cli analyze --input data/gfootball_results_games_0-2.csv --output output/custom --config path/to/config.yaml
```

Also supported:

```bash
python -m football_analytics.cli validate --input data/gfootball_results_games_0-2.csv
python -m football_analytics.cli summary --input data/gfootball_results_games_0-2.csv
```

## Updated analytics and XGBoost

The supplied `football_tracking_analytics_xgboost` implementation is integrated in
`football_analytics.audited`. The existing `analyze` command and replay outputs retain
their current behavior. The updated pipeline has different schemas and native-unit
metrics; write its outputs to a separate directory:

```bash
python -m football_analytics.cli audited-analyze --input "data/gfootball_results_games_*.csv" --output output/audited --config config-audited.yaml
python -m football_analytics.cli audited-validate --input data/
python -m pip install -e ".[xg]"
python -m football_analytics.cli xg-train --tracking "data/gfootball_results_games_*.csv" --events labelled_shots.csv --model models/xgboost_xg.joblib --config config-xg.yaml
python -m football_analytics.cli xg-predict --tracking "data/gfootball_results_games_*.csv" --events shots.csv --model models/xgboost_xg.joblib --output output/shot_xg.csv --config config-xg.yaml
```

Use `config-audited.yaml` for the updated pipeline; `config.yaml` belongs to the
existing replay pipeline. XGBoost is optional. Training requires independently
labelled shots and goal outcomes; no trained model or new match data was supplied.
Configure verified attacking directions and coordinate modes before extracting xG
features. See [xG model documentation](docs/xg_model.md) for labels, configuration,
evaluation, and model limitations. Imported synthetic fixtures live only under
`tests/fixtures/audited`; the project match data is unchanged.

## Coordinates and units

The default configuration enables conversion of the supplied centered horizontal coordinates to a 120 × 80 pitch:

- `LOCATION_X = (SOURCE_LOCATION_X + 1) × 60`
- `LOCATION_Y = (SOURCE_LOCATION_Y + 0.42) × (80 / 0.84)`
- Horizontal velocities are scaled by the matching axis factors.

The four original horizontal fields are retained as `SOURCE_LOCATION_X`, `SOURCE_LOCATION_Y`, `SOURCE_VELOCITY_X`, and `SOURCE_VELOCITY_Y`. Positions are not clipped; values beyond the nominal pitch boundary are retained for out-of-play candidate detection. Player `LOCATION_Z` and `VELOCITY_Z` values of `-1` are treated as missing sentinels in tracking diagnostics.

This is a configured coordinate mapping, not independently verified stadium calibration. Horizontal velocities are treated as metres per timestep, not metres per second. The CSV does not establish frame duration, and the Z scale is unverified; therefore time-based speeds, physical acceleration, and confirmed 3D goal outcomes are unavailable. Do not treat the legacy `frame_rate` or `seconds_per_frame` settings as validation of the source timing.

Set `normalize_source_coordinates: false` in `config.yaml` to keep source horizontal coordinates unchanged. The event detector's spatial thresholds and geometry assume the normalized 120 × 80 coordinate system, so use the supplied event-parameter YAML only with matching coordinates. Its path can be changed with `event_detection_parameters_path`.

## Event detection

`events.csv` is generated from tracking positions, horizontal velocities, player IDs, and team IDs. Event decisions do not use `ON_BALL`, score changes, player-role labels, or pre-existing event labels. The detector first estimates persistent ball control, then creates sequence-based candidate events such as touches, releases, passes and receptions, interceptions, carries, dribbles, shots, clearances, boundary exits, and restart candidates.

Rows include an event ID, frame interval, participants, inferred controllers, ball positions, confidence level, evidence flags, alternative hypotheses, and rule version. These confidence levels are qualitative heuristics, not calibrated probabilities. Intent-dependent categories, including shots versus long passes, clearances, take-ons, saves, and restart subtypes, should be treated as candidates unless independently reviewed.

The separate possession chains and player/team statistics still use the existing possession-inference path; they are not derived from the event detector's control-state timeline. Review this distinction when comparing event output with possession-based statistics.

## Validation limits

The tests include synthetic event sequences and pipeline checks, but there is no manually annotated reference set. Event precision, recall, timing error, and player attribution accuracy have not been measured. Review representative tracking windows and validate pitch geometry, attacking direction, and tracking continuity before relying on event labels. Advanced models such as xG, xPass, xT, EPV, and pitch-control outputs remain heuristic or uncalibrated without labelled training data.

## Output files

The analysis produces:

- frame_metrics.csv
- events.csv
- possession_chains.csv
- player_match_stats.csv
- team_match_stats.csv
- quality_report.json
- metric_metadata.json
- key_moments.csv: rule-based candidate moments (goals from score changes, shots, key passes, high turnovers, press escapes, build-ups, momentum shifts, rhythm changes, top speeds), each with a display `match_clock` and the `event_ids` it rests on
- match_rhythm.csv: 150-timestep windows of possession share, territory, control index (control vs chaos) and momentum
- player_tags.csv: threshold-triggered sprint and distance-milestone tags for on-screen player labels
- player_match_intel.csv: per-player broadcast stats (pass accuracy and difficulty, shots and shot speed, top speed, distance, sprints) with names from the synthetic roster

Events also carry `ball_travel_m`, `progression_m`, `lane_pressure_m` (closest opponent to the pass line), `opponents_within_5m`, `ball_speed_kmh`, `pass_difficulty` (0-100) and `completed`. Speeds in km/h use `seconds_per_timestep: 0.1` from `config.yaml`, which is inferred from plausible player and ball top speeds rather than stated by the source. Clubs and players in `data/roster.json` are fictional.

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for source fields, derived frame metrics, event records, aggregate statistics, and report schemas.

## MCP tool server

`python -m football_analytics.mcp_server` serves the analysis outputs as read-only MCP tools over streamable HTTP at `/mcp` (set `MCP_API_KEY` to require an `x-api-key` header). `validate_claims` and `validate_overlay` act as the quality gate for agent text and overlay JSON: every number must match a value in the cited records (percentages and probabilities only a cited share, index or momentum field), and penalty, card, foul, offside, VAR, injury, referee or substitution wording is rejected unless a cited record has that type. Time-indexed tools take `as_of` and only return or accept data finished by that timestep (shots count from release, everything else from its `end_timestep`). `validate_overlay` checks against data known at the overlay's `start_timestep` unless `as_of` is given.

Deploying to Azure Container Apps without Docker or ACR Tasks (blocked on Azure for Students):

```bash
uv pip install --target build/image/site --python-platform x86_64-manylinux_2_28 --python-version 3.13 --only-binary :all: -r requirements-mcp.txt
az acr import -n <registry> --source docker.io/library/python:3.13-slim --image python:3.13-slim
python scripts/push_image.py --registry <registry> --tag <tag>
az containerapp update -n ca-match-mcp -g rg-matchintel --image <registry>.azurecr.io/match-mcp:<tag>
```

The `Dockerfile` builds the same image where Docker or ACR Tasks are available.

## Live overlay

`python -m football_analytics.replay_server` (port 8765, dependencies in `requirements-replay.txt`) replays the tracking data as a live feed over server-sent events. Every key moment gets an instant overlay built from the data once the moment is finished (goals at the score change), and only data known by then is used. Every cue, derived or agent-written, is published only if it passes the gate: `validate_overlay` for overlays and `check_claims` against data known at the cue's timestep for text. Otherwise the data-built cue is used or nothing is shown. High-importance moments are then sent to the `match-orchestrator` agent in the background, and its personalized overlay and text appear in the insight feed only if both pass the same gate. Full time shows a whole-match recap built from the data, plus the agents' `match_recap` when they are on. It uses your `az login` for Foundry. Set `ENRICH=0` to run without agents. Finished agent results are cached in `build/enrich_cache/`. `AGENT_CALLS_PER_HOUR` caps live agent calls per hour (default 0, no cap). When the cap is reached, the stream sends a `rate_limited` status ("AI analysis is rate-limited this hour; instant cues keep coming.") and `/api/ask` returns 429. `POST /api/ask` with `"action": "recap"` asks `match_recap` for the match as of `timestep`; `question` is still required.

The overlay opens with five "how much I care" dials (speed, passing, pressure, possession, money; 0-100, default 50). The bands are Skip (<20), Light (20-49), Into it (50-74) and Obsessed (>=75). On a first visit the dials are shown with a Start button and nothing streams until Start (Escape also starts). Later visits start straight away with the saved dials. The dials reach the stream as `interests=speed:80,money:20,...`: cues for a stat in Skip are dropped, cues at Obsessed get a priority bump, and goals, shots and recaps have no dial, so they are never skipped. The viewer type sent to the agents (casual, analyst, newcomer or money) is derived from the dials (`persona_for` in `commentary.py`), and the bands are added to the agent profile and Ask style. The orchestrator forwards the bands (plus Ask depth, tone and tactical) to `fan-personalizer`, which leads with Obsessed/Into it stats and drops Skip stats. Change them any time in Match centre; each change reconnects the stream with the new weights.

The timeline bar along the bottom shows match progress and a diamond for each moment already shown: gold for turning points (goals, momentum shifts), cyan for moments involving the Player you picked, hollow grey for supporting moments (latest 12, only types the dials keep). Clicking a diamond rewinds to 10 seconds (100 timesteps) before the moment and plays on, so its alert and AI analysis show again. Markers never appear for moments not yet reached. Replay speed defaults to 1×.

Language: English, Español, Français, Deutsch, Português and Italiano translate the deterministic text too (cue explanations, commentary interpretation and value explanations, from the table in `src/football_analytics/i18n.py`; splash transitions, Ask labels and feedback buttons, from `I18N` in `overlay/ui/app.js`). Data-derived labels such as moment hints, stat lines and contract notes stay English. Arabic, Japanese and Chinese are marked "AI text only": only the agent-personalized text is translated.

Timing disclosure (also shown in Match centre): the match clock is normalized, 3000 timesteps scaled to 90 minutes (`timesteps_per_match`, `match_minutes`), not real match time.

Speeds and distances are inferred from position changes at an assumed `seconds_per_timestep: 0.1` (`config.yaml`); the source data does not state its frame timing.

The same UI (`overlay/ui/`) runs two ways:

- Web: open `http://127.0.0.1:8765` while the replay server runs.
- Desktop overlay: a transparent, always-on-top Tauri 2 window over any screen. Clicks pass through to the apps underneath except over its panels and tiles; `Ctrl+Shift+O` hides the controls.

```bash
cd overlay
npm install
npm run dev
```

### Hosted demo

Live at https://ca-match-replay-cons.blackocean-5801654c.norwayeast.azurecontainerapps.io (Azure Container Apps). The app's managed identity has Foundry User on the Foundry account, so it calls `match-orchestrator` without secrets. `AGENT_CALLS_PER_HOUR` caps live agent calls. Results prefilled with `scripts/prewarm_cache.py` replay without a call. Cache keys cover the game, moment, persona, language, favourite club and player, interest bands (not raw dial values), depth, tone and tactical flag. The prewarm script sets only persona and language, so it fills keys for default dials (all 50, Into it). Re-run it after any change to those keys or to the agent prompts, and pass `--languages` for other languages.

```bash
ENRICH_CONCURRENCY=1 python scripts/prewarm_cache.py --game 0 --personas casual analyst
uv pip install --target build/image/site-replay --python-platform x86_64-manylinux_2_28 --python-version 3.13 --only-binary :all: -r requirements-replay.txt
python scripts/push_image.py --app replay --registry <registry> --repository match-replay --tag <tag>
az containerapp update -n ca-match-replay-cons -g rg-matchintel --image <registry>.azurecr.io/match-replay:<tag>
```

## Testing

```bash
python -m pytest -q
```

The tests cover schema loading, data validation, tracking features, possession inference, and event detection using both the supplied CSV and synthetic sequences.
