# Graph Report - football_analytics  (2026-10-24)

## Corpus Check
- 44 files · ~31,092 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 426 nodes · 864 edges · 28 communities (25 shown, 3 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 13 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c54788d3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- mcp_server.py
- app.js
- replay_server.py
- Data Dictionary
- cli.py
- tauri.conf.json
- match_intelligence.py
- player_focus
- test_overlay_cues.py
- load_tracking_data
- overlay_cues.py
- Football tracking normalization and event-detector parameter report
- test_pipeline.py
- detect_events
- foundry_agents.py
- main.rs
- package.json
- push_image.py
- default.json
- infer_possession
- __init__.py
- model_status
- deploy.py
- estimate_pitch_control
- football-analytics
- match-overlay

## God Nodes (most connected - your core abstractions)
1. `_game()` - 23 edges
2. `run_analysis()` - 21 edges
3. `_roster()` - 19 edges
4. `respond()` - 18 edges
5. `load_tracking_data()` - 15 edges
6. `match_clock()` - 14 edges
7. `detect_events()` - 13 edges
8. `derived_cues()` - 13 edges
9. `enrich_events()` - 12 edges
10. `compute_tracking_features()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `test_tracking_only_detector_links_pass_and_ignores_labels()` --calls--> `detect_events()`  [EXTRACTED]
  tests/test_pipeline.py → src/football_analytics/event_detection.py
- `test_normalize_pitch_coordinates_preserves_sources_and_does_not_clip()` --calls--> `normalize_pitch_coordinates()`  [EXTRACTED]
  tests/test_pipeline.py → src/football_analytics/data_loader.py
- `test_both_supplied_csvs_load_all_six_games()` --calls--> `load_tracking_data()`  [EXTRACTED]
  tests/test_pipeline.py → src/football_analytics/data_loader.py
- `test_load_tracking_csv()` --calls--> `load_tracking_data()`  [EXTRACTED]
  tests/test_pipeline.py → src/football_analytics/data_loader.py
- `test_pass_quality_and_ball_speed()` --calls--> `detect_events()`  [EXTRACTED]
  tests/test_pipeline.py → src/football_analytics/event_detection.py

## Import Cycles
- None detected.

## Communities (28 total, 3 thin omitted)

### Community 0 - "mcp_server.py"
Cohesion: 0.09
Nodes (58): Action, Depth, apply_feedback(), _club(), CommentaryResponse, _current_moment(), default_depth(), detect_action() (+50 more)

### Community 1 - "app.js"
Cohesion: 0.10
Nodes (41): addEntry(), analyst(), applyFrame(), applyPersona(), ask(), buildTiles(), clubName(), compactMode() (+33 more)

### Community 2 - "replay_server.py"
Cohesion: 0.15
Nodes (29): JSONResponse, Request, RuntimeError, run_dashboard(), ask(), ask_orchestrator(), BudgetExceeded, _club() (+21 more)

### Community 3 - "Data Dictionary"
Cohesion: 0.08
Nodes (24): Conventions and Units, Data Dictionary, `events.csv`, `frame_metrics.csv`, Input Tracking CSV, Legacy inferred-possession fields, `metric_metadata.json`, `player_match_stats.csv` (+16 more)

### Community 4 - "cli.py"
Cohesion: 0.12
Nodes (17): main(), run_analysis(), AnalyticsConfig, Any, Path, export_outputs(), Any, DataFrame (+9 more)

### Community 5 - "tauri.conf.json"
Cohesion: 0.10
Nodes (20): app, security, windows, withGlobalTauri, build, frontendDist, bundle, active (+12 more)

### Community 6 - "match_intelligence.py"
Cohesion: 0.20
Nodes (20): detect_key_moments(), _direction(), enrich_events(), _frame_lookup(), _json_value(), load_roster(), match_rhythm(), _moment() (+12 more)

### Community 7 - "player_focus"
Cohesion: 0.22
Nodes (18): Agent, build_agent(), _call(), _envelope(), live_moment(), main(), match_recap(), player_focus() (+10 more)

### Community 8 - "test_overlay_cues.py"
Cohesion: 0.14
Nodes (12): main(), Fill build/enrich_cache with agent results so the hosted demo can replay them…, run(), _setup(), test_validate_claims_accepts_cited_facts_and_rejects_others(), test_validate_overlay_checks_schema_and_numbers(), _setup(), test_goal_cue_is_themed_traceable_and_gated() (+4 more)

### Community 9 - "load_tracking_data"
Cohesion: 0.15
Nodes (15): _coerce_numeric_columns(), load_tracking_data(), normalize_pitch_coordinates(), DataFrame, Path, Map centered source coordinates to metres on a 120 x 80 pitch., Load and minimally validate a football-tracking CSV file., Any (+7 more)

### Community 10 - "overlay_cues.py"
Cohesion: 0.29
Nodes (16): _club(), _confidence(), cue_schema(), derived_cues(), _goal_story(), Metric, _metrics(), moment_cue() (+8 more)

### Community 11 - "Football tracking normalization and event-detector parameter report"
Cohesion: 0.13
Nodes (14): 1. Coordinate conversion, 2. Data audit, 3. Critical timing finding: velocity is per timestep, not verified m/s, 4. Observed distributions on the 120 × 80 pitch, 5. Detected reset / discontinuity signature, 6. Recommended starting parameters, 7. Event-specific implementation guidance, 8. Must-have validation before treating results as statistics (+6 more)

### Community 12 - "test_pipeline.py"
Cohesion: 0.23
Nodes (13): Series, compute_tracking_features(), _nearest_player_distances(), DataFrame, Compute basic movement and spatial features for each entity-frame., _pass_sequence(), test_nearest_player_distances_split_by_team(), test_pass_difficulty_rewards_long_tight_progressive_passes() (+5 more)

### Community 13 - "detect_events"
Cohesion: 0.37
Nodes (12): _control_states(), detect_events(), _detect_game(), detect_pass_events(), _event(), _intervals(), _make_game_frames(), _nested() (+4 more)

### Community 14 - "foundry_agents.py"
Cohesion: 0.29
Nodes (10): ask(), _client(), create(), _definition(), _nullable(), _object(), AIProjectClient, Any (+2 more)

### Community 15 - "main.rs"
Cohesion: 0.29
Nodes (10): Mutex, Option, cursor_over_controls(), HitRegions, main(), set_hit_regions(), watch_cursor(), State (+2 more)

### Community 16 - "package.json"
Cohesion: 0.18
Nodes (10): devDependencies, @tauri-apps/cli, name, private, scripts, build, dev, tauri (+2 more)

### Community 17 - "push_image.py"
Cohesion: 0.39
Nodes (7): Client, _layer(), main(), Path, Build and push the MCP server or replay server image without Docker or ACR…, _token(), _upload()

### Community 18 - "default.json"
Cohesion: 0.25
Nodes (7): identifier, permissions, $schema, windows, core:default, core:window:allow-close, main

### Community 19 - "infer_possession"
Cohesion: 0.29
Nodes (7): build_possession_chains(), infer_possession(), DataFrame, Infer a candidate owner and team for each frame using ON_BALL and the nearest…, Create simple possession-chain groups by consecutive team owners., test_possession_and_event_creation(), test_uncontrolled_distant_ball_remains_unknown()

### Community 20 - "__init__.py"
Cohesion: 0.33
Nodes (5): Football tracking analytics package., generate_quality_report(), Any, DataFrame, Build a concise quality report describing coverage, missingness, and anomalies.

### Community 21 - "model_status"
Cohesion: 0.73
Nodes (5): model_status(), Any, xg_baseline(), xpass_baseline(), xt_baseline()

### Community 22 - "deploy.py"
Cohesion: 0.50
Nodes (4): deploy(), AIProjectClient, Deploy the orchestrator as a Foundry hosted agent from source (remote build, no…, _zip()

## Knowledge Gaps
- **67 isolated node(s):** `name`, `private`, `version`, `tauri`, `dev` (+62 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `match_clock()` connect `mcp_server.py` to `replay_server.py`, `cli.py`, `match_intelligence.py`, `overlay_cues.py`, `test_pipeline.py`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Why does `run_analysis()` connect `cli.py` to `mcp_server.py`, `match_intelligence.py`, `load_tracking_data`, `test_pipeline.py`, `detect_events`, `infer_possession`, `__init__.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Why does `load_tracking_data()` connect `load_tracking_data` to `replay_server.py`, `cli.py`, `__init__.py`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **What connects `name`, `private`, `version` to the rest of the system?**
  _67 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `mcp_server.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09344262295081968 - nodes in this community are weakly interconnected._
- **Should `app.js` be split into smaller, more focused modules?**
  _Cohesion score 0.10465116279069768 - nodes in this community are weakly interconnected._
- **Should `Data Dictionary` be split into smaller, more focused modules?**
  _Cohesion score 0.07692307692307693 - nodes in this community are weakly interconnected._