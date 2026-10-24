# Match Intelligence: pitch for the judges

## Problem

Football fans and broadcast producers get plenty of match data and little explanation. Key moments pass in seconds, and the numbers that explain them sit in tables. Viewers want to know why a moment mattered, at their own level of detail.

## What we built

A live replay of a fictional match between Kestrel Vale FC and Port Saltmere United, with a transparent desktop overlay that shows cues in the corner. An Ask the Match panel answers quick and free-text questions, viewer feedback sets the depth of detail, and a producer view queues and airs cues.

## Why it is multi-agent

A deterministic event interpreter finds moments from tracking data, so the facts come from code. Specialist Foundry agents explain them: `match-moment-explainer` (why a moment matters), `player-lens` (player profiles), `match-narrator` (commentary and overlay payloads) and `fan-personalizer` (viewer depth and language). `match-orchestrator` coordinates them. The fact and consistency checker is the `validate_claims` and `validate_overlay` tools, which reject any cited id or number not in the match record.

## Real-world impact

The intended users are fans new to the game and producers who prepare cues quickly. We have not measured either benefit.

## Honesty and limits

All clubs, players and match data are fictional and synthetic. The tracking input is a simulated tracking dataset (the source CSVs are named gfootball_results_*), not real match data, and no real player data is used. The metrics are heuristics and are not yet calibrated. The README says so. Event precision, recall and player attribution have not been measured against an annotated reference set.

## What is next

Calibrate the metrics against annotated tracking windows, add the synthetic-data extension (being added in parallel: `scripts/extend_synthetic.py` and its generated files), and test the overlay with real viewers.
