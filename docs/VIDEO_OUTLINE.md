# Demo video outline

Target length: 115 seconds (under the 120 second limit). Clubs are fictional (Kestrel Vale FC and Port Saltmere United). The data is synthetic and built on a simulated tracking dataset (the source CSVs are named gfootball_results_*), not on real match data. Put the on-screen label "SYNTHETIC DATA, FICTIONAL CLUBS" in the corner of every shot that shows match content.

## Rules for the recording

- No league or club crests, no third-party brand logos, and no trademarked artwork on screen. Only the product names used in the demo may appear, and no logos.
- Music: use none, or a royalty-free track. If a track is used, record its name, the library and its licence here before publishing: `[music: none yet]`.
- Record the overlay at 1600x900 in the Tauri build for the corner overlay shots, and in a browser for the producer view.

## Shot list

| # | Start | End | Length | On screen | Voiceover |
| --- | --- | --- | --- | --- | --- |
| 1 | 0:00 | 0:08 | 8 s | Title card: "Match Intelligence". Small label: "SYNTHETIC DATA, FICTIONAL CLUBS". | "This is Match Intelligence, a live football analysis demo built on synthetic data." |
| 2 | 0:08 | 0:20 | 12 s | Plain text cards: a match is watched by many fans, and the key moments come and go in seconds. Stats are scattered across tables. | "Fans miss the moment that matters, and the numbers behind it are buried in tables. Most viewers want the why, not the spreadsheet." |
| 3 | 0:20 | 0:35 | 15 s | Live replay of Kestrel Vale FC against Port Saltmere United. Scorebar, clock and pitch radar are visible. The Kick off splash shows at the start. | "Here is a replay of a match between Kestrel Vale FC and Port Saltmere United. Tracking data drives the radar, the score and the clock." |
| 4 | 0:35 | 0:50 | 15 s | The overlay sits in the corner of the screen. A moment alert appears in the lower left. A Goal threat sting shows in the centre of the screen in its accent colour, then the stat card. | "When something happens, a cue appears in the corner. Each one shows its evidence, its confidence band and the numbers behind it." |
| 5 | 0:50 | 1:03 | 13 s | Click the Ask the match tile. Type a question and press Ask. The answer appears in the panel. | "Ask the match is a tap away. Ask in your own words and the answer is built from the same match records." |
| 6 | 1:03 | 1:13 | 10 s | Click Too basic under the answer. The effect note appears and the depth changes. | "Feedback changes how deep the next answer goes." |
| 7 | 1:13 | 1:25 | 12 s | Open `?view=studio`. Show the Producer queue, the On air switch, the Auto toggles, the Min confidence slider, and Air and Drop on a cue. | "A producer view queues cues. The producer can air or drop each one, filter by confidence, and choose which priorities go on air automatically." |
| 8 | 1:25 | 1:39 | 14 s | A simple diagram: the event interpreter, the overlay producer, the insight explainer, the personalisation agent, the checker, and the orchestrator, as labelled boxes. | "Several agents do the work. A deterministic event interpreter finds the moments. An overlay producer and an insight explainer write the cues. A personalisation agent adapts them. An orchestrator coordinates the calls." |
| 9 | 1:39 | 1:52 | 13 s | Text cards: "Synthetic data", "Fictional clubs", "Claims checked against match records before they reach the screen", "Heuristic metrics, not yet calibrated". | "Everything here is synthetic and the clubs are fictional. Every cue is checked against the match records before it reaches the screen. Some metrics are heuristics and are not yet calibrated." |
| 10 | 1:52 | 1:55 | 3 s | End card with the product name and the hosted demo address. The SYNTHETIC DATA label stays on screen. | "Match Intelligence. Thank you." |

## Total

Shot lengths: 8 + 12 + 15 + 15 + 13 + 10 + 12 + 14 + 13 + 3 = **115 seconds**. The end time of the last shot is 1:55, which is 115 seconds. This is under the 120 second limit.

## Notes on the agents shot (shot 8)

The agent names in the video should match the code. `agents/foundry_agents.py` defines four specialist Foundry agents: `match-moment-explainer`, `player-lens`, `match-narrator` and `fan-personalizer`. `agents/orchestrator/main.py` defines the `match-orchestrator`. The deterministic event interpreter is in `src/football_analytics/` (event detection), not an agent. The fact and consistency check is done by the `validate_claims` and `validate_overlay` tools in `src/football_analytics/mcp_server.py`, not by a separate agent. Use those names on screen if the diagram is drawn from the code.
