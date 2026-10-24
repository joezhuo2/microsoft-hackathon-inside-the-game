# Football tracking normalization and event-detector parameter report

## Deliverables

- Normalized input: `gfootball_results_games_0-2_pitch120x80.csv`
- Machine-readable starter configuration: `football_event_detection_parameters_pitch120x80.yaml`
- Machine-readable data profile: `football_tracking_data_profile.json`

## 1. Coordinate conversion

The source coordinates use the usual centered normalized field convention: nominal pitch boundaries are `x ∈ [-1, 1]` and `y ∈ [-0.42, 0.42]`. The transformation maps those nominal boundaries exactly to a 120 × 80 pitch:

- `LOCATION_X = (SOURCE_LOCATION_X + 1) × 60`
- `LOCATION_Y = (SOURCE_LOCATION_Y + 0.42) × (80 / 0.84)`
- `VELOCITY_X = SOURCE_VELOCITY_X × 60`
- `VELOCITY_Y = SOURCE_VELOCITY_Y × (80 / 0.84)`

The original four horizontal fields are retained in `SOURCE_LOCATION_X`, `SOURCE_LOCATION_Y`, `SOURCE_VELOCITY_X`, and `SOURCE_VELOCITY_Y`. All other source fields are preserved. `LOCATION_Z` and `VELOCITY_Z` are not transformed because the vertical units cannot be verified from this file alone.

**Do not clip positions to [0,120] × [0,80].** The source contains genuine-looking beyond-boundary positions. Those values are useful for candidate out-of-play logic. After conversion, the observed full coordinate ranges are X = -1.178 to 121.257 m, and Y = -1.422 to 81.560 m. Values outside pitch bounds are preserved.

## 2. Data audit

| Check | Observed result |
|---|---:|
| Rows | 207,069 |
| Games | 3 (0, 1, 2) |
| Frames | 9,003 total; 3,001 per game |
| Timestep range | 0–3000 in each game |
| Rows per frame | {23: 9003} |
| Entities per frame | 22 players + 1 ball |
| Duplicate `(GAME_ID, TIMESTEP, PLAYER_ID)` keys | 0 |
| Missing source cells | 0 |
| Player vertical Z/VZ sentinel | -1 for all player rows; treat as missing |

The event-parameter analysis uses coordinates, velocities, entity IDs, and team IDs. `ON_BALL` and score columns were **not** used to tune the thresholds; they remain in the CSV for format compatibility and must be ignored by the second agent.

## 3. Critical timing finding: velocity is per timestep, not verified m/s

After coordinate conversion, the ratio between position displacement over a timestep and the magnitude of the supplied horizontal velocity has a median of 0.990 for players and 1.018 for the ball (large jumps above 10 m were excluded from this comparison). This is strong evidence that the horizontal velocity fields correspond approximately to **metres per timestep**, or an equivalent per-step quantity. It is not evidence of metres per second.

The frame rate / seconds per timestep is not present in the CSV. Therefore:

- Use `m/timestep` and `m/timestep²` (or simply Δvelocity per timestep) in the first version.
- Do not calculate metres per second, physical acceleration, sprint speed bands, duration in seconds, or per-90 stats until frame rate is independently calibrated.
- Use the supplied velocity for instantaneous direction/speed, and independently finite-difference positions as a consistency check.

## 4. Observed distributions on the 120 × 80 pitch

### Nearest player-to-ball distance (one value per frame)

- Median: 0.974 m
- 75th percentile: 1.828 m
- 90th percentile: 4.287 m
- Share of frames with nearest player within 1.0 m: 52.1%
- Within 1.25 m: 65.6%
- Within 1.5 m: 70.9%
- Within 2.0 m: 76.9%

These numbers justify 1.25 m as an initial **candidate-control acquisition radius**, but proximity alone is insufficient: players are frequently near the ball without necessarily controlling it. Require persistence and compatible relative motion.

### Horizontal movement distributions

| Measure | Median | 75th percentile | 90th percentile | 95th percentile | 99th percentile |
|---|---:|---:|---:|---:|---:|
| Player supplied horizontal speed (m/timestep) | 0.397 | 0.611 | 0.806 | 0.852 | 0.881 |
| Ball supplied horizontal speed (m/timestep) | 0.699 | 1.200 | 1.820 | 2.449 | 3.684 |
| Player displacement per timestep, excluding reset jumps >3 m | 0.397 | 0.607 | 0.803 | 0.850 | 0.880 |
| Ball displacement per timestep, excluding jumps >10 m | 0.699 | 1.152 | 1.797 | 2.412 | 3.661 |

### Ball trajectory changes

- Ball Δvelocity median: 0.036 m/timestep.
- Ball Δvelocity 90th percentile: 0.433 m/timestep.
- Ball Δvelocity 95th percentile: 0.945 m/timestep.
- A Δvelocity threshold of 0.5 m/timestep is exceeded on 8.7% of ball frames.
- A heading-change threshold of 45° while speed is above 0.75 m/timestep on both adjacent frames occurs on 0.52% of ball frames.

Use these as trajectory-change **supporting features**, not event definitions by themselves. A trajectory bend can come from a touch, collision, bounce, tracking artifact, or a sharp control change.

## 5. Detected reset / discontinuity signature

There are **40 frames out of 9003** where at least 8 of 22 players move more than 3 m in a single timestep. In this sample each such frame actually has 19–22 players over that threshold. The pattern appears in all 3 games (game 0: 14, game 1: 12, game 2: 14). At these same transition frames, the supplied ball horizontal velocity is zero or effectively zero.

This is a strong **global reset / state-transition signature**, not proof of a specific football event. The second agent should:

1. Detect a global reset when at least 8 players have a position jump over 3 m in one timestep.
2. Mark a break in movement derivatives, trajectories, and event-linking across that transition.
3. Never treat the large player/ball jump as a run, pass, shot, or measured physical distance.
4. Inspect the ball location and nearby formation after the reset to generate a restart candidate, but do not force a restart subtype from the reset alone.

Separately, the ball has 11 single-timestep displacement jumps over 10 m and 10 over 30 m. Treat an isolated ball jump over 10 m as a tracking discontinuity unless an independently validated data-specific pattern supports it.

## 6. Recommended starting parameters

These are initial implementation values, not validated ground truth. The control radii and motion thresholds are anchored to this dataset's observed distributions; event-specific distance cutoffs are explicit football-logic heuristics that must remain configurable.

| Parameter | Starter value | Why |
|---|---:|---|
| Control acquisition radius | 1.25 m | About 65.6% of frames have a player within this distance; useful as a candidate radius only |
| Control retention radius | 2.25 m | Hysteresis to reduce controller flicker; larger than acquisition radius |
| Contested radius | 2.5 m | Captures tight multi-player interaction; use only if opposing candidates are both nearby |
| Maximum distance gap for contested tie | 0.75 m | Provisional near-tie threshold |
| Minimum stable control | 3 consecutive frames | Suppresses single-frame false control |
| Touch proximity | 1.25 m | Dataset-anchored proximity candidate |
| Touch trajectory support | Δvelocity ≥0.5 m/timestep OR heading change ≥45° with speed >0.75 on both sides | Empirical tail / provisional change thresholds |
| Ball-release speed support | >1.0 m/timestep | About 28.5% of ball frames exceed it; use as support, not a mandatory rule |
| Ball-release separation growth | ≥0.75 m over 2 frames and exceeds 2.25 m within 3 frames | Provisional separation rule |
| Global reset | ≥8 players each jump >3 m in the same timestep | Captures all 40 reset-like frames in this sample; observed minimum was 19 |
| Ball discontinuity | >10 m in one timestep | Captures 11 obvious high-jump frames, including jumps >30 m |
| Progressive carry | ≥10 m net goalward gain OR cross a configured pitch-zone boundary | Football-logic heuristic; not learned from these labels |
| Cross origin | Outer 16 m channel along pitch width, pass enters target penalty area | Wide-zone heuristic on an 80 m width |
| Through ball | Trajectory/receiver passes at least 3 m beyond inferred defensive line | Provisional spatial margin |
| Switch of play | ≥30 m lateral displacement and opposite wide channels | Provisional width-shift criterion |
| Out-of-play candidate | Segment crosses x=0/120 or y=0/80; use 0.25 m outside margin as backup | Keep outside coordinates; don't rely on one marginal outside sample |
| Stationary-ball candidate | Speed ≤0.25 m/timestep for ≥3 frames PLUS restart / boundary / formation evidence | Stationary ball alone is not enough |

## 7. Event-specific implementation guidance

- **Touches:** proximity plus trajectory change or control transition; keep touch candidates separate from possession gains.
- **Possession gain/loss:** require persistent control; contested/unknown states are allowed. A loss can lead to free ball, opponent control, or a stoppage candidate.
- **Release/pass:** detect separation from the stable controller; classify the release only after following its trajectory and next stable controller.
- **Completed pass/reception:** link release to the first stable teammate controller. Count the pass once, and store reception as a linked event.
- **Incomplete pass/interception:** opponent control after a plausible pass becomes an interception candidate; a boundary exit becomes an incomplete-pass outcome; loose unresolved balls stay unresolved.
- **Carry/dribble:** same controller across frames with meaningful ball path; dribble needs a nearby defender to be bypassed or escaped.
- **Progressive carry:** use net displacement toward the configured attacking goal, not absolute X alone.
- **Cross/through ball/switch:** apply as pass subtypes using origin zone, destination/trajectory, defending line, and width displacement.
- **Shot:** release plus goal-directed projected trajectory and goal geometry. Do not use speed alone.
- **On/off target, goal, save, block:** create candidates based on goal-plane and nearby interactions. Because Z units are not calibrated, do not label a shot as *confirmed* on target or a confirmed goal from geometry alone until vertical scale is verified.
- **Clearance:** defensive-zone release away from own goal with strong progression and optional pressure support; preserve `CLEARANCE_LIKE_RELEASE` if intent is ambiguous.
- **Corner/throw-in:** infer from a verified boundary-crossing segment and which line(s) were crossed, followed by restart evidence.
- **Goal kick/penalty/free kick/kickoff:** use nominal marked areas and restart formation only as candidates. Tracking-only data cannot establish a foul or prove restart intent.
- **Other restart:** use when a stoppage/reset pattern is evident but the specific subtype is not separable.

## 8. Must-have validation before treating results as statistics

1. Create synthetic controlled sequences for pass, interception, carry, deflection, shot, and each boundary restart.
2. Visualize all global reset frames and all ball jumps over 10 m.
3. Review a manually annotated sample of real tracking windows; no labelled events exist in the current file, so threshold performance cannot be measured yet.
4. Measure event precision, recall, timing error, and player attribution separately.
5. Keep the configured heuristic parameters separate from learned/calibrated model probabilities.
6. Re-run data profiling on future files (e.g. games 3–5, 6–8) rather than assuming the same thresholds and attacking directions automatically transfer.

## 9. Output-column semantics

`LOCATION_X/Y`: position in metres on the 120 × 80 field; values can fall outside pitch boundaries. `VELOCITY_X/Y`: metres per timestep, not m/s. `LOCATION_Z` and `VELOCITY_Z`: original units; player value -1 is unavailable. `SOURCE_*` columns provide the unmodified native horizontal values for reproducibility.
