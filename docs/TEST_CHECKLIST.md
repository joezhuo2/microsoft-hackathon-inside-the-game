# Test checklist

## A. Automated tests

Run from the repo root:

```
.venv-local\Scripts\python.exe -m pytest -q
```

`pytest.ini` sets `pythonpath = src`. The results of this run are not recorded in this document.

| File | What it covers |
| --- | --- |
| `tests/test_commentary.py` | Schema files match the Pydantic models; no future events leak into a response and the headline is clean; a player name in the question routes to focus-player; action detection from question text; feedback adjusts viewer preferences; agent profile collapses viewer types. |
| `tests/test_mcp_server.py` | `validate_claims` accepts cited facts and rejects uncited ones; `validate_overlay` checks schema and that numbers are backed by the moment. |
| `tests/test_overlay_cues.py` | The overlay cue schema file matches its model; a goal cue is themed, traceable to source events and gated; rhythm changes map to control, and a favourite player gets a priority bump. |
| `tests/test_pipeline.py` | Tracking CSV loading; coordinate normalisation keeps source values and does not clip; ball and player detection in preprocessing; both supplied CSVs load all six games; stationary entities have zero speed; diagnostics do not bridge gaps or resets; possession and event creation; the tracking-only pass detector ignores labels; an uncontrolled distant ball stays unknown; nearest-player distances split by team; pass quality, ball speed and pass difficulty; a score change becomes a goal moment with its clock; sprint tags need sustained speed. |
| `tests/test_replay_server.py` | An instant overlay passes the gate; an enriched overlay falls back to gated English when agents fail; the agent call budget blocks after the hourly limit. |
| `tests/test_synthetic_extension.py` | Generated synthetic rows are flagged synthetic; event ids are unique and prefixed; player ids exist in the roster; the substitution window is respected; generation is deterministic. This file is untracked in the working tree and belongs to the synthetic-data extension being added in parallel. |

## B. Manual browser checks

Start the replay server and open the overlay in a browser (`/` for the overlay, `?view=studio` for the producer view). Tick each line when it passes.

- [ ] At 1600x900 the stat tiles sit in one row or wrap onto a second row, and no tile is clipped.
- [ ] At 390px wide the stat tiles wrap into rows and no tile is clipped.
- [ ] The Ask the match tile opens and closes the Ask panel.
- [ ] Opening Ask the match closes Match centre, and opening Match centre closes Ask the match (accordion behaviour).
- [ ] Each quick-action chip (Explain more, Show tactics, Focus on player, Make it simpler, Give me the numbers, Why did this matter?) returns a commentary response.
- [ ] A typed question in the Ask field returns a commentary response.
- [ ] The Too basic feedback button changes the depth setting and shows an effect note.
- [ ] Each feedback button (Too basic, Too technical, More tactical, More emotional, Follow this player) shows an effect note.
- [ ] On first load with no saved preferences, the onboarding dialog "How do you watch?" appears.
- [ ] Skip closes the onboarding dialog.
- [ ] After closing the onboarding dialog, reloading the page does not show it again (`match-onboarded` in localStorage).
- [ ] The Kick off splash appears when the replay starts.
- [ ] The Half time splash appears at the half time point, followed by the Second half splash.
- [ ] The Full time splash appears at the end of the match, followed by the Match recap splash.
- [ ] Splashes appear in that order: Kick off, Half time, Second half, Full time, Match recap.
- [ ] A theme sting (for example Goal threat) appears on a qualifying cue.
- [ ] No theme sting appears during the Half time splash.
- [ ] No theme sting appears during the Full time splash.
- [ ] `?view=studio` shows the Producer queue panel.
- [ ] The producer view shows the On air checkbox and it can be toggled.
- [ ] The producer view shows the Auto high, Auto medium and Auto low toggles, and each can be changed.
- [ ] The producer view shows the Min confidence slider and its value updates.
- [ ] The producer view shows Air and Drop buttons on queued cues, and each changes the cue state (Air becomes On air, Drop becomes Dropped).
- [ ] With the system setting for reduced motion on, cue alerts, splashes and feed items do not slide, blur or pulse.
- [ ] In the Tauri build, clicks on empty desktop areas outside the overlay controls reach the desktop.
- [ ] In the Tauri build, clicks on the Match centre, Ask the match and onboarding controls work.
- [ ] In the Tauri build, Ctrl+Shift+O hides the dock and makes the whole overlay click-through.
- [ ] The hosted demo loads at https://ca-match-replay-std.ashymeadow-7dc0a741.norwayeast.azurecontainerapps.io
