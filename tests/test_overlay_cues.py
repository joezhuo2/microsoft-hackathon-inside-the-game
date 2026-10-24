import json
from pathlib import Path

import pandas as pd

from football_analytics import mcp_server as server
from football_analytics import overlay_cues as cues
from football_analytics import replay_server as replay

ROOT = Path(__file__).resolve().parents[1]


def _setup(tmp_path, monkeypatch):
    folder = tmp_path / "games_0"
    folder.mkdir()
    pd.DataFrame([{"event_id": "0-000001", "GAME_ID": 0, "start_timestep": 10, "event_type": "shot_candidate", "PLAYER_ID": 9, "TEAM_ID": 0, "ball_speed_kmh": 120.5, "confidence_level": "MEDIUM"}]).to_csv(folder / "events.csv", index=False)
    pd.DataFrame([
        {"moment_id": "M0-0001", "GAME_ID": 0, "start_timestep": 5, "end_timestep": 12, "match_clock": "00:22", "moment_type": "goal", "TEAM_ID": 0, "PLAYER_ID": 9, "importance": 1.0, "event_ids": "0-000001", "metrics": json.dumps({"score_after": "1-0", "shot_speed_kmh": 120.5}), "fact_hint": "Goal"},
        {"moment_id": "M0-0002", "GAME_ID": 0, "start_timestep": 40, "end_timestep": 50, "match_clock": "01:12", "moment_type": "rhythm_change", "TEAM_ID": None, "PLAYER_ID": None, "importance": 0.45, "event_ids": None, "metrics": json.dumps({"control_index": 0.8, "from": "chaos", "to": "control", "turnovers": 0}), "fact_hint": "Game moved from chaos to control"},
    ]).to_csv(folder / "key_moments.csv", index=False)
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"clubs": {"0": {"name": "Home", "short": "HOM"}, "1": {"name": "Away", "short": "AWY"}}, "players": {"9": {"name": "Daniel Moreau", "short": "Moreau", "number": 10}}}))
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "ROSTER_PATH", roster)
    server._data.cache_clear()
    server._roster.cache_clear()


def test_schema_file_matches_model():
    assert json.loads((ROOT / "schemas" / "overlay_cue.schema.json").read_text(encoding="utf-8")) == cues.cue_schema()


def test_goal_cue_is_themed_traceable_and_gated(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    goal = replay.instant_overlay(replay.moments(0)[0])
    cue = goal["cue"]
    assert (cue["theme"], cue["priority"], cue["evidence"]) == ("goal_threat", "high", "observed")
    assert cue["why_it_matters"] == "Home take the lead, 1-0. The game state changes for both sides."
    assert cue["source_event_ids"] == ["0-000001"] and cue["synthetic"] is True
    assert goal["cue_validation"]["valid"], goal["cue_validation"]
    assert goal["overlay"]["headline"] == "GOAL Home, 1-0"


def test_rhythm_to_control_and_favourite_player_bump(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    rhythm = replay.instant_overlay(replay.moments(0)[1])["cue"]
    assert rhythm["theme"] == "control_index" and rhythm["evidence"] == "derived"
    focused = cues.moment_cue(replay.moments(0)[0], [], {"favorite_player_id": 9, "persona": "analyst"})
    assert focused.theme == "goal_threat" and focused.audience_mode == "analyst"
    assert focused.player_focus.name == "Daniel Moreau"
