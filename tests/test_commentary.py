import json
from pathlib import Path

import pandas as pd

from football_analytics import commentary as c
from football_analytics import mcp_server as server

ROOT = Path(__file__).resolve().parents[1]
STATS = {"possession": [60, 40], "ball_wins": [3, 2], "shots": [2, 1], "passes": [30, 20], "pass_accuracy": [80, 70], "sprints": [4, 3]}


def _setup(tmp_path, monkeypatch):
    folder = tmp_path / "games_0"
    folder.mkdir()
    pd.DataFrame([
        {"event_id": "0-1", "GAME_ID": 0, "start_timestep": 10, "event_type": "shot_candidate", "PLAYER_ID": 9, "TEAM_ID": 0, "ball_speed_kmh": 120.5},
        {"event_id": "0-2", "GAME_ID": 0, "start_timestep": 14, "event_type": "pass_attempt", "PLAYER_ID": 9, "TEAM_ID": 0, "ball_speed_kmh": None},
        {"event_id": "0-3", "GAME_ID": 0, "start_timestep": 400, "event_type": "pass_attempt", "PLAYER_ID": 9, "TEAM_ID": 0, "ball_speed_kmh": None},
    ]).to_csv(folder / "events.csv", index=False)
    pd.DataFrame([
        {"moment_id": "M1", "GAME_ID": 0, "start_timestep": 5, "end_timestep": 450, "match_clock": "00:22", "moment_type": "goal", "TEAM_ID": 0, "PLAYER_ID": 9, "importance": 1.0, "event_ids": "0-1;0-2;0-3", "metrics": json.dumps({"score_after": "1-0"}), "fact_hint": "Goal for team 0; score 1-0"},
    ]).to_csv(folder / "key_moments.csv", index=False)
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"clubs": {"0": {"name": "Home", "short": "HOM"}, "1": {"name": "Away", "short": "AWY"}}, "players": {"9": {"name": "Daniel Moreau", "short": "Moreau", "number": 10}}}))
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "ROSTER_PATH", roster)
    server._data.cache_clear()
    server._roster.cache_clear()


def test_schemas_match_models():
    for name, model in (("viewer_preferences", c.ViewerPreferences), ("commentary_response", c.CommentaryResponse), ("feedback_event", c.FeedbackEvent)):
        assert json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text(encoding="utf-8")) == model.model_json_schema()


def test_no_future_events_and_clean_headline(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    response = c.respond(0, 100, c.ViewerPreferences(depth="deep"), STATS, "explain_more")
    assert response.headline == "Goal, Home"
    assert [t.id for t in response.context_trail if t.kind == "event"] == ["0-1", "0-2"]
    assert all("0-3" not in t.id for t in response.context_trail)
    assert not any(c.match_clock(400) in line for line in response.observed)


def test_player_name_routes_to_focus_player(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    response = c.respond(0, 100, c.ViewerPreferences(), STATS, "question", "How is Moreau doing?")
    assert (response.action, response.player_id) == ("focus_player", 9)
    assert response.observed[0].startswith("2 involvements")


def test_detect_action():
    assert c.detect_action("Why did that matter?") == "why"
    assert c.detect_action("show me the pressing shape") == "show_tactics"
    assert c.detect_action("give me the numbers") == "numbers"
    assert c.detect_action("tell me more") == "explain_more"


def test_feedback_adjusts_preferences(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    prefs = c.ViewerPreferences()
    basic = c.apply_feedback(prefs, c.FeedbackEvent(kind="too_basic", game_id=0, timestep=1)).preferences
    assert basic.depth == "deep"
    technical = c.apply_feedback(basic, c.FeedbackEvent(kind="too_technical", game_id=0, timestep=1)).preferences
    assert technical.depth == "balanced" and not technical.tactical
    follow = c.apply_feedback(prefs, c.FeedbackEvent(kind="follow_player", game_id=0, timestep=1, player_id=9))
    assert follow.preferences.favorite_player_id == 9 and "Daniel Moreau" in follow.effects[0]
