import json

import pandas as pd

from football_analytics import mcp_server as server
from football_analytics import replay_server as replay


def _setup(tmp_path, monkeypatch):
    folder = tmp_path / "games_0"
    folder.mkdir()
    pd.DataFrame([{"event_id": "0-000001", "GAME_ID": 0, "start_timestep": 10, "event_type": "shot_candidate", "PLAYER_ID": 9, "TEAM_ID": 0, "ball_speed_kmh": 120.5}]).to_csv(folder / "events.csv", index=False)
    pd.DataFrame([{"moment_id": "M0-0001", "GAME_ID": 0, "start_timestep": 5, "end_timestep": 12, "match_clock": "00:22", "moment_type": "goal", "TEAM_ID": 0, "PLAYER_ID": 9, "importance": 1.0, "event_ids": "0-000001", "metrics": json.dumps({"score_after": "1-0", "shot_speed_kmh": 120.5}), "fact_hint": "Goal"}]).to_csv(folder / "key_moments.csv", index=False)
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"clubs": {"0": {"name": "Home"}, "1": {"name": "Away"}}, "players": {"9": {"name": "Daniel Moreau", "short": "Moreau", "number": 10}}}))
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "ROSTER_PATH", roster)
    server._data.cache_clear()
    server._roster.cache_clear()


def test_instant_overlay_passes_gate(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    result = replay.instant_overlay(replay.moments(0)[0])
    assert result["validation"]["valid"], result
    assert result["overlay"]["headline"] == "GOAL Home, 1-0"
    assert result["overlay"]["stats"] == [{"label": "shot speed", "value": 120.5, "unit": "km/h"}]


def test_enriched_overlay_falls_back_to_gated_english(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    base = {"game_id": 0, "moment_id": "M0-0001", "start_timestep": 5, "display_seconds": 6, "template": "lower_third", "language": "en", "headline": "Moreau scores, 1-0", "body": "Shot at about 120.5 km/h", "player_id": 9, "stats": []}
    good = replay.enriched_overlay({"narration": {"overlays": [base]}, "personalized": {"overlay_headline": "Moreau marque, 1-0", "overlay_body": "Tir à environ 120.5 km/h"}, "profile": {"language": "fr"}})
    assert good["overlay"]["language"] == "fr" and good["validation"]["valid"]
    bad = replay.enriched_overlay({"narration": {"overlays": [base]}, "personalized": {"overlay_headline": "Moreau marque à 150 km/h"}, "profile": {"language": "fr"}})
    assert bad["overlay"] == base
    assert replay.enriched_overlay({})["overlay"] is None


def test_agent_budget_blocks_after_hourly_limit(monkeypatch):
    monkeypatch.setattr(replay, "AGENT_CALLS_PER_HOUR", 2)
    replay.agent_calls.clear()
    replay.spend_agent_call()
    replay.spend_agent_call()
    try:
        replay.spend_agent_call()
    except replay.BudgetExceeded:
        pass
    else:
        raise AssertionError("third call should exceed the budget")
    replay.agent_calls.clear()


def test_cors_preflight_allows_private_network_for_tauri_origin():
    from starlette.testclient import TestClient

    headers = {"Origin": "http://tauri.localhost", "Access-Control-Request-Method": "GET", "Access-Control-Request-Private-Network": "true"}
    response = TestClient(replay.app).options("/api/meta", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-private-network"] == "true"
    assert response.headers["access-control-allow-origin"] == "http://tauri.localhost"


def test_profile_from_derives_persona_from_interests():
    assert replay.profile_from({"persona": "analyst"})["persona"] == "analyst"
    profile = replay.profile_from({"persona": "analyst", "interests": "money:90"})
    assert profile["persona"] == "money" and profile["interests"]["money"] == 90
    assert replay.agent_profile(profile)["interests"] == {"speed": "Into it", "passing": "Into it", "pressure": "Into it", "possession": "Into it", "money": "Obsessed"}
