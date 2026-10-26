import json

import pandas as pd

from football_analytics import mcp_server as server


def _setup(tmp_path, monkeypatch):
    folder = tmp_path / "games_0"
    folder.mkdir()
    pd.DataFrame([{"event_id": "0-000001", "GAME_ID": 0, "start_timestep": 10, "event_type": "shot_candidate", "PLAYER_ID": 9, "TEAM_ID": 0, "ball_speed_kmh": 120.5}]).to_csv(folder / "events.csv", index=False)
    pd.DataFrame([{"moment_id": "M0-0001", "GAME_ID": 0, "start_timestep": 5, "end_timestep": 12, "match_clock": "00:22", "moment_type": "goal", "TEAM_ID": 0, "PLAYER_ID": 9, "importance": 1.0, "event_ids": "0-000001", "metrics": json.dumps({"score_after": "1-0", "shot_speed_kmh": 120.5}), "fact_hint": "Goal"}]).to_csv(folder / "key_moments.csv", index=False)
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"clubs": {"0": {"name": "Home"}, "1": {"name": "Away"}}, "players": {"9": {"name": "Daniel Moreau", "short": "Moreau", "number": 10}, "2": {"name": "Kwame Asante-Boyd", "short": "Asante-Boyd", "number": 9}}}))
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "ROSTER_PATH", roster)
    server._data.cache_clear()
    server._roster.cache_clear()


def test_validate_claims_accepts_cited_facts_and_rejects_others(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    good = server.validate_claims(0, "Moreau scores at 00:22 with a 120.5 km/h shot, 1-0", moment_ids=["M0-0001"])
    assert good["valid"], good
    bad = server.validate_claims(0, "Asante-Boyd scores a 150 km/h shot", moment_ids=["M0-0001", "M0-9999"])
    assert not bad["valid"]
    assert bad["unknown_ids"] == ["M0-9999"]
    assert bad["unsupported_numbers"] == ["150"]
    assert bad["uncited_players"] == ["Kwame Asante-Boyd"]
    assert not server.validate_claims(0, "A great goal")["valid"]


def test_validate_overlay_checks_schema_and_numbers(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    overlay = {"game_id": 0, "moment_id": "M0-0001", "start_timestep": 12, "display_seconds": 6, "template": "lower_third", "language": "es", "headline": "¡GOL! 1-0", "stats": [{"label": "Velocidad", "value": 120.5, "unit": "km/h"}]}
    assert server.validate_overlay(overlay)["valid"]
    assert not server.validate_overlay({**overlay, "template": "banner"})["valid"]
    assert not server.validate_overlay({**overlay, "stats": [{"label": "Velocidad", "value": 99, "unit": "km/h"}]})["valid"]
    assert server.validate_overlay({**overlay, "start_timestep": 5})["future_ids"] == ["0-000001", "M0-0001"]


def _extra(tmp_path, events, rhythm=None):
    folder = tmp_path / "games_0"
    pd.concat([pd.read_csv(folder / "events.csv"), pd.DataFrame(events)]).to_csv(folder / "events.csv", index=False)
    if rhythm:
        pd.DataFrame(rhythm).to_csv(folder / "match_rhythm.csv", index=False)
    server._data.cache_clear()


def test_as_of_hides_data_not_finished_yet(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _extra(tmp_path, [{"event_id": "0-000002", "GAME_ID": 0, "start_timestep": 11, "end_timestep": 40, "event_type": "pass_attempt", "PLAYER_ID": 9, "TEAM_ID": 0}], [{"GAME_ID": 0, "window": 0, "start_timestep": 0, "end_timestep": 149, "possession_share_team0": 0.6}])
    assert server.get_key_moments(0, as_of=11) == [] and len(server.get_key_moments(0, as_of=12)) == 1
    assert [e["event_id"] for e in server.get_moment_context(0, "M0-0001")["events"]] == ["0-000001"]
    assert "error" in server.get_moment_context(0, "M0-0001", as_of=11)
    assert server.get_match_rhythm(0, as_of=148) == [] and server.get_match_state(0, 148)["current_window"] is None
    assert server.validate_claims(0, "A shot", ["0-000002"], as_of=20)["future_ids"] == ["0-000002"]


def test_gate_rejects_invented_events_and_numbers(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _extra(tmp_path, [
        {"event_id": "0-000002", "GAME_ID": 0, "start_timestep": 30, "event_type": "shot_candidate", "PLAYER_ID": 9, "TEAM_ID": 0, "ball_speed_kmh": 99.0},
        {"event_id": "0-000003", "GAME_ID": 0, "start_timestep": 50, "event_type": "restart_candidate", "subtype": "penalty_candidate"},
    ])
    penalty = server.validate_claims(0, "The referee awarded a penalty.", moment_ids=["M0-0001"])
    assert not penalty["valid"] and penalty["unsupported_events"] == ["penalty", "referee"]
    assert server.validate_claims(0, "A penalty is about to be taken.", ["0-000003"])["valid"]
    assert server.validate_claims(0, "Win probability is now 0.9999.", moment_ids=["M0-0001"])["unsupported_numbers"] == ["0.9999"]
    assert not server.validate_claims(0, "Home have 57% of the ball.", moment_ids=["M0-0001"])["valid"]
    assert server.validate_claims(0, "It finished 22-0, the goal came at 00:23.", moment_ids=["M0-0001"])["unsupported_numbers"] == ["00:23", "22"]
    assert server.validate_claims(0, "Moreau shot at 99 km/h.", ["0-000001"])["unsupported_numbers"] == ["99"]
    assert server.validate_claims(0, "Moreau shot at 99 km/h.", ["0-000002"])["valid"]
