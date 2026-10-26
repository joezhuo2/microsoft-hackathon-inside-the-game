import json
from pathlib import Path

import pandas as pd

from football_analytics import mcp_server as server
from football_analytics import overlay_cues as cues
from football_analytics import valuation as v

ROOT = Path(__file__).resolve().parents[1]
STEPS = 3001


def _setup(tmp_path, monkeypatch):
    folder = tmp_path / "games_0"
    folder.mkdir()
    base = {"GAME_ID": 0, "TEAM_ID": 0, "completed": None, "progression_m": None, "pass_difficulty": None, "confidence_level": "HIGH"}
    pd.DataFrame([
        {**base, "event_id": "0-1", "start_timestep": 10, "event_type": "pass_attempt", "PLAYER_ID": 9, "completed": True, "progression_m": 15.0, "pass_difficulty": 60.0},
        {**base, "event_id": "0-2", "start_timestep": 20, "event_type": "shot_candidate", "PLAYER_ID": 9},
        {**base, "event_id": "0-3", "start_timestep": 30, "event_type": "possession_loss_candidate", "PLAYER_ID": 9},
        {**base, "event_id": "0-4", "start_timestep": 2000, "event_type": "interception_candidate", "PLAYER_ID": 9},
        {**base, "event_id": "0-5", "start_timestep": 50, "event_type": "shot_candidate", "PLAYER_ID": 101, "TEAM_ID": 1},
    ]).to_csv(folder / "events.csv", index=False)
    pd.DataFrame([
        {"moment_id": "M0-0001", "GAME_ID": 0, "start_timestep": 25, "end_timestep": 30, "moment_type": "goal", "TEAM_ID": 0, "PLAYER_ID": 9, "event_ids": "0-2", "metrics": json.dumps({"score_after": "1-0"})},
    ]).to_csv(folder / "key_moments.csv", index=False)
    pd.DataFrame([
        {"event_id": "S0-1", "GAME_ID": 0, "start_timestep": 40, "event_type": "tackle_synthetic", "PLAYER_ID": 9, "TEAM_ID": 0},
    ]).to_csv(folder / "synthetic_events.csv", index=False)
    players = {
        "0": {"name": "Tomas Ellery", "short": "Ellery", "role": "goalkeeper", "weekly_wage_gbp": 45000, "contract_note": "keeper"},
        "9": {"name": "Daniel Moreau", "short": "Moreau", "role": "central midfield", "weekly_wage_gbp": 150000, "contract_note": "record signing"},
        "100": {"name": "Arvid Nystrom", "short": "Nystrom", "role": "goalkeeper", "weekly_wage_gbp": 50000, "contract_note": "keeper"},
    }
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"clubs": {"0": {"name": "Home", "short": "HOM"}, "1": {"name": "Away", "short": "AWY"}}, "players": players}))
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "ROSTER_PATH", roster)
    server._data.cache_clear()
    server._roster.cache_clear()
    v.actions.cache_clear()


def test_player_value_counts_only_actions_so_far(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    card = v.player_value(0, 9, 100, STEPS)
    lines = {line.action: line for line in card.breakdown}
    assert set(lines) == {"pass", "progressive_pass", "hard_pass", "shot", "goal", "loss", "tackle"}
    assert lines["tackle"].synthetic and not lines["goal"].synthetic
    expected = sum(v.ACTIONS[a][2] for a in lines)
    assert card.match_value_gbp == expected
    assert card.wage_so_far_gbp == int(150000 * v.MIN_SHARE)
    assert card.verdict == "bargain" and "record signing" in card.why_paid
    assert v.player_value(0, 9, 2500, STEPS).match_value_gbp == expected + v.ACTIONS["interception"][2]


def test_keeper_credited_for_shot_faced(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert v.player_value(0, 0, 100, STEPS).breakdown == []
    keeper = v.player_value(0, 0, 50 + v.SHOT_TO_GOAL, STEPS)
    assert [line.action for line in keeper.breakdown] == ["shot_faced"]
    assert v.player_value(0, 100, 100, STEPS).breakdown == []


def test_money_cues_follow_money_interest(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    found = v.money_cues(0, {}, STEPS)
    assert not [cue for cue in found if cues.wanted(cue.event_type, {"interests": "money:10"})]
    assert {cues.weighted(cue, {"interests": "money:90"}).priority for cue in found} == {"high"}
    assert {cue.event_type for cue in found} == {"value_goal", "value_shot", "value_progressive_pass", "value_interception"}
    goal = next(cue for cue in found if cue.event_type == "value_goal")
    assert goal.priority == "high" and goal.moment_id == "M0-0001" and goal.audience_mode == "money"
    assert server.validate_claims(0, f"{goal.primary_insight} {goal.why_it_matters}", goal.source_event_ids, [goal.moment_id], [9])["valid"]


def test_value_board_sorted_by_ratio(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    board = v.value_board(0, 3000, STEPS)
    assert [row["value_ratio"] for row in board] == sorted((row["value_ratio"] for row in board), reverse=True)


def test_value_explanations_follow_language(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    en, es = v.player_value(0, 9, 100, STEPS), v.player_value(0, 9, 100, STEPS, "es")
    assert es.match_value_gbp == en.match_value_gbp and es.verdict == en.verdict == "bargain"
    assert es.why_value.startswith(f"{v._gbp(es.match_value_gbp)} de valor frente a {v._gbp(es.wage_so_far_gbp)} de salario hasta ahora")
    assert "1 gol (£150k)" in es.why_value and es.why_value.endswith("Veredicto: una ganga.")
    assert es.why_paid.startswith("Daniel Moreau cobra £150k a la semana (record signing). A los mediocentros")
    goal = next(cue for cue in v.money_cues(0, {"language": "es"}, STEPS) if cue.event_type == "value_goal")
    assert goal.primary_insight == "Daniel Moreau suma valor: gol" and goal.why_it_matters.endswith("Veredicto por ahora: una ganga.")


def test_schema_file_matches_model():
    assert json.loads((ROOT / "schemas" / "player_value.schema.json").read_text(encoding="utf-8")) == v.PlayerValue.model_json_schema()
