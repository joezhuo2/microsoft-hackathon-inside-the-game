import asyncio
import json
import re
import time

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
    assert result["overlay"]["stats"] == [{"label": "shot speed", "value": 120, "unit": "km/h"}]


def test_enriched_overlay_falls_back_to_gated_english(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    base = {"game_id": 0, "moment_id": "M0-0001", "start_timestep": 5, "display_seconds": 6, "template": "lower_third", "language": "en", "headline": "Moreau scores, 1-0", "body": "Shot at about 120.5 km/h", "player_id": 9, "stats": []}
    moment = replay.moments(0)[0]
    good = replay.enriched_overlay({"narration": {"overlays": [base]}, "personalized": {"overlay_headline": "Moreau marque, 1-0", "overlay_body": "Tir à environ 120.5 km/h"}, "profile": {"language": "fr"}}, moment)
    assert good["overlay"]["language"] == "fr" and good["validation"]["valid"]
    bad = replay.enriched_overlay({"narration": {"overlays": [base]}, "personalized": {"overlay_headline": "Moreau marque à 150 km/h"}, "profile": {"language": "fr"}}, moment)
    assert bad["overlay"] == base
    assert replay.enriched_overlay({}, moment)["overlay"] is None
    assert replay.enriched_overlay({"narration": {"overlays": [{**base, "moment_id": "M0-0002"}]}}, moment)["overlay"] is None


def test_enriched_text_is_the_validated_string(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    moment = replay.moments(0)[0]
    overlay = replay.instant_overlay(moment)["overlay"]
    invented = ("A 99 km/h penalty sealed it.", "en")
    assert replay.verified_text(0, moment, overlay, [invented, ("Moreau marque, environ 120.5 km/h", "fr")]) == ("Moreau marque, environ 120.5 km/h", "fr")
    assert replay.verified_text(0, moment, overlay, [invented]) == (overlay["body"], "en")


def test_ask_returns_only_the_validated_answer(tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    _setup(tmp_path, monkeypatch)
    prompts = []
    monkeypatch.setattr(replay, "ask_orchestrator", lambda prompt: prompts.append(prompt) or {"text": "Moreau scored at about 120.5 km/h, 1-0.\nEvidence: M0-0001", "envelopes": []})
    client = TestClient(replay.app)
    data = client.post("/api/ask", json={"game": 0, "timestep": 20, "question": "What happened?"}).json()
    assert (data["answer"], data["evidence"], data["verified"]) == ("Moreau scored at about 120.5 km/h, 1-0.", ["M0-0001"], True)
    assert not client.post("/api/ask", json={"game": 0, "timestep": 11, "question": "What happened?"}).json()["verified"]
    client.post("/api/ask", json={"game": 0, "timestep": 3000, "question": "Give me the match recap.", "action": "recap"})
    assert "call match_recap with game_id=0 and as_of=3000" in prompts[-1] and "as_of=20" in prompts[0]


def test_stream_publishes_only_validated_cues(tmp_path, monkeypatch):
    import numpy as np
    from starlette.testclient import TestClient

    from football_analytics import overlay_cues as cues

    _setup(tmp_path, monkeypatch)
    steps = 30
    monkeypatch.setattr(replay, "tracks", lambda: {0: {"ids": [-1], "xy": np.zeros((steps, 1, 2)), "score": np.zeros((steps, 2), dtype=int), "steps": steps}})
    monkeypatch.setattr(replay, "running_stats", lambda game_id: {"possession": np.full((steps, 2), 50.0)})
    base = {"game_id": 0, "timestep": 20, "match_clock": "00:36", "event_type": "interception", "why_it_matters": "Cut out early.", "confidence": 0.7, "evidence": "observed", "priority": "high", "duration_ms": 4000, "theme": "defensive_masterclass", "source_event_ids": ["0-000001"]}
    derived = [cues.OverlayCue(event_id="good", primary_insight="Moreau: interception", **base), cues.OverlayCue(event_id="invented", primary_insight="Moreau: penalty won", **base), cues.OverlayCue(event_id="uncited", primary_insight="Full time", **{**base, "source_event_ids": []})]
    monkeypatch.setattr(cues, "derived_cues", lambda *args: derived)
    monkeypatch.setattr(replay.valuation, "money_cues", lambda *args: [])
    body = TestClient(replay.app).get("/api/stream?game=0&speed=120&enrich=0").text
    events = [(block.split("\n")[0][7:], json.loads(block.split("\n")[1][6:])) for block in body.strip().split("\n\n")]
    assert [data["cue"]["event_id"] for kind, data in events if kind == "cue"] == ["good"]
    assert [(data["cue"]["timestep"], data["validation"]["valid"]) for kind, data in events if kind == "moment"] == [(12, True)]


def test_feedback_tone_reaches_live_enrichment(tmp_path, monkeypatch):
    import asyncio

    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(replay, "CACHE_DIR", tmp_path / "cache")
    prompts = []
    monkeypatch.setattr(replay, "ask_orchestrator", lambda prompt: prompts.append(prompt) or {"text": "", "envelopes": []})
    asyncio.run(replay.enrich(0, replay.moments(0)[0], replay.profile_from({"tone": "emotional", "tactical": "1", "depth": "deep"})))
    assert "Tone: emotional" in prompts[0] and "Lead with tactics." in prompts[0] and '"depth": "deep"' in prompts[0]
    bad = replay.profile_from({"tone": "shouty", "depth": "x"})
    assert (bad["tone"], bad["depth"], bad["tactical"]) == ("neutral", "balanced", False)


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


def test_enrich_coalesces_identical_in_flight_calls(tmp_path, monkeypatch):
    calls = []

    def fake(prompt):
        calls.append(prompt)
        time.sleep(0.05)
        return {"text": "ok", "envelopes": []}

    monkeypatch.setattr(replay, "ask_orchestrator", fake)
    monkeypatch.setattr(replay, "CACHE_DIR", tmp_path)
    moment, profile = {"moment_id": "M0-0001", "start_timestep": 5, "known_timestep": 5}, replay.profile_from({})

    async def both():
        return await asyncio.gather(replay.enrich(0, moment, profile), replay.enrich(0, moment, profile))

    first, second = asyncio.run(both())
    assert len(calls) == 1 and first == second and not replay.inflight


def test_single_flight_cancels_work_when_last_waiter_leaves():
    async def scenario():
        started = asyncio.Event()

        async def work():
            started.set()
            await asyncio.sleep(10)

        waiter = asyncio.create_task(replay.single_flight("k", work))
        await started.wait()
        flight = replay.inflight["k"][0]
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await asyncio.sleep(0)
        return flight

    assert asyncio.run(scenario()).cancelled() and "k" not in replay.inflight


def test_stream_close_cancels_agent_tasks():
    async def scenario():
        task = asyncio.create_task(asyncio.sleep(10))

        async def chunks():
            yield "a"
            yield "b"

        stream = replay.cancel_on_close(chunks(), {task})
        assert await stream.__anext__() == "a"
        await stream.aclose()
        await asyncio.sleep(0)
        return task

    assert asyncio.run(scenario()).cancelled()


def test_stream_shows_loading_then_rate_limit_and_keeps_cues(monkeypatch):
    from starlette.testclient import TestClient

    async def over_budget(*_):
        raise replay.BudgetExceeded("hourly live agent budget reached")

    moment = {"moment_id": "M0-0001", "moment_type": "goal", "start_timestep": 0, "known_timestep": 0, "importance": 1.0}
    monkeypatch.setattr(replay, "ENRICH", True)
    monkeypatch.setattr(replay, "TICK_S", 0.01)
    monkeypatch.setattr(replay, "replay_plan", lambda *_: (3, [moment], []))
    monkeypatch.setattr(replay, "frame_at", lambda game_id, t: {"t": t})
    monkeypatch.setattr(replay, "instant_overlay", lambda m, profile: {"overlay": {"moment_id": m["moment_id"]}, "validation": {"valid": True}, "cue_validation": {"valid": True}})
    monkeypatch.setattr(replay, "enrich", over_budget)
    with TestClient(replay.app).stream("GET", "/api/stream?game=0&speed=120") as response:
        body = "".join(response.iter_text())
    names = re.findall(r"^event: (\w+)", body, re.M)
    assert names[0] == "status" and '"state": "loading"' in body.split("\n\n")[0]
    assert "moment" in names and '"state": "rate_limited"' in body and names[-1] == "done"


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
