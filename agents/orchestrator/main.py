"""Match Orchestrator: hosted Foundry agent that routes fan requests to the specialist agents.

Handoffs run as coded pipelines (explainer -> narrator + personalizer) so they are fast and
recoverable: every specialist call has a timeout, one retry, and a fallback path that is
reported back in `fallbacks`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Annotated, Any

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryAgent, FoundryChatClient
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.identity import DefaultAzureCredential
from pydantic import Field

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5.4")
MCP_URL = os.environ.get("MATCH_MCP_URL", "https://ca-match-mcp.gentlesmoke-46304b37.norwayeast.azurecontainerapps.io/mcp")
MCP_CONNECTION = os.environ.get("MATCH_MCP_CONNECTION", "match-mcp")
SPECIALIST_TIMEOUT_S = float(os.environ.get("SPECIALIST_TIMEOUT_S", "75"))
DEFAULT_PROFILE = {"persona": "casual", "language": "en", "favorite_club_id": None, "favorite_player_id": None, "metric_focus": None, "interests": {}, "depth": None, "tone": None, "tactical": False}

log = logging.getLogger("orchestrator")
credential = DefaultAzureCredential()
specialists = {name: FoundryAgent(project_endpoint=PROJECT_ENDPOINT, agent_name=name, credential=credential) for name in ("match-moment-explainer", "player-lens", "match-narrator", "fan-personalizer")}
explanations: dict[tuple[int, str], dict[str, Any]] = {}

ProfileArg = Annotated[str, Field(description='JSON fan profile: {"persona": "casual"|"analyst", "language": "en"|"es"|..., "favorite_club_id": 0|1|null, "favorite_player_id": int|null, "metric_focus": str|null, "interests": {"speed"|"passing"|"pressure"|"possession"|"money": "Obsessed"|"Into it"|"Light"|"Skip"}, "depth": "headlines"|"balanced"|"deep"|null, "tone": "neutral"|"emotional"|null, "tactical": bool}. Pass every field from the request unchanged. Use "{}" for defaults.')]
AsOfArg = Annotated[int | None, Field(description="Timestep the viewer has reached (as_of from the request). Specialists only see data finished by then. null only for a full-match request.")]


def _profile(profile_json: str) -> dict[str, Any]:
    try:
        given = json.loads(profile_json or "{}")
    except json.JSONDecodeError:
        given = {}
    return {**DEFAULT_PROFILE, **{k: v for k, v in given.items() if k in DEFAULT_PROFILE}}


async def _call(name: str, payload: dict[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    for attempt in (1, 2):
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(specialists[name].run(json.dumps(payload)), SPECIALIST_TIMEOUT_S)
            result = json.loads(response.text)
            trace.append({"agent": name, "attempt": attempt, "seconds": round(time.perf_counter() - started, 1), "valid": result.get("validation", {}).get("valid")})
            if result.get("validation", {}).get("valid", True):
                return result
        except Exception as error:  # timeouts, transport errors and malformed JSON all get the same retry
            trace.append({"agent": name, "attempt": attempt, "seconds": round(time.perf_counter() - started, 1), "error": type(error).__name__})
            log.warning("specialist %s attempt %s failed: %s", name, attempt, error)
    return None


def _envelope(kind: str, trace: list[dict[str, Any]], fallbacks: list[str], **parts: Any) -> str:
    return json.dumps({"kind": kind, **parts, "fallbacks": fallbacks, "handoffs": trace}, ensure_ascii=False)


@tool
async def live_moment(game_id: int, moment_id: str, profile_json: ProfileArg = "{}", as_of: AsOfArg = None) -> str:
    """Explain a key moment, write the broadcast line and overlay, and personalize it for the fan."""
    trace: list[dict[str, Any]] = []
    fallbacks: list[str] = []
    profile = _profile(profile_json)
    explanation = explanations.get((game_id, moment_id, as_of)) or await _call("match-moment-explainer", {"game_id": game_id, "moment_id": moment_id, "as_of": as_of}, trace)
    if explanation:
        explanations[(game_id, moment_id, as_of)] = explanation
    else:
        fallbacks.append("explainer_failed: narrator worked from raw moment data")
    narration, personalized = await asyncio.gather(
        _call("match-narrator", {"mode": "live_line", "game_id": game_id, "moment_id": moment_id, "as_of": as_of, "explanation": explanation}, trace),
        _call("fan-personalizer", {"content": explanation, "profile": profile, "as_of": as_of}, trace) if explanation else asyncio.sleep(0, None),
    )
    if narration is None:
        fallbacks.append("narrator_failed: no overlay")
    if personalized is None:
        fallbacks.append("personalizer_unavailable: reply uses unpersonalized English")
    return _envelope("live_moment", trace, fallbacks, game_id=game_id, moment_id=moment_id, profile=profile, explanation=explanation, narration=narration, personalized=personalized)


@tool
async def player_focus(game_id: int, player: Annotated[str, Field(description="Roster name or player id")], profile_json: ProfileArg = "{}", as_of: AsOfArg = None) -> str:
    """Player-focus mode: profile one player's contribution and personalize it for the fan."""
    trace: list[dict[str, Any]] = []
    fallbacks: list[str] = []
    profile = _profile(profile_json)
    lens = await _call("player-lens", {"game_id": game_id, "player": player, "as_of": as_of}, trace)
    personalized = await _call("fan-personalizer", {"content": lens, "profile": profile, "as_of": as_of}, trace) if lens else None
    if lens is None:
        fallbacks.append("player_lens_failed")
    elif personalized is None:
        fallbacks.append("personalizer_unavailable: reply uses unpersonalized English")
    return _envelope("player_focus", trace, fallbacks, game_id=game_id, profile=profile, player_profile=lens, personalized=personalized)


@tool
async def match_recap(game_id: int, profile_json: ProfileArg = "{}", as_of: AsOfArg = None) -> str:
    """Automated recap of the whole match up to as_of (full time at the end) with overlays, personalized for the fan."""
    trace: list[dict[str, Any]] = []
    fallbacks: list[str] = []
    profile = _profile(profile_json)
    recap = await _call("match-narrator", {"mode": "recap", "game_id": game_id, "as_of": as_of}, trace)
    personalized = await _call("fan-personalizer", {"content": recap, "profile": profile, "as_of": as_of}, trace) if recap else None
    if recap is None:
        fallbacks.append("narrator_failed")
    elif personalized is None:
        fallbacks.append("personalizer_unavailable: reply uses unpersonalized English")
    return _envelope("match_recap", trace, fallbacks, game_id=game_id, profile=profile, recap=recap, personalized=personalized)


INSTRUCTIONS = """
You are the Match Orchestrator for a personalized football match-intelligence experience built on synthetic tracking data.
You coordinate specialists; you never compute statistics or invent facts yourself.

Routing:
- A key moment (moment_id given, or "what just happened") -> live_moment.
- A question about one player, or player-focus mode -> player_focus.
- A recap, summary or "how did the match go" -> match_recap.
- To find game ids, moment ids or player names, use the match_intel tools (list_matches, get_key_moments, get_match_state, get_roster) first.
Pass the fan profile from the request as profile_json unchanged (keep interests, depth, tone and tactical); if none is given use "{}".
Known-at-this-time rule: when the request gives as_of (or a timestep), pass it as as_of to live_moment, player_focus, match_recap, list_matches and get_key_moments, and as timestep to get_match_state. Never mention anything after it.
Club ids: 0 = Kestrel Vale FC, 1 = Port Saltmere United. Resolve a favourite player's name to its id with get_roster; never guess ids.

Reply:
- Answer in the fan's language with the personalized text when present, otherwise the specialist's English text.
- Keep every number, name, score and match clock exactly as the tools returned them.
- If fallbacks is non-empty, still answer from whatever succeeded and add one short note that part of the analysis is unavailable.
- End with one line "Evidence: <moment and event ids>".
""".strip()


def build_agent() -> Agent:
    client = FoundryChatClient(project_endpoint=PROJECT_ENDPOINT, model=MODEL, credential=credential)
    match_intel = client.get_mcp_tool(name="match_intel", url=MCP_URL, project_connection_id=MCP_CONNECTION, approval_mode="never_require", allowed_tools=["list_matches", "get_key_moments", "get_match_state", "get_roster"])
    return Agent(client=client, name="match-orchestrator", instructions=INSTRUCTIONS, tools=[live_moment, player_focus, match_recap, match_intel], default_options={"store": False})


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ResponsesHostServer(build_agent()).run()


if __name__ == "__main__":
    main()
