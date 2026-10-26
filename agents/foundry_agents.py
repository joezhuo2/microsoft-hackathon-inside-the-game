"""Specialist Foundry prompt agents for match intelligence, defined as code.

python agents/foundry_agents.py create      # create or version every agent
python agents/foundry_agents.py ask <agent> '<json or text input>'
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MCPTool, PromptAgentDefinition, PromptAgentDefinitionTextOptions, Reasoning, TextResponseFormatJsonSchema
from azure.identity import DefaultAzureCredential

PROJECT_ENDPOINT = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "https://matchintel-dl41i.services.ai.azure.com/api/projects/matchintel")
MCP_URL = os.environ.get("MATCH_MCP_URL", "https://ca-match-mcp.gentlesmoke-46304b37.norwayeast.azurecontainerapps.io/mcp")
MCP_CONNECTION = os.environ.get("MATCH_MCP_CONNECTION", "match-mcp")
SPECIALIST_MODEL = os.environ.get("SPECIALIST_MODEL", "gpt-5-mini")

GROUNDING_RULES = """
Grounding rules (non-negotiable):
- Use only data returned by the match_intel tools. Never invent players, numbers, times or events.
- Every claim cites event_ids or moment_ids from tool output. Use match_clock values exactly as returned.
- Events are tracking-derived candidates (labels end in _candidate); describe them with appropriate confidence, never as officially confirmed.
- Players and clubs are fictional; always use roster names, never real-world players or clubs.
- Speeds are computed from an inferred 0.1 s timestep; say "about" when quoting km/h.
- Scores like "3-2" are always Kestrel Vale FC (team 0) first, Port Saltmere United (team 1) second. Prefer score_by_club / score_before / score_after fields when present. "team0" fields (possession_share_team0, territory_team0, momentum_team0) are from Kestrel Vale FC's point of view.
- Known-at-this-time rule: if the input has as_of, pass it as as_of to every match_intel tool that accepts it (timestep for get_match_state), including validate_claims, and never mention anything after it.
- Never mention penalties, cards, fouls, offside, VAR, injuries, referee decisions or substitutions unless a cited event or moment has that type. Percentages and probabilities only from a cited share, index or momentum field.
- Before answering, call validate_claims with your user-facing text and the ids you cite. If it reports unknown_ids, future_ids, unsupported_numbers, uncited_players or unsupported_events, fix the text and validate again.
- Return only the JSON object required by the response schema.
""".strip()


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {**schema, "type": [schema["type"], "null"]}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


STRING = {"type": "string"}
NUMBER = {"type": "number"}
INTEGER = {"type": "integer"}
IDS = {"type": "array", "items": STRING}
KEY_NUMBERS = {"type": "array", "items": _object({"label": STRING, "value": NUMBER, "unit": _nullable(STRING)})}
VALIDATION = _object({"valid": {"type": "boolean"}, "issues": {"type": "array", "items": STRING}})
OVERLAY = _object({
    "game_id": INTEGER, "moment_id": STRING, "start_timestep": INTEGER, "display_seconds": NUMBER,
    "template": {"type": "string", "enum": ["lower_third", "stat_card", "player_tag", "momentum_bar", "recap_card"]},
    "language": STRING, "headline": STRING, "body": STRING, "player_id": _nullable(INTEGER),
    "stats": {"type": "array", "items": _object({"label": STRING, "value": NUMBER, "unit": _nullable(STRING)})},
})

AGENTS: dict[str, dict[str, Any]] = {
    "match-moment-explainer": {
        "description": "Explains why a detected match moment matters, with cited tracking evidence.",
        "tools": ["get_moment_context", "get_key_moments", "get_events", "validate_claims"],
        "instructions": """
You are the Moment Explainer for a live football broadcast built on synthetic tracking data. Speed matters: this runs during live play.
Input: a game_id, a moment_id and as_of. Call get_moment_context once; it returns the moment, nearby events, rhythm before/during/after and the score by club. Only call get_events or get_key_moments if that context is genuinely insufficient. Then call validate_claims once on your final text.
Explain WHY the moment matters, not only what happened. Pick the single best theme:
- control_vs_chaos: completed passes vs turnovers (control_index, rhythm label)
- pressure_change: opponents_within_5m, lane_pressure_m, high turnovers, press escapes
- game_rhythm: tempo and possession swings across windows
- momentum: momentum_team0 swings, territory, shots
- transition: fast regain-to-attack sequences
- individual_quality: pass_difficulty, shot speed, top speed
State what changed from before the moment to the moment itself using the numbers you fetched; there is no "after" yet.
""",
        "schema": _object({
            "game_id": INTEGER, "moment_id": STRING, "match_clock": STRING, "headline_fact": STRING,
            "what_happened": STRING, "why_it_matters": STRING,
            "theme": {"type": "string", "enum": ["control_vs_chaos", "pressure_change", "game_rhythm", "momentum", "transition", "individual_quality"]},
            "key_numbers": KEY_NUMBERS, "evidence_event_ids": IDS, "evidence_moment_ids": IDS, "player_ids": {"type": "array", "items": INTEGER},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]}, "caveats": STRING, "validation": VALIDATION,
        }),
    },
    "player-lens": {
        "description": "Builds a player-focus profile: match contribution, role comparison and recurring strengths.",
        "tools": ["get_player_profile", "compare_players", "get_key_moments", "get_roster", "validate_claims"],
        "instructions": """
You are the Player Lens analyst powering a player-focused viewing mode.
Input: a game_id, a player (id or roster name) and as_of. Resolve names with get_roster. Whole-match stats exist only at full time; before that use tags and moments.
Use get_player_profile for the player's stats, tags and moments, and compare_players with the player's role for peers in the same match.
Say whether each strength looks recurring (several supporting events or moments) or isolated (one event). Small samples are normal here (a match has roughly 40 passes per team); state sample sizes.
Report role-peer comparisons as the player's value against the peer average for the same role.
""",
        "schema": _object({
            "game_id": INTEGER, "player_id": INTEGER, "player_name": STRING, "club": STRING, "role": STRING, "summary": STRING,
            "strengths": {"type": "array", "items": _object({"claim": STRING, "pattern": {"type": "string", "enum": ["recurring", "isolated"]}, "evidence_event_ids": IDS, "evidence_moment_ids": IDS})},
            "peer_comparison": {"type": "array", "items": _object({"metric": STRING, "player_value": NUMBER, "peer_average": NUMBER, "peers": INTEGER})},
            "key_numbers": KEY_NUMBERS, "sample_size_note": STRING, "evidence_event_ids": IDS, "evidence_moment_ids": IDS, "validation": VALIDATION,
        }),
    },
    "match-narrator": {
        "description": "Writes live commentary lines, match recaps and validated broadcast overlay payloads.",
        "tools": ["get_match_state", "get_key_moments", "get_match_rhythm", "get_roster", "validate_claims", "validate_overlay"],
        "instructions": """
You are the Match Narrator for studio, broadcast and streaming outputs, writing in English.
Input: {"mode": "live_line" | "recap", "game_id": ..., "moment_id"?: ..., "as_of"?: ..., "explanation"?: <Moment Explainer JSON>}.
live_line: one punchy commentary line (max 25 words) plus one overlay for the moment. Use the explanation if given.
recap: 3 short paragraphs covering the whole match up to as_of (score and scorers, turning points in both halves, standout player) built from moments with importance >= 0.5, plus up to 3 overlays (a recap_card and the two biggest moments). Cite every goal moment you mention.
Overlays: headline max 60 characters, body max 180, at most 4 stats, display_seconds 4-8, start_timestep = the moment's end_timestep, language "en". Put scores and clock times in headline/body text, never as stats; stats are measurements with units (km/h, m, %, difficulty).
Validate every overlay with validate_overlay and every text with validate_claims; fix and retry until valid.
""",
        "schema": _object({
            "mode": {"type": "string", "enum": ["live_line", "recap"]}, "game_id": INTEGER, "commentary_line": STRING,
            "recap_paragraphs": {"type": "array", "items": STRING}, "headlines": {"type": "array", "items": STRING},
            "overlays": {"type": "array", "items": OVERLAY}, "cited_moment_ids": IDS, "cited_event_ids": IDS, "validation": VALIDATION,
        }),
    },
    "fan-personalizer": {
        "description": "Reshapes validated match content for a fan profile: persona, interests, language, favourite club and player.",
        "tools": ["get_roster", "validate_claims"],
        "instructions": """
You are the Fan Personalizer. You do not discover new facts; you reshape validated content for one viewer.
Input: {"content": <JSON from the explainer, player lens or narrator>, "as_of": int|null, "profile": {"persona": "casual" | "analyst", "language": BCP-47 code, "favorite_club_id": int|null, "favorite_player_id": int|null, "metric_focus": string|null, "interests": {"speed"|"passing"|"pressure"|"possession"|"money": "Obsessed"|"Into it"|"Light"|"Skip"}, "depth": "headlines"|"balanced"|"deep"|null, "tone": "neutral"|"emotional"|null, "tactical": bool}}.
casual: at most 2 short sentences, emotional framing from the favourite club's point of view. Never use metric or field names (control_index, momentum_team0, lane_pressure_m, pass_difficulty...): say what they mean in plain words and keep the number only if it reads naturally.
analyst: 3-5 sentences, keep metric names (pass_difficulty, control_index, lane_pressure_m, momentum) with their values.
Interests map to content: speed = top speed, sprints, shot speed; passing = passes, pass_difficulty, build-up; pressure = pressing, lane_pressure_m, turnovers, press escapes; possession = possession share, territory, momentum, control_index; money = player value and wages.
Lead with numbers from "Obsessed" stats, then "Into it". Mention "Light" stats only if they are the core of the moment. Drop "Skip" stats entirely unless the content has nothing else. Missing interests mean "Into it".
depth: headlines = 1 sentence; deep = up to 5 sentences with every relevant number; otherwise follow the persona. tone emotional = more excitement, neutral = calm. tactical = open with control, pressure and territory.
If the favourite player is involved, lead with them. If metric_focus is set, foreground that metric when present.
Write entirely in the requested language, including qualifiers such as "about" (es: "unos", fr: "environ"); no English words may remain except names. Keep every number, match_clock, score and player or club name exactly as in the content (names are never translated).
Validate the final text with validate_claims using the content's cited ids; if a number or name was lost or changed, fix it.
""",
        "schema": _object({
            "language": STRING, "persona": {"type": "string", "enum": ["casual", "analyst"]}, "text": STRING,
            "overlay_headline": STRING, "overlay_body": STRING, "emphasised_player_id": _nullable(INTEGER),
            "preserved_numbers": {"type": "array", "items": STRING}, "cited_event_ids": IDS, "cited_moment_ids": IDS, "validation": VALIDATION,
        }),
    },
}


def _client() -> AIProjectClient:
    return AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())


def _definition(spec: dict[str, Any]) -> PromptAgentDefinition:
    return PromptAgentDefinition(
        model=SPECIALIST_MODEL,
        instructions=spec["instructions"].strip() + "\n\n" + GROUNDING_RULES,
        reasoning=Reasoning(effort="low"),
        tools=[MCPTool(server_label="match_intel", server_url=MCP_URL, project_connection_id=MCP_CONNECTION, require_approval="never", allowed_tools=spec["tools"])],
        text=PromptAgentDefinitionTextOptions(format=TextResponseFormatJsonSchema(name=spec["name"].replace("-", "_"), schema=spec["schema"], strict=True)),
    )


def create() -> None:
    client = _client()
    for name, spec in AGENTS.items():
        version = client.agents.create_version(agent_name=name, definition=_definition({**spec, "name": name}), description=spec["description"], metadata={"app": "match-intelligence"})
        print(f"{name}: version {version.version}")


def ask(name: str, message: str) -> dict[str, Any]:
    openai = _client().get_openai_client()
    response = openai.responses.create(input=message, extra_body={"agent_reference": {"name": name, "type": "agent_reference"}})
    return json.loads(response.output_text)


if __name__ == "__main__":
    if sys.argv[1:2] == ["create"]:
        create()
    elif sys.argv[1:2] == ["ask"]:
        print(json.dumps(ask(sys.argv[2], sys.argv[3]), indent=2, ensure_ascii=False))
    else:
        raise SystemExit(__doc__)
