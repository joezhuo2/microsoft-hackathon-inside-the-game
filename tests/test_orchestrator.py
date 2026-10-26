import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "agents" / "orchestrator" / "main.py"


def _load(monkeypatch):
    stub = lambda *args, **kwargs: types.SimpleNamespace(run=None)
    modules = {
        "agent_framework": {"Agent": stub, "tool": lambda fn: fn},
        "agent_framework.foundry": {"FoundryAgent": stub, "FoundryChatClient": stub},
        "agent_framework_foundry_hosting": {"ResponsesHostServer": stub},
        "azure.identity": {"DefaultAzureCredential": stub},
    }
    for name, attrs in modules.items():
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://example.invalid")
    spec = importlib.util.spec_from_file_location("orchestrator_main", MAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_moment_forwards_interests_to_personalizer(monkeypatch):
    main = _load(monkeypatch)
    sent = {}

    def agent(name):
        async def run(payload):
            sent[name] = json.loads(payload)
            return types.SimpleNamespace(text=json.dumps({"validation": {"valid": True}}))
        return types.SimpleNamespace(run=run)

    monkeypatch.setattr(main, "specialists", {name: agent(name) for name in main.specialists})
    given = {"persona": "analyst", "language": "es", "interests": {"pressure": "Obsessed", "speed": "Skip"}, "depth": "deep", "tone": "emotional", "tactical": True}
    envelope = json.loads(asyncio.run(main.live_moment(0, "M0-0001", json.dumps({**given, "junk": 1}))))
    forwarded = sent["fan-personalizer"]["profile"]
    assert {k: forwarded[k] for k in given} == given
    assert "junk" not in forwarded and envelope["profile"] == forwarded
    assert main._profile("{}")["interests"] == {}
