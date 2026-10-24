"""Deploy the orchestrator as a Foundry hosted agent from source (remote build, no container registry).

python agents/orchestrator/deploy.py           # upload a new version
python agents/orchestrator/deploy.py status    # show the latest version
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CodeConfiguration, HostedAgentDefinition, ProtocolVersionRecord
from azure.identity import DefaultAzureCredential

HERE = Path(__file__).resolve().parent
PROJECT_ENDPOINT = "https://matchintel-dl41i.services.ai.azure.com/api/projects/matchintel"
AGENT_NAME = "match-orchestrator"


def _zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("main.py", "requirements.txt"):
            archive.write(HERE / name, name)
    return buffer.getvalue()


def deploy(client: AIProjectClient) -> None:
    code = _zip()
    definition = HostedAgentDefinition(
        cpu="1",
        memory="2Gi",
        protocol_versions=[ProtocolVersionRecord(protocol="responses", version="2.0.0")],
        code_configuration=CodeConfiguration(runtime="python_3_13", entry_point=["python", "main.py"], dependency_resolution="remote_build"),
        environment_variables={"AZURE_AI_MODEL_DEPLOYMENT_NAME": "gpt-5.4", "MATCH_MCP_CONNECTION": "match-mcp"},
    )
    version = client.agents.create_version_from_code(
        agent_name=AGENT_NAME,
        definition=definition,
        code=("orchestrator.zip", code, "application/zip"),
        code_zip_sha256=hashlib.sha256(code).hexdigest(),
        description="Routes fan requests to the moment explainer, player lens, narrator and personalizer with timeouts, retries and fallbacks.",
        metadata={"app": "match-intelligence"},
    )
    print(f"uploaded version {version.version}")
    for _ in range(60):
        details = client.agents.get_version(agent_name=AGENT_NAME, agent_version=version.version)
        status = details.as_dict().get("status")
        print("status:", status)
        if status not in {"creating", "building", "pending", "provisioning", None}:
            break
        time.sleep(15)
    print(json.dumps(details.as_dict(), indent=2, default=str)[:3000])


if __name__ == "__main__":
    project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())
    if sys.argv[1:2] == ["status"]:
        latest = next(iter(project.agents.list_versions(agent_name=AGENT_NAME, order="desc", limit=1)))
        print(json.dumps(latest.as_dict(), indent=2, default=str)[:3000])
    else:
        deploy(project)
