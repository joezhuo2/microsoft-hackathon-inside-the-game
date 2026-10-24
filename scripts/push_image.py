"""Build and push the MCP server or replay server image without Docker or ACR Tasks.

Azure for Students subscriptions block ACR Tasks, so this appends one layer (Linux
wheels from build/image/site, the package source and the served tables) onto a base
image already imported into ACR, then pushes it through the registry HTTP API.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
TABLES = ["events", "key_moments", "match_rhythm", "player_tags", "player_match_intel"]
INDEX_TYPES = {"application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json"}
APPS = {
    "mcp": {"site": "site", "module": "football_analytics.mcp_server"},
    "replay": {"site": "site-replay", "module": "football_analytics.replay_server"},
}
ACCEPT = ", ".join(sorted(INDEX_TYPES | {"application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"}))


def _token(login_server: str, registry: str, repository: str, base_repository: str) -> str:
    az = shutil.which("az") or shutil.which("az.cmd")
    refresh = subprocess.run([az, "acr", "login", "-n", registry, "--expose-token", "--query", "accessToken", "-o", "tsv"], capture_output=True, text=True, check=True).stdout.strip()
    response = httpx.post(f"https://{login_server}/oauth2/token", data={
        "grant_type": "refresh_token", "service": login_server, "refresh_token": refresh,
        "scope": [f"repository:{repository}:pull,push", f"repository:{base_repository}:pull"],
    })
    response.raise_for_status()
    return response.json()["access_token"]


def _layer(work: Path, app: str) -> tuple[Path, str, str, int]:
    def clean(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if "__pycache__" in info.name or info.name.endswith(".pyc"):
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        info.mtime = 0
        info.mode = 0o755 if info.isdir() else 0o644
        if "enrich_cache" in info.name:
            info.mode = 0o777 if info.isdir() else 0o666
        return info

    raw = work / "layer.tar"
    with tarfile.open(raw, "w", format=tarfile.PAX_FORMAT) as tar:
        tar.add(ROOT / "build" / "image" / APPS[app]["site"], "app/site", filter=clean)
        tar.add(ROOT / "src" / "football_analytics", "app/src/football_analytics", filter=clean)
        tar.add(ROOT / "data" / "roster.json", "app/data/roster.json", filter=clean)
        for folder in sorted((ROOT / "output").glob("games_*")):
            for table in TABLES:
                tar.add(folder / f"{table}.csv", f"app/output/{folder.name}/{table}.csv", filter=clean)
        if app == "replay":
            for tracking in sorted((ROOT / "data").glob("gfootball_results_games_*.csv")):
                tar.add(tracking, f"app/data/{tracking.name}", filter=clean)
            tar.add(ROOT / "overlay" / "ui", "app/overlay/ui", filter=clean)
            cache = ROOT / "build" / "enrich_cache"
            cache.mkdir(parents=True, exist_ok=True)
            tar.add(cache, "app/build/enrich_cache", filter=clean)
    diff_id = "sha256:" + hashlib.sha256(raw.read_bytes()).hexdigest()
    compressed = work / "layer.tar.gz"
    with raw.open("rb") as source, gzip.GzipFile(compressed, "wb", mtime=0) as target:
        shutil.copyfileobj(source, target)
    data = compressed.read_bytes()
    return compressed, diff_id, "sha256:" + hashlib.sha256(data).hexdigest(), len(data)


def _upload(client: httpx.Client, base: str, repository: str, digest: str, body: bytes) -> None:
    if client.head(f"{base}/v2/{repository}/blobs/{digest}").status_code == 200:
        return
    start = client.post(f"{base}/v2/{repository}/blobs/uploads/")
    start.raise_for_status()
    location = start.headers["Location"]
    location = location if location.startswith("http") else base + location
    separator = "&" if "?" in location else "?"
    client.put(f"{location}{separator}digest={digest}", content=body, headers={"Content-Type": "application/octet-stream"}).raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--app", choices=sorted(APPS), default="mcp")
    parser.add_argument("--repository", default="match-mcp")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--base", default="python:3.13-slim")
    args = parser.parse_args()
    login_server = f"{args.registry}.azurecr.io"
    base_repository, base_tag = args.base.split(":")
    base = f"https://{login_server}"
    client = httpx.Client(headers={"Authorization": f"Bearer {_token(login_server, args.registry, args.repository, base_repository)}"}, follow_redirects=True, timeout=600)

    manifest = client.get(f"{base}/v2/{base_repository}/manifests/{base_tag}", headers={"Accept": ACCEPT})
    manifest.raise_for_status()
    if manifest.headers["Content-Type"] in INDEX_TYPES:
        entry = next(m for m in manifest.json()["manifests"] if m["platform"]["os"] == "linux" and m["platform"]["architecture"] == "amd64")
        manifest = client.get(f"{base}/v2/{base_repository}/manifests/{entry['digest']}", headers={"Accept": entry["mediaType"]})
        manifest.raise_for_status()
    media_type = manifest.headers["Content-Type"]
    image = manifest.json()
    config = client.get(f"{base}/v2/{base_repository}/blobs/{image['config']['digest']}").json()

    for layer in image["layers"]:
        mounted = client.post(f"{base}/v2/{args.repository}/blobs/uploads/", params={"mount": layer["digest"], "from": base_repository})
        if mounted.status_code != 201:
            raise SystemExit(f"could not mount {layer['digest']}: {mounted.status_code}")

    with tempfile.TemporaryDirectory() as temp:
        layer_path, diff_id, digest, size = _layer(Path(temp), args.app)
        _upload(client, base, args.repository, digest, layer_path.read_bytes())

    settings = config.setdefault("config", {})
    settings["Env"] = [item for item in settings.get("Env", []) if not item.startswith("PYTHONPATH=")] + ["PYTHONPATH=/app/site:/app/src", "HOST=0.0.0.0", "PORT=8000", "PYTHONUNBUFFERED=1"]
    settings.update({"Cmd": ["python", "-m", APPS[args.app]["module"]], "WorkingDir": "/app", "User": "65534:65534", "ExposedPorts": {"8000/tcp": {}}})
    config.pop("container_config", None)
    config["rootfs"]["diff_ids"].append(diff_id)
    config.setdefault("history", []).append({"created_by": "scripts/push_image.py: app layer"})
    config_bytes = json.dumps(config, separators=(",", ":")).encode()
    config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
    _upload(client, base, args.repository, config_digest, config_bytes)

    layer_type = "application/vnd.oci.image.layer.v1.tar+gzip" if "oci" in media_type else "application/vnd.docker.image.rootfs.diff.tar.gzip"
    image["config"] = {**image["config"], "digest": config_digest, "size": len(config_bytes)}
    image["layers"].append({"mediaType": layer_type, "digest": digest, "size": size})
    image.pop("annotations", None)
    pushed = client.put(f"{base}/v2/{args.repository}/manifests/{args.tag}", content=json.dumps(image).encode(), headers={"Content-Type": media_type})
    pushed.raise_for_status()
    print(f"{login_server}/{args.repository}:{args.tag} {pushed.headers.get('Docker-Content-Digest')}")


if __name__ == "__main__":
    main()
