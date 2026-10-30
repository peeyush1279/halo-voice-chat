"""Create and deploy the Render service for this repo through the Render API.

    python deploy/render/deploy.py --token rnd_xxx [--region singapore] [--dry-run]

Why this exists: a Blueprint deploy needs a browser and the Render GitHub app.
The API can do the same job from a terminal, because Render builds services
straight from a public repository URL when no GitHub connection is present.

Token: Render dashboard -> Account Settings -> API Keys -> Create API Key.
Read from --token, .render_token (gitignored), or RENDER_API_KEY.

Idempotent: if a service with the same name already exists in the workspace it
is reused, its env vars are refreshed, and a new deploy is triggered.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API = "https://api.render.com/v1"
REPO_URL = "https://github.com/peeyush1279/halo-voice-chat"
NAME = "halo-voice-chat"
BRANCH = "main"
BUILD = "pip install -r requirements.txt"
START = "uvicorn server.main:app --host 0.0.0.0 --port $PORT"


def fail(message: str, response: httpx.Response | None = None) -> int:
    print(f"\n{message}", file=sys.stderr)
    if response is not None:
        print(f"  HTTP {response.status_code}: {response.text[:1200]}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="deploy to Render through the API")
    parser.add_argument("--token", default=None)
    parser.add_argument("--owner", default=None, help="workspace id (default: first one)")
    parser.add_argument("--region", default="singapore")
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = args.token
    if not token:
        candidate = ROOT / ".render_token"
        if candidate.exists():
            token = candidate.read_text(encoding="utf-8").strip()
    if not token:
        token = os.getenv("RENDER_API_KEY", "").strip()
    if not token:
        print(
            "no token: pass --token, write it to .render_token, or set RENDER_API_KEY\n"
            "create one at https://dashboard.render.com/settings#api-keys",
            file=sys.stderr,
        )
        return 2

    from server.config import get_settings  # noqa: E402  (repo root on path)

    settings = get_settings()
    groq_key = settings.groq_api_key
    if not groq_key:
        return fail("GROQ_API_KEY is empty in .env - the service would have no key")

    env_vars = [
        {"key": "GROQ_API_KEY", "value": groq_key},
        {"key": "PYTHON_VERSION", "value": "3.12.7"},
    ]

    client = httpx.Client(
        base_url=API,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=60,
    )

    owners = client.get("/owners", params={"limit": 20})
    if owners.status_code != 200:
        return fail("could not read workspaces - is the API key valid?", owners)
    owner_list = [row.get("owner", row) for row in owners.json()]
    if not owner_list:
        return fail("the API key has no workspace")
    owner = next((o for o in owner_list if o.get("id") == args.owner), None)
    if args.owner and owner is None:
        return fail(f"workspace {args.owner} not found")
    owner = owner or owner_list[0]
    print(f"workspace: {owner.get('name')} ({owner.get('id')})")

    existing = client.get("/services", params={"name": args.name, "limit": 20})
    if existing.status_code != 200:
        return fail("could not list services", existing)
    match = None
    for row in existing.json():
        service = row.get("service", row)
        if service.get("name") == args.name:
            match = service
            break

    if match:
        service_id = match["id"]
        print(f"reusing service {service_id}")
    else:
        body = {
            "type": "web_service",
            "name": args.name,
            "ownerId": owner["id"],
            "repo": REPO_URL,
            "branch": BRANCH,
            "autoDeploy": "yes",
            "envVars": env_vars,
            "serviceDetails": {
                "env": "python",
                "plan": "free",
                "region": args.region,
                "healthCheckPath": "/api/health",
                "envSpecificDetails": {
                    "buildCommand": BUILD,
                    "startCommand": START,
                },
            },
        }
        if args.dry_run:
            print(json.dumps(body, indent=2))
            return 0
        created = client.post("/services", json=body)
        if created.status_code >= 300:
            return fail("service creation failed", created)
        payload = created.json()
        service = payload.get("service", payload)
        service_id = service["id"]
        print(f"created service {service_id}")

    if args.dry_run:
        return 0

    refreshed = client.put(f"/services/{service_id}/env-vars", json=env_vars)
    if refreshed.status_code >= 300:
        return fail("could not set env vars", refreshed)
    print(f"env vars: {', '.join(v['key'] for v in env_vars)}")

    deploy = client.post(f"/services/{service_id}/deploys", json={"clearCache": "do_not_clear"})
    if deploy.status_code >= 300:
        return fail("could not trigger a deploy", deploy)
    deploy_id = (deploy.json().get("deploy") or {}).get("id")
    print(f"deploy {deploy_id} started")

    url = None
    status = "unknown"
    deadline = time.time() + 900
    while time.time() < deadline:
        time.sleep(15)
        info = client.get(f"/services/{service_id}").json()
        service = info.get("service", info)
        details = service.get("serviceDetails") or {}
        url = details.get("url") or url
        status = service.get("suspended") and "suspended" or status
        latest = client.get(f"/services/{service_id}/deploys", params={"limit": 1})
        if latest.status_code == 200 and latest.json():
            deploy_row = latest.json()[0].get("deploy", latest.json()[0])
            status = deploy_row.get("status", status)
        print(f"  {status:<22} {url or ''}")
        if status in {"live", "build_failed", "update_failed", "canceled", "deactivated"}:
            break

    if status != "live":
        return fail(f"deploy ended in state {status!r}", None)

    print(f"\nlive:\n  {url}")
    print(f"dashboard:\n  https://dashboard.render.com/web/{service_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
