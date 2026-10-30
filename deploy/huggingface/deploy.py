"""Deploy this repo to a free Hugging Face Docker Space.

    python deploy/huggingface/deploy.py --token hf_xxx [--space halo-voice-chat] [--dry-run]

The Space is a separate git repo, so this stages a clean copy of the app (no
.git, no .venv, no local models, no .env), drops in the Space card and
Dockerfile, uploads it, and sets GROQ_API_KEY as a Space secret.

Free CPU hardware, 16 GB RAM, ~48 h of inactivity before it sleeps.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

HF_DIR = Path(__file__).resolve().parent
ROOT = HF_DIR.parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Everything the Space needs, and nothing that would bloat or leak.
INCLUDE = ["server", "web", "requirements.txt"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", "*.pyc"}


def stage(target: Path) -> list[str]:
    written: list[str] = []
    for name in INCLUDE:
        source = ROOT / name
        if source.is_dir():
            for item in source.rglob("*"):
                if not item.is_file():
                    continue
                if "__pycache__" in item.parts or item.suffix == ".pyc":
                    continue
                dest = target / item.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item, dest)
                written.append(str(dest.relative_to(target)))
        else:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            written.append(name)

    for name in ("Dockerfile", "README.md"):
        shutil.copyfile(HF_DIR / name, target / name)
        written.append(name)

    (target / ".dockerignore").write_text(
        ".git\n.venv\nmodels\n.env\n**/__pycache__\n*.pyc\n*.log\n*.wav\n*.mp3\n",
        encoding="utf-8",
    )
    written.append(".dockerignore")
    return sorted(written)


def main() -> int:
    parser = argparse.ArgumentParser(description="deploy to a Hugging Face Space")
    parser.add_argument("--token", default=None, help="HF write token (or set HF_TOKEN)")
    parser.add_argument("--space", default="halo-voice-chat", help="Space repo name")
    parser.add_argument("--user", default="peeyush1279", help="HF username or org")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="stage the files, upload nothing")
    args = parser.parse_args()

    repo_id = f"{args.user}/{args.space}"
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        files = stage(target)
        print(f"staged {len(files)} files for {repo_id}:")
        for name in files:
            print("  ", name)
        if args.dry_run:
            return 0

        token = args.token
        if not token:
            candidate = ROOT / ".hf_token"
            if candidate.exists():
                token = candidate.read_text(encoding="utf-8").strip()
        if not token:
            token = os.getenv("HF_TOKEN")
        if not token:
            print(
                "\nno token: pass --token, or write it to .hf_token, or set HF_TOKEN\n"
                "(create one at https://huggingface.co/settings/tokens - 'write' scope)",
                file=sys.stderr,
            )
            return 2

        # The key that the running app needs; read from .env exactly like the
        # server does, so there is one source of truth.
        from server.config import get_settings  # noqa: E402  (repo root on path)

        groq_key = get_settings().groq_api_key
        if not groq_key:
            print("\nGROQ_API_KEY is empty in .env - the Space would have no key", file=sys.stderr)
            return 2

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        who = api.whoami()
        print(f"\nlogged in as {who.get('name')}")

        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )
        print(f"space ready: {repo_id}")

        api.add_space_secret(repo_id=repo_id, key="GROQ_API_KEY", value=groq_key)
        print("secret GROQ_API_KEY set")

        api.upload_folder(
            folder_path=str(target),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy Halo voice chat",
        )
        print("uploaded")

        url = f"https://{repo_id.replace('/', '-').lower()}.hf.space"
        print(f"\nlive (build takes ~2-4 min on first push):\n  {url}")
        print(f"logs:\n  https://huggingface.co/spaces/{repo_id}?logs=container")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
