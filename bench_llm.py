"""Benchmark the installed Ollama models the way this app actually calls them.

    .\\.venv\\Scripts\\python.exe bench_llm.py
    .\\.venv\\Scripts\\python.exe bench_llm.py --models qwen2.5:3b-instruct
    .\\.venv\\Scripts\\python.exe bench_llm.py --question "your own prompt"

Reproduces the server's request exactly: same system prompt, temperature,
top_p, num_ctx, num_predict and `think: false`, so the numbers match what a
voice turn really costs.  Reports:

  cold   ms  time from request to first token when the model had to load
  ttft   ms  time to first token (what the user hears as dead air)
  tok/s      generation speed (what makes a reply feel fast or slow)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import httpx

from server.config import get_settings

OLLAMA = get_settings().ollama_url
SYSTEM = get_settings().system_prompt

# Short, spoken, and it has to actually reason to answer - the kind of turn a
# voice companion gets asked all day.
DEFAULT_QUESTION = (
    "A shirt is on sale for $40 after a 20% discount. "
    "What was the original price? Answer in one short sentence."
)


async def ask(client: httpx.AsyncClient, model: str, question: str, *, max_tokens: int):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    options = {
        "temperature": 0.7,
        "top_p": 0.9,
        "num_predict": max_tokens,
        "num_ctx": 4096,
    }
    # The server tries `think: false` first and retries without it; mirror that.
    variants = [
        {"model": model, "messages": messages, "stream": True, "keep_alive": "30m", "think": False, "options": options},
        {"model": model, "messages": messages, "stream": True, "keep_alive": "30m", "options": options},
    ]

    last_error = "no response"
    for payload in variants:
        t0 = time.perf_counter()
        first_token = None
        text: list[str] = []
        stats: dict = {}
        timeout = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)
        async with client.stream("POST", f"{OLLAMA}/api/chat", json=payload, timeout=timeout) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")[:200]
                last_error = f"HTTP {resp.status_code}: {body}"
                continue
            async for line in resp.aiter_lines():
                first_token, text, stats = _consume(line, t0, first_token, text, stats)

        total = time.perf_counter() - t0
        return {
            "text": "".join(text).strip(),
            "total_ms": round(total * 1000),
            "ttft_ms": round((first_token - t0) * 1000) if first_token else None,
            "tok_s": round(stats["eval_count"] / (stats["eval_duration"] / 1e9), 1)
            if stats.get("eval_count") and stats.get("eval_duration")
            else None,
            "eval_count": stats.get("eval_count"),
            "load_ms": round(stats["load_duration"] / 1e6) if stats.get("load_duration") else None,
            "prompt_tokens": stats.get("prompt_eval_count"),
        }
    return {"error": last_error}


def _consume(line, t0, first_token, text, stats):
    if not line.strip():
        return first_token, text, stats
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return first_token, text, stats
    if obj.get("done"):
        stats = {k: obj.get(k) for k in ("eval_count", "eval_duration", "load_duration", "prompt_eval_count")}
        return first_token, text, stats
    piece = (obj.get("message") or {}).get("content")
    if piece:
        if first_token is None:
            first_token = time.perf_counter()
        text.append(piece)
    return first_token, text, stats


async def main() -> int:
    ap = argparse.ArgumentParser(description="benchmark Ollama models for voice latency")
    ap.add_argument("--models", nargs="*", default=None, help="defaults to every installed model")
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--max-tokens", type=int, default=120, help="short answers keep voice latency honest")
    ap.add_argument("--rounds", type=int, default=2, help="first round is cold, later rounds are warm")
    args = ap.parse_args()

    async with httpx.AsyncClient() as client:
        tags = await client.get(f"{OLLAMA}/api/tags", timeout=5)
        if tags.status_code >= 400:
            print(f"Ollama is not reachable at {OLLAMA} - start it with `ollama serve`.", file=sys.stderr)
            return 1
        available = [m["name"] for m in tags.json().get("models", [])]
        models = args.models or available

        print(f"question: {args.question}\n")
        rows = []
        for model in models:
            if model not in available:
                print(f"  {model}: not installed (available: {', '.join(available)})")
                continue
            for rnd in range(args.rounds):
                res = await ask(client, model, args.question, max_tokens=args.max_tokens)
                if "error" in res:
                    print(f"  {model:44s} round {rnd}: ERROR {res['error']}")
                    break
                tag = "cold" if rnd == 0 else "warm"
                rows.append((res["ttft_ms"] or 0, model, rnd, res))
                print(
                    f"  {model:44s} {tag:4s}  ttft {res['ttft_ms']:>5} ms   "
                    f"gen {res['tok_s'] or 0:>6.1f} tok/s   total {res['total_ms']:>5} ms   "
                    f"load {res['load_ms']} ms"
                )
                if rnd == args.rounds - 1:
                    print(f"      {res['text'][:220]!r}")

        if rows:
            print("\nwarm first-token (what the user waits through):")
            for ttft, model, rnd, res in sorted(
                [(t, m, r, x) for t, m, r, x in rows if r > 0], key=lambda x: x[0]
            ):
                print(f"  {ttft:>5} ms  {res['tok_s'] or 0:>6.1f} tok/s   {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
