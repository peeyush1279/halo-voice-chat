"""Streaming chat client for a local Ollama server.

Uses the native /api/chat NDJSON stream.  `keep_alive` is important: it stops
Ollama from unloading the model between turns, which would otherwise add
several seconds to the first reply after every pause.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

log = logging.getLogger("v2v.llm")


class OllamaError(RuntimeError):
    pass


def _cannot_reach(base_url: str, exc: Exception) -> OllamaError:
    """The single most confusing failure in the app, so explain it in full.

    Ollama is a separate process that has to be started by hand (or by its
    tray app); when it disappears the chat pipeline just stops responding and
    a bare ConnectError tells the user nothing useful.
    """
    detail = str(exc).strip() or exc.__class__.__name__
    return OllamaError(
        f"cannot reach Ollama at {base_url} ({detail}). "
        "Ollama is not running - start it with `ollama serve` or by opening "
        "the Ollama desktop app, then retry."
    )


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        system_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 220,
        keep_alive: str = "30m",
        num_ctx: int = 4096,
        no_think: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.no_think = no_think
        # Streaming replies can be long; only connect/write need a deadline.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=None, write=15.0, pool=5.0))

    # ---- housekeeping ------------------------------------------------------
    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[dict]:
        try:
            r = await self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            r.raise_for_status()
        except Exception as exc:
            raise _cannot_reach(self.base_url, exc) from exc
        return r.json().get("models", [])

    def _keep_alive(self):
        """Ollama only treats the *integer* -1 as "keep resident forever".

        A bare "-1" string is not a valid Go duration, so it would be rejected
        (or ignored) and the model would get evicted on the next idle window,
        which is exactly the 17 s cold-start penalty we are trying to avoid.
        """
        raw = str(self.keep_alive).strip().lower()
        if raw in {"-1", "forever", "always", "inf", "none", "unlimited"}:
            return -1
        return self.keep_alive

    def _payload(self, messages: list[dict], *, think: bool | None) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self._keep_alive(),
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        if think is not None:
            payload["think"] = think
        return payload

    async def _open_stream(self, messages: list[dict]):
        # Ask for thinking-free output where supported; retry once without the
        # flag if the server/model does not understand it.
        attempts = [False] if self.no_think else [None]
        if self.no_think:
            attempts.append(None)
        last_exc: Exception | None = None
        for think in attempts:
            req = self._client.build_request(
                "POST", f"{self.base_url}/api/chat", json=self._payload(messages, think=think)
            )
            try:
                resp = await self._client.send(req, stream=True)
            except Exception as exc:
                raise _cannot_reach(self.base_url, exc) from exc
            if resp.status_code < 400:
                return resp
            body = (await resp.aread()).decode("utf-8", "replace")[:300]
            await resp.aclose()
            last_exc = OllamaError(f"Ollama returned {resp.status_code}: {body}")
            log.debug("chat request rejected (status %s), retrying: %s", resp.status_code, body)
        raise last_exc or OllamaError("Ollama request failed")

    # ---- the interesting bit ----------------------------------------------
    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        """Yield content pieces as they are generated (thinking is not spoken)."""
        resp = await self._open_stream(messages)
        try:
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if err := obj.get("error"):
                    raise OllamaError(str(err))
                piece = (obj.get("message") or {}).get("content")
                if piece:
                    yield piece
                if obj.get("done"):
                    break
        finally:
            await resp.aclose()

    async def warmup(self) -> None:
        """Force the model into memory so turn #1 is not slow."""
        try:
            async for _ in self.stream_chat(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Say hi."},
                ]
            ):
                break
        except OllamaError:
            # Not a warmup quirk: Ollama is down or lacks the model.  Rethrow so
            # engines.py marks the LLM as failed - otherwise /api/health reports
            # all-green and the user only learns the truth on their first sentence.
            raise
        except Exception as exc:  # pragma: no cover
            log.warning("LLM warmup failed: %s", exc)


class GroqError(RuntimeError):
    pass


class GroqClient:
    """OpenAI-compatible streaming chat on Groq's free tier.

    Same `stream_chat(messages)` generator as `OllamaClient`, so the pipeline
    is unaware of which one it is holding.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 220,
        timeout: float = 30.0,
    ):
        if not api_key:
            raise GroqError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and set it in the environment."
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=timeout, write=15.0, pool=5.0)
        )

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _explain(self, status: int, body: str) -> str:
        if status == 401:
            return "Groq rejected the API key (401). Check GROQ_API_KEY."
        if status == 429:
            return (
                "Groq rate limit hit (429). The free tier is limited per minute "
                "and per day; wait a moment and try again."
            )
        if status == 404:
            return f"Groq does not know the model {self.model!r} (404). {body[:160]}"
        return f"Groq returned {status}: {body[:200]}"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[dict]:
        response = await self._client.get(f"{self.base_url}/models", headers=self._headers())
        if response.status_code >= 400:
            raise GroqError(self._explain(response.status_code, response.text))
        return [{"name": item.get("id")} for item in response.json().get("data", [])]

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise GroqError(self._explain(response.status_code, body))
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if err := obj.get("error"):
                        raise GroqError(str(err))
                    choices = obj.get("choices") or [{}]
                    piece = (choices[0].get("delta") or {}).get("content")
                    if piece:
                        yield piece
        except httpx.HTTPError as exc:
            raise GroqError(f"cannot reach Groq ({exc})") from exc

    async def warmup(self) -> None:
        # A hosted API is already warm; a probe request would only spend quota.
        pass
