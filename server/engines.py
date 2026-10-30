"""Process-wide model holder.

Models are heavy and stateful, so we load them exactly once and share them
across websocket connections.  Loading happens in a background thread at
startup, which keeps the web UI responsive and reports progress via /api/health.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from .llm import GroqClient, OllamaClient
from .stt import GroqTranscriber, Transcriber
from .tts import EdgeTTS, GttsTTS, KokoroTTS
from .vad import SileroVAD

log = logging.getLogger("v2v.engines")


class Engines:
    def __init__(self, settings, models_dir):
        self.settings = settings
        self.models_dir = models_dir
        self.vad: SileroVAD | None = None
        self.stt: Transcriber | None = None
        self.tts: KokoroTTS | None = None
        # A second instance used only for the live partial transcript. It gets
        # fewer threads, and because it is a separate model a slow partial can
        # never sit in front of the authoritative transcription of your turn.
        self.stt_partial: Transcriber | None = None
        # One Whisper instance -> one worker, otherwise we thrash the CPU.
        self.stt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self.partial_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt-partial")
        self.finals_in_flight = 0
        self.status: dict[str, str] = {"vad": "pending", "stt": "pending", "tts": "pending", "llm": "pending"}
        self.errors: dict[str, str] = {}
        self._lock = threading.Lock()
        self.loaded = threading.Event()
        self.load_seconds: float | None = None
        self._llm_retry: asyncio.Task | None = None

    # ---- loading -----------------------------------------------------------
    def _set(self, key: str, state: str, error: str | None = None) -> None:
        with self._lock:
            self.status[key] = state
            if error:
                self.errors[key] = error
        log.info("engine %s: %s%s", key, state, f" ({error})" if error else "")

    def load_all(self) -> None:
        import time

        started = time.perf_counter()
        self._load_vad()
        self._load_stt()
        self._load_tts()
        self.load_seconds = time.perf_counter() - started
        self.loaded.set()
        log.info("all engines loaded in %.1fs", self.load_seconds)

    def _load_vad(self) -> None:
        try:
            from .download_models import ensure_silero

            path = ensure_silero()
            self.vad = SileroVAD(path)
            self._set("vad", "ready")
        except Exception as exc:  # pragma: no cover
            log.exception("VAD load failed")
            self._set("vad", "error", str(exc))

    def _load_stt(self) -> None:
        s = self.settings
        try:
            self._set("stt", "loading")
            if s.stt_backend == "groq":
                self.stt = GroqTranscriber(
                    api_key=s.groq_api_key,
                    model=s.groq_stt_model,
                    language=s.whisper_language,
                    base_url=s.groq_api_base,
                    timeout=s.groq_timeout_s,
                )
                # No second model: partials are off in cloud mode, see
                # partials_are_affordable().
                self.stt_partial = None
                self._set("stt", "ready")
                return
            self.stt = Transcriber(
                model_name=s.whisper_model,
                device=s.whisper_device,
                compute_type=s.whisper_compute_type,
                language=s.whisper_language,
                beam_size=s.whisper_beam,
                cpu_threads=s.whisper_threads,
                initial_prompt=s.whisper_initial_prompt,
            )
            self.stt.warmup()
            # Best effort: a second, cheaper instance just for live partials.
            try:
                self.stt_partial = Transcriber(
                    model_name=s.whisper_partial_model,
                    device=s.whisper_device,
                    compute_type=s.whisper_compute_type,
                    language=s.whisper_language,
                    beam_size=1,
                    cpu_threads=max(1, s.whisper_threads // 2),
                )
            except Exception as exc:
                log.warning("no dedicated partial-transcription model (%s); sharing the main one", exc)
                self.stt_partial = self.stt
            self._set("stt", "ready")
        except Exception as exc:
            log.exception("STT load failed")
            self._set("stt", "error", str(exc))

    def _load_tts(self) -> None:
        s = self.settings
        try:
            self._set("tts", "loading")
            if s.tts_engine == "gtts":
                self.tts = GttsTTS(lang=s.gtts_lang, tld=s.gtts_tld)
            elif s.tts_engine == "edge":
                self.tts = EdgeTTS(voice=s.edge_voice)
            else:
                self.tts = KokoroTTS(
                    models_dir=self.models_dir,
                    backend=s.tts_backend,
                    default_voice=s.tts_voice,
                    speed=s.tts_speed,
                    trim_silence=s.tts_trim_silence,
                )
            self.tts.warmup()
            self._set("tts", "ready")
        except Exception as exc:
            log.exception("TTS load failed")
            self._set("tts", "error", str(exc))

    # ---- transcription helpers --------------------------------------------
    def transcribe_final(self, audio) -> str:
        """Authoritative pass for a finished utterance."""
        if self.stt is None:
            raise RuntimeError("speech recognition is not available")
        self.finals_in_flight += 1
        try:
            return self.stt.transcribe(audio)
        finally:
            self.finals_in_flight -= 1

    def transcribe_partial(self, audio) -> str:
        worker = self.stt_partial or self.stt
        return worker.transcribe(audio, partial=True)

    def partials_are_affordable(self, max_rtf: float = 0.75) -> bool:
        """Skip the live transcript if the partial model cannot keep up.

        Partials are a luxury: on a fast model they make the UI feel instant, on
        a slow one they would just burn CPU that the real transcript needs.
        This measures the *partial* worker, which is deliberately cheaper than
        the one used for finals - otherwise the accurate final model would
        switch partials off even though the cheap one keeps up fine.

        On Groq they are switched off outright: a partial every 0.7 s would eat
        a 30-requests-per-minute free quota to render text that gets thrown
        away when the turn ends.
        """
        if self.settings.stt_backend == "groq":
            return False
        if self.finals_in_flight:
            return False
        worker = self.stt_partial or self.stt
        rtf = worker.last_rtf if worker else None
        return rtf is None or rtf <= max_rtf

    async def warm_llm(self, model: str | None = None) -> None:
        try:
            client = self.make_llm(model)
            try:
                await client.warmup()
            finally:
                await client.aclose()
            self._set("llm", "ready")
        except Exception as exc:
            log.warning("LLM warmup failed: %s", exc)
            self._set("llm", "error", str(exc))
            # Ollama is a separate process that is often not running yet.  Retry
            # quietly in the background so /api/health recovers (and the model
            # gets resident) as soon as it comes up, instead of staying red.
            # A hosted API never "comes up later", so there is nothing to poll.
            if self.settings.llm_backend != "groq":
                self._schedule_llm_retry(model)

    def _schedule_llm_retry(self, model: str | None) -> None:
        if self._llm_retry and not self._llm_retry.done():
            return
        self._llm_retry = asyncio.create_task(self._llm_retry_loop(model), name="llm-retry")

    async def _llm_retry_loop(
        self, model: str | None, *, interval: float = 15.0, attempts: int = 80
    ) -> None:
        """Poll until Ollama answers again (~20 minutes of trying)."""
        for _ in range(attempts):
            await asyncio.sleep(interval)
            if self.status.get("llm") == "ready":
                return
            client = self.make_llm(model)
            try:
                await client.warmup()
                self._set("llm", "ready")
                log.info("LLM recovered: %s is reachable", model or self.settings.llm_model)
                return
            except Exception as exc:
                self._set("llm", "error", str(exc))
            finally:
                await client.aclose()

    # ---- factories ---------------------------------------------------------
    def make_llm(self, model: str | None = None) -> OllamaClient | GroqClient:
        s = self.settings
        if s.llm_backend == "groq":
            return GroqClient(
                base_url=s.groq_api_base,
                api_key=s.groq_api_key,
                model=model or s.groq_llm_model,
                system_prompt=s.system_prompt,
                temperature=s.llm_temperature,
                top_p=s.llm_top_p,
                max_tokens=s.llm_max_tokens,
                timeout=s.groq_timeout_s,
            )
        return OllamaClient(
            base_url=s.ollama_url,
            model=model or s.llm_model,
            system_prompt=s.system_prompt,
            temperature=s.llm_temperature,
            top_p=s.llm_top_p,
            max_tokens=s.llm_max_tokens,
            keep_alive=s.llm_keep_alive,
            num_ctx=s.llm_num_ctx,
            no_think=s.llm_no_think,
        )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "ready": self.loaded.is_set(),
                "status": dict(self.status),
                "errors": dict(self.errors),
                "load_seconds": self.load_seconds,
            }
