"""One live conversation over a websocket.

Design notes (all of these exist to keep the round trip snappy):

* Audio arrives as raw 16 kHz mono int16 frames.  Silero VAD runs inline on the
  event loop (it is ~0.2 ms per 32 ms frame) and decides utterance boundaries.
* STT runs once per finished utterance in a single-thread executor.  While you
  are still talking we re-transcribe the growing buffer every ~0.7 s to show a
  live transcript; those partials are dropped rather than queued if the worker
  is busy, so they can never delay the final pass.
* The LLM streams tokens; a chunker flushes the first *clause* so TTS can start
  speaking while the rest of the reply is still being generated.
* TTS runs on a worker thread and pushes ordered PCM into an asyncio outbox so
  audio chunks can never overtake each other.
* Everything is tagged with a turn id.  When you start talking over the
  assistant we bump the id, cancel the LLM request and drain the TTS queue -
  that is barge-in, and it is why the id checks are everywhere.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from .config import INPUT_SR, OUTPUT_SR, VAD_FRAME
from .textchunk import SpeechChunker
from .vad import TurnDetector

log = logging.getLogger("v2v.pipeline")


class Conversation:
    def __init__(self, ws: WebSocket, engines):
        self.ws = ws
        self.engines = engines
        self.s = engines.settings
        self.loop = asyncio.get_running_loop()

        self.detector = TurnDetector(
            vad=engines.vad,
            sample_rate=INPUT_SR,
            frame=VAD_FRAME,
            threshold=self.s.vad_threshold,
            neg_threshold=self.s.vad_neg_threshold,
            min_speech_s=self.s.vad_min_speech_ms / 1000,
            min_silence_s=self.s.vad_min_silence_ms / 1000,
            pad_s=self.s.vad_speech_pad_ms / 1000,
            max_utterance_s=self.s.vad_max_utterance_s,
        )

        self._history: list[dict] = [{"role": "system", "content": self.s.system_prompt}]
        self._outbox: asyncio.Queue = asyncio.Queue()
        self._sender_task: asyncio.Task | None = None
        self._turn_task: asyncio.Task | None = None

        self._turn = 0
        self._generating = False
        self._audio_pending = 0
        self._partial_busy = False
        self._last_partial = 0.0
        self._timing: dict[int, dict] = {}

        self._tts_q: queue.Queue = queue.Queue()
        self._tts_thread: threading.Thread | None = None
        self._tts_voice = self.s.tts_voice
        self._tts_speed = self.s.tts_speed
        self._idle_emitted_for = -1

        self._model = self.s.active_llm_model
        self._llm = engines.make_llm(self._model)
        self._closed = False

        # Diagnostics: without these, a client that never sends audio and a
        # client sending a silent stream look identical from the server.
        self._frames = 0
        self._peak = 0.0
        self._first_frame_at: float | None = None
        self._connected_at = time.monotonic()

    # ------------------------------------------------------------------ setup
    async def run(self) -> None:
        self._sender_task = asyncio.create_task(self._sender(), name="ws-sender")
        self._tts_thread = threading.Thread(target=self._tts_loop, name="tts", daemon=True)
        self._tts_thread.start()
        self._emit({"type": "ready", "config": self._public_state()})
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("bytes")
                text = message.get("text")
                if data is not None:
                    self.on_audio(data)
                elif text is not None:
                    await self.on_text(text)
        except WebSocketDisconnect:
            pass
        except Exception:  # pragma: no cover - connection level failure
            log.exception("websocket loop failed")
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        self._closed = True
        self._turn += 1
        self._drain_tts()
        self._tts_q.put(None)
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        if self._sender_task and not self._sender_task.done():
            self._outbox.put_nowait(("stop", None))
            self._sender_task.cancel()
        try:
            await self._llm.aclose()
        except Exception:
            pass

    def _public_state(self) -> dict:
        return {
            "llm_model": self._model,
            "voice": self._tts_voice,
            "speed": self._tts_speed,
            "system_prompt": self.s.system_prompt,
            "input_sample_rate": INPUT_SR,
            "output_sample_rate": OUTPUT_SR,
            "audio_format": "pcm" if self.s.tts_engine == "kokoro" else "mp3",
            "stt_backend": self.s.stt_backend,
            "llm_backend": self.s.llm_backend,
            "tts_engine": self.s.tts_engine,
        }

    # ------------------------------------------------------------------ outbox
    def _emit(self, event: dict) -> None:
        if not self._closed:
            self._outbox.put_nowait(("json", event))

    def _put_from_thread(self, item: tuple) -> None:
        try:
            self.loop.call_soon_threadsafe(self._outbox.put_nowait, item)
        except RuntimeError:  # loop already closed
            pass

    async def _sender(self) -> None:
        try:
            while True:
                kind, payload = await self._outbox.get()
                if kind == "stop":
                    return
                if kind == "json":
                    await self.ws.send_text(json.dumps(payload, ensure_ascii=False))
                elif kind == "bytes":
                    await self.ws.send_bytes(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("sender stopped: %s", exc)

    # ------------------------------------------------------------------ audio in
    def on_audio(self, data: bytes) -> None:
        if self._closed:
            return
        pcm = np.frombuffer(data, dtype=np.int16)
        if pcm.size == 0:
            return
        audio = pcm.astype(np.float32) / 32768.0
        self._audio_stats(audio)

        for event in self.detector.push(audio):
            if event.kind == "start":
                log.info("speech started (peak so far %.3f)", self._peak)
                self._emit({"type": "speech_start"})
            elif event.kind == "confirm":
                # Real speech: this is the moment we are allowed to barge in.
                if self._assistant_busy():
                    self._interrupt("barge_in")
            elif event.kind == "end":
                log.info("speech ended after %d frames", self._frames)
                self._emit({"type": "speech_end"})
                if event.audio is not None and event.audio.size:
                    self._begin_turn(event.audio)

        if self.detector.speaking:
            self._maybe_partial()

    def _audio_stats(self, audio: np.ndarray) -> None:
        """Log that audio is arriving, and whether it is actually audible.

        Three failure modes look the same in the browser ("nothing happens"):
        no frames at all (capture never started), frames of pure silence (wrong
        or muted input device), and audio the VAD refuses. This separates them
        from the server side. Levels only - never the audio itself.
        """
        self._frames += 1
        if self._first_frame_at is None:
            self._first_frame_at = time.monotonic()
            log.info(
                "audio: first frame after %.1fs idle (%d samples, %d bytes)",
                self._first_frame_at - self._connected_at,
                audio.size,
                audio.size * 2,
            )
        peak = float(np.abs(audio).max())
        if peak > self._peak:
            self._peak = peak
        if self._frames in (50, 500):
            elapsed = time.monotonic() - (self._first_frame_at or 0)
            muted = " - SILENT: device muted or wrong input" if self._peak < 0.01 else ""
            log.info(
                "audio: %d frames in %.1fs (%.1f/s), peak %.3f%s",
                self._frames,
                elapsed,
                self._frames / max(elapsed, 1e-3),
                self._peak,
                muted,
            )

    def _assistant_busy(self) -> bool:
        return self._generating or self._audio_pending > 0

    def _maybe_emit_idle(self, turn: int) -> None:
        """Signal "back to listening" exactly once, and only when the reply is
        fully spoken *and* the LLM has stopped producing text."""
        if turn != self._turn or self._closed:
            return
        if self._generating or self._audio_pending > 0:
            return
        if self._idle_emitted_for == turn:
            return
        self._idle_emitted_for = turn
        self._emit({"type": "idle"})

    def _drain_tts(self) -> None:
        while True:
            try:
                self._tts_q.get_nowait()
            except queue.Empty:
                return

    def _interrupt(self, reason: str) -> None:
        self._generating = False
        self._turn += 1  # invalidates any in-flight turn
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        self._drain_tts()
        self._audio_pending = 0
        self._emit({"type": "cancel", "reason": reason})

    def _begin_turn(self, audio: np.ndarray | None, preset_text: str | None = None) -> None:
        if self._closed:
            return
        self._turn += 1
        turn = self._turn
        self._turn_task = asyncio.create_task(
            self._run_turn(turn, audio, preset_text), name=f"turn-{turn}"
        )

    def _force_endpoint(self) -> None:
        """End the current utterance right now (used by push-to-talk)."""
        if not self.detector.speaking:
            return
        audio = self.detector.current_audio()
        self.detector.reset()
        if audio is not None and audio.size:
            self._emit({"type": "speech_end"})
            self._begin_turn(audio)

    # ------------------------------------------------------------------ partials
    def _maybe_partial(self) -> None:
        if self._partial_busy or self.engines.stt is None:
            return
        if not self.engines.partials_are_affordable():
            return
        now = time.perf_counter()
        if now - self._last_partial < self.s.partial_interval_s:
            return
        audio = self.detector.current_audio()
        if audio is None:
            return
        if audio.size / INPUT_SR < self.s.partial_min_audio_s:
            return
        max_samples = int(self.s.partial_max_audio_s * INPUT_SR)
        if audio.size > max_samples:
            audio = audio[-max_samples:]
        self._last_partial = now
        self._partial_busy = True
        asyncio.create_task(self._partial_task(self._turn, audio))

    async def _partial_task(self, turn: int, audio: np.ndarray) -> None:
        try:
            text = await self.loop.run_in_executor(
                self.engines.partial_pool, lambda: self.engines.transcribe_partial(audio)
            )
            if turn == self._turn and text and not self._closed:
                self._emit({"type": "partial", "text": text})
        except Exception as exc:
            log.debug("partial failed: %s", exc)
        finally:
            self._partial_busy = False

    # ------------------------------------------------------------------ a turn
    async def _run_turn(self, turn: int, audio: np.ndarray | None, preset_text: str | None = None) -> None:
        t0 = time.perf_counter()
        if preset_text is not None:
            text = preset_text
        else:
            try:
                text = await self.loop.run_in_executor(
                    self.engines.stt_pool, lambda: self.engines.transcribe_final(audio)
                )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.exception("STT failed")
                self._emit({"type": "error", "message": f"Speech recognition failed: {exc}"})
                return

        if turn != self._turn:
            return

        t_stt = time.perf_counter()
        text = " ".join(text.split())
        self._timing[turn] = {"t0": t0, "t_stt": t_stt, "t_first_token": None, "t_first_audio": None}
        # Content stays at DEBUG: this is a shared link, and the owner can turn
        # it on with V2V_LOG_LEVEL=debug when something needs explaining.
        log.debug("transcript: %s", text)
        log.info("transcribed %d chars in %.0f ms", len(text), (t_stt - t0) * 1000)
        self._emit({"type": "user", "text": text, "final": True})

        if len(text) < 2:
            self._maybe_emit_idle(turn)
            return

        self._history.append({"role": "user", "content": text})
        keep = max(1, self.s.history_turns * 2)
        messages = [self._history[0]] + self._history[1:][-keep:]

        self._generating = True
        self._emit({"type": "assistant_start"})

        chunker = SpeechChunker(self.s.tts_min_chars, self.s.tts_max_chars, self.s.tts_first_chars)
        pieces: list[str] = []
        seq = 0
        pending_delta: list[str] = []
        last_flush = 0.0

        try:
            async for piece in self._llm.stream_chat(messages):
                if turn != self._turn:
                    return
                if self._timing[turn]["t_first_token"] is None:
                    self._timing[turn]["t_first_token"] = time.perf_counter()
                pieces.append(piece)
                pending_delta.append(piece)
                now = time.perf_counter()
                if now - last_flush >= 0.05:
                    self._emit({"type": "assistant_delta", "text": "".join(pending_delta)})
                    pending_delta.clear()
                    last_flush = now
                for chunk in chunker.push(piece):
                    self._enqueue_tts(turn, chunk, seq)
                    seq += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("LLM failed")
            self._emit({"type": "error", "message": f"LLM error: {exc}"})
        finally:
            if turn == self._turn:
                self._generating = False

        if turn != self._turn:
            return

        if pending_delta:
            self._emit({"type": "assistant_delta", "text": "".join(pending_delta)})
        for chunk in chunker.flush():
            self._enqueue_tts(turn, chunk, seq)
            seq += 1

        reply = "".join(pieces).strip()
        if reply:
            self._history.append({"role": "assistant", "content": reply})
        self._emit({"type": "assistant_done", "text": reply})
        # Unconditional: TTS may already have drained its queue while we were
        # still streaming, in which case this is the only chance to go idle.
        self._maybe_emit_idle(turn)

    def _enqueue_tts(self, turn: int, text: str, seq: int) -> None:
        if not text.strip() or self.engines.tts is None:
            return
        self._audio_pending += 1
        self._tts_q.put((turn, text, seq))

    # ------------------------------------------------------------------ TTS thread
    def _tts_loop(self) -> None:
        while True:
            item = self._tts_q.get()
            if item is None:
                return
            turn, text, seq = item
            if turn != self._turn or self._closed:
                continue
            try:
                payload, sample_rate, audio_format = self.engines.tts.audio_payload(
                    text, voice=self._tts_voice, speed=self._tts_speed
                )
            except Exception as exc:
                log.exception("TTS failed")
                self._put_from_thread(("json", {"type": "error", "message": f"Speech synthesis failed: {exc}"}))
                if turn == self._turn:
                    self._audio_pending -= 1
                continue

            if turn != self._turn or self._closed:
                continue

            self._put_from_thread(
                (
                    "json",
                    {
                        "type": "audio_start",
                        "turn": turn,
                        "id": seq,
                        "sample_rate": sample_rate,
                        "format": audio_format,
                        "text": text,
                    },
                )
            )
            if seq == 0:
                self._emit_timing(turn)

            if audio_format == "pcm":
                # ~200 ms per frame keeps the playback queue smooth without chatter.
                step = max(1, int(sample_rate * 0.2 * 2))  # int16 -> 2 bytes/sample
                for i in range(0, len(payload), step):
                    self._put_from_thread(("bytes", payload[i : i + step]))
            else:
                # MP3 arrives as one self-contained file per clause; the browser
                # decodes it with decodeAudioData and queues the result.
                self._put_from_thread(("bytes", payload))
            self._put_from_thread(("json", {"type": "audio_end", "turn": turn, "id": seq}))

            if turn == self._turn:
                self._audio_pending -= 1
                # Re-evaluate "back to listening" on the event loop thread.
                try:
                    self.loop.call_soon_threadsafe(self._maybe_emit_idle, turn)
                except RuntimeError:
                    pass

    def _emit_timing(self, turn: int) -> None:
        t = self._timing.get(turn)
        if not t or t["t_first_audio"] is not None:
            return
        t["t_first_audio"] = time.perf_counter()
        ttft = t["t_first_token"]
        self._put_from_thread(
            (
                "json",
                {
                    "type": "timing",
                    "turn": turn,
                    "stt_ms": round((t["t_stt"] - t["t0"]) * 1000),
                    "ttft_ms": round((ttft - t["t_stt"]) * 1000) if ttft else None,
                    "tts_ms": round((t["t_first_audio"] - ttft) * 1000) if ttft else None,
                    "total_ms": round((t["t_first_audio"] - t["t0"]) * 1000),
                },
            )
        )

    # ------------------------------------------------------------------ control
    async def on_text(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        kind = msg.get("type")

        if kind == "ping":
            self._emit({"type": "pong", "t": msg.get("t")})

        elif kind == "config":
            await self._apply_config(msg)

        elif kind == "interrupt":
            self._interrupt("client")

        elif kind == "flush":
            self._force_endpoint()

        elif kind == "text_input":
            text = (msg.get("text") or "").strip()
            if text:
                self._interrupt("text_input")
                self._begin_turn(None, preset_text=text)

        elif kind == "reset":
            self._interrupt("reset")
            self._history = [{"role": "system", "content": self.s.system_prompt}]
            self.detector.reset()
            self._emit({"type": "reset_done"})

    async def _apply_config(self, msg: dict) -> None:
        changed: list[str] = []

        voice = msg.get("voice")
        if isinstance(voice, str) and voice and voice != self._tts_voice and self.engines.tts is not None:
            self._tts_voice = voice
            self.engines.tts.default_voice = voice
            changed.append("voice")

        speed = msg.get("speed")
        if isinstance(speed, (int, float)) and 0.5 <= float(speed) <= 2.0:
            self._tts_speed = float(speed)
            self.engines.tts.speed = self._tts_speed
            changed.append("speed")

        model = msg.get("llm_model")
        if isinstance(model, str) and model and model != self._model:
            self._interrupt("model_switch")
            await self._llm.aclose()
            self._model = model
            self._llm = self.engines.make_llm(model)
            changed.append("llm_model")
            asyncio.create_task(self.engines.warm_llm(model))

        system_prompt = msg.get("system_prompt")
        if isinstance(system_prompt, str) and system_prompt.strip():
            self._history[0] = {"role": "system", "content": system_prompt}
            self._llm.system_prompt = system_prompt
            changed.append("system_prompt")

        if changed:
            self._emit({"type": "config_applied", "changed": changed, "config": {
                "voice": self._tts_voice,
                "speed": self._tts_speed,
                "llm_model": self._model,
            }})
