"""Speech to text with faster-whisper (CTranslate2 - no torch required)."""

from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger("v2v.stt")


class Transcriber:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        beam_size: int = 1,
        cpu_threads: int = 0,
        initial_prompt: str | None = None,
    ):
        from faster_whisper import WhisperModel

        self.language = language or None
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt or None
        # Seconds of compute per second of audio, measured on the last call.
        # Note that Whisper always encodes a fixed 30 s mel window, so short
        # utterances are dominated by that fixed cost - this is why model size
        # matters so much more than utterance length on a CPU.
        self.last_rtf: float | None = None

        kwargs = {}
        if device == "cpu" and cpu_threads:
            kwargs["cpu_threads"] = cpu_threads
        log.info("loading faster-whisper %s (%s/%s, %d threads)", model_name, device, compute_type, cpu_threads or 0)
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type, **kwargs)
        log.info("faster-whisper ready")

    def transcribe(self, audio: np.ndarray, *, partial: bool = False) -> str:
        """`audio` is mono float32 in [-1, 1] at 16 kHz."""
        if audio is None or audio.size < 1600:  # < 100 ms
            return ""
        duration = audio.size / 16_000
        started = time.perf_counter()
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=1 if partial else self.beam_size,
            best_of=1,
            # Partials stay greedy and deterministic. The authoritative pass
            # also falls back to warmer sampling when its confidence checks
            # fail, which rescues a noisy or heavily accented clip instead of
            # returning a garbled sentence.
            temperature=0.0 if partial else (0.0, 0.2, 0.4),
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
            initial_prompt=None if partial else self.initial_prompt,
            no_speech_threshold=0.75 if partial else 0.5,
            log_prob_threshold=-1.5 if partial else -1.0,
        )
        # The generator is lazy: all the work happens here, in the join.
        text = " ".join(s.text.strip() for s in segments).strip()
        if duration > 0:
            self.last_rtf = (time.perf_counter() - started) / duration
        return text

    def warmup(self) -> None:
        try:
            self.transcribe(np.zeros(16_000, dtype=np.float32))
        except Exception as exc:  # pragma: no cover
            log.debug("warmup failed (harmless): %s", exc)
