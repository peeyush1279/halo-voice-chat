"""Kokoro-82M text to speech.

Runs on the ONNX runtime by default (no torch, fast enough on CPU) and can fall
back to the `kokoro` torch package if that is what you have installed.

Kokoro needs grapheme-to-phoneme conversion for English, which goes through
eSpeak.  `espeakng-loader` ships a Windows DLL so you do not have to install
eSpeak system-wide.
"""

from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger("v2v.tts")


def _setup_espeak() -> tuple[str | None, str | None]:
    """Make eSpeak discoverable for misaki/phonemizer. Returns (data, library)."""
    try:
        import espeakng_loader  # type: ignore

        data_path = str(espeakng_loader.get_data_path())
        lib_path = str(espeakng_loader.get_library_path())
        os.environ.setdefault("ESPEAK_DATA_PATH", data_path)
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", lib_path)
        log.debug("espeak-ng ready: %s", lib_path)
        return data_path, lib_path
    except Exception as exc:  # pragma: no cover - platform dependent
        log.debug("espeakng-loader unavailable: %s", exc)
        return None, None


def _trim_silence(audio: np.ndarray, sample_rate: int, threshold: float = 0.004, pad_ms: int = 20) -> np.ndarray:
    if audio.size == 0:
        return audio
    loud = np.where(np.abs(audio) > threshold)[0]
    if loud.size == 0:
        return audio
    pad = int(sample_rate * pad_ms / 1000)
    start = max(0, int(loud[0]) - pad)
    end = min(audio.size, int(loud[-1]) + pad)
    return audio[start:end]


class KokoroTTS:
    def __init__(self, models_dir, backend: str = "auto", default_voice: str = "af_heart", speed: float = 1.0, trim_silence: bool = True):
        self.models_dir = models_dir
        self.default_voice = default_voice
        self.speed = speed
        self.trim_silence = trim_silence
        self.backend = backend

        onnx_model = models_dir / "kokoro-v1.0.onnx"
        onnx_voices = models_dir / "voices-v1.0.bin"

        if backend == "auto":
            backend = "onnx" if onnx_model.exists() and onnx_voices.exists() else "torch"

        if backend == "onnx":
            self._load_onnx(onnx_model, onnx_voices)
        elif backend == "torch":
            self._load_torch()
        else:
            raise ValueError(f"unknown TTS backend: {backend!r}")

    # ---- backends ----------------------------------------------------------
    def _load_onnx(self, model_path, voices_path) -> None:
        from kokoro_onnx import Kokoro

        data_path, lib_path = _setup_espeak()
        kwargs = {}
        try:
            from kokoro_onnx import EspeakConfig  # type: ignore

            if data_path and lib_path:
                kwargs["espeak_config"] = EspeakConfig(data_path=data_path, library_path=lib_path)
        except Exception:
            pass
        log.info("loading Kokoro (onnx) %s", model_path.name)
        self._kokoro = Kokoro(str(model_path), str(voices_path), **kwargs)
        self.backend = "onnx"

    def _load_torch(self) -> None:
        from kokoro import KPipeline  # type: ignore

        _setup_espeak()
        log.info("loading Kokoro (torch)")
        self._pipelines = {}
        self._KPipeline = KPipeline
        self.backend = "torch"

    # ---- synthesis ---------------------------------------------------------
    def synthesize(self, text: str, voice: str | None = None, speed: float | None = None) -> tuple[np.ndarray, int]:
        """Return (float32 mono audio, sample_rate)."""
        voice = voice or self.default_voice
        speed = speed or self.speed
        if self.backend == "onnx":
            audio, sample_rate = self._synth_onnx(text, voice, speed)
        else:
            audio, sample_rate = self._synth_torch(text, voice, speed)
        audio = np.asarray(audio, dtype=np.float32)
        if self.trim_silence:
            audio = _trim_silence(audio, sample_rate)
        return audio, sample_rate

    def _synth_onnx(self, text: str, voice: str, speed: float):
        lang = "en-us" if voice[0] in "ab" else None
        if lang is None:
            try:
                audio, sr = self._kokoro.create(text, voice=voice, speed=speed)
                return audio, sr
            except Exception:
                pass
        return self._kokoro.create(text, voice=voice, speed=speed, lang=lang)

    def _synth_torch(self, text: str, voice: str, speed: float):
        lang_code = voice[0] if voice else "a"
        pipe = self._pipelines.get(lang_code)
        if pipe is None:
            pipe = self._KPipeline(lang_code=lang_code)
            self._pipelines[lang_code] = pipe
        pieces: list[np.ndarray] = []
        for _gs, _ps, audio in pipe(text, voice=voice, speed=speed):
            if audio is not None:
                pieces.append(np.asarray(audio, dtype=np.float32))
        if not pieces:
            return np.zeros(0, dtype=np.float32), 24_000
        return np.concatenate(pieces), 24_000

    def warmup(self) -> None:
        try:
            self.synthesize("Ready.")
        except Exception as exc:  # pragma: no cover
            log.warning("TTS warmup failed: %s", exc)
