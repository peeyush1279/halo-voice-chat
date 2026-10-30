"""Speech to text.

Two interchangeable engines behind the same `transcribe()` call:

* `Transcriber`     - faster-whisper on the local CPU (CTranslate2, no torch).
* `GroqTranscriber` - Whisper large-v3-turbo on Groq's free tier. Used when the
                      app is hosted, where there is no CPU budget for a local
                      model.
"""

from __future__ import annotations

import io
import logging
import time
import wave

import numpy as np

log = logging.getLogger("v2v.stt")


def wav_bytes(audio: np.ndarray, sample_rate: int = 16_000) -> bytes:
    """Wrap mono float32 [-1, 1] as a 16-bit PCM WAV, in memory."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buf.getvalue()



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


class GroqTranscriber:
    """Whisper on Groq's free tier. Same interface as `Transcriber`.

    Free tier is rate-limited per minute and per day, which is fine for one
    final transcript per turn and hopeless for the live partial transcript, so
    partials are switched off rather than spending the quota on text we throw
    away.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        language: str = "en",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 30.0,
    ):
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and set it in the environment."
            )
        import httpx  # imported here so local-only installs do not need it at import time

        self._httpx = httpx
        self.api_key = api_key
        self.model = model
        self.language = language or None
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_rtf: float | None = None

    def transcribe(self, audio: np.ndarray, *, partial: bool = False) -> str:
        if partial:
            return ""
        if audio is None or audio.size < 1600:  # < 100 ms
            return ""

        started = time.perf_counter()
        data = {"model": self.model, "response_format": "json", "temperature": "0"}
        if self.language:
            data["language"] = self.language

        try:
            response = self._httpx.post(
                f"{self.base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": ("speech.wav", wav_bytes(audio), "audio/wav")},
                data=data,
                timeout=self.timeout,
            )
        except self._httpx.HTTPError as exc:
            raise RuntimeError(f"cannot reach Groq speech-to-text ({exc})") from exc

        if response.status_code == 401:
            raise RuntimeError("Groq rejected the API key (401). Check GROQ_API_KEY.")
        if response.status_code == 429:
            raise RuntimeError(
                "Groq speech-to-text rate limit hit (429). Wait a few seconds "
                "and try again."
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Groq speech-to-text returned {response.status_code}: {response.text[:200]}"
            )

        text = (response.json().get("text") or "").strip()
        duration = audio.size / 16_000
        if duration > 0:
            self.last_rtf = (time.perf_counter() - started) / duration
        return text

    def warmup(self) -> None:
        # No warmup request: it would cost a call against a 30/minute quota.
        pass
