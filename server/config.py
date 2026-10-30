"""Central configuration.

Every knob is overridable via an environment variable so you can tune
latency vs. quality without editing code.  See README.md for the whole list.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines into the environment, without overwriting anything.

    A real environment variable always wins, which is what a host like Render
    injects, so the same file works locally and is simply absent in production.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

MODELS_DIR = Path(os.getenv("V2V_MODELS_DIR") or (ROOT / "models")).resolve()

# Audio formats.  The browser always streams mono int16 @ 16 kHz (what Silero VAD
# and Whisper want) and we always synthesise at 24 kHz (what Kokoro outputs).
INPUT_SR = 16_000
OUTPUT_SR = 24_000
VAD_FRAME = 512  # Silero v5 expects 512-sample windows at 16 kHz


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or value.strip() == "" else value.strip()


def _int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _bool(key: str, default: bool) -> bool:
    return _env(key, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


DEFAULT_SYSTEM_PROMPT = (
    "You are Halo, a warm, quick-witted voice companion. You are being spoken to "
    "out loud and your answer is read aloud by a text-to-speech engine, so you must "
    "write only plain spoken English: no markdown, no bullet lists, no emoji, no "
    "asterisks, no stage directions, no code blocks. "
    "Keep answers short and conversational - usually one or two sentences, "
    "unless the user explicitly asks you to go deeper. "
    "Never mention that you are an AI language model unless asked directly."
)

# Kokoro v1.0 voice packs.  First letter = language, second = gender.
VOICES = [
    {"id": "af_heart", "label": "Heart - US female, warm (default)"},
    {"id": "af_bella", "label": "Bella - US female, expressive"},
    {"id": "af_nicole", "label": "Nicole - US female, soft"},
    {"id": "af_sarah", "label": "Sarah - US female, clear"},
    {"id": "af_sky", "label": "Sky - US female, bright"},
    {"id": "am_michael", "label": "Michael - US male, steady (default)"},
    {"id": "am_adam", "label": "Adam - US male"},
    {"id": "am_onyx", "label": "Onyx - US male, deep"},
    {"id": "am_puck", "label": "Puck - US male, playful"},
    {"id": "bf_emma", "label": "Emma - UK female"},
    {"id": "bf_isabella", "label": "Isabella - UK female"},
    {"id": "bm_george", "label": "George - UK male"},
    {"id": "bm_lewis", "label": "Lewis - UK male"},
    {"id": "jf_alpha", "label": "Alpha - Japanese female"},
    {"id": "zf_xiaoxiao", "label": "Xiaoxiao - Chinese female"},
]


@dataclass(frozen=True)
class Settings:
    # ---- server -------------------------------------------------------------
    host: str = _env("V2V_HOST", "127.0.0.1")
    # Render, Fly and Railway all inject PORT. V2V_PORT still wins when both
    # are set, so the local workflow is unchanged.
    port: int = _int("V2V_PORT", _int("PORT", 8000))

    # ---- which backends to use ----------------------------------------------
    # The app ships as a hosted service: a free 512 MB instance cannot hold a
    # 4 GB GPU model, so speech and chat default to free cloud APIs. Set these
    # to "local"/"kokoro" to run the on-device stack instead (see
    # requirements-local.txt).
    stt_backend: str = _env("V2V_STT_BACKEND", "groq").lower()   # groq | local
    llm_backend: str = _env("V2V_LLM_BACKEND", "groq").lower()   # groq | ollama
    tts_engine: str = _env("V2V_TTS_ENGINE", "gtts").lower()     # gtts | edge | kokoro

    # ---- Groq (free tier, no credit card) -----------------------------------
    # One key covers both Whisper and chat: https://console.groq.com/keys
    groq_api_key: str = _env("GROQ_API_KEY", "")
    groq_api_base: str = _env("V2V_GROQ_API_BASE", "https://api.groq.com/openai/v1")
    groq_stt_model: str = _env("V2V_GROQ_STT_MODEL", "whisper-large-v3-turbo")
    # 70b for better prose, 8b-instant for lower latency and higher free limits.
    groq_llm_model: str = _env("V2V_GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
    groq_timeout_s: float = _float("V2V_GROQ_TIMEOUT_S", 30.0)


    # ---- LLM (Ollama) -------------------------------------------------------
    ollama_url: str = _env("V2V_OLLAMA_URL", "http://127.0.0.1:11434")
    # Uncensored abliterated Llama 3.1 8B - the model this app is tuned around.
    # It is also the faster of the two 8B models on this box (15.4 vs 12.7 tok/s).
    # The one model for this build. It is the only installed model that fits
    # entirely in the RX 5500M's 4 GB, so Ollama keeps it 100% on GPU:
    # measured 68.5 tok/s and ~30 ms to first token, vs 8-15 tok/s for the 8B
    # models that have to straddle CPU/GPU.
    #
    # It also must NOT be a "thinking" model. qwen3:4b was tried here and is
    # unusable for voice: it burns the whole num_predict budget reasoning, so
    # with thinking on it says nothing, and with `think:false` it just relabels
    # its chain of thought as content (41 s, no answer, and Halo would speak
    # the reasoning out loud). qwen2.5 answers directly.
    llm_model: str = _env("V2V_LLM_MODEL", "qwen2.5:3b-instruct")
    llm_temperature: float = _float("V2V_LLM_TEMPERATURE", 0.7)
    llm_top_p: float = _float("V2V_LLM_TOP_P", 0.9)
    llm_max_tokens: int = _int("V2V_LLM_MAX_TOKENS", 220)
    # "-1" = never evict. This is the single biggest latency decision in the app:
    # Ollama charges 17.5 s to reload an 8B model from disk after idle, and 30
    # minutes is a long conversation gap. Always resident => warm TTFT ~100 ms
    # on every single turn. Costs ~5.4 GB of RAM while it sits there.
    llm_keep_alive: str = _env("V2V_LLM_KEEP_ALIVE", "-1")
    llm_num_ctx: int = _int("V2V_LLM_NUM_CTX", 4096)
    # ASK for non-thinking mode (Ollama ignores/errors it on non-thinking models;
    # we transparently retry without the flag if the server rejects it).
    llm_no_think: bool = _bool("V2V_LLM_NO_THINK", True)
    system_prompt: str = _env("V2V_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    history_turns: int = _int("V2V_HISTORY_TURNS", 12)

    # ---- STT ----------------------------------------------------------------
    # On this class of CPU, faster-whisper cost is dominated by the fixed 30 s
    # mel encode, so model size - not utterance length - decides latency.
    # Measured on a Ryzen 5 5600H (6 threads, int8, 3.9 s clip):
    #   tiny.en 0.10 RTF | base.en 0.23 | distil-small.en 0.54 | small 0.62
    #   large-v3-turbo 2.61 (unusable).
    #
    # Accuracy first: base.en mishears ordinary conversational speech, and those
    # errors are the ones the LLM cannot recover from. distil-small.en is
    # roughly small.en quality at about half the cost, and the app is English-
    # only anyway, so it is the right default for a 6-core laptop.
    whisper_model: str = _env("V2V_WHISPER_MODEL", "Systran/faster-distil-whisper-small.en")
    # The live "as you speak" transcript only has to look right, not be right,
    # so it keeps running on the cheap base model. A separate instance means a
    # slow partial can never sit in front of the authoritative transcription.
    whisper_partial_model: str = _env(
        "V2V_WHISPER_PARTIAL_MODEL", "Systran/faster-whisper-base.en"
    )
    whisper_device: str = _env("V2V_WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = _env("V2V_WHISPER_COMPUTE_TYPE", "int8")
    # "en" skips language detection, which saves a little time and removes a
    # nonsense generator on very short clips. "" = auto-detect (only useful
    # with a multilingual model).
    whisper_language: str = _env("V2V_WHISPER_LANGUAGE", "en")
    # Beam search on the final pass - a real accuracy win over greedy decoding.
    # Drop to 3 if finals feel slow on a busy machine.
    whisper_beam: int = _int("V2V_WHISPER_BEAM", 5)
    whisper_threads: int = _int("V2V_WHISPER_THREADS", max(2, (os.cpu_count() or 4) // 2))
    whisper_initial_prompt: str = _env(
        "V2V_WHISPER_INITIAL_PROMPT", "A casual spoken conversation. Use punctuation."
    )

    # ---- VAD (endpointing) --------------------------------------------------
    vad_threshold: float = _float("V2V_VAD_THRESHOLD", 0.5)
    vad_neg_threshold: float = _float("V2V_VAD_NEG_THRESHOLD", 0.35)
    vad_min_speech_ms: int = _int("V2V_VAD_MIN_SPEECH_MS", 220)
    # 400 ms is the usual Silero default, but measured here it splits an
    # ordinary two-sentence utterance at the breath between sentences and
    # throws away the second half. 500 ms keeps the sentence together for
    # 100 ms more endpointing delay - a clear net accuracy win.
    vad_min_silence_ms: int = _int("V2V_VAD_MIN_SILENCE_MS", 500)
    vad_speech_pad_ms: int = _int("V2V_VAD_SPEECH_PAD_MS", 160)
    vad_max_utterance_s: float = _float("V2V_VAD_MAX_UTTERANCE_S", 30.0)

    # ---- streaming partials (live "as you speak" transcript) ----------------
    partial_interval_s: float = _float("V2V_PARTIAL_INTERVAL_S", 0.7)
    partial_min_audio_s: float = _float("V2V_PARTIAL_MIN_AUDIO_S", 0.9)
    partial_max_audio_s: float = _float("V2V_PARTIAL_MAX_AUDIO_S", 14.0)

    # ---- TTS (Kokoro / gTTS / Edge) -----------------------------------------
    # Kokoro sub-backend when tts_engine == "kokoro": auto | onnx | torch
    tts_backend: str = _env("V2V_TTS_BACKEND", "auto")
    # gTTS: lang is the spoken language, tld picks the accent
    # (com = US, co.uk = UK, com.au = AU, ie = IE, ca = CA, co.in = IN).
    gtts_lang: str = _env("V2V_GTTS_LANG", "en")
    gtts_tld: str = _env("V2V_GTTS_TLD", "com")
    # Edge: any https://speech.platform.bing.com voice name.
    edge_voice: str = _env("V2V_EDGE_VOICE", "en-US-AriaNeural")
    tts_voice: str = _env("V2V_TTS_VOICE", "af_heart")
    tts_speed: float = _float("V2V_TTS_SPEED", 1.0)
    tts_lang: str = _env("V2V_TTS_LANG", "en-us")
    # Text is flushed to the TTS engine at these boundaries.  Small = lower
    # latency to first audio, too small = choppy prosody.  10 still covers the
    # first short clause of a reply, which is what decides perceived speed.
    tts_min_chars: int = _int("V2V_TTS_MIN_CHARS", 10)
    tts_max_chars: int = _int("V2V_TTS_MAX_CHARS", 150)
    # First chunk only: speak as soon as this many characters exist rather than
    # waiting for the first comma/period.  0 = wait for real punctuation.
    tts_first_chars: int = _int("V2V_TTS_FIRST_CHARS", 40)
    tts_trim_silence: bool = _bool("V2V_TTS_TRIM_SILENCE", True)

    def public(self) -> dict:
        data = asdict(self)
        # Never ship the key to the browser, even though it is only a free one.
        data.pop("groq_api_key", None)
        data["input_sample_rate"] = INPUT_SR
        data["output_sample_rate"] = OUTPUT_SR
        # gTTS and Edge hand back MP3, Kokoro hands back raw PCM. The client
        # needs to know which before the first audio_start arrives.
        data["audio_format"] = "pcm" if self.tts_engine == "kokoro" else "mp3"
        data["voices"] = VOICES if self.tts_engine == "kokoro" else []
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
