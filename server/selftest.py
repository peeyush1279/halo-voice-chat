"""Component smoke test - no browser, no server.

Run:  python -m server.selftest

It exercises every stage of the pipeline individually and prints the numbers
that matter for latency:

  1. Silero VAD loads
  2. Kokoro synthesises speech (and reports real-time factor)
  3. Its own audio is fed back through the VAD + Whisper (round-trip check)
  4. Ollama streams a reply (time to first token)

Whisper weights (~0.8-1.5 GB) download on the first run.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from .config import MODELS_DIR, get_settings
from .textchunk import SpeechChunker, clean_for_speech
from .tts import KokoroTTS
from .vad import SileroVAD, TurnDetector

PHRASE = "Hey, can you hear me clearly? I am testing the voice pipeline right now."


def hr(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main() -> int:
    settings = get_settings()
    ok = True

    # ---------------------------------------------------------------- chunker
    hr("1. Chunker (how fast we hand text to TTS)")
    chunker = SpeechChunker(settings.tts_min_chars, settings.tts_max_chars)
    pieces = ["Sure, ", "I can ", "do that. ", "What else ", "would you ", "like ", "to know?"]
    out: list[str] = []
    for piece in pieces:
        out.extend(chunker.push(piece))
    out.extend(chunker.flush())
    for i, chunk in enumerate(out):
        print(f"   chunk {i}: {chunk!r}")
    print(f"   markdown strip: {clean_for_speech('**Bold** and `code` and [link](x) 🙂 (*smiles*)')!r}")

    # -------------------------------------------------------------------- VAD
    hr("2. Silero VAD")
    try:
        t = time.perf_counter()
        vad = SileroVAD(MODELS_DIR / "silero_vad.onnx")
        silence = np.zeros(512, dtype=np.float32)
        probs = [vad(silence) for _ in range(5)]
        print(f"   loaded in {(time.perf_counter() - t) * 1000:.0f} ms")
        print(f"   p(speech) on digital silence: {max(probs):.4f}  (want ~0)")
    except Exception as exc:
        ok = False
        print(f"   FAILED: {exc}")
        vad = None

    # -------------------------------------------------------------------- TTS
    hr("3. Kokoro TTS")
    tts = None
    audio = None
    try:
        t = time.perf_counter()
        tts = KokoroTTS(
            models_dir=MODELS_DIR,
            backend=settings.tts_backend,
            default_voice=settings.tts_voice,
            speed=settings.tts_speed,
        )
        load_s = time.perf_counter() - t
        t = time.perf_counter()
        audio, sample_rate = tts.synthesize("Ready when you are.")
        warm = time.perf_counter() - t
        t = time.perf_counter()
        audio, sample_rate = tts.synthesize(PHRASE)
        synth_s = time.perf_counter() - t
        dur = audio.size / sample_rate
        print(f"   backend={tts.backend}  loaded in {load_s:.1f}s")
        print(f"   1st call (warmup+g2p): {warm * 1000:.0f} ms")
        print(f"   '{PHRASE[:34]}...'")
        print(f"   -> {dur:.2f}s of audio in {synth_s * 1000:.0f} ms  (RTF {synth_s / dur:.2f}, lower is better)")
        print(f"   peak amplitude {np.abs(audio).max():.3f}, sample rate {sample_rate}")
        if synth_s / dur > 1.0:
            print("   WARNING: slower than real time - use a smaller Whisper model or close other apps")
    except Exception as exc:
        ok = False
        print(f"   FAILED: {exc}")
        audio = None

    # ------------------------------------------------------ 16 kHz reference
    # Everything downstream (VAD, Whisper) works at 16 kHz mono float32, so
    # resample Kokoro's 24 kHz output once and reuse it.
    speech16 = None
    if audio is not None:
        target = int(audio.size * 16_000 / sample_rate)
        speech16 = np.interp(
            np.linspace(0, audio.size - 1, target), np.arange(audio.size), audio
        ).astype(np.float32)

    # ------------------------------------------------------- VAD round trip
    if vad is not None and speech16 is not None:
        hr("4. Endpointing (VAD on real speech, in 32 ms frames)")
        detector = TurnDetector(
            vad=SileroVAD(MODELS_DIR / "silero_vad.onnx"),
            sample_rate=16_000,
            threshold=settings.vad_threshold,
            neg_threshold=settings.vad_neg_threshold,
            min_speech_s=settings.vad_min_speech_ms / 1000,
            min_silence_s=settings.vad_min_silence_ms / 1000,
            pad_s=settings.vad_speech_pad_ms / 1000,
        )
        # Append trailing silence, exactly like the browser stream continues to
        # send silence after you stop talking.
        mono = np.concatenate([np.zeros(int(0.2 * 16_000), np.float32), speech16, np.zeros(16_000, np.float32)])

        events = []
        t = time.perf_counter()
        for i in range(0, mono.size, 512):
            events.extend(detector.push(mono[i : i + 512]))
        took = time.perf_counter() - t
        realtime = mono.size / 16_000
        print(f"   scanned {realtime:.2f}s of audio in {took * 1000:.0f} ms ({realtime / took:.0f}x realtime)")
        kinds = [e.kind for e in events]
        print(f"   events: {kinds}")
        for event in events:
            if event.audio is not None:
                print(f"     end -> {event.audio.size / 16_000:.2f}s utterance captured")
        if kinds != ["start", "confirm", "end"]:
            print("   WARNING: expected exactly [start, confirm, end]")
            ok = False

    # -------------------------------------------------------------------- STT
    hr("5. faster-whisper")
    transcriber = None
    if speech16 is not None:
        try:
            from .stt import Transcriber

            t = time.perf_counter()
            transcriber = Transcriber(
                model_name=settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                language=settings.whisper_language,
                beam_size=settings.whisper_beam,
                cpu_threads=settings.whisper_threads,
                initial_prompt=settings.whisper_initial_prompt,
            )
            print(f"   loaded in {time.perf_counter() - t:.1f}s  ({settings.whisper_model} on {settings.whisper_device})")
            transcriber.transcribe(speech16)  # warm up
            t = time.perf_counter()
            heard = transcriber.transcribe(speech16)
            stt_s = time.perf_counter() - t
            dur = speech16.size / 16_000
            print(f"   said : {PHRASE}")
            print(f"   heard: {heard}")
            print(f"   transcribe took {stt_s * 1000:.0f} ms for {dur:.2f}s audio (RTF {stt_s / dur:.2f}, want < 1.0)")
            if stt_s / dur > 1.0:
                ok = False
                print("   TOO SLOW: pick a smaller V2V_WHISPER_MODEL (see README)")
        except Exception as exc:
            ok = False
            print(f"   FAILED: {exc}")

    # -------------------------------------------------------------------- LLM
    hr("6. Ollama streaming")
    try:
        import asyncio

        from .llm import OllamaClient

        async def probe() -> None:
            client = OllamaClient(
                base_url=settings.ollama_url,
                model=settings.llm_model,
                system_prompt=settings.system_prompt,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                keep_alive=settings.llm_keep_alive,
                no_think=settings.llm_no_think,
            )
            models = await client.list_models()
            print(f"   installed models: {[m.get('name') for m in models]}")
            if settings.llm_model not in [m.get("name") for m in models]:
                print(f"   WARNING: {settings.llm_model!r} is not pulled. Run: ollama pull {settings.llm_model}")
                await client.aclose()
                return

            messages = [
                {"role": "system", "content": settings.system_prompt},
                {"role": "user", "content": "In one short sentence, say hello and ask how my day is going."},
            ]
            await client.warmup()
            t = time.perf_counter()
            first = None
            tokens = []
            async for piece in client.stream_chat(messages):
                if first is None:
                    first = time.perf_counter() - t
                tokens.append(piece)
            total = time.perf_counter() - t
            reply = "".join(tokens)
            print(f"   model: {settings.llm_model}")
            print(f"   first token in {first * 1000:.0f} ms" if first else "   no tokens!")
            print(f"   full reply in {total:.2f}s: {reply[:140]!r}")
            await client.aclose()

        asyncio.run(probe())
    except Exception as exc:
        ok = False
        print(f"   FAILED: {exc}")

    hr("Summary")
    print("   all components OK" if ok else "   some components failed - see above")
    print("\n   Next:  python -m server.main    then open http://127.0.0.1:8000")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
