"""Benchmark faster-whisper models on *this* CPU, to pick a good default.

    .\\.venv\\Scripts\\python.exe bench_whisper.py
    .\\.venv\\Scripts\\python.exe bench_whisper.py --models Systran/faster-whisper-tiny.en
    .\\.venv\\Scripts\\python.exe bench_whisper.py --wav me.wav        # your own 16 kHz mono clip

Whisper always encodes a fixed 30 s mel window, so on a CPU the *model size*,
not how long you spoke, decides latency.  The number that matters is the
real-time factor (RTF): compute-seconds per second of audio.  Aim for < 0.5.

Results on a Ryzen 5 5600H (6 threads, int8, 3.9 s clip) are in README.md.
"""

from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np

from server.config import MODELS_DIR, INPUT_SR

DEFAULT_MODELS = [
    "Systran/faster-whisper-tiny.en",
    "Systran/faster-whisper-base.en",
    "Systran/faster-distil-whisper-small.en",
    "Systran/faster-whisper-small",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
]

SENTENCE = "Hey, can you hear me clearly? I am testing the voice pipeline right now."


def load_clip(wav: str | None) -> np.ndarray:
    """A mono float32 @ 16 kHz test clip: --wav, or Kokoro speaking SENTENCE."""
    if wav:
        with wave.open(wav, "rb") as handle:
            if handle.getsampwidth() != 2:
                raise SystemExit("only 16-bit PCM wav is supported")
            data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
            data = data.astype(np.float32) / 32768.0
            if handle.getnchannels() > 1:
                data = data.reshape(-1, handle.getnchannels()).mean(axis=1)
            if handle.getframerate() != INPUT_SR:
                out = int(data.size * INPUT_SR / handle.getframerate())
                data = np.interp(np.linspace(0, data.size - 1, out), np.arange(data.size), data)
            return data.astype(np.float32)

    from server.tts import KokoroTTS

    print("  synthesising the test clip with Kokoro...")
    tts = KokoroTTS(MODELS_DIR, backend="onnx")
    audio, rate = tts.synthesize(SENTENCE)
    if rate != INPUT_SR:
        out = int(audio.size * INPUT_SR / rate)
        audio = np.interp(np.linspace(0, audio.size - 1, out), np.arange(audio.size), audio)
    return audio.astype(np.float32)


def bench(model_name: str, clip: np.ndarray, dur: float, compute_type: str, threads: int, runs: int):
    """Return (best_seconds, load_seconds, text) or (None, None, error)."""
    from faster_whisper import WhisperModel

    try:
        t = time.perf_counter()
        model = WhisperModel(model_name, device="cpu", compute_type=compute_type, cpu_threads=threads)
        load = time.perf_counter() - t
        list(model.transcribe(clip, beam_size=1)[0])  # warm up / build kernels
        best, text = None, ""
        for _ in range(runs):
            t = time.perf_counter()
            text = " ".join(
                s.text
                for s in model.transcribe(
                    clip,
                    beam_size=1,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                    language="en",
                )[0]
            ).strip()
            elapsed = time.perf_counter() - t
            best = elapsed if best is None else min(best, elapsed)
        del model
        return best, load, text
    except Exception as exc:  # noqa: BLE001 - report and keep going
        return None, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="benchmark faster-whisper on this CPU")
    parser.add_argument("--models", nargs="*", default=None, help="model names to compare")
    parser.add_argument("--wav", default=None, help="16 kHz mono wav to use instead of Kokoro")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--thread-scaling", action="store_true", help="also sweep cpu_threads")
    args = parser.parse_args()

    clip = load_clip(args.wav)
    dur = clip.size / INPUT_SR
    models = args.models or DEFAULT_MODELS
    print(f"clip: {dur:.2f}s of speech, compute_type={args.compute_type}\n")

    if args.thread_scaling:
        print("thread scaling (first model):")
        for threads in (1, 2, 3, 6, 12):
            sec, _load, text = bench(models[0], clip, dur, args.compute_type, threads, runs=1)
            if sec is None:
                print(f"  threads={threads:2d}  FAILED: {text}")
            else:
                print(f"  threads={threads:2d}  {sec * 1000:7.0f} ms  RTF {sec / dur:5.2f}")
        print()

    print(f"models ({args.threads} threads):")
    results = []
    for name in models:
        sec, load, text = bench(name, clip, dur, args.compute_type, args.threads, args.runs)
        if sec is None:
            print(f"  {name:52s} FAILED: {text[:44]}")
            continue
        results.append((sec / dur, name))
        print(f"  {name:52s} {sec * 1000:7.0f} ms  RTF {sec / dur:5.2f}  load {load:5.1f}s")
        print(f"       {text[:70]!r}")

    if results:
        results.sort()
        print("\nfastest that still reads correctly:")
        for rtf, name in results:
            verdict = "GOOD" if rtf < 0.5 else ("ok" if rtf < 1.0 else "too slow")
            print(f"  RTF {rtf:5.2f}  {verdict:8s}  {name}")
        print(f"\nset the winner with  $env:V2V_WHISPER_MODEL=\"{results[0][1]}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
