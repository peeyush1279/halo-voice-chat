"""End-to-end test: talks to the running server exactly like the browser does.

    python -m server.wstest                 # synthesise a sentence, feed it in
    python -m server.wstest --wav me.wav    # use your own recording (16 kHz mono)

It streams int16 PCM at 20 ms per frame over the websocket, prints every
control event it receives, and writes the assistant's spoken reply to
`reply.wav` in the project root so you can listen to it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np

from .config import INPUT_SR, MODELS_DIR, ROOT, get_settings


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise SystemExit("only 16-bit PCM wav is supported")
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != INPUT_SR:
        out_len = int(data.size * INPUT_SR / rate)
        data = np.interp(np.linspace(0, data.size - 1, out_len), np.arange(data.size), data).astype(np.float32)
    return data


def make_speech(text: str) -> np.ndarray:
    try:
        from .tts import KokoroTTS
    except Exception as exc:
        raise SystemExit(
            "  Kokoro is not installed (this is the hosted stack), so there is no\n"
            "  local voice to synthesise a test input with. Pass a 16 kHz mono WAV:\n"
            "    python -m server.wstest --wav me.wav"
        ) from exc

    settings = get_settings()
    print("  synthesising test input with Kokoro…")
    tts = KokoroTTS(MODELS_DIR, backend=settings.tts_backend, default_voice=settings.tts_voice)
    audio, rate = tts.synthesize(text)
    if rate != INPUT_SR:
        out_len = int(audio.size * INPUT_SR / rate)
        audio = np.interp(np.linspace(0, audio.size - 1, out_len), np.arange(audio.size), audio).astype(np.float32)
    return audio.astype(np.float32)


async def run(args: argparse.Namespace) -> int:
    import websockets

    settings = get_settings()
    url = args.url
    silence_head = np.zeros(int(0.3 * INPUT_SR), dtype=np.float32)
    silence_tail = np.zeros(int(1.0 * INPUT_SR), dtype=np.float32)

    speech = read_wav(Path(args.wav)) if args.wav else make_speech(args.text)
    audio = np.concatenate([silence_head, speech, silence_tail])
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    frame = int(INPUT_SR * 0.02)  # 20 ms

    print(f"  input: {speech.size / INPUT_SR:.2f}s of speech -> {pcm.size / INPUT_SR:.2f}s of stream")
    print(f"  connecting to {url}\n")

    reply_audio: list[np.ndarray] = []
    reply_mp3 = bytearray()
    fmt = "pcm"
    rate = 24_000
    done = asyncio.Event()
    t_sent_end = None

    async with websockets.connect(url, max_size=None) as ws:
        async def sender() -> None:
            nonlocal t_sent_end
            for i in range(0, pcm.size, frame):
                await ws.send(pcm[i : i + frame].tobytes())
                await asyncio.sleep(0.02)
            t_sent_end = time.perf_counter()
            print("  [sent] end of speech")

        async def receiver() -> None:
            nonlocal rate, fmt
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    if fmt == "mp3":
                        reply_mp3.extend(message)
                    else:
                        reply_audio.append(np.frombuffer(message, dtype=np.int16))
                    continue
                event = json.loads(message)
                kind = event.get("type")
                if kind == "ready":
                    print(f"  [ready] model={event['config']['llm_model']} voice={event['config']['voice']}")
                    asyncio.create_task(sender())
                elif kind == "speech_start":
                    print("  [vad  ] speech started")
                elif kind == "speech_end":
                    print("  [vad  ] speech ended -> transcribing")
                elif kind == "partial":
                    print(f"  [part ] {event['text']}")
                elif kind == "user":
                    print(f"  [user ] {event['text']}")
                elif kind == "assistant_start":
                    print("  [llm  ] generating…")
                elif kind == "assistant_delta":
                    print(event["text"], end="", flush=True)
                elif kind == "assistant_done":
                    print("\n  [llm  ] done")
                elif kind == "audio_start":
                    rate = event.get("sample_rate", rate)
                    fmt = event.get("format", "pcm")
                    if t_sent_end:
                        print(f"  [tts  ] first audio {time.perf_counter() - t_sent_end:.2f}s after I stopped talking")
                    print(f"  [tts  ] speaking {event['text']!r}")
                elif kind == "timing":
                    print(
                        f"  [time ] stt {event['stt_ms']} ms | first token {event['ttft_ms']} ms | "
                        f"first audio {event['tts_ms']} ms | total {event['total_ms']} ms"
                    )
                elif kind == "idle":
                    print("  [idle ] back to listening\n")
                    done.set()
                    return
                elif kind == "error":
                    print(f"  [ERR  ] {event['message']}")
                elif kind == "cancel":
                    print(f"  [cancel] {event.get('reason')}")

        receiver_task = asyncio.create_task(receiver())
        try:
            await asyncio.wait_for(done.wait(), timeout=args.timeout)
        except asyncio.TimeoutError:
            print(f"\n  timed out after {args.timeout}s")
        receiver_task.cancel()

    if reply_mp3:
        out_path = ROOT / "reply.mp3"
        out_path.write_bytes(bytes(reply_mp3))
        print(f"  wrote {len(reply_mp3) / 1024:.0f} KB of reply audio to {out_path}")
        return 0
    if reply_audio:
        audio_out = np.concatenate(reply_audio).astype(np.float32) / 32768.0
        out_path = ROOT / "reply.wav"
        with wave.open(str(out_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes((np.clip(audio_out, -1, 1) * 32767).astype(np.int16).tobytes())
        print(f"  wrote {audio_out.size / rate:.2f}s of reply audio to {out_path}")
    else:
        print("  no reply audio received")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="end-to-end websocket test")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    parser.add_argument("--wav", default=None, help="16 kHz mono wav to use instead of synthesising")
    parser.add_argument("--text", default="Hey Halo, can you hear me? Tell me something interesting in one sentence.")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
