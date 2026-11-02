"""End-to-end phone test: imitates a carrier's media stream.

    python -m server.phonetest --wav me.wav            # fake Plivo call
    python -m server.phonetest --wav me.wav --provider twilio

It opens the same websocket a carrier opens, sends the recording as 20 ms
mu-law frames in real time (exactly what Plivo and Twilio do), and decodes the
bot's audio back into `phone-reply.wav` at 8 kHz so you can hear the call.

No telephone network is involved, which is the point: this isolates our side
from the carrier side, so a bad call can be blamed correctly.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import wave
from pathlib import Path

import numpy as np

from .config import INPUT_SR, ROOT, get_settings
from .phone import PHONE_SR, resample, ulaw_decode, ulaw_encode


def read_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise SystemExit("only 16-bit PCM wav is supported")
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != INPUT_SR:
        data = resample(data, rate, INPUT_SR)
    return data


async def run(args: argparse.Namespace) -> int:
    import websockets

    settings = get_settings()
    token = args.token if args.token is not None else settings.phone_token
    url = args.url
    if "token=" not in url and token:
        url += f"&token={token}"

    speech = read_wav_16k(Path(args.wav))
    audio = np.concatenate(
        [
            np.zeros(int(0.4 * INPUT_SR), np.float32),
            speech,
            np.zeros(int(2.0 * INPUT_SR), np.float32),
        ]
    )
    mu = ulaw_encode(resample(audio, INPUT_SR, PHONE_SR).astype(np.int16))
    frame = PHONE_SR * 20 // 1000           # 160 bytes == 20 ms
    frames = [mu[i : i + frame] for i in range(0, len(mu), frame)]

    print(f"  caller audio: {audio.size / INPUT_SR:.2f}s -> {len(frames)} frames of 20 ms")
    print(f"  connecting to {url}\n")

    reply = bytearray()
    t_start = time.perf_counter()
    t_first_audio = None
    last_audio = None

    async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:
        stream_id = "test-stream"
        await ws.send(
            json.dumps(
                {
                    "event": "start",
                    "sequenceNumber": 1,
                    "start": {
                        "callId": "test-call",
                        "streamId": stream_id,
                        "accountId": "test",
                        "tracks": ["inbound"],
                        "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": PHONE_SR},
                    },
                    "extra_headers": "",
                }
            )
        )

        async def receiver() -> None:
            nonlocal t_first_audio, last_audio
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                event = msg.get("event")
                if event in ("playAudio", "media"):
                    media = msg.get("media") or {}
                    payload = media.get("payload")
                    if not payload:
                        continue
                    chunk = base64.b64decode(payload)
                    if t_first_audio is None:
                        t_first_audio = time.perf_counter() - t_start
                        print(f"  first audio from bot after {t_first_audio:.2f}s")
                    reply.extend(chunk)
                    last_audio = time.perf_counter()
                elif event in ("clearedAudio", "clear"):
                    print("  carrier acknowledged clear (barge-in)")
                elif event == "stop":
                    return

        recv_task = asyncio.create_task(receiver())

        next_at = time.perf_counter()
        for i, chunk in enumerate(frames):
            if recv_task.done():
                break
            await ws.send(
                json.dumps(
                    {
                        "event": "media",
                        "sequenceNumber": i + 2,
                        "streamId": stream_id,
                        "media": {
                            "track": "inbound",
                            "timestamp": str(int((i * 20))),
                            "chunk": i,
                            "payload": base64.b64encode(chunk).decode("ascii"),
                        },
                        "extra_headers": "",
                    }
                )
            )
            next_at += 0.02
            delay = next_at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

        print("  caller finished speaking, waiting for the reply to drain")
        deadline = time.perf_counter() + args.wait
        while time.perf_counter() < deadline:
            if last_audio and time.perf_counter() - last_audio > 1.5 and t_first_audio:
                break
            await asyncio.sleep(0.1)

        await ws.send(json.dumps({"event": "stop", "streamId": stream_id}))
        await asyncio.sleep(0.3)
        recv_task.cancel()

    if not reply:
        print("\n  NO AUDIO CAME BACK. Check the server log for the caller transcript.")
        return 1

    out = ROOT / "phone-reply.wav"
    with wave.open(str(out), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(PHONE_SR)
        handle.writeframes(np.frombuffer(ulaw_decode(bytes(reply)).tobytes(), dtype=np.int16).tobytes())

    peak = int(np.abs(ulaw_decode(bytes(reply))).max())
    print(
        f"\n  reply: {len(reply) / PHONE_SR:.2f}s  first audio {t_first_audio:.2f}s  peak {peak}"
        f"\n  wrote {out.name} (8 kHz, listen to it)"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    settings = get_settings()
    parser.add_argument("--wav", required=True, help="16 kHz mono recording of the caller")
    parser.add_argument(
        "--url",
        default=f"ws://127.0.0.1:{settings.port}/ws/phone?provider=plivo",
        help="media websocket (default: local Plivo-shaped endpoint)",
    )
    parser.add_argument("--provider", choices=["plivo", "twilio"], default="plivo")
    parser.add_argument("--token", default=None, help="phone token, defaults to V2V_PHONE_TOKEN")
    parser.add_argument("--wait", type=float, default=25.0, help="seconds to wait for the reply")
    args = parser.parse_args()
    args.url = args.url.replace("{provider}", args.provider)
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
