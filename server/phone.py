"""Answer real phone calls with the same conversation engine.

A carrier hands a live call to us as a websocket of 8 kHz mono mu-law frames -
the native telephone format.  Everything else in this project speaks 16 kHz
PCM16, so this module is the adapter:

    mu-law 8 kHz  ->  PCM16 16 kHz  ->  Conversation (VAD/STT/LLM/TTS)
    PCM16 24 kHz  <-  TTS             ->  mu-law 8 kHz -> paced frames

The adapter implements the three methods `Conversation` uses on a websocket
(`receive`, `send_text`, `send_bytes`), so the pipeline does not know a phone
is on the other end and stays identical for the browser.

Two carriers are supported because they share the same shape:

* Plivo  - `start` / `media` in, `playAudio` / `clearAudio` out.
* Twilio - `connected` / `start` / `media` in, `media` / `clear` out.

Both are mu-law 8 kHz, both need us to pace outbound audio in 20 ms frames.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from xml.sax.saxutils import escape

import numpy as np

from .config import INPUT_SR

log = logging.getLogger("v2v.phone")

PHONE_SR = 8000
FRAME_MS = 20
FRAME_SAMPLES = PHONE_SR * FRAME_MS // 1000        # 160 samples == 160 mu-law bytes
OUTPUT_RATE_HINT = 24_000


# --------------------------------------------------------------------- codecs
def _ulaw_decode_table() -> np.ndarray:
    """G.711 mu-law -> linear16, the usual 256-entry expansion."""
    table = np.zeros(256, dtype=np.int16)
    for i in range(256):
        u = (~i) & 0xFF
        t = ((u & 0x0F) << 3) + 0x84
        t <<= (u & 0x70) >> 4
        t -= 0x84
        table[i] = -t if (u & 0x80) else t
    return table


_ULAW_DECODE = _ulaw_decode_table()


def _ulaw_encode_table() -> np.ndarray:
    """Nearest-code map for every int16 value.

    Built from the decode table instead of re-deriving the bit twiddling, so
    encode and decode can never disagree - a mismatch here is inaudible until
    you are on a call, which is exactly the wrong place to find out.
    """
    values = _ULAW_DECODE.astype(np.int32)
    order = np.argsort(values)
    ordered = values[order]

    xs = np.arange(-32768, 32768, dtype=np.int32)
    idx = np.clip(np.searchsorted(ordered, xs), 1, len(ordered) - 1)
    lo, hi = ordered[idx - 1], ordered[idx]
    pick_high = (hi - xs) < (xs - lo)
    return np.where(pick_high, order[idx], order[idx - 1]).astype(np.uint8)


_ULAW_ENCODE = _ulaw_encode_table()


def ulaw_decode(data: bytes) -> np.ndarray:
    """bytes of mu-law -> int16 samples."""
    return _ULAW_DECODE[np.frombuffer(data, dtype=np.uint8)]


def ulaw_encode(pcm: np.ndarray) -> bytes:
    """int16 samples -> bytes of mu-law."""
    x = np.clip(np.asarray(pcm, dtype=np.int32), -32768, 32767) + 32768
    return _ULAW_ENCODE[x].tobytes()


def _lowpass(x: np.ndarray, rate: float, cutoff: float = 3400.0) -> np.ndarray:
    """Windowed-sinc low pass, applied before decimating.

    Telephone audio is band limited to ~3.4 kHz; without this, 24 kHz speech
    folded down to 8 kHz aliases into a metallic hiss that costs nothing to
    avoid (63 taps, a few milliseconds per clause).
    """
    if x.size < 64 or cutoff >= rate * 0.45:
        return x
    taps = 63
    k = np.arange(taps) - (taps - 1) / 2
    fc = cutoff / rate
    h = 2 * fc * np.sinc(2 * fc * k) * np.hamming(taps)
    h /= h.sum()
    return np.convolve(x, h, mode="same").astype(np.float32)


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear resample with an anti-alias filter when going down."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or src == dst:
        return x
    if dst < src:
        x = _lowpass(x, src)
    n = int(round(x.size * dst / src))
    if n <= 1:
        return x[:n]
    pos = np.arange(n, dtype=np.float64) * (src / dst)
    return np.interp(pos, np.arange(x.size, dtype=np.float64), x).astype(np.float32)


def decode_tts_mp3(data: bytes) -> np.ndarray:
    """An MP3 clause from gTTS/Edge -> int16 8 kHz mono, ready for mu-law."""
    import miniaudio  # imported lazily: the phone path is optional

    decoded = miniaudio.decode(
        data,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=OUTPUT_RATE_HINT,
    )
    pcm = np.frombuffer(decoded.samples, dtype=np.int16)
    return resample(pcm, decoded.sample_rate, PHONE_SR).astype(np.int16)


# ------------------------------------------------------------------- the call
class PhoneSocket:
    """The slice of `starlette.WebSocket` that `Conversation` actually uses."""

    def __init__(self, provider: str, send, max_seconds: float = 600.0):
        self.provider = provider
        self._send_raw = send                    # async (dict) -> None, serialised
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._out: asyncio.Queue = asyncio.Queue()
        self._player: asyncio.Task | None = None
        self._gen = 0                            # bumps on barge-in; stale frames drop
        self._next_at = 0.0
        self._fmt: str | None = None
        self._rate = OUTPUT_RATE_HINT
        self._closed = False
        self._started_at = time.monotonic()
        self.stream_id = ""
        self.sent_frames = 0
        self.recv_frames = 0

    # ---- starlette surface -------------------------------------------------
    async def receive(self) -> dict:
        return await self._inbox.get()

    async def send_text(self, text: str) -> None:
        try:
            msg = json.loads(text)
        except (TypeError, ValueError):
            return
        kind = msg.get("type")
        if kind == "audio_start":
            self._fmt = msg.get("format") or "pcm"
            self._rate = int(msg.get("sample_rate") or OUTPUT_RATE_HINT)
        elif kind == "cancel":
            await self.clear()
        elif kind == "user":
            log.info("caller: %s", str(msg.get("text", ""))[:160])
        elif kind == "assistant_done":
            log.info("halo: %s", str(msg.get("text", ""))[:160])
        elif kind == "error":
            log.warning("pipeline error: %s", msg.get("message"))
        elif kind == "ready":
            log.info(
                "call ready  model=%s  voice=%s",
                (msg.get("config") or {}).get("llm_model"),
                (msg.get("config") or {}).get("voice"),
            )

    async def send_bytes(self, data: bytes) -> None:
        if self._closed or not data:
            return
        # Remember which "generation" this clause belongs to: decoding a few
        # seconds of MP3 takes time, and if the caller barges in meanwhile the
        # result must be dropped rather than spoken over them.
        gen = self._gen
        if self._fmt == "mp3":
            pcm = await asyncio.to_thread(decode_tts_mp3, data)
        else:
            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            pcm = await asyncio.to_thread(resample, pcm, self._rate, PHONE_SR)
            pcm = pcm.astype(np.int16)
        if pcm.size and gen == self._gen:
            self._enqueue(ulaw_encode(pcm), gen)

    # ---- carrier input -----------------------------------------------------
    def feed(self, payload_b64: str) -> None:
        """One 20 ms mu-law frame from the carrier."""
        if self._closed:
            return
        try:
            raw = base64.b64decode(payload_b64)
        except Exception:
            return
        if not raw:
            return
        self.recv_frames += 1
        up = resample(ulaw_decode(raw).astype(np.float32), PHONE_SR, INPUT_SR)
        pcm = np.clip(up, -32768, 32767).astype(np.int16)
        self._inbox.put_nowait({"type": "websocket.receive", "bytes": pcm.tobytes()})

    def disconnect(self) -> None:
        self._inbox.put_nowait({"type": "websocket.disconnect"})

    # ---- output ------------------------------------------------------------
    def _enqueue(self, mu: bytes, gen: int) -> None:
        for i in range(0, len(mu), FRAME_SAMPLES):
            self._out.put_nowait((gen, mu[i : i + FRAME_SAMPLES]))
        if self._player is None or self._player.done():
            self._player = asyncio.create_task(self._play(), name="phone-play")

    async def _play(self) -> None:
        self._next_at = time.monotonic()
        while not self._closed:
            gen, frame = await self._out.get()
            if frame is None:
                return
            if gen != self._gen:
                continue
            await self._send_raw(self._media(frame))
            self.sent_frames += 1
            self._next_at = max(self._next_at, time.monotonic()) + FRAME_MS / 1000
            delay = self._next_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

    async def clear(self) -> None:
        """Barge-in: drop what is queued and tell the carrier to stop too."""
        self._gen += 1
        while not self._out.empty():
            try:
                self._out.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self.stream_id:
            await self._send_raw(self._clear())

    # ---- wire formats ------------------------------------------------------
    def _media(self, payload: bytes) -> dict:
        b64 = base64.b64encode(payload).decode("ascii")
        if self.provider == "twilio":
            return {"event": "media", "streamSid": self.stream_id, "media": {"payload": b64}}
        return {
            "event": "playAudio",
            "media": {"contentType": "audio/x-mulaw", "sampleRate": PHONE_SR, "payload": b64},
        }

    def _clear(self) -> dict:
        if self.provider == "twilio":
            return {"event": "clear", "streamSid": self.stream_id}
        return {"event": "clearAudio", "streamId": self.stream_id}

    # ---- lifecycle ---------------------------------------------------------
    async def close(self) -> None:
        self._closed = True
        if self._player and not self._player.done():
            self._player.cancel()
        self._inbox.put_nowait({"type": "websocket.disconnect"})


async def run_call(ws, engines, settings) -> None:
    """Bridge one carrier websocket to one Conversation."""
    from .pipeline import Conversation   # local import: keeps the module optional

    provider = (ws.query_params.get("provider") or "plivo").lower()
    lock = asyncio.Lock()

    async def send(obj: dict) -> None:
        async with lock:
            await ws.send_json(obj)

    phone = PhoneSocket(provider, send, max_seconds=settings.phone_max_s)
    reader = asyncio.create_task(_read_loop(ws, phone, provider), name=f"phone-read-{provider}")
    watchdog = asyncio.create_task(_limit(phone, settings.phone_max_s), name="phone-limit")

    conversation = Conversation(phone, engines)
    log.info("call started  provider=%s", provider)
    try:
        await conversation.run()
    except Exception:
        log.exception("call failed")
    finally:
        for task in (reader, watchdog):
            task.cancel()
        await phone.close()
        log.info(
            "call ended after %.0fs  in=%d frames  out=%d frames",
            time.monotonic() - phone._started_at,
            phone.recv_frames,
            phone.sent_frames,
        )


async def _read_loop(ws, phone: PhoneSocket, provider: str) -> None:
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                continue
            event = msg.get("event")
            if event == "start":
                start = msg.get("start") or {}
                phone.stream_id = start.get("streamId") or start.get("streamSid") or ""
                fmt = start.get("mediaFormat") or {}
                log.info(
                    "call media  stream=%s  encoding=%s  rate=%s",
                    phone.stream_id[:12],
                    fmt.get("encoding"),
                    fmt.get("sampleRate"),
                )
            elif event == "media":
                payload = (msg.get("media") or {}).get("payload")
                if payload:
                    phone.feed(payload)
            elif event in ("stop", "closed"):
                log.info("caller hung up")
                phone.disconnect()
                return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.info("call reader stopped: %s", exc)
        phone.disconnect()


async def _limit(phone: PhoneSocket, seconds: float) -> None:
    try:
        await asyncio.sleep(max(30.0, seconds))
    except asyncio.CancelledError:
        return
    log.info("call time limit reached (%.0fs)", seconds)
    phone.disconnect()


# --------------------------------------------------------------------- webhook
def public_base(request, configured: str = "") -> str:
    if configured:
        return configured.rstrip("/")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme or "https"
    host = request.headers.get("host") or request.url.netloc or "localhost"
    return f"{proto}://{host}"


def answer_xml(provider: str, base: str, token: str) -> str:
    """The document a carrier fetches when the number rings."""
    ws_url = base.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/ws/phone?provider={provider}"
    if token:
        ws_url += f"&token={token}"

    if provider == "twilio":
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<Response><Connect><Stream url="{escape(ws_url)}"/></Connect></Response>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        '<Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">'
        f"{escape(ws_url)}"
        "</Stream></Response>"
    )


# --------------------------------------------------------------------- selftest
def _selftest() -> int:
    """`python -m server.phone` - codec and resampler sanity, no network."""
    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    silent = ulaw_decode(b"\xff")[0]
    check("0 decodes to silence", silent == 0, f"decode(0xFF)={silent}")
    check("silence survives a round trip", int(ulaw_decode(ulaw_encode(np.zeros(1, np.int16)))[0]) == 0)

    # G.711 is a logarithmic 8-bit codec: at the top of each segment the
    # quantisation step is large, so the pass condition is "SNR of a real
    # signal", not "smallest possible error".
    t = np.arange(24000) / 24000.0
    probe = (16000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    restored = ulaw_decode(ulaw_encode(probe)).astype(np.float32)
    noise = restored - probe.astype(np.float32)
    snr_db = 10 * np.log10(float(np.mean(probe.astype(np.float32) ** 2)) / max(float(np.mean(noise ** 2)), 1e-9))
    check("mu-law SNR > 30 dB", snr_db > 30, f"{snr_db:.1f} dB")

    t = np.arange(24000) / 24000.0
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16)
    down = resample(tone, 24000, PHONE_SR)
    check("1s @24k -> 8000 samples", down.size == 8000, f"got {down.size}")
    check("tone survives the low pass", float(np.abs(down).max()) > 12000, f"peak {np.abs(down).max():.0f}")
    spec = np.abs(np.fft.rfft(down * np.hanning(down.size)))
    peak_hz = np.argmax(spec) * PHONE_SR / down.size
    check("1 kHz stays 1 kHz", abs(peak_hz - 1000) < 25, f"{peak_hz:.0f} Hz")

    up = resample(down.astype(np.float32), PHONE_SR, INPUT_SR)
    check("1s @8k -> 16000 samples", up.size == 16000, f"got {up.size}")
    check("50 frames per second of audio", (PHONE_SR * 40 + FRAME_SAMPLES - 1) // FRAME_SAMPLES == 2000)

    try:
        import miniaudio  # noqa: F401

        check("miniaudio available for MP3 clauses", True)
    except Exception as exc:
        check("miniaudio available for MP3 clauses", False, str(exc))

    print("\n  all good" if not fails else f"\n  {fails} check(s) failed")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
