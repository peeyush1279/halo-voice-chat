# Halo — fully local voice-to-voice chat

Talk to an LLM in the browser. Everything runs on your machine: no API keys, no
cloud, nothing leaves the laptop.

```
 microphone ──▶ Silero VAD ──▶ faster-whisper ──▶ Ollama ──▶ Kokoro-82M ──▶ speaker
   16 kHz         endpointing       speech→text     LLM         text→speech   24 kHz
                        │                                            ▲
                        └────── barge-in: you talk, it shuts up ──────┘
```

Built to feel like a conversation rather than a walkie-talkie: it starts
speaking the first *clause* of the reply while the rest is still being
generated, and you can talk over it at any moment.

---

## Quick start

```powershell
git clone https://github.com/peeyush1279/halo-voice-chat
cd halo-voice-chat
.\setup.ps1          # venv + deps + Kokoro/Silero model files  (~350 MB)
.\run.ps1            # starts on http://127.0.0.1:8000
```

First launch also pulls the Whisper weights (~470 MB, once). Press
**Start talking** in the browser and grant the microphone.

> **Use headphones.** Without them the microphone hears the assistant and the
> two of you will talk over each other forever. If you have to use speakers,
> turn on *Settings → Speaker mode*, which mutes the mic while Halo talks.

`.run.ps1 -Check` runs an offline self-test of every stage
(VAD → TTS → STT round-trip → Ollama) and prints the latency of each one.
It's the fastest way to find out which stage is slow on your machine.

---

## Picking a model (this is what makes it fast or slow)

Your machine has an **AMD Radeon RX 5500M (4 GB)**. Ollama offloads whole layers
into that VRAM, and the rule that decides everything is: **a model that fits
entirely in those 4 GB runs on the GPU; a model that does not runs at a tenth
of the speed.** `ollama ps` shows you the split — the default sits at 100 % GPU,
the 8Bs come out at roughly half and half.

That is why the default is the small one:

| Model | GPU | Warm first token | Warm speed | Why this or that |
|---|---|---|---|---|
| `qwen2.5:3b-instruct` ← *default* | **100 %** | **28 ms** | **68.5 tok/s** | answers directly; ~3 s end to end |
| `mannix/llama3.1-8b-abliterated` | ~53 % | 91 ms | ~13 tok/s | uncensored, better prose, but ~14 s per reply |
| `qwen3:4b` | 78 % | — | ~5 tok/s | **thinking model, unusable for voice (below)** |

`keep_alive=-1` is what holds the default at those numbers. It is what stops
Ollama evicting an idle model, which otherwise costs 8–18 s of dead air
reloading it on your next turn.

**Never use a "thinking" model for voice** (`qwen3`, `deepseek-r1`, …). They
spend hundreds of tokens reasoning before saying anything. Measured here on
`qwen3:4b` with the app's real payload (`num_predict=220`):

| `qwen3:4b` | Time | Tokens | What came back |
|---|---|---|---|
| thinking on | 38.8 s | 220 (capped) | 716 chars of reasoning, **0 chars of answer** |
| `think: false` | 41.1 s | 220 (capped) | the reasoning *relabelled as the answer* |

So `V2V_LLM_NO_THINK=1` does not rescue a thinking model — it only moves the
chain of thought into the text Halo speaks out loud. Choose a model that answers
directly.

You can still switch at runtime from the ⚙︎ settings dialog; the server warms the
new model in the background.

---

## What the latency numbers mean

Roughly what to expect on this laptop (headphones, the default model, **nothing
else hogging the CPU**):

| Stage | Typical | What controls it |
|---|---|---|
| `stt` — speech → text | 0.9–1.7 s | Whisper model size, `V2V_WHISPER_COMPUTE_TYPE`, `V2V_WHISPER_BEAM` |
| `ttft` — first LLM token | 100–300 ms | Ollama model size; `keep_alive=-1` is why it stays low |
| `tts` — first audio chunk | 400–1300 ms | Kokoro RTF (~0.5–0.8) × how long the first chunk speaks |
| **total to first sound** | **~1–2 s** | and you can tune every one of them |

> **Background load is the hidden variable.** Measured here: with Valorant
> running in the background (352 % CPU across 3.5 cores) *every* stage roughly
> doubled. Close games/benchmarks before judging latency.

Knobs, in order of impact:

* **Close background CPU hogs** — bigger than any setting below.
* **A larger LLM** — the default is already the fastest model that fits in the 4 GB GPU. An 8B buys better prose at ~5× the wait.
* `V2V_TTS_FIRST_CHARS=40` *(default)* — say the opening words as soon as they
  exist instead of waiting for the first comma. Replies commonly open with a
  67-character clause; measured A/B on this box: **1995 ms → 1266 ms** to first
  audio, i.e. **−37 %**, purely from speaking a 39-character chunk first. Set
  `0` to wait for real punctuation everywhere (slower, marginally smoother).
* `V2V_TTS_MIN_CHARS=10` *(default)* — smallest clause handed to Kokoro.
* `V2V_VAD_MIN_SILENCE_MS=500` *(default)* — the end-of-turn threshold. At
  `400` an ordinary two-sentence utterance gets cut at the breath between the
  sentences (measured: the transcript loses the second half). Raise it further
  if Halo jumps in during thoughtful pauses.
* `V2V_WHISPER_MODEL=Systran/faster-whisper-base.en` — ~2× faster than the
  default (0.23 vs 0.54 RTF) but noticeably less accurate.
* `V2V_WHISPER_BEAM=3` — cheaper beam search if the final transcript feels slow.

### Choosing the Whisper model

Whisper always encodes a fixed 30 s mel window, so on a CPU the **model size** —
not how long you spoke — decides transcription latency. The ladder below is the
greedy-decode baseline measured here (Ryzen 5 5600H, 6 threads, `int8`, a 3.9 s
utterance); read it as a relative order, not as absolute truth:

| `V2V_WHISPER_MODEL` | RTF | Wall time | Verdict |
|---|---|---|---|
| `tiny.en` | 0.10 | ~0.4 s | instant, more mistakes |
| `base.en` | 0.23 | ~0.9 s | fast; enough for the live partial transcript |
| **`distil-small.en`** *(default)* | **0.54** | **~2.1 s** | **best accuracy/CPU trade, English-only** |
| `small` | 0.62 | ~2.4 s | multilingual; no accuracy gain over the distilled model |
| `large-v3-turbo` | 2.61 | ~10 s | too slow for live voice on CPU |

The app's final pass runs `beam_size=5`; that measured **1.6 s wall / RTF
0.34–0.49** for 3–5 s utterances on an idle machine, so the ladder's absolute
numbers are pessimistic but its ordering holds. Partials run greedy on a
separate, cheaper model (`V2V_WHISPER_PARTIAL_MODEL`, default `base.en`, RTF
~0.25), so raising the accuracy of finals does not make the on-screen transcript
lag. `V2V_WHISPER_BEAM=3` is the middle ground if finals feel slow.

The `.en` models are English-only (they silently assume English). For other
languages use `small`. Thread count barely matters here: 6 threads (0.61 RTF)
beat both 3 threads (0.77) and 12 (0.66) on `small` — the extra cores don't
help a single transcription.

---

## How it works

```
web/                       browser
  index.html               UI shell
  app.js                   websocket client, playback scheduler, orb, transcript
  mic-worklet.js           packs 16 kHz int16 frames (filtered resampler fallback)
  style.css

server/
  main.py                  FastAPI app, static files, /ws endpoint, /api/*
  config.py                every tunable, all env-overridable
  engines.py               loads models once, in a background thread
  vad.py                   Silero VAD (ONNX) + the streaming turn detector
  stt.py                   faster-whisper wrapper
  llm.py                   Ollama /api/chat streaming client
  tts.py                   Kokoro-82M (ONNX)
  textchunk.py             token stream -> speakable clauses
  pipeline.py              the actual conversation: turn-taking, barge-in, timing
  download_models.py       fetches model files
  selftest.py              offline per-stage check
  wstest.py                end-to-end test over the websocket

bench_whisper.py           measures Whisper model/thread latency on your CPU
```

### Why it feels responsive

1. **The browser sends raw 16 kHz mono PCM** — no containers, no encoding, no
   round trip through an audio codec. Frames go out every 20 ms.
2. **Endpoints are detected, not timed out.** Silero VAD runs in-process at
   ~200× real-time, with hysteresis so a breath mid-sentence doesn't cut you off,
   and pre-roll padding so your first phoneme isn't clipped.
3. **You see your own words live.** Every ~0.7 s the growing utterance is
   re-transcribed for a partial transcript. Partial jobs are *dropped* when the
   worker is busy, so they can never delay the real transcript.
4. **The reply is spoken clause by clause.** The chunker flushes on `.`/`?`/`!`,
   or on a comma once there are enough characters, or forcibly at a length cap.
   The first clause is usually out of the LLM within ~200 ms of the reply
   starting, and TTS happens while later tokens are still arriving.
5. **Ordered, gapless audio.** TTS runs on a worker thread and pushes into a
   single asyncio outbox, so chunks can't overtake each other. The browser
   schedules them onto a 24 kHz audio context with a ~70 ms jitter buffer.
6. **Barge-in is instant.** The moment VAD hears you, the server bumps the turn
   id (which invalidates everything in flight), cancels the LLM request, drains
   the TTS queue, and tells the browser to flush its playback queue. You never
   have to wait for it to finish a sentence.

---

## Configuration

Every setting is an environment variable with a `V2V_` prefix. Set them in
PowerShell before launching, e.g. `$env:V2V_TTS_VOICE="am_onyx"; .\run.ps1`.

| Variable | Default | Notes |
|---|---|---|
| `V2V_LLM_MODEL` | `qwen2.5:3b-instruct` | any `ollama list` name; it has to fit in 4 GB *and* not be a thinking model |
| `V2V_LLM_TEMPERATURE` | `0.7` | |
| `V2V_LLM_MAX_TOKENS` | `220` | caps reply length → caps latency |
| `V2V_LLM_KEEP_ALIVE` | `-1` | `-1` = **never evict**; `30m` = reload after idle (17.5 s cold hit) |
| `V2V_HISTORY_TURNS` | `12` | conversation memory |
| `V2V_SYSTEM_PROMPT` | Halo persona | editable live in settings |
| `V2V_WHISPER_MODEL` | `Systran/faster-distil-whisper-small.en` | accurate finals, English (see table above) |
| `V2V_WHISPER_PARTIAL_MODEL` | `Systran/faster-whisper-base.en` | cheap model for the live transcript |
| `V2V_WHISPER_COMPUTE_TYPE` | `int8` | `int8` is right for CPU |
| `V2V_WHISPER_LANGUAGE` | `en` | `""` = auto-detect (multilingual models only) |
| `V2V_WHISPER_BEAM` | `5` | beam search width for finals; `1` = greedy, fastest |
| `V2V_WHISPER_THREADS` | half your cores | |
| `V2V_VAD_THRESHOLD` | `0.5` | raise if it triggers on noise |
| `V2V_VAD_MIN_SILENCE_MS` | `500` | **main "feels slow" knob**; lower only if you speak in whole sentences without pausing |
| `V2V_VAD_MIN_SPEECH_MS` | `220` | |
| `V2V_PARTIAL_INTERVAL_S` | `0.7` | live transcript update rate |
| `V2V_TTS_VOICE` | `af_heart` | see the settings dropdown |
| `V2V_TTS_SPEED` | `1.0` | 0.6–2.0 |
| `V2V_TTS_MIN_CHARS` | `10` | smallest clause handed to Kokoro |
| `V2V_TTS_MAX_CHARS` | `150` | |
| `V2V_TTS_FIRST_CHARS` | `40` | **main "first audio" knob**; opening chunk only, `0` = off |
| `V2V_OLLAMA_URL` | `http://127.0.0.1:11434` | point at another host if you like |
| `V2V_PORT` | `8000` | |

---

## Testing without a browser

```powershell
.\run.ps1                        # terminal 1
.\.venv\Scripts\python.exe -m server.wstest    # terminal 2
```

`wstest` synthesises a sentence with Kokoro, streams it into the websocket
exactly like the microphone would, prints every event and timing, and writes
the assistant's spoken answer to `reply.wav`. You can also feed it your own
recording: `python -m server.wstest --wav me.wav` (16 kHz mono).

To see how fast speech recognition is on *your* machine, run:

```powershell
.\.venv\Scripts\python.exe bench_whisper.py --thread-scaling
```

It benchmarks each Whisper size and prints the real-time factor, so you can pick
the smallest model that keeps RTF under ~0.5.

---

## Troubleshooting

**"Microphone blocked"** — the page must be on `http://127.0.0.1` or
`https://`; `http://<lan-ip>` counts as an insecure origin in Chrome. Use
`localhost`.

**It hears itself / talks over me** — use headphones, or enable *Speaker mode*
in settings.

**Nothing is ever transcribed** — check the diagnostics: if `VAD` shows
`failed`, run `python -m server.download_models`. If your mic is very quiet,
lower `V2V_VAD_THRESHOLD` to `0.3`.

**Every reply takes several seconds** — that's the LLM, and it means either the
model got evicted or something bigger than the default is selected. Check
`ollama ps`: `qwen2.5:3b-instruct` should read 100 % GPU. Run `ollama pull
qwen2.5:3b-instruct` if it is missing, and select it in settings.

**`spacy` / `misaki` install errors** — you don't need them;
`kokoro-onnx` 0.4.7 phonemises with plain eSpeak. `requirements.txt` deliberately
avoids `misaki[en]` because `spacy`→`blis` has no prebuilt wheel for Python 3.14.

**eSpeak errors from Kokoro** — `espeakng-loader` ships the DLL; if it still
fails, install eSpeak NG system-wide and set `PHONEMIZER_ESPEAK_LIBRARY` to the
DLL path.

**Port already in use** — `.\run.ps1 -Port 8100`.

---

## Optional upgrades

* **Other languages**: set `V2V_WHISPER_MODEL=Systran/faster-whisper-small`,
  `V2V_WHISPER_PARTIAL_MODEL=Systran/faster-whisper-small`, and
  `V2V_WHISPER_LANGUAGE=` (auto-detect). Both default models are English-only.
* **Better English pronunciation for Kokoro**: on Python ≤ 3.13 you can
  `pip install "misaki[en]"` and set `V2V_TTS_BACKEND=torch` to use the `kokoro`
  torch pipeline, which uses misaki's dictionary G2P instead of eSpeak.
* **GPU**: install `onnxruntime-gpu` and set `V2V_WHISPER_DEVICE=cuda` +
  `V2V_WHISPER_COMPUTE_TYPE=float16` if you get an NVIDIA card. AMD RDNA1 on
  Windows has no practical acceleration path for these stacks.
* **Server-side conversation persistence** — `pipeline.py` keeps history in
  memory only; it resets when the tab closes.
