# Halo — voice-to-voice chat in the browser

Talk to an LLM out loud. It answers out loud, starts speaking before it has
finished thinking, and shuts up the moment you talk over it.

Two ways to run it:

* **Hosted (default)** — free APIs, no GPU, no local weights. Small enough for a
  free 512 MB instance; this is what ships.
* **Fully offline** — Silero VAD → faster-whisper → Ollama → Kokoro-82M, all on
  your own machine, no keys, nothing leaves the laptop.

```
 hosted ──  microphone ──▶ Silero VAD ──▶ Groq Whisper ──▶ Groq LLM ──▶ gTTS ──▶ speaker
             16 kHz        endpointing     speech→text        chat       mp3
 local  ──  microphone ──▶ Silero VAD ──▶ faster-whisper ──▶ Ollama ──▶ Kokoro ──▶ speaker
             16 kHz        endpointing      speech→text        LLM      24 kHz PCM
                                    │
                                    └──── barge-in: you talk, it shuts up ────┘
```

Both run the same pipeline; only the three engines are swapped. The browser
streams raw 16 kHz PCM over a WebSocket either way.

---

## Deploy it free on Render

You need one free key, no credit card: **<https://console.groq.com/keys>**. The
same key covers speech recognition and chat.

1. Push this repo to your GitHub account (it already is, if you cloned it).
2. On Render: **New → Blueprint**, pick the repo. `render.yaml` is in the root,
   so the service, build command and environment are filled in for you.
3. Paste `GROQ_API_KEY` when Render asks. Deploy.

You get `https://<service-name>.onrender.com` with HTTPS, which is also what the
microphone requires — a plain-HTTP IP address will not get mic access in Chrome.

Things worth knowing about the free tier:

* The instance **sleeps after ~15 minutes idle**. The next visit takes ~30 s to
  wake up (it re-imports FastAPI + onnxruntime and re-downloads the 2 MB VAD
  model).
* 512 MB RAM. That is why nothing local is installed: no Whisper weights, no
  Kokoro, no Ollama. Peak RSS here is ~200 MB.
* WebSockets are supported on free services; serverless platforms (Vercel,
  Netlify, Lambda) are not an option for this app.

### Free-tier limits, honestly

Groq's free tier is rate-limited per minute and per day, and Google's speak
endpoint (gTTS) is an undocumented public endpoint that occasionally answers 429.
For a demo or a link you share with friends this is fine. For real traffic you
would add your own key/quota — nothing in the code assumes the free tier.

---

## Run it locally against the same free APIs

```powershell
git clone https://github.com/peeyush1279/halo-voice-chat
cd halo-voice-chat
.\setup.ps1
Copy-Item .env.example .env      # then paste your Groq key into .env
.\run.ps1                        # http://127.0.0.1:8000
```

Press **Start talking** and grant the microphone.

> **Use headphones.** Without them the microphone hears the assistant and the
> two of you talk over each other forever. On speakers, turn on *Settings →
> Speaker mode*, which mutes the mic while Halo talks.

---

## Fully offline mode (the original build)

If you want the local stack, install it and flip the three backends:

```powershell
.\setup.ps1 -Local        # adds faster-whisper + Kokoro (~360 MB of model files)
$env:V2V_STT_BACKEND  = "local"
$env:V2V_LLM_BACKEND  = "ollama"
$env:V2V_TTS_ENGINE   = "kokoro"
.\run.ps1                 # starts Ollama if it is not already running
```

First launch also pulls the Whisper weights (~470 MB, once). `.\run.ps1 -Check`
runs an offline self-test of every stage (VAD → TTS → STT round-trip → Ollama)
and prints each latency.

### Picking a local model (this is what makes it fast or slow)

The reference machine has an **AMD Radeon RX 5500M (4 GB)**. Ollama offloads
whole layers into VRAM, and one rule decides everything: **a model that fits
entirely in those 4 GB runs on the GPU; a model that does not runs at a tenth of
the speed.** `ollama ps` shows the split — the default sits at 100 % GPU, the 8Bs
come out at roughly half and half.

| Model | GPU | Warm first token | Warm speed | Why this or that |
|---|---|---|---|---|
| `qwen2.5:3b-instruct` ← *default* | **100 %** | **28 ms** | **68.5 tok/s** | answers directly; ~3 s end to end |
| `mannix/llama3.1-8b-abliterated` | ~53 % | 91 ms | ~13 tok/s | uncensored, better prose, but ~14 s per reply |
| `qwen3:4b` | 78 % | — | ~5 tok/s | **thinking model, unusable for voice (below)** |

`keep_alive=-1` is what holds the default at those numbers. It stops Ollama
evicting an idle model, which otherwise costs 8–18 s of dead air reloading it on
your next turn.

**Never use a "thinking" model for voice** (`qwen3`, `deepseek-r1`, …). They
spend hundreds of tokens reasoning before saying anything. Measured on
`qwen3:4b` with the app's real payload (`num_predict=220`):

| `qwen3:4b` | Time | Tokens | What came back |
|---|---|---|---|
| thinking on | 38.8 s | 220 (capped) | 716 chars of reasoning, **0 chars of answer** |
| `think: false` | 41.1 s | 220 (capped) | the reasoning *relabelled as the answer* |

So `V2V_LLM_NO_THINK=1` does not rescue a thinking model — it only moves the
chain of thought into the text Halo speaks out loud. Choose a model that answers
directly.

### Local latency (measured here)

Headphones, default model, nothing else hogging the CPU:

| Stage | Typical | What controls it |
|---|---|---|
| `stt` — speech → text | 0.9–1.7 s | Whisper size, `V2V_WHISPER_COMPUTE_TYPE`, `V2V_WHISPER_BEAM` |
| `ttft` — first LLM token | 100–300 ms | Ollama model size; `keep_alive=-1` keeps it low |
| `tts` — first audio chunk | 400–1300 ms | Kokoro RTF (~0.5–0.8) × length of the first clause |
| **total to first sound** | **~1–2 s** | every one of them is tunable |

> **Background load is the hidden variable.** Measured here: with Valorant
> running in the background (352 % CPU across 3.5 cores) *every* stage roughly
> doubled. Close games/benchmarks before judging latency.

Knobs, in order of impact:

* **Close background CPU hogs** — bigger than any setting below.
* **A larger LLM** — the default is already the fastest model that fits in 4 GB.
* `V2V_TTS_FIRST_CHARS=40` *(default)* — say the opening words as soon as they
  exist instead of waiting for the first comma. Measured A/B: **1995 ms → 1266 ms**
  to first audio, i.e. **−37 %**, purely from speaking a 39-character chunk
  first. Set `0` to wait for real punctuation (slower, marginally smoother).
* `V2V_VAD_MIN_SILENCE_MS=500` *(default)* — the end-of-turn threshold. At `400`
  a normal two-sentence utterance is cut at the breath between sentences
  (measured: the transcript loses the second half). Raise it if Halo keeps
  jumping in during thoughtful pauses.
* `V2V_WHISPER_MODEL=Systran/faster-whisper-base.en` — ~2× faster than the
  default (0.23 vs 0.54 RTF) and noticeably less accurate.

### Choosing the local Whisper model

Whisper always encodes a fixed 30 s mel window, so on a CPU the **model size** —
not how long you spoke — decides transcription latency. Measured here (Ryzen 5
5600H, 6 threads, `int8`, a 3.9 s utterance), greedy decode:

| `V2V_WHISPER_MODEL` | RTF | Wall time | Verdict |
|---|---|---|---|
| `tiny.en` | 0.10 | ~0.4 s | instant, more mistakes |
| `base.en` | 0.23 | ~0.9 s | fast; enough for the live partial transcript |
| **`distil-small.en`** *(default)* | **0.54** | **~2.1 s** | **best accuracy/CPU trade, English-only** |
| `small` | 0.62 | ~2.4 s | multilingual; no accuracy gain over the distilled model |
| `large-v3-turbo` | 2.61 | ~10 s | too slow for live voice on CPU |

Finals run `beam_size=5` (**1.6 s wall / RTF 0.34–0.49** for 3–5 s utterances on
an idle machine); partials run greedy on a separate, cheaper model
(`V2V_WHISPER_PARTIAL_MODEL`, default `base.en`, RTF ~0.25), so raising the
accuracy of finals does not make the on-screen transcript lag.

---

## How it works

```
web/                       browser
  index.html               UI shell
  app.js                   websocket client, playback scheduler, rings, transcript
  mic-worklet.js           packs 16 kHz int16 frames (filtered resampler fallback)
  style.css

server/
  main.py                  FastAPI app, static files, /ws endpoint, /api/*
  config.py                every tunable, all env-overridable (.env supported)
  engines.py               loads the chosen engines once, in a background thread
  vad.py                   Silero VAD (ONNX) + the streaming turn detector
  stt.py                   faster-whisper (local) / Groq Whisper (hosted)
  llm.py                   Ollama /api/chat, Groq OpenAI-compatible SSE
  tts.py                   Kokoro-82M (local) / gTTS / Edge TTS (hosted)
  textchunk.py             token stream -> speakable clauses
  pipeline.py              the actual conversation: turn-taking, barge-in, timing
  download_models.py       fetches the VAD model (and Kokoro with --kokoro)
  selftest.py              offline per-stage check
  wstest.py                end-to-end test over the websocket

bench_whisper.py           measures Whisper model/thread latency on your CPU
render.yaml                one-click free deploy
```

### Why it feels responsive

1. **The browser sends raw 16 kHz mono PCM** — no containers, no encoding, no
   round trip through an audio codec. Frames go out every 20 ms.
2. **Endpoints are detected, not timed out.** Silero VAD runs in-process at
   ~200× real-time, with hysteresis so a breath mid-sentence doesn't cut you off,
   and pre-roll padding so your first phoneme isn't clipped.
3. **You see your own words live** (local mode). Every ~0.7 s the growing
   utterance is re-transcribed for a partial transcript; partial jobs are
   *dropped* when the worker is busy, so they can never delay the real
   transcript. In hosted mode partials are off — they would burn a per-minute
   API quota to render text that is thrown away.
4. **The reply is spoken clause by clause.** The chunker flushes on `.`/`?`/`!`,
   or on a comma once there are enough characters, or forcibly at a length cap.
   The first clause is usually out of the LLM within ~200 ms, and speech
   synthesis happens while later tokens are still arriving.
5. **Ordered, gapless audio.** TTS runs on a worker thread and pushes into a
   single asyncio outbox, so chunks can't overtake each other. PCM is decoded
   straight into a 24 kHz audio context; the MP3 from gTTS/Edge is decoded with
   `decodeAudioData` through a serial chain (parallel decodes can finish out of
   order and play the reply backwards). Both are scheduled with a ~70 ms jitter
   buffer.
6. **Barge-in is instant.** The moment VAD hears you, the server bumps the turn
   id (which invalidates everything in flight), cancels the LLM request, drains
   the TTS queue, and tells the browser to flush its playback queue. You never
   have to wait for it to finish a sentence.

---

## Configuration

Every setting is an environment variable with a `V2V_` prefix. Put them in `.env`
next to `run.ps1` (a real environment variable always wins), or set them in
PowerShell: `$env:V2V_TTS_ENGINE="edge"; .\run.ps1`.

### Backends

| Variable | Default | Options |
|---|---|---|
| `V2V_STT_BACKEND` | `groq` | `groq` (Whisper large-v3-turbo) · `local` (faster-whisper) |
| `V2V_LLM_BACKEND` | `groq` | `groq` · `ollama` |
| `V2V_TTS_ENGINE` | `gtts` | `gtts` · `edge` (better prosody) · `kokoro` (offline) |
| `GROQ_API_KEY` | — | free key from <https://console.groq.com/keys> |
| `V2V_GROQ_LLM_MODEL` | `llama-3.3-70b-versatile` | `llama-3.1-8b-instant` is faster with higher free limits |
| `V2V_GROQ_STT_MODEL` | `whisper-large-v3-turbo` | |
| `V2V_GTTS_LANG` / `V2V_GTTS_TLD` | `en` / `com` | `co.uk`, `com.au`, `ie`, `ca`, `co.in` change the accent |
| `V2V_EDGE_VOICE` | `en-US-AriaNeural` | any Edge neural voice, e.g. `en-GB-SoniaNeural` |
| `V2V_HOST` | `127.0.0.1` | `0.0.0.0` on a host |
| `V2V_PORT` | `8000` | falls back to `PORT`, which hosts inject |

### Pipeline (both modes)

| Variable | Default | Notes |
|---|---|---|
| `V2V_LLM_TEMPERATURE` | `0.7` | |
| `V2V_LLM_MAX_TOKENS` | `220` | caps reply length → caps latency |
| `V2V_HISTORY_TURNS` | `12` | conversation memory |
| `V2V_SYSTEM_PROMPT` | Halo persona | editable live in settings |
| `V2V_VAD_THRESHOLD` | `0.5` | raise if it triggers on noise |
| `V2V_VAD_MIN_SILENCE_MS` | `500` | **main "feels slow" knob** |
| `V2V_TTS_MIN_CHARS` | `10` | smallest clause handed to the voice |
| `V2V_TTS_MAX_CHARS` | `150` | |
| `V2V_TTS_FIRST_CHARS` | `40` | **main "first audio" knob**; `0` = off |

### Local-only

| Variable | Default | Notes |
|---|---|---|
| `V2V_LLM_MODEL` | `qwen2.5:3b-instruct` | any `ollama list` name that fits 4 GB and does not think |
| `V2V_LLM_KEEP_ALIVE` | `-1` | `-1` = never evict |
| `V2V_OLLAMA_URL` | `http://127.0.0.1:11434` | |
| `V2V_WHISPER_MODEL` | `Systran/faster-distil-whisper-small.en` | accurate finals, English |
| `V2V_WHISPER_PARTIAL_MODEL` | `Systran/faster-whisper-base.en` | cheap model for the live transcript |
| `V2V_WHISPER_COMPUTE_TYPE` | `int8` | right for CPU |
| `V2V_WHISPER_LANGUAGE` | `en` | `""` = auto-detect (multilingual models only) |
| `V2V_WHISPER_BEAM` | `5` | finals; `1` = greedy, fastest |
| `V2V_WHISPER_THREADS` | half your cores | |
| `V2V_PARTIAL_INTERVAL_S` | `0.7` | live transcript update rate |
| `V2V_TTS_VOICE` | `af_heart` | see the settings dropdown |
| `V2V_TTS_SPEED` | `1.0` | 0.6–2.0 |
| `V2V_TTS_BACKEND` | `auto` | Kokoro sub-backend: `auto` · `onnx` · `torch` |

---

## Testing without a browser

```powershell
.\run.ps1                                       # terminal 1
.\.venv\Scripts\python.exe -m server.wstest     # terminal 2
```

`wstest` synthesises a sentence with Kokoro, streams it into the websocket
exactly like the microphone would, prints every event and timing, and writes the
assistant's spoken answer to `reply.wav`. You can also feed it your own
recording: `python -m server.wstest --wav me.wav` (16 kHz mono).

To see how fast speech recognition is on *your* machine (local mode):

```powershell
.\.venv\Scripts\python.exe bench_whisper.py --thread-scaling
```

---

## Troubleshooting

**`Groq rejected the API key (401)`** — `GROQ_API_KEY` is missing or wrong. In
hosted mode it is an environment variable on the service; locally put it in
`.env` and restart.

**`Groq rate limit hit (429)`** — you are going faster than the free tier allows
for a minute. Wait a few seconds. Hosted mode already disables the live partial
transcript for exactly this reason.

**Nothing is ever transcribed** — check *Settings → diagnostics*. If `VAD` shows
`failed`, run `python -m server.download_models`. If your mic is very quiet,
lower `V2V_VAD_THRESHOLD` to `0.3`.

**"Microphone blocked"** — the page must be on `https://` or `http://127.0.0.1`;
`http://<lan-ip>` counts as an insecure origin in Chrome. Render gives you HTTPS
for free.

**First visit after a while takes ~30 s** — the free instance was asleep. It
cannot be kept warm for free.

**It hears itself / talks over me** — use headphones, or enable *Speaker mode*.

**Every local reply takes several seconds** — that's the LLM, and it means either
the model got evicted or something bigger than the default is selected. Check
`ollama ps`: `qwen2.5:3b-instruct` should read 100 % GPU.

**`spacy` / `misaki` install errors** — you don't need them; `kokoro-onnx` 0.4.7
phonemises with plain eSpeak. The requirements deliberately avoid `misaki[en]`
because `spacy`→`blis` has no prebuilt wheel for Python 3.14.

**eSpeak errors from Kokoro** — `espeakng-loader` ships the DLL; if it still
fails, install eSpeak NG system-wide and set `PHONEMIZER_ESPEAK_LIBRARY` to the
DLL path.

**Port already in use** — `.\run.ps1 -Port 8100`.

---

## Optional upgrades

* **Better hosted voice**: `V2V_TTS_ENGINE=edge` (also free, keyless, better
  prosody). Pick a voice with `V2V_EDGE_VOICE`.
* **Lower hosted latency**: `V2V_GROQ_LLM_MODEL=llama-3.1-8b-instant`.
* **Other languages**: `V2V_GTTS_LANG=de` (hosted), or in local mode
  `V2V_WHISPER_MODEL=Systran/faster-whisper-small` +
  `V2V_WHISPER_PARTIAL_MODEL=Systran/faster-whisper-small` +
  `V2V_WHISPER_LANGUAGE=` (auto-detect).
* **Better English pronunciation for Kokoro**: on Python ≤ 3.13 you can
  `pip install "misaki[en]"` and set `V2V_TTS_BACKEND=torch`.
* **GPU**: install `onnxruntime-gpu` and set `V2V_WHISPER_DEVICE=cuda` +
  `V2V_WHISPER_COMPUTE_TYPE=float16` with an NVIDIA card. RDNA1 on Windows has no
  practical acceleration path for these stacks.
* **Conversation persistence** — `pipeline.py` keeps history in memory only; it
  resets when the tab closes.
