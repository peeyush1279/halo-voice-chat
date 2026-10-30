/* Halo - browser client for the local voice-to-voice loop.
 *
 * Responsibilities:
 *   - capture the mic, resample to 16 kHz mono int16, stream over the socket
 *   - play streamed 24 kHz PCM gaplessly and stop instantly on barge-in
 *   - render the live transcript and the visualiser
 */

const $ = (id) => document.getElementById(id);

const els = {
  badges: $('badges'),
  orb: $('orb'),
  status: $('status'),
  substatus: $('substatus'),
  start: $('start-btn'),
  mic: $('mic-btn'),
  stop: $('stop-btn'),
  reset: $('reset-btn'),
  transcript: $('transcript'),
  empty: $('transcript-empty'),
  composer: $('composer'),
  textInput: $('text-input'),
  send: $('send-btn'),
  settingsBtn: $('settings-btn'),
  settings: $('settings'),
  applyBtn: $('apply-btn'),
  setModel: $('set-model'),
  setVoice: $('set-voice'),
  setSpeed: $('set-speed'),
  setSpeedVal: $('set-speed-val'),
  setMode: $('set-mode'),
  setEcho: $('set-echo'),
  setPrompt: $('set-prompt'),
  stackInfo: $('stack-info'),
};

const state = {
  ws: null,
  connected: false,
  started: false,
  streaming: false,       // assistant audio currently arriving
  mode: 'handsfree',
  muted: false,
  pttHeld: false,
  echoGuard: false,
  audioRate: 24000,
  audioFormat: 'pcm',        // 'pcm' (local Kokoro) or 'mp3' (gTTS / Edge)
  mp3Parts: null,            // bytes of the clause currently arriving as MP3
  cfg: null,
  userBubble: null,
  botBubble: null,
  micLevel: 0,
  outLevel: 0,
  micPeak: 0,        // loudest sample seen since this mic was opened
  micStartedAt: 0,
  micWarned: false,
  draining: false,   // server said idle, but the last audio is still audible
  phase: 'idle',          // idle | listening | thinking | speaking
  micBlocked: false,
  backend: null,
};

/* ------------------------------------------------------------------ player */
class Player {
  constructor(sampleRate) {
    this.ctx = new AudioContext({ sampleRate });
    this.gain = this.ctx.createGain();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.gain.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);
    this._buf = new Float32Array(this.analyser.fftSize);
    this.queue = [];
    this.active = new Set();
    this.nextTime = 0;
    this.playing = false;
    this.pending = 0;        // MP3 clauses handed to decodeAudioData, not queued yet
    this._decode = Promise.resolve();
    this.pump = this.pump.bind(this);
    this.timer = setInterval(this.pump, 20);
  }

  enqueue(int16, sampleRate) {
    let data = int16;
    if (sampleRate !== this.ctx.sampleRate) {
      data = resampleInt16(int16, sampleRate, this.ctx.sampleRate);
    }
    const f32 = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) f32[i] = data[i] / 32768;
    const buffer = this.ctx.createBuffer(1, f32.length, this.ctx.sampleRate);
    buffer.copyToChannel(f32, 0);
    this.queue.push(buffer);
    if (this.queue.length <= 2) this.pump();
  }

  // Hosted voices arrive as one complete MP3 per clause. decodeAudioData is
  // async, so the chunks go through a chain: decoding two clauses in parallel
  // can finish out of order and play the reply backwards.
  enqueueMp3(arrayBuffer) {
    this.pending += 1;
    this._decode = this._decode
      .then(() => this.ctx.decodeAudioData(arrayBuffer))
      .then((buffer) => {
        this.pending -= 1;
        this.queue.push(buffer);
        this.pump();
      })
      .catch(() => { this.pending -= 1; });
  }

  busy() {
    return this.playing || this.queue.length > 0 || this.pending > 0;
  }

  pump() {
    const now = this.ctx.currentTime;
    while (this.queue.length && this.nextTime < now + 0.4) {
      const buffer = this.queue.shift();
      if (this.nextTime < now + 0.02) this.nextTime = now + 0.07; // small jitter buffer
      const src = this.ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(this.gain);
      src.start(this.nextTime);
      this.nextTime += buffer.duration;
      this.playing = true;
      this.active.add(src);
      src.onended = () => {
        this.active.delete(src);
        if (!this.active.size && !this.queue.length) {
          this.playing = false;
          this.nextTime = 0;
        }
      };
    }
  }

  stop() {
    this.queue.length = 0;
    this.pending = 0;
    this._decode = Promise.resolve();
    for (const src of this.active) { try { src.stop(); } catch { /* already done */ } }
    this.active.clear();
    this.playing = false;
    this.nextTime = 0;
  }

  level() {
    this.analyser.getFloatTimeDomainData(this._buf);
    let peak = 0;
    for (let i = 0; i < this._buf.length; i += 4) {
      const v = Math.abs(this._buf[i]);
      if (v > peak) peak = v;
    }
    return Math.min(1, peak * 1.8);
  }
}

function resampleInt16(input, from, to) {
  const outLen = Math.max(1, Math.round((input.length * to) / from));
  const out = new Int16Array(outLen);
  const ratio = (input.length - 1) / Math.max(1, outLen - 1);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(input.length - 1, i0 + 1);
    const frac = pos - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}

/* --------------------------------------------------------------------- mic */
class Mic {
  constructor(stream) {
    this.stream = stream;
    this.ctx = null;
    this.send = () => {};
  }

  async init() {
    // Ask the context for 16 kHz directly. Chrome then resamples the device
    // stream with its own filter and the worklet can simply forward frames.
    // If a browser refuses the rate, fall back to the default and let the
    // worklet do an anti-aliased resample instead.
    try {
      this.ctx = new AudioContext({ sampleRate: 16000, latencyHint: 'interactive' });
    } catch {
      this.ctx = new AudioContext({ latencyHint: 'interactive' });
    }
    await this.ctx.audioWorklet.addModule('/static/mic-worklet.js');
    const source = this.ctx.createMediaStreamSource(this.stream);

    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this._buf = new Float32Array(this.analyser.fftSize);
    source.connect(this.analyser);

    this.node = new AudioWorkletNode(this.ctx, 'mic-processor');
    source.connect(this.node);

    // The worklet must be pulled by something, but we want silence locally.
    const sink = this.ctx.createGain();
    sink.gain.value = 0;
    this.node.connect(sink);
    sink.connect(this.ctx.destination);

    this.node.port.onmessage = (event) => this.send(event.data);
    if (this.ctx.state === 'suspended') await this.ctx.resume();
  }

  level() {
    if (!this.analyser) return 0;
    this.analyser.getFloatTimeDomainData(this._buf);
    let peak = 0;
    for (let i = 0; i < this._buf.length; i += 4) {
      const v = Math.abs(this._buf[i]);
      if (v > peak) peak = v;
    }
    return Math.min(1, peak * 1.8);
  }
}

let player = null;
let mic = null;

/* --------------------------------------------------------------- websocket */
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onopen = () => {
    state.connected = true;
    setStatus('Connected', 'Models may still be loading…');
  };
  ws.onclose = () => {
    state.connected = false;
    setStatus('Disconnected', 'Reconnecting…');
    setTimeout(connect, 1200);
  };
  ws.onerror = () => { /* onclose handles it */ };
  ws.onmessage = (event) => {
    if (typeof event.data === 'string') onEvent(JSON.parse(event.data));
    else onAudio(event.data);
  };
}

function send(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
}

/* -------------------------------------------------------------- server msgs */
function onEvent(msg) {
  switch (msg.type) {
    case 'ready':
      state.audioRate = msg.config.output_sample_rate || 24000;
      state.audioFormat = msg.config.audio_format || 'pcm';
      els.start.disabled = false;
      setStatus('Ready to talk', 'Press Start, then just speak.');
      break;

    case 'speech_start':
      // Optimistic: only the server's `cancel` actually stops playback, so a
      // stray noise can't cut Halo off.
      state.draining = false;
      state.phase = 'listening';
      ensureUserBubble(true);
      setStatus('Listening…', '');
      break;

    case 'speech_end':
      setStatus('Transcribing…', '');
      break;

    case 'partial':
      ensureUserBubble().textContent = msg.text;
      break;

    case 'user': {
      const bubble = ensureUserBubble();
      bubble.classList.remove('provisional');
      bubble.textContent = msg.text || '…';
      state.userBubble = null;
      break;
    }

    case 'assistant_start':
      state.draining = false;
      state.phase = 'thinking';
      state.botBubble = addBubble('bot', '');
      setStatus('Thinking…', '');
      break;

    case 'assistant_delta':
      if (!state.botBubble) state.botBubble = addBubble('bot', '');
      state.botBubble.dataset.raw = (state.botBubble.dataset.raw || '') + msg.text;
      state.botBubble.textContent = state.botBubble.dataset.raw;
      scrollDown();
      break;

    case 'assistant_done':
      state.phase = 'speaking';
      if (state.botBubble && !state.botBubble.dataset.raw) state.botBubble.textContent = msg.text || '(no reply)';
      break;

    case 'audio_start':
      state.streaming = true;
      state.phase = 'speaking';
      state.audioRate = msg.sample_rate || state.audioRate;
      if (msg.format) state.audioFormat = msg.format;
      state.mp3Parts = state.audioFormat === 'mp3' ? [] : null;
      setStatus('Speaking', '');
      break;

    case 'audio_end':
      if (state.mp3Parts) {
        const parts = state.mp3Parts;
        state.mp3Parts = null;
        if (parts.length && player) {
          const blob = new Blob(parts, { type: 'audio/mpeg' });
          blob.arrayBuffer().then((buf) => player.enqueueMp3(buf)).catch(() => {});
        }
      }
      break;

    case 'idle':
      state.streaming = false;
      state.botBubble = null;
      // Done *sending* is not done *speaking*: the player still holds audio.
      // Stay in "speaking" and let the meter loop flip us over when it drains.
      if (player && player.busy()) {
        state.draining = true;
        setStatus('Speaking', '');
      } else {
        state.phase = 'idle';
        setStatus('Ready to talk', '');
      }
      break;

    case 'cancel':
      if (player) player.stop();
      state.streaming = false;
      state.draining = false;
      state.mp3Parts = null;
      state.phase = 'listening';
      if (state.botBubble) { state.botBubble.classList.add('interrupted'); state.botBubble = null; }
      break;

    case 'reset_done':
      if (player) player.stop();
      state.draining = false;
      els.transcript.innerHTML = '';
      els.empty.style.display = '';
      setStatus('Conversation reset', '');
      break;

    case 'config_applied':
      if (msg.config) applyConfigToUI(msg.config);
      setStatus('Settings applied', '');
      break;

    case 'error':
      addBubble('bot', msg.message);
      setStatus('Error', msg.message, 'bad');
      break;

    default:
      break;
  }
}

function onAudio(buf) {
  if (!player) return;
  state.streaming = true;
  if (state.audioFormat === 'mp3') {
    if (state.mp3Parts) state.mp3Parts.push(buf);
    return;
  }
  player.enqueue(new Int16Array(buf), state.audioRate);
}

/* -------------------------------------------------------------- transcript */
function addBubble(kind, text) {
  els.empty.style.display = 'none';
  const bubble = document.createElement('div');
  bubble.className = `bubble ${kind}`;
  const who = document.createElement('span');
  who.className = 'who';
  who.textContent = kind === 'user' ? 'You' : 'Halo';
  const body = document.createElement('span');
  body.textContent = text;
  bubble.append(who, body);
  els.transcript.appendChild(bubble);
  scrollDown();
  return body;
}

function ensureUserBubble(provisional = false) {
  if (!state.userBubble) {
    state.userBubble = addBubble('user', provisional ? '…' : '');
    if (provisional) state.userBubble.parentElement.classList.add('provisional');
  }
  return state.userBubble;
}

function scrollDown() {
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

/* ---------------------------------------------------------------- status UI */
function setStatus(main, sub) {
  els.status.textContent = main;
  els.substatus.textContent = sub || '';
}

function renderBadges(health) {
  if (!health || !health.status) return;
  const order = [['vad', 'VAD'], ['stt', 'Speech'], ['tts', 'Voice'], ['llm', 'Chat']];
  els.badges.innerHTML = order
    .map(([key, label]) => {
      const st = health.status[key] || 'pending';
      const cls = st === 'ready' ? 'ready' : st === 'error' ? 'error' : st === 'pending' ? '' : 'loading';
      const text = st === 'ready' ? 'ready' : st === 'error' ? 'failed' : st;
      return `<span class="badge ${cls}" title="${(health.errors && health.errors[key]) || ''}">${label}<i>${text}</i></span>`;
    })
    .join('');
}

async function pollHealth() {
  try {
    const health = await (await fetch('/api/health')).json();
    renderBadges(health);
    if (!health.ready) setStatus('Waking up…', 'Loading the engines; the first visit after a pause takes a moment.');
    else if (state.phase === 'idle' && !state.started) {
      els.start.disabled = false;
      const llmDown = health.status && health.status.llm === 'error';
      // Don't let the health poll clobber the microphone warning.
      if (state.micBlocked) setStatus('Microphone blocked', MIC_HINT, 'bad');
      else if (llmDown) {
        // The server's message explains the fix (API key, or `ollama serve`).
        setStatus('Chat unavailable', (health.errors && health.errors.llm) || 'The language model is not reachable.');
      } else setStatus('Ready to talk', 'Press Start, then just speak.');
    }
  } catch { /* server restarting */ }
  setTimeout(pollHealth, 2500);
}

/* --------------------------------------------------------------- start/stop */
const MIC_ERRORS = {
  NotAllowedError:
    'Permission denied. Click the icon at the far left of the address bar → ' +
    '“Allow” (or “Reset permissions”), then press Start again.',
  SecurityError:
    'The browser only grants microphone access on https:// or http://127.0.0.1 / http://localhost.',
  NotFoundError:
    'No microphone found. Plug one in, or check Windows → Privacy → Microphone.',
  NotReadableError:
    'The microphone is busy in another app (Teams, Discord, a recorder). Close it and retry.',
  OverconstrainedError:
    'This microphone cannot provide the requested audio format. Try a different input device.',
  AbortError:
    'The microphone request was interrupted. Press Start again.',
};

/* Ask the browser what it thinks of our microphone permission *before* the user
   presses Start, so a blocked mic is explained instead of just failing. */
const MIC_HINT =
  'This site is not allowed to use your microphone. Click the icon at the far left ' +
  'of the address bar → “Allow” (or “Reset permissions”), then press Start again.';

function setMicBlocked(blocked) {
  state.micBlocked = blocked;
  if (blocked && state.phase === 'idle' && !state.started) {
    setStatus('Microphone blocked', MIC_HINT, 'bad');
  }
}

async function checkMicPermission() {
  if (!navigator.permissions || !navigator.mediaDevices) return 'unknown';
  try {
    const status = await navigator.permissions.query({ name: 'microphone' });
    setMicBlocked(status.state === 'denied');
    // Chrome re-fires this when the user flips the permission in the address bar,
    // so the UI heals itself without a reload.
    status.onchange = () => setMicBlocked(status.state === 'denied');
    return status.state;
  } catch {
    return 'unknown';
  }
}

async function startTalking() {
  try {
    setStatus('Requesting microphone…', '');
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        // Capture at the device's native rate and let the 16 kHz context do
        // the resampling - the driver's downmix to 16 kHz is usually worse.
        sampleRate: { ideal: 48000 },
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    if (!player) player = new Player(24000);
    if (player.ctx.state === 'suspended') await player.ctx.resume();

    mic = new Mic(stream);
    mic.send = (int16) => {
      if (!canSendAudio()) return;
      if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(int16.buffer);
    };
    await mic.init();

    state.started = true;
    state.micBlocked = false;
    state.micPeak = 0;
    state.micStartedAt = 0;
    state.micWarned = false;
    els.start.disabled = true;
    els.start.textContent = 'Running';
    els.mic.disabled = false;
    els.stop.disabled = false;
    els.reset.disabled = false;
    els.textInput.disabled = false;
    els.send.disabled = false;
    setStatus('Ready to talk', 'Speak whenever you like.');
  } catch (err) {
    const name = err && err.name ? err.name : 'Error';
    setMicBlocked(name === 'NotAllowedError');
    setStatus('Microphone unavailable', MIC_ERRORS[name] || String(err.message || err), 'bad');
    // Leave Start clickable so a retry works once the permission is fixed.
    els.start.disabled = false;
    els.start.textContent = 'Start talking';
    console.warn('getUserMedia failed:', name, err);
  }
}

function canSendAudio() {
  if (!state.started || state.muted) return false;
  if (state.mode === 'ptt' && !state.pttHeld) return false;
  if (state.echoGuard && player && (player.busy() || state.phase === 'speaking')) return false;
  return true;
}

/* ----------------------------------------------------------------- meter
 * Two soft rings instead of a gauge: terracotta follows your microphone, warm
 * charcoal follows the assistant, and the centre swells gently with whichever
 * channel is live. Every mark here tracks a real level.
 */
const orbCanvas = els.orb;
const orb = orbCanvas.getContext('2d');

function sizeOrb() {
  const rect = orbCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const size = Math.max(1, Math.round(rect.width * dpr));
  if (orbCanvas.width === size && orbCanvas.height === size) return;
  orbCanvas.width = size;
  orbCanvas.height = size;
  orb.setTransform(dpr, 0, 0, dpr, 0, 0);
}
sizeOrb();
window.addEventListener('resize', sizeOrb);

let lastSig = '';

function drawOrb() {
  const dpr = window.devicePixelRatio || 1;
  const size = orbCanvas.width / dpr;
  const c = size / 2;
  const R = size * 0.33;
  const TAU = Math.PI * 2;
  const top = -Math.PI / 2;

  const micLvl = mic ? mic.level() : 0;
  const outLvl = player ? player.level() : 0;
  // Asymmetric smoothing: snap up to a syllable quickly, decay off it slowly.
  // Symmetric smoothing made the ring trail the voice noticeably.
  const micK = micLvl > state.micLevel ? 0.5 : 0.2;
  const outK = outLvl > state.outLevel ? 0.6 : 0.2;
  state.micLevel += (micLvl - state.micLevel) * micK;
  state.outLevel += (outLvl - state.outLevel) * outK;

  // "Nothing happens when I speak" has three very different causes: the page
  // never got a mic, the device delivers silence (muted, wrong input, hardware
  // switch), or the audio never leaves the browser. The analyser is on the raw
  // device stream, so this catches the second case and says so instead of
  // leaving a still ring and no explanation.
  if (state.started && !state.muted) {
    if (!state.micStartedAt) state.micStartedAt = performance.now();
    if (micLvl > state.micPeak) state.micPeak = micLvl;
    const silentFor = (performance.now() - state.micStartedAt) / 1000;
    const warn = silentFor > 8 && state.micPeak < 0.02;
    if (warn !== state.micWarned) {
      state.micWarned = warn;
      if (warn) {
        setStatus(
          'Waiting for sound',
          'The ring is still because nothing has reached the page yet. Check that the right input is selected and that the mic is not muted, then reload and press Start again.'
        );
      } else {
        setStatus('Ready to talk', '');
      }
    }
  }

  // The server sends `idle` as soon as it stops *sending*, but the player
  // schedules up to 0.4 s of audio ahead, so Halo is usually still audible.
  // Promote the phase only once playback has genuinely drained, otherwise the
  // ring freezes while she is mid-sentence.
  if (state.draining && (!player || !player.busy())) {
    state.draining = false;
    state.phase = 'idle';
    setStatus('Ready to talk', '');
  }

  const inLevel = state.phase === 'listening' ? Math.min(1, state.micLevel) : 0;
  const outLevel = state.phase === 'speaking' ? Math.min(1, state.outLevel) : 0;

  // Skip the repaint when nothing has changed. An idle 60 fps redraw loop
  // steals CPU/GPU from Ollama on a laptop this size.
  const sig = `${state.phase}|${Math.round((inLevel + outLevel) * 24)}`;
  if (sig === lastSig) {
    requestAnimationFrame(drawOrb);
    return;
  }
  lastSig = sig;

  orb.clearRect(0, 0, size, size);
  orb.lineCap = 'round';

  // the resting ring
  orb.strokeStyle = 'rgba(27,26,24,0.10)';
  orb.lineWidth = size * 0.030;
  orb.beginPath();
  orb.arc(c, c, R, 0, TAU);
  orb.stroke();

  // assistant, inside
  if (outLevel > 0.004) {
    orb.strokeStyle = 'rgba(27,26,24,0.45)';
    orb.lineWidth = size * 0.019;
    orb.beginPath();
    orb.arc(c, c, R - size * 0.06, top, top + Math.max(0.06, outLevel) * TAU);
    orb.stroke();
  }

  // you, outside
  if (inLevel > 0.004) {
    orb.strokeStyle = '#c8102e';
    orb.lineWidth = size * 0.030;
    orb.beginPath();
    orb.arc(c, c, R, top, top + Math.max(0.06, inLevel) * TAU);
    orb.stroke();
  }

  // a soft centre that swells with whichever channel is live
  const active = Math.max(inLevel, outLevel);
  if (active > 0.01) {
    orb.fillStyle = 'rgba(200,16,46,0.13)';
    orb.beginPath();
    orb.arc(c, c, size * 0.10 * (1 + active * 0.4), 0, TAU);
    orb.fill();
  }

  requestAnimationFrame(drawOrb);
}

/* ------------------------------------------------------------------ settings */
async function loadConfig() {
  const cfg = await (await fetch('/api/config')).json();
  state.cfg = cfg;
  const voices = cfg.voices || [];
  els.setVoice.innerHTML = voices.length
    ? voices.map((v) => `<option value="${v.id}">${v.label}</option>`).join('')
    : '<option value="default">Automatic</option>';
  if (voices.length) els.setVoice.value = cfg.tts_voice;
  els.setPrompt.value = cfg.system_prompt;
  els.setSpeed.value = cfg.tts_speed;
  els.setSpeedVal.textContent = Number(cfg.tts_speed).toFixed(2);
  els.setModel.innerHTML = `<option value="${cfg.llm_model}">${cfg.llm_model}</option>`;
  els.setModel.value = cfg.llm_model;
  const stt = cfg.stt_backend === 'groq'
    ? `Groq ${cfg.groq_stt_model}`
    : `faster-whisper ${cfg.whisper_model}`;
  const tts = cfg.tts_engine === 'kokoro'
    ? `Kokoro ${cfg.tts_backend}/${cfg.tts_voice}`
    : cfg.tts_engine === 'edge'
      ? `Edge ${cfg.edge_voice}`
      : `gTTS ${cfg.gtts_lang}/${cfg.gtts_tld}`;
  const llm = cfg.llm_backend === 'groq'
    ? `Groq ${cfg.llm_model}`
    : `Ollama ${cfg.llm_model} @ ${cfg.ollama_url}`;
  els.stackInfo.textContent = `${stt} · ${tts} · ${llm}`;

  try {
    const r = await (await fetch('/api/models')).json();
    if (r.ok && r.models.length) {
      els.setModel.innerHTML = r.models
        .map((m) => `<option value="${m}">${m}</option>`)
        .join('');
      els.setModel.value = cfg.llm_model;
      if (!r.models.includes(cfg.llm_model)) {
        els.setModel.insertAdjacentHTML('afterbegin', `<option value="${cfg.llm_model}">${cfg.llm_model} (not pulled)</option>`);
        els.setModel.value = cfg.llm_model;
      }
    }
  } catch { /* keep default option */ }
}

function applyConfigToUI(cfg) {
  if (cfg.voice) els.setVoice.value = cfg.voice;
  if (cfg.llm_model) els.setModel.value = cfg.llm_model;
  if (cfg.speed) { els.setSpeed.value = cfg.speed; els.setSpeedVal.textContent = Number(cfg.speed).toFixed(2); }
}

function wireUI() {
  els.start.addEventListener('click', startTalking);

  els.mic.addEventListener('click', () => {
    state.muted = !state.muted;
    els.mic.classList.toggle('muted', state.muted);
    els.mic.classList.toggle('on', !state.muted);
    setStatus(state.muted ? 'Microphone muted' : 'Ready to talk', '');
  });

  els.stop.addEventListener('click', () => {
    if (player) player.stop();
    send({ type: 'interrupt' });
    setStatus('Stopped', '');
  });

  els.reset.addEventListener('click', () => send({ type: 'reset' }));

  els.composer.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = els.textInput.value.trim();
    if (!text) return;
    els.textInput.value = '';
    send({ type: 'text_input', text });
  });

  els.settingsBtn.addEventListener('click', () => {
    if (typeof els.settings.showModal === 'function') els.settings.showModal();
  });

  els.setSpeed.addEventListener('input', () => {
    els.setSpeedVal.textContent = Number(els.setSpeed.value).toFixed(2);
  });

  els.applyBtn.addEventListener('click', () => {
    send({
      type: 'config',
      voice: els.setVoice.value,
      speed: Number(els.setSpeed.value),
      llm_model: els.setModel.value,
      system_prompt: els.setPrompt.value,
    });
  });

  els.setMode.addEventListener('change', () => {
    state.mode = els.setMode.value;
    setStatus(state.mode === 'ptt' ? 'Push to talk' : 'Hands-free', state.mode === 'ptt' ? 'Hold Space to speak.' : '');
  });

  els.setEcho.addEventListener('change', () => { state.echoGuard = els.setEcho.checked; });

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && state.mode === 'ptt' && !isTyping(e.target)) {
      e.preventDefault();
      if (!state.pttHeld) { state.pttHeld = true; els.mic.classList.add('on'); }
    }
  });
  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && state.mode === 'ptt' && !isTyping(e.target)) {
      e.preventDefault();
      if (state.pttHeld) {
        state.pttHeld = false;
        els.mic.classList.remove('on');
        send({ type: 'flush' }); // don't make the user wait for the silence timeout
      }
    }
  });
}

function isTyping(el) {
  return el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA');
}

/* -------------------------------------------------------------------- boot */
(async function boot() {
  wireUI();
  requestAnimationFrame(drawOrb);
  await loadConfig();
  connect();
  pollHealth();
  checkMicPermission();
})();
