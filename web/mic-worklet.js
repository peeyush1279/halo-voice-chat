// Mic capture worklet.
//
// app.js opens the AudioContext at 16 kHz, so normally the browser has already
// resampled the device stream for us and this processor is a straight
// pass-through that packs 20 ms int16 frames.
//
// When a browser refuses that rate we still have to resample here, and the
// naive version of that is a quality bug: dropping 48 kHz to 16 kHz with plain
// linear interpolation folds everything above 8 kHz back down into the speech
// band as aliasing. So the fallback runs a low-pass FIR before it decimates.
//
// Both paths work out of preallocated buffers. An audio thread that allocates
// is an audio thread that eventually glitches.

const OUT_RATE = 16000;
const CHUNK = 320; // 20 ms @ 16 kHz
const TAPS = 33;   // low-pass length for the fallback path

class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / OUT_RATE;
    this.passthrough = Math.abs(this.ratio - 1) < 1e-6;

    this.out = new Int16Array(CHUNK);
    this.outLen = 0;

    // Scratch for the stereo -> mono fold, allocated up front. A 2-channel
    // device stream reaches downmix() on the pass-through path too, and an
    // undefined buffer here threw on the first render quantum - which kills a
    // worklet for the whole session, so the mic went silent with no error
    // anywhere except the browser console.
    this.mono = new Float32Array(0);
    this.work = new Float32Array(0);
    this.filt = new Float32Array(0);

    if (!this.passthrough) {
      // Source samples waiting to be read, plus a fractional cursor into them.
      this.src = new Float32Array(4096);
      this.len = 0;
      this.pos = 0;
      // Low-pass state, carried across render quanta.
      this.taps = designLowpass(TAPS, 7200 / sampleRate);
      this.hist = new Float32Array(TAPS - 1);
    }
  }

  process(inputs) {
    try {
      return this.render(inputs);
    } catch (err) {
      // Never let one bad quantum silence the microphone forever.
      if (!this.failed) {
        this.failed = true;
        this.port.postMessage({ error: String((err && err.message) || err) });
      }
      return true;
    }
  }

  render(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    let block = input[0];
    if (input.length > 1) block = this.downmix(input);

    if (this.passthrough) {
      for (let i = 0; i < block.length; i++) this.emit(block[i]);
      return true;
    }

    this.append(this.lowpass(block));

    while (this.pos + 1 < this.len) {
      const i = this.pos | 0;
      const frac = this.pos - i;
      this.emit(this.src[i] * (1 - frac) + this.src[i + 1] * frac);
      this.pos += this.ratio;
    }

    const used = Math.min(this.pos | 0, this.len);
    if (used > 0) {
      this.src.copyWithin(0, used, this.len);
      this.len -= used;
      this.pos -= used;
    }
    return true;
  }

  // ---- fallback path helpers ---------------------------------------------

  downmix(input) {
    const first = input[0];
    const n = first ? first.length : 0;
    if (!this.mono || this.mono.length !== n) this.mono = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      let used = 0;
      for (let c = 0; c < input.length; c++) {
        const ch = input[c];
        if (!ch) continue;          // a channel can be absent, not just silent
        sum += ch[i] || 0;
        used++;
      }
      this.mono[i] = used ? sum / used : 0;
    }
    return this.mono;
  }

  lowpass(block) {
    const n = this.taps.length;
    const need = block.length + n - 1;
    if (this.work.length < need) this.work = new Float32Array(need);
    if (this.filt.length < block.length) this.filt = new Float32Array(block.length);

    const work = this.work;
    work.set(this.hist, 0);
    work.set(block, n - 1);
    for (let i = 0; i < block.length; i++) {
      let acc = 0;
      for (let k = 0; k < n; k++) acc += work[i + k] * this.taps[k];
      this.filt[i] = acc;
    }
    this.hist.set(work.subarray(block.length, need));
    return this.filt.subarray(0, block.length);
  }

  append(block) {
    if (this.len + block.length > this.src.length) {
      // Only reachable on an unusual render size; drop the oldest audio rather
      // than grow, so the audio thread can never allocate.
      const drop = Math.min(this.len, this.len + block.length - this.src.length);
      this.src.copyWithin(0, drop, this.len);
      this.len -= drop;
      this.pos = Math.max(0, this.pos - drop);
    }
    this.src.set(block, this.len);
    this.len += block.length;
  }

  emit(sample) {
    const v = sample < -1 ? -1 : sample > 1 ? 1 : sample;
    this.out[this.outLen++] = v < 0 ? v * 0x8000 : v * 0x7fff;
    if (this.outLen === CHUNK) {
      const packet = this.out.slice(0);
      this.port.postMessage(packet, [packet.buffer]);
      this.outLen = 0;
    }
  }
}

// Blackman-windowed sinc, normalised to unity DC gain. `fc` is cycles/sample.
function designLowpass(n, fc) {
  const taps = new Float32Array(n);
  const mid = (n - 1) / 2;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const x = i - mid;
    const sinc = x === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x);
    const w =
      0.42 -
      0.5 * Math.cos((2 * Math.PI * i) / (n - 1)) +
      0.08 * Math.cos((4 * Math.PI * i) / (n - 1));
    taps[i] = sinc * w;
    sum += taps[i];
  }
  for (let i = 0; i < n; i++) taps[i] /= sum;
  return taps;
}

registerProcessor('mic-processor', MicProcessor);
