"""Silero VAD (ONNX) + a streaming turn detector.

`SileroVAD` is a tiny dependency-free wrapper around the ONNX graph, so we do
not need torch for voice activity detection.

`TurnDetector` turns a continuous stream of 16 kHz audio into utterance
start/end events, which is what actually decides "the user finished talking".
Getting this right is the single biggest factor in a natural voice loop:
too eager and you cut people off, too lazy and every reply feels sluggish.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import VAD_FRAME

log = logging.getLogger("v2v.vad")


class SileroVAD:
    """Single-window (512 samples @ 16 kHz) speech probability."""

    def __init__(self, model_path, sample_rate: int = 16_000):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        # The graph is tiny; extra threads cost more than they save at 32 ms/frame.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.sample_rate = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        # v5+ uses a single fused `state`; v4 used separate h/c LSTMs.
        if "state" in self.input_names:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            # v5 also wants the previous 64 samples glued to the front of every
            # 512-sample window - without this the graph outputs ~0 for speech.
            self._context = np.zeros((1, 64), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        frame = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        if "state" in self.input_names:
            x = np.concatenate([self._context, frame], axis=1)
            feeds = {"input": x, "state": self._state, "sr": self.sample_rate}
            out, self._state = self.session.run(None, feeds)
            self._context = x[:, -64:]
        else:
            feeds = {"input": frame, "sr": self.sample_rate, "h": self._h, "c": self._c}
            out, self._h, self._c = self.session.run(None, feeds)
        return float(np.asarray(out).reshape(-1)[0])


@dataclass
class Event:
    """`kind` is one of:

    start   - first frame above threshold. Optimistic, for UI feedback only.
    confirm - enough consecutive speech to believe it (min_speech_s). This is
              what is allowed to interrupt the assistant, so that a cough or a
              door slam cannot cut Halo off mid-sentence.
    end     - the utterance finished; `audio` holds the mono float32 utterance.
    """

    kind: str
    audio: np.ndarray | None = None


@dataclass
class TurnDetector:
    """Streaming utterance segmenter with hysteresis and pre-roll padding."""

    vad: SileroVAD
    sample_rate: int = 16_000
    frame: int = VAD_FRAME
    threshold: float = 0.5
    neg_threshold: float = 0.35
    min_speech_s: float = 0.22
    min_silence_s: float = 0.52
    pad_s: float = 0.16
    max_utterance_s: float = 30.0

    speaking: bool = field(default=False, init=False)
    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32), init=False)
    _preroll: deque = field(default_factory=deque, init=False)
    _utt: list = field(default_factory=list, init=False)
    _speech_run: int = field(default=0, init=False)
    _silence_run: int = field(default=0, init=False)
    _started: bool = field(default=False, init=False)

    # ---- helpers -----------------------------------------------------------
    @property
    def _frame_s(self) -> float:
        return self.frame / self.sample_rate

    def _frames_for(self, seconds: float) -> int:
        return max(1, int(round(seconds / self._frame_s)))

    def current_audio(self) -> np.ndarray | None:
        """Audio captured so far for the in-progress utterance (for partial STT)."""
        if not self.speaking or not self._utt:
            return None
        return np.concatenate(self._utt)

    def reset(self) -> None:
        self.speaking = False
        self._pending = np.zeros(0, np.float32)
        self._preroll.clear()
        self._utt.clear()
        self._speech_run = 0
        self._silence_run = 0
        self._started = False
        self.vad.reset()

    # ---- main loop ---------------------------------------------------------
    def push(self, audio: np.ndarray) -> list[Event]:
        events: list[Event] = []
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        self._pending = np.concatenate([self._pending, audio]) if self._pending.size else audio

        pos = 0
        n = self._pending.size
        min_speech_frames = self._frames_for(self.min_speech_s)
        min_silence_frames = self._frames_for(self.min_silence_s)
        pad_frames = self._frames_for(self.pad_s)
        max_frames = int(self.max_utterance_s / self._frame_s)

        while n - pos >= self.frame:
            window = self._pending[pos : pos + self.frame]
            pos += self.frame
            prob = self.vad(window)

            if not self.speaking:
                self._preroll.append(window)
                while len(self._preroll) > pad_frames:
                    self._preroll.popleft()
                if prob >= self.threshold:
                    self._speech_run += 1
                    if not self._started:
                        self._started = True
                        events.append(Event("start"))
                    if self._speech_run >= min_speech_frames:
                        self.speaking = True
                        self._utt = list(self._preroll)
                        self._silence_run = 0
                        events.append(Event("confirm"))
                else:
                    self._speech_run = 0
            else:
                self._utt.append(window)
                if prob < self.neg_threshold:
                    self._silence_run += 1
                else:
                    self._silence_run = 0

                forced = len(self._utt) >= max_frames
                if self._silence_run >= min_silence_frames or forced:
                    if not forced:
                        # Drop most of the trailing silence but keep a little pad.
                        keep = max(1, self._silence_run - pad_frames)
                        del self._utt[len(self._utt) - keep :]
                    utterance = np.concatenate(self._utt)
                    self._utt = []
                    self.speaking = False
                    self._speech_run = 0
                    self._silence_run = 0
                    self._started = False
                    self._preroll.clear()
                    self.vad.reset()
                    events.append(Event("end", utterance))

        self._pending = self._pending[pos:] if pos < n else np.zeros(0, np.float32)
        return events
