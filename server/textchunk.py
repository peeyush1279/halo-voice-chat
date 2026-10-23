"""Turn a token stream into speakable chunks.

Low latency to first audio comes from speaking the *first clause* as soon as it
exists instead of waiting for the whole reply.  At the same time we must not
hand the TTS engine fragments so small that prosody falls apart, so we look for
punctuation boundaries with a minimum length and force a cut at a maximum.
"""

from __future__ import annotations

import re

# Markdown / non-speech artefacts we strip before synthesis.
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.*?)\1")
_STAGE_DIRECTION = re.compile(r"(?<![\w])\((?:\*[^)]*|[a-z ]{1,20})\)", re.IGNORECASE)
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002600-\U000026FF"
    "\U0000FE0F"
    "]+"
)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")

_SENTENCE_END = re.compile(r"[.!?\u2026\u3002\uFF01\uFF1F]+(?=\s|$)")
_CLAUSE_END = re.compile(r"[,;:\u2014\u2013](?=\s)")


def clean_for_speech(text: str) -> str:
    text = _CODE_FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _BOLD_ITALIC.sub(r"\2", text)
    text = _STAGE_DIRECTION.sub(" ", text)
    text = _EMOJI.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def _find_split(buf: str, min_chars: int, max_chars: int, first_max: int = 0, is_first: bool = False) -> int:
    """Return an exclusive end index to cut `buf` at, or -1 to keep buffering."""
    n = len(buf)

    for m in _SENTENCE_END.finditer(buf):
        end = m.end()
        if end < min_chars:
            continue
        # Don't break "3.5" or "1,000" style tokens.
        if buf[m.start()] == "." and m.start() > 0 and buf[m.start() - 1].isdigit():
            if end < n and buf[end].isdigit():
                continue
        return end

    if n >= max_chars:
        cut = buf.rfind(" ", 0, max_chars)
        return cut + 1 if cut > 0 else max_chars

    if n >= min_chars:
        for m in _CLAUSE_END.finditer(buf):
            if m.end() >= min_chars:
                return m.end()

    # Fast path for the *first* chunk only: replies routinely open with a 60-70
    # character clause ("Did you know that there is a species of jellyfish
    # that is immortal,"), and waiting for that comma costs more than a second
    # of generation *plus* the whole Kokoro synthesis of it.  Say the opening
    # words as soon as they are speakable, then fall back to normal
    # punctuation-driven chunking so the rest of the reply keeps its prosody.
    if is_first and first_max and n >= first_max:
        cut = buf.rfind(" ", 0, first_max)
        if cut + 1 >= min_chars:
            return cut + 1

    return -1


class SpeechChunker:
    def __init__(self, min_chars: int = 14, max_chars: int = 150, first_chars: int = 0):
        self.min_chars = min_chars
        self.max_chars = max_chars
        # Only used for the opening chunk; 0 (or < 0) disables the fast path so
        # every boundary waits for punctuation exactly as before.
        self.first_chars = first_chars
        self._emitted = 0
        self._buf = ""

    def push(self, piece: str) -> list[str]:
        self._buf += piece
        out: list[str] = []
        while True:
            idx = _find_split(
                self._buf,
                self.min_chars,
                self.max_chars,
                first_max=self.first_chars,
                is_first=not self._emitted,
            )
            if idx < 0:
                break
            chunk, self._buf = self._buf[:idx], self._buf[idx:]
            cleaned = clean_for_speech(chunk)
            if cleaned:
                self._emitted += 1
                out.append(cleaned)
        return out

    def flush(self) -> list[str]:
        cleaned = clean_for_speech(self._buf)
        self._buf = ""
        return [cleaned] if cleaned else []
