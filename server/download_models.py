"""Download the small model files the server needs.

Run once after installing requirements:  python -m server.download_models

* Silero VAD  (~2 MB)  - endpointing.  Usually already shipped inside the
                         `silero-vad` wheel, in which case we just copy it.
* Kokoro-82M   (~330 MB) - the text-to-speech voice.  Downloaded from the
                         kokoro-onnx GitHub release.

Whisper weights are fetched by faster-whisper itself on first use.
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent / ".." / "models"
MODELS_DIR = MODELS_DIR.resolve()

SILERO_URLS = [
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx",
]

KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {dest.name}  <-  {url}")

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100.0, blocks * block_size * 100.0 / total)
            sys.stdout.write(f"\r         {pct:5.1f}%  ({blocks * block_size / 1e6:7.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    sys.stdout.write("\r" + " " * 60 + "\r")
    tmp.replace(dest)


def ensure_silero() -> Path:
    """Return a path to a silero_vad.onnx, copying it out of the wheel if we can."""
    dest = MODELS_DIR / "silero_vad.onnx"
    if dest.exists():
        return dest
    try:
        import silero_vad  # type: ignore

        pkg = Path(silero_vad.__file__).parent
        for candidate in (pkg / "data" / "silero_vad.onnx", pkg / "silero_vad.onnx"):
            if candidate.exists():
                shutil.copyfile(candidate, dest)
                print(f"  [copy] silero_vad.onnx  <-  {candidate}")
                return dest
    except Exception as exc:  # pragma: no cover - best effort
        print(f"  [warn] could not inspect silero-vad package: {exc}")
    last_error: Exception | None = None
    for url in SILERO_URLS:
        try:
            _download(url, dest)
            return dest
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"could not obtain silero_vad.onnx: {last_error}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download model files.")
    parser.add_argument(
        "--kokoro",
        action="store_true",
        help="also fetch Kokoro-82M (~330 MB); only needed for local TTS",
    )
    args = parser.parse_args(argv)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {MODELS_DIR}\n")
    print("Silero VAD (voice activity detection):")
    ensure_silero()
    if args.kokoro:
        print("\nKokoro-82M (text to speech):")
        for name, url in KOKORO_FILES.items():
            _download(url, MODELS_DIR / name)
    else:
        print("\nSkipping Kokoro: hosted mode speaks through gTTS/Edge (`--kokoro` to fetch).")
    print("\nDone.  Whisper is only downloaded in local mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
