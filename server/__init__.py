"""Fully-local voice-to-voice chat server.

Pipeline:  microphone -> Silero VAD -> faster-whisper -> Ollama -> Kokoro-82M -> speaker
"""

__version__ = "0.1.0"
