---
title: Halo Voice Chat
emoji: 🎙️
colorFrom: yellow
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Talk to a model out loud, in the browser.
---

# Halo

Voice-to-voice chat in the browser. You speak, it answers out loud, and you can
talk over it mid-sentence.

```
microphone -> Silero VAD -> Groq Whisper -> Groq LLM -> gTTS -> speaker
   16 kHz       endpointing    speech->text      chat      mp3
```

All the code and the fully-offline variant (faster-whisper + Ollama + Kokoro)
live at **https://github.com/peeyush1279/halo-voice-chat**.

Press **Start talking** and click **Allow** on the microphone prompt. Headphones
are strongly recommended: on speakers the mic hears the assistant and the two of
you interrupt each other forever.
