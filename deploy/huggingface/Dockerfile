FROM python:3.12-slim

# Hosted stack only: no Whisper weights, no Kokoro, no Ollama in the image.
# Model files are never baked in - the 2 MB Silero VAD graph is fetched by
# server/download_models.py on first start (see engines._load_vad).

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    V2V_HOST=0.0.0.0 \
    V2V_PORT=7860 \
    V2V_STT_BACKEND=groq \
    V2V_LLM_BACKEND=groq \
    V2V_TTS_ENGINE=gtts

WORKDIR /app

# onnxruntime links against libgomp; slim does not ship it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server
COPY web ./web

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "server.main"]
