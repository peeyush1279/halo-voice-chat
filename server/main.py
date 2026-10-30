"""FastAPI app: static UI + websocket endpoint + model management."""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import MODELS_DIR, WEB_DIR, get_settings
from .engines import Engines
from .pipeline import Conversation

log = logging.getLogger("v2v")

settings = get_settings()
engines = Engines(settings, MODELS_DIR)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(name)-14s %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.ERROR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info(
        "voice-chat starting   stt=%s  llm=%s  tts=%s",
        settings.stt_backend,
        settings.llm_backend,
        settings.tts_engine,
    )
    if settings.llm_backend == "ollama":
        log.info("Ollama backend: %s  model=%s", settings.ollama_url, settings.llm_model)
    else:
        log.info("Groq backend: model=%s  key=%s", settings.groq_llm_model, "set" if settings.groq_api_key else "MISSING")

    async def boot() -> None:
        await asyncio.to_thread(engines.load_all)
        log.info("engines: %s", engines.status)
        await engines.warm_llm(settings.llm_model)

    boot_task = asyncio.create_task(boot())
    try:
        yield
    finally:
        boot_task.cancel()


app = FastAPI(title="Voice Chat", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.middleware("http")
async def no_cache_in_dev(request: Request, call_next):
    """Web assets must never be stale.

    The static mount sets a 30s/ETag cache policy, which means an edit to
    app.js/style.css keeps running the old copy until the browser decides to
    revalidate - very confusing while tuning the UI.  These files are a few KB
    and served from localhost, so bypass the HTTP cache entirely.
    """
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/config")
async def api_config() -> JSONResponse:
    return JSONResponse(settings.public())


@app.get("/api/health")
async def api_health() -> JSONResponse:
    return JSONResponse({**engines.snapshot(), "settings": {"llm_model": settings.llm_model}})


@app.get("/api/models")
async def api_models() -> JSONResponse:
    try:
        client = engines.make_llm()
        models = await client.list_models()
        await client.aclose()
        return JSONResponse({"ok": True, "models": [m.get("name") for m in models]})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc), "models": []}, status_code=200)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    conversation = Conversation(ws, engines)
    log.info("client connected")
    try:
        await conversation.run()
    finally:
        log.info("client disconnected")


def main() -> None:
    import uvicorn

    configure_logging()
    # Pass the app object (not an import string): the module is already loaded,
    # and "server.main:app" would import it a second time as a separate object.
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        ws="websockets",
    )


if __name__ == "__main__":
    main()
