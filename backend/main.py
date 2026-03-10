"""
FastAPI application entry point for the Telugu Voice Agent (full-duplex mode).

Startup sequence:
  1. Apply HF_TOKEN to environment
  2. Verify CUDA is available
  3. Load Silero VAD model (CPU)
  4. Load STT model (faster-whisper, GPU)
  5. Load LLM engine (vLLM, GPU)
  6. Load TTS model (Coqui XTTS v2, GPU)
  7. Serve static frontend files
  8. Accept WebSocket connections at /ws/audio

Each WebSocket connection creates an isolated VoicePipeline instance that
shares the globally pre-loaded model objects.
"""

import os
import sys
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from loguru import logger

# config must be imported first — it sets HF_TOKEN env vars at module level
from backend.config import settings, get_gpu_info, is_cuda_available
from backend.vad.silero_vad_engine import VADEngine
from backend.stt.whisper_stt import TeluguSTT
from backend.llm.vllm_engine import LLMEngine
from backend.tts.xtts_tts import TeluguTTS
from backend.pipeline.voice_pipeline import VoicePipeline


# ------------------------------------------------------------------ #
# Logger configuration
# ------------------------------------------------------------------ #
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
)
logger.add(
    "logs/telugu_agent.log",
    level="DEBUG",
    rotation="100 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} | {message}",
)


# ------------------------------------------------------------------ #
# Shared model singletons — loaded once, reused across all sessions
# ------------------------------------------------------------------ #
_vad: VADEngine = None
_stt: TeluguSTT = None
_llm: LLMEngine = None
_tts: TeluguTTS = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup; release references on shutdown."""
    global _vad, _stt, _llm, _tts

    logger.info("=" * 60)
    logger.info("Telugu Voice Agent — full-duplex mode — startup")
    logger.info("=" * 60)

    # Confirm HF_TOKEN is set
    hf_tok = os.environ.get("HF_TOKEN", "")
    if hf_tok:
        logger.info(f"HF_TOKEN set ({hf_tok[:8]}...)")
    else:
        logger.warning("HF_TOKEN not set — Llama-3 download will fail (gated model)")

    # Verify CUDA
    if not is_cuda_available():
        logger.error("CUDA is not available. Cannot start without a GPU.")
        raise RuntimeError("CUDA not available")

    gpu = get_gpu_info()
    logger.info(
        f"GPU: {gpu['name']}  "
        f"VRAM: {gpu['total_gb']}GB total / {gpu['free_gb']:.1f}GB free"
    )

    # Load VAD (CPU — fast)
    logger.info("[1/4] Loading Silero VAD...")
    _vad = VADEngine()
    _vad.load()

    # Load STT
    logger.info("[2/4] Loading STT model (faster-whisper)...")
    _stt = TeluguSTT()
    _stt.load()

    # Load LLM
    logger.info("[3/4] Loading LLM engine (vLLM)...")
    _llm = LLMEngine()
    _llm.load()

    # Load TTS
    logger.info("[4/4] Loading TTS model (Coqui XTTS v2)...")
    _tts = TeluguTTS()
    _tts.load()

    gpu_after = get_gpu_info()
    logger.info(
        f"All models loaded. "
        f"VRAM: {gpu_after['allocated_gb']:.1f}GB allocated / "
        f"{gpu_after['total_gb']}GB total"
    )
    logger.info("Telugu Voice Agent ready — accepting WebSocket connections")

    yield

    # Shutdown
    logger.info("Shutting down Telugu Voice Agent")
    _vad = None
    _stt = None
    _llm = None
    _tts = None


# ------------------------------------------------------------------ #
# FastAPI application
# ------------------------------------------------------------------ #
app = FastAPI(
    title="Telugu Voice Agent",
    description="Real-time full-duplex Telugu conversational voice agent",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# REST endpoints
# ------------------------------------------------------------------ #

@app.get("/health")
async def health_check():
    """Return service health and GPU utilisation."""
    gpu = get_gpu_info()
    models_loaded = all([_vad, _stt, _llm, _tts])
    return JSONResponse({
        "status": "healthy" if models_loaded else "loading",
        "models_loaded": models_loaded,
        "gpu": gpu,
        "mode": "full_duplex_phone_call",
        "config": {
            "stt_model": settings.STT_MODEL,
            "llm_model": settings.LLM_MODEL,
            "tts_model": settings.TTS_MODEL,
            "stt_language": settings.STT_LANGUAGE,
            "tts_language": settings.TTS_LANGUAGE,
            "audio_chunk_ms": settings.AUDIO_CHUNK_MS,
            "vad_speech_threshold": settings.VAD_SPEECH_THRESHOLD,
            "vad_silence_duration_ms": settings.VAD_SILENCE_DURATION_MS,
        },
    })


@app.get("/config")
async def get_config():
    """Return audio configuration for the frontend."""
    return JSONResponse({
        "tts_sample_rate": settings.TTS_SAMPLE_RATE,
        "stt_sample_rate": settings.STT_SAMPLE_RATE,
        "audio_chunk_ms": settings.AUDIO_CHUNK_MS,
        "audio_chunk_samples": settings.AUDIO_CHUNK_SAMPLES,
        "audio_encoding": "pcm_s16le",
    })


# ------------------------------------------------------------------ #
# WebSocket endpoint
# ------------------------------------------------------------------ #

@app.websocket("/ws/audio")
async def websocket_audio(ws: WebSocket):
    """
    Full-duplex audio WebSocket endpoint (phone-call mode).

    Client → Server:
      binary: raw Int16 PCM, 20ms chunks, 16kHz mono (AudioWorklet output)
      text:   JSON {"type": "interrupt"}

    Server → Client:
      binary: raw Int16 PCM, 22050Hz mono (TTS output, no WAV header)
      text:   JSON {"type": "ready"|"vad_state"|"transcript"|"error", ...}
    """
    await ws.accept()
    client_host = ws.client.host if ws.client else "unknown"
    logger.info(f"WebSocket connected: {client_host}")

    if not all([_vad, _stt, _llm, _tts]):
        import json as _json
        await ws.send_text(
            _json.dumps({
                "type": "error",
                "message": "Models still loading — try again shortly",
            })
        )
        await ws.close(code=1013)
        return

    pipeline = VoicePipeline(stt=_stt, llm=_llm, tts=_tts, vad=_vad)
    try:
        await pipeline.process_audio_stream(ws)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_host}")
    except Exception as exc:
        logger.error(f"WebSocket session error ({client_host}): {exc}")
    finally:
        logger.info(f"WebSocket session closed: {client_host}")


# ------------------------------------------------------------------ #
# Static file serving (frontend)
# ------------------------------------------------------------------ #

_frontend_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)

if os.path.isdir(_frontend_path):
    app.mount(
        "/",
        StaticFiles(directory=_frontend_path, html=True),
        name="frontend",
    )
    logger.info(f"Serving frontend from: {_frontend_path}")
else:
    @app.get("/")
    async def root():
        return HTMLResponse(
            "<h1>Telugu Voice Agent API</h1>"
            "<p>Frontend directory not found.</p>"
            "<p>WebSocket: <code>ws://localhost:8000/ws/audio</code></p>"
            "<p>Health: <a href='/health'>/health</a></p>"
        )


# ------------------------------------------------------------------ #
# Development server entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import uvicorn
    os.makedirs("logs", exist_ok=True)
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,           # reload=True breaks vLLM's process model
        log_level="info",
        access_log=True,
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )
