"""
Central configuration for the Telugu Voice Agent system.

All model paths, GPU settings, audio parameters, VAD tuning, and latency
targets are defined here. Import this module in all backend components.
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional
import torch


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    # HuggingFace authentication (required for gated models like Llama-3)
    # ------------------------------------------------------------------ #
    HF_TOKEN: str = ""  # Set via HF_TOKEN env var or .env file

    # ------------------------------------------------------------------ #
    # Model identifiers
    # ------------------------------------------------------------------ #
    STT_MODEL: str = "/workspace/telugu-voice-agent/models/whisper-telugu-base-ct2"
    LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    TTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"

    # ------------------------------------------------------------------ #
    # GPU / compute settings
    # ------------------------------------------------------------------ #
    CUDA_DEVICE: str = "cuda:0"
    # float16 gives best speed/memory balance on A40 / RTX 3060+
    TORCH_DTYPE: str = "float16"
    # faster-whisper compute type (mirrors TORCH_DTYPE)
    STT_COMPUTE_TYPE: str = "float16"

    # vLLM GPU memory fraction (leave headroom for STT + TTS on same GPU)
    VLLM_GPU_MEMORY_UTILIZATION: float = 0.55
    # Tensor parallel degree — 1 for single-GPU setups
    VLLM_TENSOR_PARALLEL_SIZE: int = 1
    # Max concurrent sequences vLLM will schedule
    VLLM_MAX_NUM_SEQS: int = 4

    # ------------------------------------------------------------------ #
    # Audio settings — phone-call / full-duplex mode
    # ------------------------------------------------------------------ #
    # STT expects 16 kHz mono PCM
    STT_SAMPLE_RATE: int = 16000
    # MMS-TTS outputs 16 kHz natively (matches STT input rate)
    TTS_SAMPLE_RATE: int = 16000
    # AudioWorklet sends 20ms chunks of Int16 PCM
    AUDIO_CHUNK_MS: int = 20
    # Samples per 20ms chunk at 16kHz: 320
    AUDIO_CHUNK_SAMPLES: int = 320
    # Bytes per 20ms chunk (320 samples * 2 bytes/sample = 640)
    AUDIO_CHUNK_BYTES: int = 640

    # ------------------------------------------------------------------ #
    # VAD (Silero VAD) settings
    # ------------------------------------------------------------------ #
    # Speech probability threshold to enter LISTENING state
    VAD_SPEECH_THRESHOLD: float = 0.15
    # Probability below which we consider a frame silent (hysteresis)
    VAD_SILENCE_THRESHOLD: float = 0.05
    # Consecutive ms of speech required to confirm utterance start
    VAD_MIN_SPEECH_DURATION_MS: int = 200
    # Consecutive ms of silence after speech to trigger end-of-utterance
    VAD_SILENCE_DURATION_MS: int = 300
    # Maximum utterance length before forced processing (ms)
    VAD_MAX_UTTERANCE_MS: int = 15000

    # ------------------------------------------------------------------ #
    # LLM sampling parameters
    # ------------------------------------------------------------------ #
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 150
    LLM_TOP_P: float = 0.95
    LLM_REPETITION_PENALTY: float = 1.1
    # Keep last N turns (each turn = 1 user + 1 assistant message)
    LLM_MAX_HISTORY_TURNS: int = 4

    # ------------------------------------------------------------------ #
    # Latency targets (milliseconds — informational)
    # ------------------------------------------------------------------ #
    TARGET_STT_MS: int = 200
    TARGET_LLM_FIRST_TOKEN_MS: int = 300
    TARGET_TTS_FIRST_CHUNK_MS: int = 200
    TARGET_TOTAL_MS: int = 700

    # ------------------------------------------------------------------ #
    # Server settings
    # ------------------------------------------------------------------ #
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]

    # ------------------------------------------------------------------ #
    # Optional: path to a reference WAV for XTTS voice cloning.
    # If None, XTTS uses its built-in Telugu speaker embedding.
    # ------------------------------------------------------------------ #
    TTS_REFERENCE_AUDIO: Optional[str] = None
    TTS_REFERENCE_SPEAKER: str = "Kumar Dahl"

    # ------------------------------------------------------------------ #
    # Telugu language codes
    # ------------------------------------------------------------------ #
    STT_LANGUAGE: str = "te"
    TTS_LANGUAGE: str = "hi"  # XTTS v2 doesn't support 'te'; use 'hi' for Indian accent
    LLM_RESPONSE_LANGUAGE: str = "Telugu"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# ------------------------------------------------------------------ #
# Apply HF_TOKEN to environment immediately so all HuggingFace calls
# (vLLM, faster-whisper hub downloads) can authenticate.
# ------------------------------------------------------------------ #
os.environ.setdefault("HF_TOKEN", settings.HF_TOKEN)
os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", settings.HF_TOKEN)


# ------------------------------------------------------------------ #
# Derived helpers (not pydantic fields)
# ------------------------------------------------------------------ #

def get_torch_dtype() -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(settings.TORCH_DTYPE, torch.float16)


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_gpu_info() -> dict:
    if not is_cuda_available():
        return {"available": False}
    device = torch.device(settings.CUDA_DEVICE)
    props = torch.cuda.get_device_properties(device)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    total = props.total_memory / (1024 ** 3)
    return {
        "available": True,
        "device": settings.CUDA_DEVICE,
        "name": props.name,
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(total - allocated, 2),
        "cuda_version": torch.version.cuda,
    }
