"""
Telugu Speech-to-Text using faster-whisper.

Model: vasista22/whisper-telugu-base (HuggingFace fine-tune of Whisper Base)

In full-duplex phone-call mode the backend receives raw Int16 PCM bytes
(16kHz, mono, little-endian) accumulated by the VAD engine.  This module
converts those bytes directly to a float32 numpy array — no audio container
decoding required, eliminating the ffmpeg/soundfile overhead of the old
webm/opus path.

The legacy _decode_audio() method (soundfile + librosa fallback) is kept
for compatibility with any non-PCM test callers.
"""

import asyncio
import io
import time
from typing import Optional

import numpy as np
import soundfile as sf
import librosa
from faster_whisper import WhisperModel
from loguru import logger

from backend.config import settings


class TeluguSTT:
    """
    Wraps faster-whisper for real-time Telugu transcription.

    Primary entrypoint in full-duplex mode:
        text = stt.transcribe_pcm(pcm_int16_bytes)

    Legacy entrypoint (webm/opus bytes):
        text = stt.transcribe_chunk(audio_bytes)
    """

    def __init__(self):
        self._model: Optional[WhisperModel] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load the model onto GPU. Call once at startup."""
        logger.info(f"Loading STT model: {settings.STT_MODEL}")
        start = time.perf_counter()
        self._model = WhisperModel(
            settings.STT_MODEL,
            device="cuda",
            compute_type=settings.STT_COMPUTE_TYPE,
            download_root=None,
            local_files_only=False,
        )
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"STT model loaded in {elapsed:.0f}ms")
        self._warmup()

    def _warmup(self) -> None:
        silence = np.zeros(settings.STT_SAMPLE_RATE, dtype=np.float32)
        try:
            segments, _ = self._model.transcribe(
                silence,
                language=settings.STT_LANGUAGE,
                beam_size=1,
            )
            _ = list(segments)
            logger.info("STT warm-up complete")
        except Exception as exc:
            logger.warning(f"STT warm-up failed (non-fatal): {exc}")

    # ------------------------------------------------------------------ #
    # Public API — full-duplex mode (raw Int16 PCM input)
    # ------------------------------------------------------------------ #

    def transcribe_pcm(self, pcm_int16_bytes: bytes) -> str:
        """
        Transcribe a raw Int16 PCM utterance.

        Args:
            pcm_int16_bytes: Raw signed 16-bit little-endian PCM bytes at
                             16000 Hz mono.  Produced by the VAD engine after
                             an utterance is complete.

        Returns:
            Transcribed Telugu string, or empty string for silence/noise.
        """
        if not self._model:
            raise RuntimeError("STT model not loaded. Call load() first.")

        if not pcm_int16_bytes or len(pcm_int16_bytes) < settings.AUDIO_CHUNK_BYTES:
            return ""

        # Direct conversion: Int16 bytes → float32 [-1, 1]
        pcm_int16 = np.frombuffer(pcm_int16_bytes, dtype=np.int16)
        pcm_f32 = pcm_int16.astype(np.float32) / 32768.0

        # Minimum duration check: skip if less than 200ms of audio
        min_samples = int(0.2 * settings.STT_SAMPLE_RATE)
        if len(pcm_f32) < min_samples:
            return ""

        return self._transcribe_numpy(pcm_f32)

    async def transcribe_pcm_async(self, pcm_int16_bytes: bytes) -> str:
        """
        Async wrapper around transcribe_pcm. Runs in the default thread pool
        executor to avoid blocking the event loop during inference.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe_pcm, pcm_int16_bytes)

    # ------------------------------------------------------------------ #
    # Legacy API — accepts webm/opus/wav bytes (kept for compatibility)
    # ------------------------------------------------------------------ #

    def transcribe_chunk(self, audio_bytes: bytes) -> str:
        """
        Decode audio_bytes (webm/opus/wav) and transcribe.
        Kept for backward compatibility and testing.
        """
        if not self._model:
            raise RuntimeError("STT model not loaded. Call load() first.")

        pcm = self._decode_audio(audio_bytes)
        if pcm is None or len(pcm) < 1600:
            return ""

        return self._transcribe_numpy(pcm)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _transcribe_numpy(self, pcm: np.ndarray) -> str:
        """Synchronous transcription of a float32 numpy array at 16 kHz."""
        try:
            start = time.perf_counter()
            segments, info = self._model.transcribe(
                pcm,
                language=settings.STT_LANGUAGE,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=False,
                word_timestamps=False,
                chunk_length=5,
                condition_on_previous_text=False,
            )
            tokens = [seg.text.strip() for seg in segments if seg.text.strip()]
            result = " ".join(tokens)
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug(f"STT transcribed in {elapsed:.0f}ms: '{result[:60]}'")
            return result
        except Exception as exc:
            logger.error(f"STT transcription error: {exc}")
            return ""

    def _decode_audio(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decode arbitrary audio bytes to 16 kHz mono float32 numpy array.
        Tries soundfile first (wav/ogg/flac) then librosa (webm/opus via ffmpeg).
        """
        if not audio_bytes:
            return None
        try:
            buf = io.BytesIO(audio_bytes)
            audio, sr = sf.read(buf, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != settings.STT_SAMPLE_RATE:
                audio = librosa.resample(
                    audio, orig_sr=sr, target_sr=settings.STT_SAMPLE_RATE
                )
            return audio.astype(np.float32)
        except Exception:
            pass

        try:
            buf = io.BytesIO(audio_bytes)
            audio, sr = librosa.load(
                buf, sr=settings.STT_SAMPLE_RATE, mono=True
            )
            return audio.astype(np.float32)
        except Exception as exc:
            logger.warning(f"Audio decode failed: {exc}")
            return None
