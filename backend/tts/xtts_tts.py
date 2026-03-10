"""
Telugu Text-to-Speech using Coqui XTTS v2.

Model: tts_models/multilingual/multi-dataset/xtts_v2

Full-duplex phone-call mode changes vs. previous version:
- synthesize_stream() now yields raw Int16 PCM bytes (no WAV header) after
  the first chunk, so the browser can receive and schedule them immediately
  without container parsing overhead.
- The first chunk still has no WAV header — the client knows the format from
  the "ready" JSON message sent at connection time (Int16, 22050Hz, mono).
- XTTS native output is 24kHz float32; resampled to 22050Hz then Int16.
- Sentence boundary chunking retained: split at ।?!.\n with min 15 chars.
"""

import asyncio
import re
import time
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from loguru import logger
from TTS.api import TTS

from backend.config import settings, get_torch_dtype


# Telugu / Latin sentence boundary pattern.
# Matches after: Devanagari danda (।), full stop, ?, !, newline.
_SENTENCE_RE = re.compile(r"(?<=[।.?!\n])\s*")

# Minimum characters to accumulate before synthesizing a chunk.
_MIN_CHUNK_CHARS = 15

# XTTS v2 native output sample rate
_XTTS_NATIVE_SR = 24000


class TeluguTTS:
    """
    Wraps Coqui XTTS v2 for streaming Telugu synthesis.

    Full-duplex mode usage:
        async for pcm_bytes in tts.synthesize_stream(token_generator, interrupt_event):
            await ws.send_bytes(pcm_bytes)   # raw Int16 PCM, 22050Hz mono

    Each yielded chunk is raw signed 16-bit little-endian PCM at TTS_SAMPLE_RATE.
    No WAV headers are included; the client decodes raw samples directly.
    """

    def __init__(self):
        self._tts: Optional[TTS] = None
        self._speaker_wav: Optional[str] = None
        self._speaker_name: Optional[str] = None
        self._device: str = settings.CUDA_DEVICE

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load XTTS v2 onto GPU. Call once at startup."""
        logger.info(f"Loading TTS model: {settings.TTS_MODEL}")
        start = time.perf_counter()

        self._tts = TTS(
            model_name=settings.TTS_MODEL,
            progress_bar=True,
            gpu=True,
        )
        self._tts.to(self._device)

        if settings.TTS_REFERENCE_AUDIO:
            self._speaker_wav = settings.TTS_REFERENCE_AUDIO
            self._speaker_name = None
            logger.info(f"TTS using reference audio: {self._speaker_wav}")
        else:
            self._speaker_wav = None
            self._speaker_name = settings.TTS_REFERENCE_SPEAKER
            logger.info(f"TTS using built-in speaker: {self._speaker_name}")

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"TTS model loaded in {elapsed:.0f}ms")
        self._warmup()

    def _warmup(self) -> None:
        try:
            _ = self._synthesize_sync("నమస్కారం.")
            logger.info("TTS warm-up complete")
        except Exception as exc:
            logger.warning(f"TTS warm-up failed (non-fatal): {exc}")

    # ------------------------------------------------------------------ #
    # Public API — streaming (primary path)
    # ------------------------------------------------------------------ #

    async def synthesize_stream(
        self,
        text_generator: AsyncGenerator[str, None],
        interrupt_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Consume an async token generator and yield raw Int16 PCM audio chunks
        at sentence boundaries.

        Args:
            text_generator:  Async generator of text tokens from the LLM.
            interrupt_event: If set, synthesis aborts at the next sentence
                             boundary.

        Yields:
            bytes: Raw signed 16-bit little-endian PCM at settings.TTS_SAMPLE_RATE.
                   No WAV header.
        """
        if not self._tts:
            raise RuntimeError("TTS model not loaded. Call load() first.")

        buffer = ""
        first_chunk_logged = False
        pipeline_start = time.perf_counter()

        async for token in text_generator:
            if interrupt_event and interrupt_event.is_set():
                logger.info("TTS interrupted — stopping synthesis")
                break

            buffer += token

            sentences = _split_at_boundaries(buffer)
            if len(sentences) < 2:
                continue

            # Complete sentences are all but the last fragment
            complete = sentences[:-1]
            buffer = sentences[-1]

            for sentence in complete:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 3:
                    continue

                if interrupt_event and interrupt_event.is_set():
                    logger.info("TTS interrupted mid-sentence")
                    return

                pcm_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self._synthesize_sync, sentence
                )
                if pcm_bytes:
                    if not first_chunk_logged:
                        elapsed = (time.perf_counter() - pipeline_start) * 1000
                        logger.debug(f"TTS first chunk ready in {elapsed:.0f}ms")
                        first_chunk_logged = True
                    yield pcm_bytes

        # Flush remaining buffer after generator exhausted
        remainder = buffer.strip()
        if remainder and len(remainder) >= 3:
            if not (interrupt_event and interrupt_event.is_set()):
                pcm_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self._synthesize_sync, remainder
                )
                if pcm_bytes:
                    yield pcm_bytes

    async def synthesize_text(self, text: str) -> bytes:
        """
        One-shot synthesis of a complete text string.
        Returns raw Int16 PCM bytes at TTS_SAMPLE_RATE.
        """
        if not self._tts:
            raise RuntimeError("TTS model not loaded. Call load() first.")
        return await asyncio.get_event_loop().run_in_executor(
            None, self._synthesize_sync, text
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _synthesize_sync(self, text: str) -> Optional[bytes]:
        """
        Synchronous synthesis of a single sentence.
        Returns raw Int16 PCM bytes at settings.TTS_SAMPLE_RATE, or None on error.
        No WAV header included.
        """
        if not text or not text.strip():
            return None
        try:
            start = time.perf_counter()

            with torch.inference_mode():
                if self._speaker_wav:
                    wav = self._tts.tts(
                        text=text,
                        speaker_wav=self._speaker_wav,
                        language=settings.TTS_LANGUAGE,
                    )
                else:
                    wav = self._tts.tts(
                        text=text,
                        speaker=self._speaker_name,
                        language=settings.TTS_LANGUAGE,
                    )

            # wav is a list of floats or numpy array from XTTS
            wav_np = np.array(wav, dtype=np.float32)
            wav_np = np.clip(wav_np, -1.0, 1.0)

            # Resample 24kHz → 22050Hz
            if _XTTS_NATIVE_SR != settings.TTS_SAMPLE_RATE:
                import librosa
                wav_np = librosa.resample(
                    wav_np,
                    orig_sr=_XTTS_NATIVE_SR,
                    target_sr=settings.TTS_SAMPLE_RATE,
                )

            # Convert float32 → int16 PCM
            pcm_int16 = (wav_np * 32767).astype(np.int16)

            # Return raw bytes (no WAV header) — client decodes raw samples
            raw_bytes = pcm_int16.tobytes()

            elapsed = (time.perf_counter() - start) * 1000
            audio_ms = len(pcm_int16) / settings.TTS_SAMPLE_RATE * 1000
            logger.debug(
                f"TTS synthesized {len(text)} chars / {audio_ms:.0f}ms audio "
                f"in {elapsed:.0f}ms ({len(raw_bytes)} bytes)"
            )
            return raw_bytes

        except Exception as exc:
            logger.error(f"TTS synthesis error for '{text[:40]}': {exc}")
            return None


# ------------------------------------------------------------------ #
# Module-level helpers
# ------------------------------------------------------------------ #

def _split_at_boundaries(text: str) -> list[str]:
    """
    Split text at Telugu/Latin sentence boundaries.
    Returns a list of fragments; the last may be incomplete.
    """
    parts = _SENTENCE_RE.split(text)
    return [p for p in parts if p] or [text]
