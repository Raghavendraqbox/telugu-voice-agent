"""
Telugu Text-to-Speech using Facebook MMS-TTS (Massively Multilingual Speech).

Model: facebook/mms-tts-tel  — natively supports Telugu (ISO 639-3: tel)

Advantages over XTTS v2:
- Native Telugu language support (XTTS only supports Hindi as closest Indian language)
- Much faster synthesis (~10x): single forward pass, no autoregressive decoding
- Lower VRAM (~200MB vs ~6GB for XTTS)
- 16 kHz output (matches STT input rate)

Output: raw Int16 PCM at 16000 Hz mono (no WAV header).
"""

import asyncio
import re
import time
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from loguru import logger
from transformers import VitsModel, AutoTokenizer

from backend.config import settings


# Sentence boundary pattern for Telugu + Latin punctuation
_SENTENCE_RE = re.compile(r"(?<=[।.?!\n])\s*")

# Minimum chars to accumulate before synthesizing
_MIN_CHUNK_CHARS = 8

_MMS_MODEL_ID = "facebook/mms-tts-tel"
_MMS_SAMPLE_RATE = 16000


class TeluguTTS:
    """
    Wraps facebook/mms-tts-tel for streaming Telugu synthesis.

    Full-duplex mode usage:
        async for pcm_bytes in tts.synthesize_stream(token_generator, interrupt_event):
            await ws.send_bytes(pcm_bytes)   # raw Int16 PCM, 16000Hz mono

    Each yielded chunk is raw signed 16-bit little-endian PCM at 16000 Hz.
    No WAV headers are included; the client decodes raw samples directly.
    """

    def __init__(self):
        self._model: Optional[VitsModel] = None
        self._tokenizer = None
        self._device: str = settings.CUDA_DEVICE

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load MMS-TTS Telugu onto GPU. Call once at startup."""
        logger.info(f"Loading TTS model: {_MMS_MODEL_ID}")
        start = time.perf_counter()

        self._tokenizer = AutoTokenizer.from_pretrained(_MMS_MODEL_ID)
        self._model = VitsModel.from_pretrained(_MMS_MODEL_ID)
        self._model = self._model.to(self._device)
        self._model.eval()

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"TTS model loaded in {elapsed:.0f}ms (MMS-TTS Telugu, {_MMS_SAMPLE_RATE}Hz)")
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

        Yields:
            bytes: Raw signed 16-bit little-endian PCM at 16000 Hz. No WAV header.
        """
        if not self._model:
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

        # Flush remaining buffer
        remainder = buffer.strip()
        if remainder and len(remainder) >= 3:
            if not (interrupt_event and interrupt_event.is_set()):
                pcm_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self._synthesize_sync, remainder
                )
                if pcm_bytes:
                    yield pcm_bytes

    async def synthesize_text(self, text: str) -> bytes:
        """One-shot synthesis. Returns raw Int16 PCM bytes at 16000 Hz."""
        if not self._model:
            raise RuntimeError("TTS model not loaded. Call load() first.")
        return await asyncio.get_event_loop().run_in_executor(
            None, self._synthesize_sync, text
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _synthesize_sync(self, text: str) -> Optional[bytes]:
        """Synchronous synthesis of a single sentence → raw Int16 PCM bytes."""
        if not text or not text.strip():
            return None
        try:
            start = time.perf_counter()

            inputs = self._tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self._model(**inputs).waveform

            wav_np = output.squeeze().cpu().float().numpy()
            wav_np = np.clip(wav_np, -1.0, 1.0)
            pcm_int16 = (wav_np * 32767).astype(np.int16)
            raw_bytes = pcm_int16.tobytes()

            elapsed = (time.perf_counter() - start) * 1000
            audio_ms = len(pcm_int16) / _MMS_SAMPLE_RATE * 1000
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
    """Split text at Telugu/Latin sentence boundaries."""
    parts = _SENTENCE_RE.split(text)
    return [p for p in parts if p] or [text]
