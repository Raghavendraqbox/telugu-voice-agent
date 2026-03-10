"""
Silero VAD engine for real-time phone-call-style voice activity detection.

Processes 20ms Int16 PCM frames continuously and maintains a state machine:

  IDLE       — no speech detected, waiting
  LISTENING  — speech confirmed (>MIN_SPEECH_DURATION_MS of speech chunks)
  SILENCE    — speech ended; silence timer running (>VAD_SILENCE_DURATION_MS)

The pipeline layer reads VADResult.speech_ended == True to trigger STT.
The pipeline layer reads VADResult.speech_started == True to interrupt TTS
(barge-in detection).

Silero VAD runs on CPU; it processes a 20ms frame in ~0.5-1ms, fast enough
to run inline before enqueuing audio for the pipeline.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch
from loguru import logger

from backend.config import settings


class VADState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"   # active speech confirmed
    SILENCE = "silence"       # speech ended, silence accumulating


@dataclass
class VADResult:
    """Outcome of processing a single 20ms audio frame."""
    is_speech: bool           # True if this frame contains speech (raw prob)
    speech_prob: float        # Raw Silero speech probability 0.0–1.0
    speech_started: bool      # Transition: IDLE → LISTENING (first frame of confirmed speech)
    speech_ended: bool        # Transition: LISTENING → IDLE after silence timeout
    state: VADState           # Current state after processing this frame
    # Accumulated audio so far in the current utterance (all speech frames)
    utterance_audio: Optional[bytes] = None


class VADEngine:
    """
    Stateful per-connection VAD processor using Silero VAD.

    One instance per WebSocket connection. Not thread-safe; use from a
    single asyncio coroutine or protect externally.

    Parameters (from config):
      SPEECH_THRESHOLD   — probability to mark a frame as speech (0.6)
      SILENCE_THRESHOLD  — probability below which a frame is silence (0.35)
      MIN_SPEECH_DURATION_MS  — ms of continuous speech to confirm start (300)
      SILENCE_DURATION_MS     — ms of silence after speech to confirm end (500)
      VAD_MAX_UTTERANCE_MS    — hard cap on utterance length (15000)
      AUDIO_CHUNK_MS          — frame duration (20)
      AUDIO_CHUNK_SAMPLES     — samples per frame (320)
    """

    def __init__(self):
        self._model = None
        self._speech_threshold: float = settings.VAD_SPEECH_THRESHOLD
        self._silence_threshold: float = settings.VAD_SILENCE_THRESHOLD
        self._chunk_ms: int = settings.AUDIO_CHUNK_MS
        self._chunk_samples: int = settings.AUDIO_CHUNK_SAMPLES
        self._sample_rate: int = settings.STT_SAMPLE_RATE

        # State machine
        self._state: VADState = VADState.IDLE

        # Counters (in frames)
        self._frames_per_ms = 1.0 / self._chunk_ms
        self._min_speech_frames: int = max(
            1, settings.VAD_MIN_SPEECH_DURATION_MS // self._chunk_ms
        )
        self._silence_frames_needed: int = max(
            1, settings.VAD_SILENCE_DURATION_MS // self._chunk_ms
        )
        self._max_utterance_frames: int = max(
            1, settings.VAD_MAX_UTTERANCE_MS // self._chunk_ms
        )

        # Rolling counters
        self._speech_frame_count: int = 0     # consecutive speech frames
        self._silence_frame_count: int = 0    # consecutive silence frames after speech
        self._total_utterance_frames: int = 0

        # Buffer of Int16 PCM bytes for the current utterance
        # Includes pre-roll (speech frames before confirmation) and all frames
        # once in LISTENING state.
        self._pre_roll: list[bytes] = []      # frames buffered before speech confirmed
        self._pre_roll_max: int = self._min_speech_frames + 2
        self._utterance_chunks: list[bytes] = []  # confirmed utterance audio

        # Buffer for accumulating frames before VAD inference
        # silero-vad v5 requires >= 512 samples; we buffer two 20ms frames (640)
        self._vad_buffer: bytes = b""
        self._vad_buffer_samples: int = 0
        self._vad_min_samples: int = 512

    def load(self) -> None:
        """Load Silero VAD model onto CPU. Call once at startup."""
        logger.info("Loading Silero VAD model...")
        start = time.perf_counter()
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad()
            self._model.eval()
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Silero VAD loaded in {elapsed:.0f}ms (CPU)")
        except ImportError:
            logger.error("silero-vad not installed. Run: pip install silero-vad")
            raise
        except Exception as exc:
            logger.error(f"Failed to load Silero VAD: {exc}")
            raise

    def reset(self) -> None:
        """Reset state machine for a new connection or after utterance ends."""
        self._state = VADState.IDLE
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._total_utterance_frames = 0
        self._pre_roll.clear()
        self._utterance_chunks.clear()
        self._vad_buffer = b""
        self._vad_buffer_samples = 0

    def process_chunk(self, pcm_int16_bytes: bytes) -> VADResult:
        """
        Process one 20ms Int16 PCM frame (640 bytes at 16kHz).

        Returns a VADResult describing the outcome and any state transition.
        The caller should collect frames and act when speech_started or
        speech_ended is True.

        If pcm_int16_bytes is shorter than expected (partial frame), it is
        zero-padded to avoid model errors.
        """
        if not self._model:
            raise RuntimeError("VADEngine not loaded. Call load() first.")

        # Ensure correct length — pad if partial frame arrived
        expected = self._chunk_samples * 2  # 2 bytes per int16 sample
        if len(pcm_int16_bytes) < expected:
            pcm_int16_bytes = pcm_int16_bytes + b"\x00" * (
                expected - len(pcm_int16_bytes)
            )
        elif len(pcm_int16_bytes) > expected:
            pcm_int16_bytes = pcm_int16_bytes[:expected]

        # Buffer incoming frames — silero-vad v5 requires EXACTLY 512 samples.
        # Each frontend frame is 320 samples (20ms). We accumulate until we
        # have >= 512, slice off exactly 512 for inference, keep remainder.
        self._vad_buffer += pcm_int16_bytes
        self._vad_buffer_samples = len(self._vad_buffer) // 2

        if self._vad_buffer_samples < self._vad_min_samples:
            # Not enough samples yet — hold, no state change
            return VADResult(
                is_speech=False,
                speech_prob=0.0,
                speech_started=False,
                speech_ended=False,
                state=self._state,
                utterance_audio=None,
            )

        # Slice exactly 512 samples (1024 bytes), keep the rest
        run_bytes = self._vad_buffer[:self._vad_min_samples * 2]
        self._vad_buffer = self._vad_buffer[self._vad_min_samples * 2:]
        self._vad_buffer_samples = len(self._vad_buffer) // 2

        # Convert Int16 bytes → Float32 tensor in [-1, 1]
        pcm_int16 = np.frombuffer(run_bytes, dtype=np.int16)
        audio_f32 = pcm_int16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_f32).unsqueeze(0)  # shape: [1, 512]

        # Run Silero VAD inference — v5 API: model(audio, sr) → Tensor
        with torch.no_grad():
            result = self._model(tensor, self._sample_rate)
        prob = float(result.item())

        # Debug: log every 25th inference to see live probability values
        if not hasattr(self, '_dbg_count'):
            self._dbg_count = 0
        self._dbg_count += 1
        if self._dbg_count % 25 == 0:
            rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
            logger.debug(f"VAD prob={prob:.3f} rms={rms:.4f} state={self._state.value}")

        is_speech_frame = prob >= self._speech_threshold
        is_silence_frame = prob < self._silence_threshold

        speech_started = False
        speech_ended = False
        utterance_audio: Optional[bytes] = None

        if self._state == VADState.IDLE:
            # Collect pre-roll regardless so we capture the onset
            self._pre_roll.append(pcm_int16_bytes)
            if len(self._pre_roll) > self._pre_roll_max:
                self._pre_roll.pop(0)

            if is_speech_frame:
                self._speech_frame_count += 1
                if self._speech_frame_count >= self._min_speech_frames:
                    # Speech confirmed — transition to LISTENING
                    self._state = VADState.LISTENING
                    speech_started = True
                    self._silence_frame_count = 0
                    self._total_utterance_frames = 0
                    # Include pre-roll frames so STT gets clean onset
                    self._utterance_chunks = list(self._pre_roll)
                    self._pre_roll.clear()
                    logger.debug(
                        f"VAD: IDLE→LISTENING (prob={prob:.2f})"
                    )
            else:
                # Reset speech counter on non-speech frame
                self._speech_frame_count = max(0, self._speech_frame_count - 1)

        elif self._state == VADState.LISTENING:
            self._utterance_chunks.append(pcm_int16_bytes)
            self._total_utterance_frames += 1

            if is_silence_frame:
                self._silence_frame_count += 1
            else:
                self._silence_frame_count = 0

            # Silence timeout → end of utterance
            if self._silence_frame_count >= self._silence_frames_needed:
                self._state = VADState.IDLE
                speech_ended = True
                utterance_audio = b"".join(self._utterance_chunks)
                logger.debug(
                    f"VAD: LISTENING→IDLE (silence timeout, "
                    f"{self._total_utterance_frames * self._chunk_ms}ms audio, "
                    f"prob={prob:.2f})"
                )
                # Reset for next utterance
                self._utterance_chunks.clear()
                self._speech_frame_count = 0
                self._silence_frame_count = 0
                self._total_utterance_frames = 0

            # Hard cap: force end-of-utterance at maximum length
            elif self._total_utterance_frames >= self._max_utterance_frames:
                self._state = VADState.IDLE
                speech_ended = True
                utterance_audio = b"".join(self._utterance_chunks)
                logger.info(
                    f"VAD: max utterance length reached "
                    f"({settings.VAD_MAX_UTTERANCE_MS}ms) — forcing end"
                )
                self._utterance_chunks.clear()
                self._speech_frame_count = 0
                self._silence_frame_count = 0
                self._total_utterance_frames = 0

        return VADResult(
            is_speech=is_speech_frame,
            speech_prob=prob,
            speech_started=speech_started,
            speech_ended=speech_ended,
            state=self._state,
            utterance_audio=utterance_audio,
        )

