"""
VoicePipeline: Full-duplex phone-call-style voice agent pipeline.

Architecture — two concurrent coroutines per WebSocket connection:

  _receive_loop(ws):
    - Reads binary frames (raw Int16 PCM, 20ms chunks) continuously
    - Runs Silero VAD on every chunk inline
    - On speech_started:  sets _speech_active event; during SPEAKING state,
                          triggers barge-in interrupt
    - On speech_ended:    puts complete utterance bytes onto _utterance_queue
    - Handles JSON control frames (explicit interrupt)

  _pipeline_loop(ws):
    - Waits on _utterance_queue for a complete utterance
    - Runs STT → LLM → TTS pipeline
    - Streams raw Int16 PCM audio back via WebSocket
    - Checks _interrupt_event at every TTS sentence boundary
    - After pipeline completes, returns to IDLE and sends vad_state update

VAD state machine transitions:
  IDLE      → (speech_started)  → LISTENING
  LISTENING → (speech_ended)    → PROCESSING
  PROCESSING→ (STT done)        → SPEAKING  (if transcript non-empty)
              (empty transcript) → IDLE (listening)
  SPEAKING  → (utterance_ended) → IDLE
  SPEAKING  → (barge-in)        → interrupt → LISTENING

WebSocket protocol (full-duplex mode):
  Client → Server  binary:  raw Int16 PCM (20ms, 16kHz, mono)
  Client → Server  text:    JSON {"type": "interrupt"}
  Server → Client  binary:  raw Int16 PCM (TTS audio, 22050Hz, mono)
  Server → Client  text:    JSON {"type": "vad_state",  "state": "..."}
  Server → Client  text:    JSON {"type": "transcript", "role": "user"|"agent", "text": "..."}
  Server → Client  text:    JSON {"type": "ready"}
  Server → Client  text:    JSON {"type": "error",      "message": "..."}
"""

import asyncio
import json
import time
from enum import Enum
from typing import AsyncGenerator, Optional

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from backend.config import settings
from backend.stt.whisper_stt import TeluguSTT
from backend.llm.vllm_engine import LLMEngine
from backend.tts.xtts_tts import TeluguTTS
from backend.vad.silero_vad_engine import VADEngine, VADState


class ConnectionState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"       # VAD confirmed speech onset
    PROCESSING = "processing"     # STT + LLM + TTS running
    SPEAKING = "speaking"         # TTS audio being streamed to client


# How many bytes to send per WebSocket binary frame (tunable).
# 4096 bytes = ~93ms of audio at 22050Hz 16-bit mono.
_TTS_FRAME_BYTES = 4096


class VoicePipeline:
    """
    Manages the full-duplex voice agent lifecycle for one WebSocket connection.

    One instance is created per connected client. Shared model objects
    (stt, llm, tts, vad) are injected so they are loaded once globally.
    """

    def __init__(
        self,
        stt: TeluguSTT,
        llm: LLMEngine,
        tts: TeluguTTS,
        vad: VADEngine,
    ):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad

        # Connection-level state
        self._state: ConnectionState = ConnectionState.IDLE

        # VAD signals a complete utterance by placing bytes here
        self._utterance_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=4)

        # Set by receive_loop when barge-in speech is detected during SPEAKING
        self._interrupt_event = asyncio.Event()

        # Set by receive_loop when VAD detects speech onset (for UI feedback)
        self._vad_speech_active = False

        # Conversation history for multi-turn context
        self._history: list[dict] = []
        self._max_history_turns: int = settings.LLM_MAX_HISTORY_TURNS

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    async def process_audio_stream(self, ws: WebSocket) -> None:
        """
        Run receive and pipeline loops concurrently.
        Exits when the WebSocket closes or either loop raises a fatal error.
        """
        # Reset VAD state for this new connection
        self.vad.reset()

        # Notify client that the agent is ready and announce audio format
        await self._send_json(ws, {
            "type": "ready",
            "audio_format": {
                "encoding": "pcm_s16le",
                "sample_rate": settings.TTS_SAMPLE_RATE,
                "channels": 1,
            },
        })
        await self._send_vad_state(ws, "listening")

        recv_task = asyncio.create_task(
            self._receive_loop(ws), name="recv_loop"
        )
        pipe_task = asyncio.create_task(
            self._pipeline_loop(ws), name="pipeline_loop"
        )

        try:
            done, pending = await asyncio.wait(
                [recv_task, pipe_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        except Exception as exc:
            logger.error(f"Pipeline top-level error: {exc}")
        finally:
            # Unblock pipeline_loop if it is waiting on utterance_queue
            try:
                self._utterance_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            logger.info("VoicePipeline session ended")

    # ------------------------------------------------------------------ #
    # Receive loop — WebSocket → VAD → utterance_queue
    # ------------------------------------------------------------------ #

    async def _receive_loop(self, ws: WebSocket) -> None:
        """
        Continuously receive frames from the WebSocket.

        Binary frames (20ms Int16 PCM) are fed to VAD inline.
        VAD state transitions produce events that drive the pipeline.
        Text frames handle explicit interrupts.
        """
        try:
            while True:
                message = await ws.receive()

                if message["type"] == "websocket.disconnect":
                    logger.info("WebSocket disconnected (receive_loop)")
                    break

                if message.get("bytes"):
                    await self._handle_audio_frame(ws, message["bytes"])

                elif message.get("text"):
                    try:
                        ctrl = json.loads(message["text"])
                        if ctrl.get("type") == "interrupt":
                            logger.info("Client sent explicit interrupt")
                            self._interrupt_event.set()
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Invalid text frame: {message['text'][:80]}"
                        )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnect in receive loop")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"Receive loop error: {exc}")
        finally:
            # Signal pipeline_loop to exit
            try:
                self._utterance_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def _handle_audio_frame(self, ws: WebSocket, pcm_bytes: bytes) -> None:
        """
        Process one incoming 20ms PCM frame through VAD.
        Manages state transitions and barge-in detection.
        """
        result = self.vad.process_chunk(pcm_bytes)

        # Barge-in: new speech detected while agent is speaking → interrupt
        # Add 2s cooldown to prevent echo/mic noise from immediately killing TTS
        if result.speech_started and self._state == ConnectionState.SPEAKING:
            cooldown = getattr(self, '_speaking_start_time', 0)
            if time.time() - cooldown > 2.0:
                logger.info("Barge-in detected — interrupting agent")
                self._interrupt_event.set()

        # VAD indicator update: send vad_state "listening" vs "silent" for UI
        speech_active = result.is_speech
        if speech_active != self._vad_speech_active:
            self._vad_speech_active = speech_active
            if self._state == ConnectionState.IDLE or self._state == ConnectionState.LISTENING:
                vad_ui_state = "listening" if speech_active else "silent"
                await self._send_vad_state(ws, vad_ui_state)

        # Utterance complete → enqueue for pipeline
        if result.speech_ended and result.utterance_audio:
            logger.info(
                f"VAD: utterance complete "
                f"({len(result.utterance_audio)} bytes, "
                f"{len(result.utterance_audio) // settings.AUDIO_CHUNK_BYTES * settings.AUDIO_CHUNK_MS}ms)"
            )
            try:
                self._utterance_queue.put_nowait(result.utterance_audio)
            except asyncio.QueueFull:
                logger.warning("Utterance queue full — dropping utterance")

    # ------------------------------------------------------------------ #
    # Pipeline loop — utterance_queue → STT → LLM → TTS → WebSocket
    # ------------------------------------------------------------------ #

    async def _pipeline_loop(self, ws: WebSocket) -> None:
        """
        Wait for complete utterances from VAD and run the STT→LLM→TTS pipeline.
        """
        try:
            while True:
                # Block until VAD delivers a complete utterance
                utterance_bytes = await self._utterance_queue.get()

                if utterance_bytes is None:
                    logger.info("Pipeline received end-of-stream sentinel")
                    break

                if len(utterance_bytes) < settings.AUDIO_CHUNK_BYTES * 2:
                    # Too short — likely noise or mis-trigger
                    await self._send_vad_state(ws, "listening")
                    continue

                await self._run_pipeline(utterance_bytes, ws)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnect in pipeline loop")
        except asyncio.CancelledError:
            logger.info("Pipeline loop cancelled")
        except Exception as exc:
            logger.error(f"Pipeline loop fatal error: {exc}")
            try:
                await self._send_json(ws, {"type": "error", "message": str(exc)})
            except Exception:
                pass

    async def _run_pipeline(self, utterance_bytes: bytes, ws: WebSocket) -> None:
        """
        Execute one full turn: STT → LLM → TTS → send audio.
        """
        self._state = ConnectionState.PROCESSING
        await self._send_vad_state(ws, "processing")
        turn_start = time.perf_counter()

        # ---- STT ----
        t0 = time.perf_counter()
        transcript = await asyncio.get_event_loop().run_in_executor(
            None, self.stt.transcribe_pcm, utterance_bytes
        )
        stt_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"STT ({stt_ms:.0f}ms): '{transcript}'")

        if not transcript or not transcript.strip():
            logger.info("Empty STT result — returning to listening")
            self._state = ConnectionState.IDLE
            await self._send_vad_state(ws, "listening")
            return

        await self._send_json(ws, {
            "type": "transcript",
            "role": "user",
            "text": transcript,
        })

        # ---- LLM ----
        self._history.append({"role": "user", "content": transcript})
        # Trim history to last N turns (N user + N assistant = 2N messages)
        max_messages = self._max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

        # Build history context (exclude the current user turn — passed separately)
        history_context = self._history[:-1]

        full_response = ""

        async def _llm_tokens() -> AsyncGenerator[str, None]:
            nonlocal full_response
            async for token in self.llm.stream_response(
                transcript, conversation_history=history_context
            ):
                full_response += token
                yield token

        # ---- TTS streaming ----
        self._interrupt_event.clear()
        self._state = ConnectionState.SPEAKING
        self._speaking_start_time = time.time()
        await self._send_vad_state(ws, "speaking")

        tts_first_sent = False
        t_tts_start = time.perf_counter()

        try:
            async for pcm_bytes in self.tts.synthesize_stream(
                _llm_tokens(),
                interrupt_event=self._interrupt_event,
            ):
                if not tts_first_sent:
                    tts_ms = (time.perf_counter() - t_tts_start) * 1000
                    total_ms = (time.perf_counter() - turn_start) * 1000
                    logger.info(
                        f"TTS first chunk ({tts_ms:.0f}ms) | "
                        f"Total to first audio: {total_ms:.0f}ms"
                    )
                    tts_first_sent = True

                # Stream in frames to avoid large single sends
                for offset in range(0, len(pcm_bytes), _TTS_FRAME_BYTES):
                    frame = pcm_bytes[offset: offset + _TTS_FRAME_BYTES]
                    try:
                        await ws.send_bytes(frame)
                    except Exception:
                        logger.warning("Failed to send TTS frame — client disconnected?")
                        self._interrupt_event.set()
                        break

                if self._interrupt_event.is_set():
                    logger.info("TTS streaming interrupted (barge-in or explicit)")
                    break

        except Exception as exc:
            logger.error(f"TTS/LLM pipeline error: {exc}")
            await self._send_json(ws, {"type": "error", "message": str(exc)})
        finally:
            self._state = ConnectionState.IDLE

        # Store assistant turn in history
        if full_response.strip():
            self._history.append({
                "role": "assistant",
                "content": full_response.strip(),
            })
            await self._send_json(ws, {
                "type": "transcript",
                "role": "agent",
                "text": full_response.strip(),
            })

        self._interrupt_event.clear()
        await self._send_vad_state(ws, "listening")

    # ------------------------------------------------------------------ #
    # WebSocket send helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _send_json(ws: WebSocket, data: dict) -> None:
        try:
            await ws.send_text(json.dumps(data, ensure_ascii=False))
        except Exception as exc:
            logger.warning(f"Failed to send JSON frame: {exc}")

    @staticmethod
    async def _send_vad_state(ws: WebSocket, state: str) -> None:
        """
        Send a vad_state message to update the frontend UI.
        state: "listening" | "silent" | "processing" | "speaking"
        """
        await VoicePipeline._send_json(ws, {"type": "vad_state", "state": state})
