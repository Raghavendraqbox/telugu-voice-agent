"""
LLM inference engine using vLLM AsyncLLMEngine.

Model: meta-llama/Meta-Llama-3-8B-Instruct
Runs streaming token generation so TTS can start synthesizing while the
LLM is still generating the rest of the response.

Key design choices:
- AsyncLLMEngine with async generator output → zero-blocking token streaming
- System prompt written in Telugu to anchor response language
- Temperature 0.7 balances creativity vs. coherence for conversational use
- gpu_memory_utilization=0.55 leaves headroom for STT + TTS on same GPU
"""

import asyncio
import time
import uuid
from typing import AsyncGenerator, Optional

from loguru import logger
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.outputs import RequestOutput

from backend.config import settings

# Telugu system prompt — instructs the model to respond naturally in Telugu
# and behave as a friendly, helpful voice assistant.
SYSTEM_PROMPT_TE = (
    "మీరు ఒక సహాయకరమైన తెలుగు వాయిస్ అసిస్టెంట్. "
    "దయచేసి ప్రతి సమాధానం తెలుగులో ఇవ్వండి. "
    "మీ సమాధానాలు స్పష్టంగా, సంక్షిప్తంగా మరియు సహాయకరంగా ఉండాలి. "
    "వాయిస్ సంభాషణకు అనుకూలంగా జవాబులు ఇవ్వండి — "
    "చిన్న వాక్యాలు వాడండి, జాబితాలు లేదా మార్క్‌డౌన్ ఉపయోగించకండి."
)


class LLMEngine:
    """
    Async wrapper around vLLM's AsyncLLMEngine for streaming Telugu responses.

    Usage:
        engine = LLMEngine()
        engine.load()
        async for token in engine.stream_response("నీకు ఎలా ఉంది?"):
            print(token, end="", flush=True)
    """

    def __init__(self):
        self._engine: Optional[AsyncLLMEngine] = None
        self._tokenizer = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Initialize vLLM AsyncLLMEngine. Call once at startup."""
        logger.info(f"Loading LLM: {settings.LLM_MODEL}")
        start = time.perf_counter()

        engine_args = AsyncEngineArgs(
            model=settings.LLM_MODEL,
            dtype="float16",
            gpu_memory_utilization=settings.VLLM_GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=settings.VLLM_TENSOR_PARALLEL_SIZE,
            max_num_seqs=settings.VLLM_MAX_NUM_SEQS,
            # Trust remote code for Llama-3 tokenizer
            trust_remote_code=True,
            # Disable prefix caching for simplicity (enable for production)
            enable_prefix_caching=False,
            # Quantization: comment out if using full fp16 on A40
            # quantization="awq",
        )

        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"LLM engine initialized in {elapsed:.0f}ms")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def stream_response(
        self,
        user_text: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM tokens for the given user utterance.

        Yields individual text tokens as they are generated. The caller
        (TTS stage) should buffer tokens at sentence boundaries before
        synthesizing to balance latency vs. naturalness.

        Args:
            user_text: The transcribed user utterance in Telugu.
            conversation_history: Optional list of prior turns for multi-turn
                                  context.  Each item: {"role": ..., "content": ...}

        Yields:
            str: Individual token strings (may be sub-word pieces).
        """
        if not self._engine:
            raise RuntimeError("LLM engine not loaded. Call load() first.")

        if not user_text or not user_text.strip():
            logger.warning("LLM received empty input — skipping")
            return

        # Build the chat messages list
        messages = [{"role": "system", "content": SYSTEM_PROMPT_TE}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_text.strip()})

        prompt = self._build_chat_prompt(messages)

        sampling_params = SamplingParams(
            temperature=settings.LLM_TEMPERATURE,
            top_p=settings.LLM_TOP_P,
            max_tokens=settings.LLM_MAX_TOKENS,
            repetition_penalty=settings.LLM_REPETITION_PENALTY,
            stop=["<|eot_id|>", "<|end_of_text|>", "</s>"],
        )

        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        first_token_logged = False

        # vLLM async generator yields RequestOutput objects.
        # Each output contains the cumulative text; we diff adjacent outputs
        # to extract the new token(s) added in each step.
        prev_text = ""

        async for request_output in self._engine.generate(
            prompt, sampling_params, request_id
        ):
            request_output: RequestOutput
            if not request_output.outputs:
                continue

            current_text = request_output.outputs[0].text
            new_token = current_text[len(prev_text):]
            prev_text = current_text

            if not first_token_logged and new_token.strip():
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug(f"LLM first token in {elapsed:.0f}ms")
                first_token_logged = True

            if new_token:
                yield new_token

            if request_output.finished:
                total_elapsed = (time.perf_counter() - start) * 1000
                token_count = len(request_output.outputs[0].token_ids)
                logger.debug(
                    f"LLM finished: {token_count} tokens in {total_elapsed:.0f}ms "
                    f"({token_count / (total_elapsed / 1000):.0f} tok/s)"
                )
                break

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_chat_prompt(self, messages: list[dict]) -> str:
        """
        Format messages list into Llama-3 chat template.

        Llama-3-Instruct uses the following special tokens:
          <|begin_of_text|>
          <|start_header_id|>role<|end_header_id|>
          content
          <|eot_id|>
        """
        prompt = "<|begin_of_text|>"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += (
                f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                f"{content}<|eot_id|>"
            )
        # Append the assistant header to prime generation
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt
