# SPDX-License-Identifier: Apache-2.0
"""Stage input processor for Step-Audio2: Thinker → Token2Wav transition."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.step_audio2.step_audio2_constants import (
    DEFAULT_STREAM_CONFIG,
    DEFAULT_TOKEN_CONFIG,
)

logger = init_logger(__name__)


def _ensure_list(x):
    """Convert ConstantList / tensor / iterable to Python list."""
    if hasattr(x, "_x"):
        return list(x._x)
    elif isinstance(x, torch.Tensor):
        return x.tolist()
    elif not isinstance(x, list):
        return list(x) if hasattr(x, "__iter__") else [x]
    return list(x)


# =========================
# Async chunk processor
# =========================


def thinker2token2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """
    Async chunk processor: stream audio tokens from Thinker to Token2Wav.

    Unlike Qwen3-Omni which passes hidden states via pooling_output,
    Step Audio2 generates audio tokens as part of the LLM autoregressive
    output (token IDs >= audio_start). This processor extracts those audio
    tokens from the running token stream and sends them in chunks.

    The flow model's conformer encoder requires ``pre_lookahead_len`` extra
    tokens beyond the chunk boundary.  Each chunk therefore contains
    ``chunk_size + pre_lookahead_len`` tokens, but the consumed pointer
    only advances by ``chunk_size`` so the lookahead tokens are re-used as
    the start of the next chunk's context (cached inside the encoder).

    Payload adapts to the framework's ``_poll_single_request`` non-AR path:
    - ``code_predictor_codes`` → ``request.prompt_token_ids`` → ``input_ids``
    - ``left_context_size``   → ``request.additional_information``
      → ``runtime_additional_information`` (0 = not last, 1 = last chunk)
    - ``finished``            → marks request as finished in adapter

    Args:
        transfer_manager: OmniChunkTransferAdapter instance (has
            ``code_prompt_token_ids`` defaultdict for state tracking).
        pooling_output: Pooler output dict (unused — audio tokens are in
            the token ID stream, not in hidden states).
        request: Current engine request with access to all_token_ids.
        is_finished: Whether the upstream request has finished generating.

    Returns:
        dict with framework-compatible fields, or None if the chunk is
        not yet ready.
    """
    audio_start = DEFAULT_TOKEN_CONFIG.audio_start
    audio_eos = DEFAULT_TOKEN_CONFIG.audio_eos
    finished = bool(is_finished or request.is_finished())

    # Only look at decode (generated) tokens — the prompt may contain
    # historical audio tokens from prior conversation turns.
    all_token_ids = _ensure_list(request.all_token_ids)
    prompt_len = len(_ensure_list(request.prompt_token_ids))
    generated_ids = all_token_ids[prompt_len:]

    # Extract audio tokens and convert to 0-based IDs for Token2Wav
    audio_tokens = [tid - audio_start for tid in generated_ids if tid >= audio_start]
    # Remove padding / EOS tokens
    audio_tokens = [t for t in audio_tokens if t < audio_eos]

    # Flow model streaming parameters (from centralised config)
    chunk_size = DEFAULT_STREAM_CONFIG.chunk_size
    pre_lookahead_len = DEFAULT_STREAM_CONFIG.pre_lookahead_len

    # consumed = number of tokens whose mel output has been produced.
    # We track this via transfer_manager.code_prompt_token_ids[request_id].
    request_id = request.external_req_id
    consumed = len(transfer_manager.code_prompt_token_ids[request_id])
    available = len(audio_tokens) - consumed

    if finished:
        # Last chunk: send all remaining tokens
        if available <= 0:
            # No audio tokens at all (text-only response) — send EOF marker
            return {
                "code_predictor_codes": [],
                "left_context_size": 1,  # 1 = last chunk
                "finished": torch.tensor(True, dtype=torch.bool),
            }
        remaining_tokens = audio_tokens[consumed:]
        transfer_manager.code_prompt_token_ids[request_id].extend(remaining_tokens)
        return {
            "code_predictor_codes": remaining_tokens,
            "left_context_size": 1,  # 1 = last chunk
            "finished": torch.tensor(True, dtype=torch.bool),
        }
    else:
        # Non-last chunk: need chunk_size + pre_lookahead_len tokens
        required = chunk_size + pre_lookahead_len
        if available < required:
            return None
        chunk_tokens = audio_tokens[consumed : consumed + required]
        # Only consume chunk_size tokens; the lookahead portion is re-used
        transfer_manager.code_prompt_token_ids[request_id].extend(audio_tokens[consumed : consumed + chunk_size])
        return {
            "code_predictor_codes": chunk_tokens,
            "left_context_size": 0,  # 0 = not last chunk
            "finished": torch.tensor(False, dtype=torch.bool),
        }


def thinker2token2wav(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create token2wav inputs.

    Workflow:
    1. Extract generated token IDs from thinker output
    2. Separate audio tokens from text tokens
    3. Package audio tokens for token2wav stage

    Step-Audio2 token ranges:
    - Text tokens: 0 - 151688
    - Audio tokens: 151696 - 158257 (vocab size 6562)

    Args:
        source_outputs: Resolved list of upstream (thinker) engine outputs
        prompt: Original prompt data (unused for Token2Wav in default-voice mode)
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for token2wav stage

    """
    if not source_outputs:
        raise ValueError("source_outputs cannot be empty")

    thinker_outputs = source_outputs
    token2wav_inputs = []

    # Token configuration uses the fixed Step-Audio2 model constants.
    audio_start = DEFAULT_TOKEN_CONFIG.audio_start
    audio_eos = DEFAULT_TOKEN_CONFIG.audio_eos  # Relative to audio start

    # Process each thinker output
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]

        # Only look at decode (generated) tokens — the prompt may contain
        # historical audio tokens from prior conversation turns.
        gen_ids = output.token_ids

        # Convert to CPU list if needed (for multi-process/multi-GPU setup)
        if isinstance(gen_ids, torch.Tensor):
            gen_ids = gen_ids.cpu().tolist()

        # Separate audio tokens from text tokens
        # Audio tokens are >= audio_start
        audio_tokens = [
            tid - audio_start  # Convert to 0-based for Token2Wav
            for tid in gen_ids
            if tid >= audio_start
        ]

        # Remove padding tokens (anything >= EOS)
        audio_tokens = [t for t in audio_tokens if t < audio_eos]

        if not audio_tokens:
            # No audio tokens generated, skip Token2Wav for this request
            logger.info(f"Request {i}: No audio tokens generated, skipping Token2Wav stage")
            continue

        logger.debug("Creating Token2Wav input with %d audio tokens", len(audio_tokens))

        token2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=audio_tokens,  # Pass original tokens (no encoding)
                multi_modal_data=None,
                additional_information=None,  # Not used due to framework limitation
            )
        )

    return token2wav_inputs
