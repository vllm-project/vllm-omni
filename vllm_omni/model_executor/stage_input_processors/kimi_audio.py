# Copyright 2025 vLLM-Omni Team
"""Stage input processor for Kimi Audio async chunk streaming."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
)
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.kimi_audio.constants import (
    CODEC_CHUNK_FRAMES,
    CODEC_LEFT_CONTEXT_FRAMES,
)

logger = init_logger(__name__)


def _extract_audio_token_ids(multimodal_output: dict[str, Any] | None) -> torch.Tensor | None:
    """Extract the [B, 1] audio token tensor from a Stage-0 multimodal output.

    Returns None when the output carries no audio tokens (e.g. text-only path).
    The returned IDs are raw unified-vocabulary IDs; callers must NOT subtract
    ``KIMI_AUDIO_TOKEN_OFFSET`` because the detokenizer performs that mapping
    itself.
    """
    if multimodal_output is None or not isinstance(multimodal_output, dict):
        return None
    audio_token_ids = multimodal_output.get("audio_tokens")
    if audio_token_ids is None:
        return None
    if not isinstance(audio_token_ids, torch.Tensor):
        audio_token_ids = torch.tensor(audio_token_ids, dtype=torch.long)
    if audio_token_ids.numel() == 0:
        return None
    # Normalize to [B, 1].
    if audio_token_ids.dim() == 1:
        audio_token_ids = audio_token_ids.unsqueeze(1)
    elif audio_token_ids.dim() > 2:
        audio_token_ids = audio_token_ids.reshape(-1, 1)
    return audio_token_ids


def _normalize_audio_tokens_for_payload(audio_token_ids: torch.Tensor) -> torch.Tensor:
    """Return a [1, L] tensor of raw unified-vocabulary audio token IDs."""
    # Detokenizer expects the original unified IDs and filters/subsamples itself.
    # Do NOT subtract KIMI_AUDIO_TOKEN_OFFSET here.
    return audio_token_ids.reshape(1, -1)


def llm2detokenizer_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool,
) -> OmniPayloadStruct | None:
    """Convert LLM audio tokens to detokenizer input (async chunk streaming).

    Called by the async-chunk transfer adapter for every Stage-0 emit step.
    Accumulates raw unified-vocabulary audio token IDs per request and flushes
    a chunk of ``CODEC_CHUNK_FRAMES`` tokens (or the tail on finish) to Stage 1.

    Args:
        transfer_manager: Holds per-request accumulated token state.
        multimodal_output: Dict from Stage-0 ``make_omni_output`` with an
            ``audio_tokens`` tensor of raw unified-vocabulary IDs.
        request: Current request object.
        is_finished: Whether this is the final chunk for the request.

    Returns:
        OmniPayloadStruct with raw audio token IDs, or None if no chunk ready.
    """
    audio_token_ids = _extract_audio_token_ids(multimodal_output)
    if audio_token_ids is None:
        if is_finished:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(
                    finished=torch.tensor(True, dtype=torch.bool),
                    codec_chunk_frames=CODEC_CHUNK_FRAMES,
                    codec_left_context_frames=CODEC_LEFT_CONTEXT_FRAMES,
                ),
            )
        return None

    logger.debug(
        "[KimiAudio Stage Transfer] Received audio_tokens: shape=%s, min=%s, max=%s, mean=%.2f",
        audio_token_ids.shape,
        audio_token_ids.min(),
        audio_token_ids.max(),
        audio_token_ids.float().mean(),
    )

    audio_token_ids = _normalize_audio_tokens_for_payload(audio_token_ids)

    # Accumulate tokens per request.
    request_id = getattr(request, "external_req_id", getattr(request, "request_id", None))
    if request_id is None:
        logger.warning("[KimiAudio Stage Transfer] request has no id; dropping audio tokens")
        return None

    if not hasattr(transfer_manager, "audio_tokens"):
        transfer_manager.audio_tokens = {}
    if request_id not in transfer_manager.audio_tokens:
        transfer_manager.audio_tokens[request_id] = []

    transfer_manager.audio_tokens[request_id].append(audio_token_ids)

    accumulated = torch.cat(transfer_manager.audio_tokens[request_id], dim=1)
    chunk_size = CODEC_CHUNK_FRAMES

    if accumulated.shape[1] >= chunk_size or is_finished:
        if accumulated.shape[1] >= chunk_size:
            chunk = accumulated[:, :chunk_size]
            leftover = accumulated[:, chunk_size:]
        else:
            chunk = accumulated
            leftover = torch.zeros((1, 0), dtype=accumulated.dtype, device=accumulated.device)

        payload = OmniPayloadStruct(
            codes=CodesStruct(audio=chunk.reshape(-1)),
            meta=MetaStruct(
                finished=torch.tensor(is_finished, dtype=torch.bool),
                codec_chunk_frames=chunk_size,
                codec_left_context_frames=CODEC_LEFT_CONTEXT_FRAMES,
            ),
        )

        if is_finished and leftover.shape[1] == 0:
            transfer_manager.audio_tokens.pop(request_id, None)
        else:
            transfer_manager.audio_tokens[request_id] = [leftover]

        return payload

    return None


def llm2detokenizer(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build Stage-1 detokenizer prompts from Stage-0 outputs (orchestrator path).

    For async-chunk pipelines this seeds the Stage-1 request; subsequent codec
    frames are delivered directly by the chunk adapter.  For non-async-chunk
    pipelines this must carry the full audio token sequence.

    Args:
        source_outputs: List of Stage-0 EngineCoreOutputs.
        prompt: Original user prompt (unused, kept for signature compatibility).
        requires_multimodal_data: Whether multimodal data should be forwarded.

    Returns:
        List of OmniTokensPrompt, one per source output.
    """
    del prompt
    detokenizer_inputs: list[OmniTokensPrompt] = []

    if not isinstance(source_outputs, list):
        source_outputs = [source_outputs]

    for source_output in source_outputs:
        output = source_output.outputs[0]
        mm = getattr(output, "multimodal_output", None)
        mm = mm if isinstance(mm, dict) else {}
        audio_token_ids = _extract_audio_token_ids(mm)

        if audio_token_ids is None or audio_token_ids.numel() == 0:
            detokenizer_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[],
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                )
            )
            continue

        token_ids = _normalize_audio_tokens_for_payload(audio_token_ids)
        token_ids = token_ids.reshape(-1).tolist()
        detokenizer_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=token_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return detokenizer_inputs


def llm2detokenizer_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Sync-side placeholder for the non-async-chunk Stage-1 input.

    In non-async-chunk mode the actual audio tokens are delivered via the
    worker-connector full payload built by ``llm2detokenizer_full_payload``.
    This function only allocates correctly-sized prompt slots so the runtime
    can schedule Stage-1.
    """
    del prompt
    detokenizer_inputs: list[OmniTokensPrompt] = []

    if not isinstance(source_outputs, list):
        source_outputs = [source_outputs]

    for source_output in source_outputs:
        output = source_output.outputs[0]
        mm = getattr(output, "multimodal_output", None)
        mm = mm if isinstance(mm, dict) else {}
        audio_token_ids = _extract_audio_token_ids(mm)

        prompt_len = 0
        if audio_token_ids is not None and audio_token_ids.numel() > 0:
            prompt_len = int(audio_token_ids.numel())

        detokenizer_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return detokenizer_inputs


def llm2detokenizer_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Producer-side payload builder for the worker-connector data plane.

    In non-async-chunk mode the worker connector accumulates per-step
    ``pooling_output["audio_tokens"]`` tensors and calls this hook at flush
    time.  The returned payload is passed to Stage 1 as ``codes.audio``.

    Args:
        transfer_manager: Connector state manager.
        pooling_output: Accumulated dict with ``audio_tokens`` tensor.
        request: Current request object.
        is_finished: Whether this is the terminal payload.

    Returns:
        OmniPayloadStruct with raw audio token IDs, or None if empty.
    """
    del transfer_manager
    if not isinstance(pooling_output, dict):
        logger.warning(
            "kimi_audio.llm2detokenizer_full_payload: pooling_output not a dict (type=%s); skipping.",
            type(pooling_output).__name__,
        )
        return None

    audio_token_ids = _extract_audio_token_ids(pooling_output)
    if audio_token_ids is None or audio_token_ids.numel() == 0:
        logger.warning(
            "kimi_audio.llm2detokenizer_full_payload: missing/empty audio_tokens (keys=%s); skipping.",
            list(pooling_output.keys()) if isinstance(pooling_output, dict) else None,
        )
        return None

    token_ids = _normalize_audio_tokens_for_payload(audio_token_ids)
    return OmniPayloadStruct(
        codes=CodesStruct(audio=token_ids.reshape(-1)),
        meta=MetaStruct(
            finished=torch.tensor(is_finished, dtype=torch.bool),
            codec_chunk_frames=CODEC_CHUNK_FRAMES,
            codec_left_context_frames=CODEC_LEFT_CONTEXT_FRAMES,
        ),
    )
