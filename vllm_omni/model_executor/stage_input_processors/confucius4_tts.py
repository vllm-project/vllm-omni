# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Confucius4-TTS Stage 0 to Stage 1 input conversion."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.confucius4_tts.configuration_confucius4_tts import (
    MAX_GENERATED_SEMANTIC_TOKENS,
    MAX_SEMANTIC_CODE,
    SEMANTIC_BOS_TOKEN_ID,
    SEMANTIC_EOS_TOKEN_ID,
)

logger = init_logger(__name__)

_CONDITIONING_KEYS = ("text_token_ids", "condition_vector", "ref_mel", "style")


def _cpu_copy(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone().contiguous()


def _payload_value(
    payload: dict[str, Any],
    dotted_key: str,
    parent: str,
    child: str,
) -> Any:
    value = payload.get(dotted_key)
    if value is not None:
        return value
    nested = payload.get(parent)
    if isinstance(nested, dict):
        return nested.get(child)
    return None


def _sanitize_semantic_sequence(
    semantic: torch.Tensor,
    bos_token_id: int = SEMANTIC_BOS_TOKEN_ID,
    eos_token_id: int = SEMANTIC_EOS_TOKEN_ID,
    max_token_id: int = MAX_SEMANTIC_CODE,
) -> torch.Tensor:
    """Match the official post-processing of sampled semantic ids.

    Cut at the first EOS, drop BOS, and clamp stray special ids into the
    codebook range. The latent is recomputed by Stage 1 from this exact
    sequence, so no per-token state needs to stay aligned here.
    """
    stop = (semantic == eos_token_id).nonzero(as_tuple=False)
    length = int(stop[0].item()) if stop.numel() else int(semantic.shape[0])
    semantic = semantic[:length]
    semantic = semantic[semantic != bos_token_id]
    return semantic.clamp(max=max_token_id)


def _request_seed(request: Any) -> int | None:
    sampling_params = getattr(request, "sampling_params", None)
    seed = getattr(sampling_params, "seed", None)
    return int(seed) if seed is not None else None


def talker2decoder_token_only(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    from vllm_omni.inputs.data import OmniTokensPrompt

    del prompt
    outputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        if not source_output.finished:
            continue
        outputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return outputs


def talker2decoder_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
    **_: Any,
) -> dict[str, Any]:
    del transfer_manager
    request_id = str(
        getattr(request, "external_req_id", None)
        or getattr(request, "request_id", None)
        or getattr(request, "req_id", "?")
    )
    if not isinstance(pooling_output, dict):
        raise ValueError(f"Confucius4-TTS Stage 0 payload must be a dict for req={request_id}")

    output_token_ids = list(getattr(request, "output_token_ids", None) or [])
    if not output_token_ids:
        raise ValueError(f"Confucius4-TTS request has no completed output_token_ids for req={request_id}")
    spoken = output_token_ids
    if SEMANTIC_EOS_TOKEN_ID in output_token_ids:
        spoken = output_token_ids[: output_token_ids.index(SEMANTIC_EOS_TOKEN_ID)]
    if len(spoken) >= MAX_GENERATED_SEMANTIC_TOKENS:
        # About 50 semantic tokens per second of audio, so the full budget
        # caps one request at roughly 30 s. The utterance is still
        # synthesized, but the text past the budget is not spoken. A smaller
        # caller-chosen max_new_tokens is an intentional cut and stays quiet.
        logger.warning(
            "Confucius4-TTS req=%s used the whole %d-token semantic budget without EOS; "
            "the audio is truncated. Split long text into shorter requests.",
            request_id,
            MAX_GENERATED_SEMANTIC_TOKENS,
        )
    semantic_seq = _sanitize_semantic_sequence(torch.tensor(output_token_ids, dtype=torch.long))
    if semantic_seq.numel() == 0:
        raise ValueError(f"Confucius4-TTS generated no semantic codes before EOS for req={request_id}")

    result: dict[str, Any] = {"semantic_codes": _cpu_copy(semantic_seq)}
    for key in _CONDITIONING_KEYS:
        value = _payload_value(pooling_output, f"meta.{key}", "meta", key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            raise ValueError(f"Confucius4-TTS payload requires non-empty {key} for req={request_id}")
        result[key] = _cpu_copy(value)

    seed = _request_seed(request)
    if seed is not None:
        result["seed"] = seed
    return result
