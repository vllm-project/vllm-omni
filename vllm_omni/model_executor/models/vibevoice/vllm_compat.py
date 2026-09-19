# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Narrow compatibility boundary for private vLLM multimodal helpers.

Keep private upstream API usage in this module so vLLM upgrades have one
explicit review point and smoke tests fail before behavior can drift silently.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.models.utils import _merge_multimodal_embeddings


def get_audio_with_sr_from_parent(
    parent_parser: Any,
    audio: Any,
) -> tuple[Any, float | None]:
    """Call ``MultiModalDataParser._get_audio_with_sr`` through one boundary."""
    method = getattr(parent_parser, "_get_audio_with_sr", None)
    if not callable(method):
        raise RuntimeError(
            "The installed vLLM no longer exposes "
            "MultiModalDataParser._get_audio_with_sr; review VibeVoice parser "
            "compatibility before serving audio."
        )
    result = method(audio)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"Unexpected MultiModalDataParser._get_audio_with_sr return value: {type(result).__name__}.")
    return result


def get_stage0_tokenizer(engine_client: Any) -> Any:
    """Resolve the initialized stage-0 tokenizer through the engine boundary."""
    engine = getattr(engine_client, "engine", engine_client)
    input_processor = getattr(engine, "input_processor", None)
    renderer = getattr(input_processor, "renderer", None)
    get_tokenizer = getattr(renderer, "get_tokenizer", None)
    if not callable(get_tokenizer):
        raise RuntimeError("VibeVoice serving could not access the stage-0 tokenizer")
    return get_tokenizer()


def merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: Any,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """Merge MM embeddings via vLLM's model-library compatibility helper."""
    return _merge_multimodal_embeddings(
        inputs_embeds,
        multimodal_embeddings,
        is_multimodal,
    )


__all__ = [
    "get_audio_with_sr_from_parent",
    "get_stage0_tokenizer",
    "merge_multimodal_embeddings",
]
