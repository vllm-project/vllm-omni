# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for explicit thinker decode span metadata."""

from collections.abc import Mapping
from typing import Any

import torch

THINKER_DECODE_EMBEDDINGS_KEY = "thinker_decode_embeddings"
THINKER_OUTPUT_TOKEN_IDS_KEY = "thinker_output_token_ids"
THINKER_DECODE_TOKEN_START_KEY = "thinker_decode_embeddings_token_start"
THINKER_DECODE_TOKEN_END_KEY = "thinker_decode_embeddings_token_end"

CACHED_THINKER_DECODE_EMBEDDINGS_KEY = "cached_thinker_decode_embeddings"
CACHED_THINKER_DECODE_TOKEN_START_KEY = "cached_thinker_decode_embeddings_token_start"
CACHED_THINKER_DECODE_TOKEN_END_KEY = "cached_thinker_decode_embeddings_token_end"

TensorSpan = tuple[torch.Tensor, int, int]


def get_tensor_span(payload: Mapping[str, Any], *, tensor_key: str, start_key: str, end_key: str) -> TensorSpan | None:
    tensor = payload.get(tensor_key)
    start = payload.get(start_key)
    end = payload.get(end_key)
    if not isinstance(tensor, torch.Tensor):
        return None
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end < start or (end - start) != int(tensor.shape[0]):
        return None
    return tensor, start, end


def merge_tensor_spans(existing_span: TensorSpan | None, incoming_span: TensorSpan | None) -> TensorSpan | None:
    if existing_span is None or incoming_span is None:
        return None

    existing_tensor, existing_start, existing_end = existing_span
    incoming_tensor, incoming_start, incoming_end = incoming_span
    if incoming_tensor.device != existing_tensor.device or incoming_tensor.dtype != existing_tensor.dtype:
        incoming_tensor = incoming_tensor.to(device=existing_tensor.device, dtype=existing_tensor.dtype)
    if incoming_start == existing_end:
        return torch.cat([existing_tensor, incoming_tensor], dim=0), existing_start, incoming_end
    if incoming_start < existing_end:
        overlap = existing_end - incoming_start
        if overlap >= int(incoming_tensor.shape[0]):
            return existing_tensor, existing_start, existing_end
        trimmed_tensor = incoming_tensor[overlap:]
        return (
            torch.cat([existing_tensor, trimmed_tensor], dim=0),
            existing_start,
            existing_end + int(trimmed_tensor.shape[0]),
        )
    return None


def get_tensor_span_row(span: TensorSpan | None, index: int) -> torch.Tensor | None:
    if span is None:
        return None
    tensor, start, end = span
    if index < start or index >= end:
        return None
    return tensor[index - start]
