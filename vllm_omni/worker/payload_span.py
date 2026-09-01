# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility imports for model-runner payload span helpers."""

from vllm_omni.utils.payload_span import (
    CACHED_THINKER_DECODE_EMBEDDINGS_KEY,
    CACHED_THINKER_DECODE_TOKEN_END_KEY,
    CACHED_THINKER_DECODE_TOKEN_START_KEY,
    THINKER_DECODE_EMBEDDINGS_KEY,
    THINKER_DECODE_TOKEN_END_KEY,
    THINKER_DECODE_TOKEN_START_KEY,
    THINKER_OUTPUT_TOKEN_IDS_KEY,
    TensorSpan,
    get_tensor_span,
    get_tensor_span_row,
    merge_tensor_spans,
)

__all__ = [
    "CACHED_THINKER_DECODE_EMBEDDINGS_KEY",
    "CACHED_THINKER_DECODE_TOKEN_END_KEY",
    "CACHED_THINKER_DECODE_TOKEN_START_KEY",
    "THINKER_DECODE_EMBEDDINGS_KEY",
    "THINKER_DECODE_TOKEN_END_KEY",
    "THINKER_DECODE_TOKEN_START_KEY",
    "THINKER_OUTPUT_TOKEN_IDS_KEY",
    "TensorSpan",
    "get_tensor_span",
    "get_tensor_span_row",
    "merge_tensor_spans",
]
