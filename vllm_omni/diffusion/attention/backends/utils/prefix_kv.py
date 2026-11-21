# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Helpers for contiguous valid-K/V prefix slicing."""

from __future__ import annotations

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata


def validate_valid_kv_length(
    attn_metadata: AttentionMetadata | None,
    kv_len: int,
) -> int | None:
    """Return validated ``valid_kv_length`` from ``attn_metadata.extra``, or ``None``."""
    if attn_metadata is None:
        return None
    valid_kv_length = attn_metadata.extra.get("valid_kv_length")
    if valid_kv_length is None:
        return None
    if isinstance(valid_kv_length, bool) or not isinstance(valid_kv_length, int):
        raise ValueError("valid_kv_length must be a Python integer")
    if not 0 < valid_kv_length <= kv_len:
        raise ValueError(f"valid_kv_length must be within the K/V sequence, got {valid_kv_length} for length {kv_len}")
    return valid_kv_length


def slice_valid_kv_prefix(
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AttentionMetadata | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice ``key`` and ``value`` to ``extra['valid_kv_length']`` without copying."""
    valid_kv_length = validate_valid_kv_length(attn_metadata, key.shape[1])
    if valid_kv_length is None or valid_kv_length == key.shape[1]:
        return key, value
    return key[:, :valid_kv_length], value[:, :valid_kv_length]
