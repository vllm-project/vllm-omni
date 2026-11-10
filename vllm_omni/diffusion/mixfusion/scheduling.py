# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-only MixFusion admission and the scheduler's execution contract.

Attention backend selection is deliberately absent: dense attention buckets
can execute the same heterogeneous batch as a native varlen kernel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MixFusionCapability:
    """Loaded model's request-preparation configuration, or an unavailable reason."""

    request_config: object | None = None
    reason: str = "MixFusion is unavailable"


@dataclass(frozen=True)
class MixFusionRequestInfo:
    """Model-resolved latent shape and packing policy, before admission."""

    shape: tuple[int, int, int]  # images, image tokens, channels
    min_chunk_tokens: int
    max_chunks: int
    condition_key: tuple = ()


@dataclass(frozen=True)
class MixFusionBatchSpec:
    """One approved step batch. Worker preparation may remove failed requests."""

    chunk_size: int
    request_shapes: dict[str, tuple[int, int, int]]


def plan_chunk_size(
    shapes: Sequence[tuple[int, int, int]], *, min_chunk_tokens: int, max_chunks: int
) -> tuple[bool, str, int, int]:
    """Choose equal chunks using only host metadata, never device tensors."""
    if not shapes or any(min(shape) <= 0 for shape in shapes):
        raise ValueError("MixFusion requires positive [images, tokens, channels] shapes")
    if min_chunk_tokens <= 0 or max_chunks <= 0:
        raise ValueError("MixFusion chunk limits must be positive")
    chunk_size = math.gcd(*(shape[1] for shape in shapes))
    chunk_count = sum(images * (tokens // chunk_size) for images, tokens, _ in shapes)
    if chunk_size < min_chunk_tokens:
        return False, f"chunk_size={chunk_size} < min_chunk_tokens={min_chunk_tokens}", chunk_size, chunk_count
    if chunk_count > max_chunks:
        return False, f"chunk_count={chunk_count} > max_chunks={max_chunks}", chunk_size, chunk_count
    return True, "ok", chunk_size, chunk_count


def plan_batch(requests: Sequence[tuple[str, MixFusionRequestInfo]]) -> MixFusionBatchSpec | None:
    """Return an execution contract only if the complete candidate batch fits."""
    accepted, _, chunk_size, _ = plan_chunk_size(
        [info.shape for _, info in requests],
        min_chunk_tokens=max(info.min_chunk_tokens for _, info in requests),
        max_chunks=min(info.max_chunks for _, info in requests),
    )
    if not accepted:
        return None
    return MixFusionBatchSpec(chunk_size, {request_id: info.shape for request_id, info in requests})
