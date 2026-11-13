# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-only MixFusion eligibility and chunk-budget admission.

Attention backend selection is deliberately absent: dense attention buckets
can execute the same heterogeneous batch as a native varlen kernel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from vllm_omni.diffusion.mixfusion.request import MixFusionImageRequestConfig


@dataclass(frozen=True)
class MixFusionCapability:
    """Loaded model's request-preparation configuration, or an unavailable reason."""

    request_config: MixFusionImageRequestConfig | None = None
    reason: str = "MixFusion is unavailable"


@dataclass(frozen=True)
class MixFusionRequestInfo:
    """Request cost and compatibility, without a tensor execution layout."""

    num_images: int
    image_tokens: int
    min_chunk_tokens: int
    max_chunks: int
    condition_key: tuple = ()


def fits_chunk_budget(requests: Sequence[MixFusionRequestInfo]) -> bool:
    """Whether the complete candidate batch fits every request's limits.

    GCD is a cost estimate here, not an execution plan. The adapter chooses
    its layout from actual prepared inputs, which may exclude failed requests.
    """
    if not requests or any(min(info.num_images, info.image_tokens) <= 0 for info in requests):
        raise ValueError("MixFusion requires positive image counts and token lengths")
    min_chunk_tokens = max(info.min_chunk_tokens for info in requests)
    max_chunks = min(info.max_chunks for info in requests)
    if any(min(info.min_chunk_tokens, info.max_chunks) <= 0 for info in requests):
        raise ValueError("MixFusion chunk limits must be positive")
    chunk_size = math.gcd(*(info.image_tokens for info in requests))
    chunk_count = sum(info.num_images * (info.image_tokens // chunk_size) for info in requests)
    return chunk_size >= min_chunk_tokens and chunk_count <= max_chunks
