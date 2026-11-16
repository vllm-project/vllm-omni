# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-only MixFusion eligibility and layout-specific admission.

Attention backend selection is deliberately absent: dense attention buckets
can execute the same heterogeneous batch as a native varlen kernel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from vllm_omni.diffusion.mixfusion.plan import MixFusionSequenceRequest
from vllm_omni.diffusion.mixfusion.request import MixFusionImageRequestConfig


@dataclass(frozen=True)
class MixFusionCapability:
    """Loaded model's request-preparation configuration, or an unavailable reason."""

    request_config: MixFusionImageRequestConfig | MixFusionSequenceRequest | None = None
    reason: str = "MixFusion is unavailable"


@dataclass(frozen=True)
class MixFusionRequestInfo:
    """Request cost and compatibility, without a tensor execution layout."""

    num_images: int
    image_tokens: int
    min_chunk_tokens: int
    max_chunks: int
    condition_key: tuple = ()


@dataclass(frozen=True)
class MixFusionSequenceInfo:
    """Host-only spatial and document geometry for a flat prefix-KV layout.

    Sequence lengths describe the logical first step, before any imported
    prefix is removed from the worker's queries. No chunk budget applies:
    StepScheduler's normal request-count limit controls this layout.
    """

    latent_shape: tuple[int, int, int, int]
    image_tokens: int
    prefix_lengths: tuple[int, ...]
    query_lengths: tuple[int, ...]
    special_tokens: int

    @property
    def condition_key(self) -> tuple[str, int]:
        return ("flat_prefix_kv", len(self.prefix_lengths))


def fits_mixfusion_budget(requests: Sequence[MixFusionRequestInfo | MixFusionSequenceInfo]) -> bool:
    """Apply the declared layout's admission policy to the whole candidate batch."""
    if not requests:
        raise ValueError("MixFusion requires a nonempty candidate batch")
    sequences = [info for info in requests if isinstance(info, MixFusionSequenceInfo)]
    if len(sequences) == len(requests):
        return all(
            min(info.latent_shape) > 0
            and info.image_tokens > 0
            and info.special_tokens > 0
            and bool(info.prefix_lengths)
            and len(info.prefix_lengths) == len(info.query_lengths)
            and all(
                0 <= p and p + info.special_tokens + info.image_tokens <= q
                for p, q in zip(info.prefix_lengths, info.query_lengths, strict=True)
            )
            for info in sequences
        )
    chunks = [info for info in requests if isinstance(info, MixFusionRequestInfo)]
    if len(chunks) == len(requests):
        return fits_chunk_budget(chunks)
    return False


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
