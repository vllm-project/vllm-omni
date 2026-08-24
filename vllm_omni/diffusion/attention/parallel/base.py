# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata


@dataclass(frozen=True, slots=True)
class ParallelAttentionContext:
    """Opaque per-forward context returned by a parallel strategy.

    Strategies may stash whatever they need here to finish post-processing after
    the attention kernel runs (e.g. reverse resharding, slicing metadata, etc.).
    """

    name: str


class ParallelAttentionStrategy(Protocol):
    """Pluggable strategy for parallel attention communication/resharding.

    This is intentionally orthogonal to the attention *kernel* backend.
    The kernel backend implements `AttentionImpl.forward()` for a given device,
    while the parallel strategy implements how Q/K/V and outputs are sharded /
    communicated across ranks.
    """

    @property
    def enabled(self) -> bool: ...

    @property
    def name(self) -> str: ...

    def pre_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, AttentionMetadata | None, ParallelAttentionContext | None]:
        """Runs before the attention kernel.

        Returns possibly transformed Q/K/V and metadata, and an optional context
        for `post_attention`.
        """

    def post_attention(
        self,
        attn_output: torch.Tensor,
        ctx: ParallelAttentionContext | None,
    ) -> torch.Tensor:
        """Runs after the attention kernel."""


class NoParallelAttention:
    """Default strategy: do nothing (single device / no SP)."""

    @property
    def enabled(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "none"

    def pre_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ):
        return query, key, value, attn_metadata, None

    def post_attention(self, attn_output: torch.Tensor, ctx: ParallelAttentionContext | None) -> torch.Tensor:
        return attn_output
