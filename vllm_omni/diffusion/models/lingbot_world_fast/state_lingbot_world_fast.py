# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lingbot World Fast pipeline persistent state."""

from __future__ import annotations

import logging
from enum import IntEnum

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CacheIndex(IntEnum):
    K = 0
    V = 1


class LingbotWorldFastState:
    """Pipeline persistent state across forward() calls.

    Lifecycle:
        - Created once in LingbotWorldFastPipeline.__init__()
        - Mutated every forward() call (frame append, KV cache grow)
        - reset() on new session / local_attn_size exceeded
    """

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Reset / should_reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state."""
        self.kv_cache: list[torch.Tensor] | None = None
        self.crossattn_cache: list[dict[str, bool | torch.Tensor | None]] | None = None
        self.current_start_frame: int = 0
        self.local_end_index: list[torch.Tensor] | None = None
        self.global_end_index: list[torch.Tensor] | None = None

    def should_reset(self, text_tokens: torch.Tensor | None, num_video_frames: int, local_attn_size: int) -> bool:
        """Determine if state should be reset before this forward()."""
        # NOTE: after accumulate_frames, num_video_frames is the accumulated T
        # (1 for first call, 4 for subsequent). Only reset on true single-frame
        # which happens when the stitched_buffer was cleared externally.
        if num_video_frames == 1 and self.call_count > 1:
            logger.info("single frame input after first call, resetting")
            return True

        if local_attn_size != -1 and self.current_start_frame >= local_attn_size:
            logger.info(
                "current_start_frame %d >= local_attn_size %d, resetting", self.current_start_frame, local_attn_size
            )
            return True

        return False

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def create_kv_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kv_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Initialize empty KV caches and cross-attention caches."""
        self.kv_cache = [
            torch.zeros(2, batch_size, kv_size, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        self.local_end_index = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(num_layers)]
        self.global_end_index = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(num_layers)]

        self.crossattn_cache = [{"is_init": False, "k": None, "v": None} for _ in range(num_layers)]

    def update_kv_cache(
        self,
        layer_index: int,
        updated_kv: torch.Tensor,
        is_negative: bool = False,
    ) -> None:
        """Update a single layer's KV cache after prefill."""
        cache = self.kv_cache_neg if is_negative else self.kv_cache
        assert cache is not None, "KV caches not initialized, call create_kv_caches first"
        cache[layer_index] = updated_kv.clone()

    def get_kv_caches(self) -> list[torch.Tensor]:
        """Get KV caches for the specified branch."""
        assert self.kv_cache is not None, "KV caches not initialized"
        return self.kv_cache

    def get_crossattn_caches(self, is_negative: bool = False) -> list[dict[str, bool | torch.Tensor | None]]:
        """Get cross-attention caches for the specified branch."""
        assert self.crossattn_cache is not None, "Cross-attn caches not initialized"
        return self.crossattn_cache
