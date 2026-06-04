# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lingbot World Fast pipeline persistent state."""

from __future__ import annotations

import logging
from enum import IntEnum

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
        self.is_initialized = False
        self.reset()

    # ------------------------------------------------------------------
    # Reset / should_reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state."""

        if self.is_initialized:
            for cache in self.kv_cache:
                del cache
            for cache in self.crossattn_cache:
                if isinstance(cache["k"], torch.Tensor):
                    del cache["k"]
                    del cache["v"]

        self.kv_cache: list[torch.Tensor] | None = None
        self.crossattn_cache: list[dict[str, bool | torch.Tensor | None]] | None = None
        self.current_start_frame: int = 0
        self.local_end_index: list[torch.Tensor] | None = None
        self.global_end_index: list[torch.Tensor] | None = None

        self.is_initialized: bool = False
        self.current_lat_f: int = 0
        self.session_id: str | None = None

        self.batch_size: int | None = None
        self.num_layers: int | None = None
        self.num_heads: int | None = None
        self.head_dim: int | None = None

        # Shape constants captured on the first call of a session and reused
        # on extension calls, where multi_modal_data["image"] is absent.
        self.h: int | None = None
        self.w: int | None = None
        self.lat_h: int | None = None
        self.lat_w: int | None = None
        self.frame_seqlen: int | None = None

        # Last few latents emitted by the diffusion loop on the previous call.
        # Prepended to pred_latent_chunks on extension so the Wan VAE decoder's
        # stacked temporal feat_maps are fully warmed before the first NEW
        # latent is decoded. The decoder's temporal receptive field spans
        # ~2 latents, so we cache the last 2.
        self.last_decoded_latent: torch.Tensor | None = None

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
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        """Initialize empty KV caches and cross-attention caches."""
        self.kv_cache = [
            torch.zeros(2, batch_size, kv_size, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        self.local_end_index = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(num_layers)]
        self.global_end_index = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(num_layers)]

        self.crossattn_cache = [{"is_init": False, "k": None, "v": None} for _ in range(num_layers)]

        self.is_initialized = True

    def extend_kv_caches(self, extra_kv_size: int):
        assert self.is_initialized, "Cannot extend uninitialized kv cache"

        dtype = self.kv_cache[0].dtype
        device = self.kv_cache[0].device

        self.kv_cache = [
            torch.cat(
                [
                    self.kv_cache[i],
                    torch.zeros(
                        2, self.batch_size, extra_kv_size, self.num_heads, self.head_dim, dtype=dtype, device=device
                    ),
                ],
                dim=2,
            )
            for i in range(self.num_layers)
        ]

    def update_kv_cache(self, layer_index: int, updated_kv: torch.Tensor) -> None:
        """Update a single layer's KV cache after prefill."""
        assert self.kv_cache is not None, "KV caches not initialized, call create_kv_caches first"
        self.kv_cache[layer_index] = updated_kv.clone()

    def get_kv_cache(self) -> list[torch.Tensor]:
        """Get KV caches for the specified branch."""
        assert self.kv_cache is not None, "KV caches not initialized"
        return self.kv_cache

    def get_crossattn_caches(self) -> list[dict[str, bool | torch.Tensor | None]]:
        """Get cross-attention caches for the specified branch."""
        assert self.crossattn_cache is not None, "Cross-attn caches not initialized"
        return self.crossattn_cache

    def advance(self, delta: int):
        self.current_lat_f += delta
