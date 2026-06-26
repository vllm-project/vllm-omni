# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lingbot World Fast pipeline persistent state."""

from __future__ import annotations

import logging
from collections import deque
from enum import IntEnum

import torch

from vllm_omni.platforms import current_omni_platform

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
        """Clear all state and return the cached device memory to the allocator."""

        had_caches = self.is_initialized
        cache_device = self.cache_device if had_caches else None

        self.kv_cache: list[torch.Tensor] | None = None
        self.crossattn_cache: list[dict[str, bool | torch.Tensor | None]] | None = None
        self.current_start_frame: int = 0
        self.local_end_index: list[torch.Tensor] | None = None
        self.global_end_index: list[torch.Tensor] | None = None
        self.evict_queues: list[list[deque[int]]] | None = None

        self.is_initialized: bool = False
        self.current_lat_f: int = 0
        self.session_id: str | None = None

        self.batch_size: int | None = None
        self.num_layers: int | None = None
        self.num_heads: int | None = None
        self.head_dim: int | None = None
        self.cache_device: torch.device | None = None

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

        # Session-level rolling-cache config captured on the first call of a
        # session. Extension calls must match these (the cache is sized once
        # and reused; changing these mid-session would corrupt cache layout).
        self.session_chunk_latent_frames: int | None = None
        self.session_sink_size: int | None = None
        self.session_local_attn_size: int | None = None

        if cache_device is not None and cache_device.type != "cpu":
            current_omni_platform.empty_cache()

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
        num_slots: int = 1,
    ) -> None:
        """Initialize empty self-attn and cross-attn caches.

        - ``num_slots=1`` (default): all forward calls read/write the same per-layer cache regardless of noise level.
        - ``num_slots > 1``: each step has its own isolated history.

        Cross-attn is single-slot per layer since text context is independent
        of denoising step.
        """
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.cache_device = device

        # kv_cache[layer_idx][slot_idx] -> tensor [2, B, kv_size, H, D]
        self.kv_cache = [
            [
                torch.zeros(2, batch_size, kv_size, num_heads, head_dim, dtype=dtype, device=device)
                for _ in range(num_slots)
            ]
            for _ in range(num_layers)
        ]
        # Per-(layer, slot) scalar tensors tracking fill length.
        self.local_end_index = [
            [torch.tensor([0], dtype=torch.long, device=device) for _ in range(num_slots)]
            for _ in range(num_layers)
        ]
        self.global_end_index = [
            [torch.tensor([0], dtype=torch.long, device=device) for _ in range(num_slots)]
            for _ in range(num_layers)
        ]

        # Ring-buffer eviction queue per (layer, slot).
        self.evict_queues = [
            [deque() for _ in range(num_slots)]
            for _ in range(num_layers)
        ]

        self.crossattn_cache = [{"is_init": False, "k": None, "v": None} for _ in range(num_layers)]

        self.is_initialized = True

    def extend_kv_caches(self, extra_kv_size: int):
        """Grow each (layer, slot) cache by ``extra_kv_size`` tokens."""
        assert self.is_initialized, "Cannot extend uninitialized kv cache"

        dtype = self.kv_cache[0][0].dtype
        device = self.kv_cache[0][0].device

        for li in range(self.num_layers):
            for si in range(self.num_slots):
                self.kv_cache[li][si] = torch.cat(
                    [
                        self.kv_cache[li][si],
                        torch.zeros(
                            2, self.batch_size, extra_kv_size, self.num_heads, self.head_dim,
                            dtype=dtype, device=device,
                        ),
                    ],
                    dim=2,
                )

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

    def seed_all_slots_from(self, src_slot: int) -> None:
        """Replicate one slot's self-attn KV state into every other slot."""
        assert self.kv_cache is not None, "KV caches not initialized"
        if self.num_slots <= 1:
            return
        for layer in range(self.num_layers):
            src_kv = self.kv_cache[layer][src_slot]
            src_local = self.local_end_index[layer][src_slot]
            src_global = self.global_end_index[layer][src_slot]
            src_evict = self.evict_queues[layer][src_slot]
            for slot in range(self.num_slots):
                if slot == src_slot:
                    continue
                self.kv_cache[layer][slot].copy_(src_kv)
                self.local_end_index[layer][slot].copy_(src_local)
                self.global_end_index[layer][slot].copy_(src_global)
                self.evict_queues[layer][slot] = deque(src_evict)

    def advance(self, delta: int):
        self.current_lat_f += delta
