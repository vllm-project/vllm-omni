# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CausVid pipeline persistent state."""

from __future__ import annotations

import logging
from collections import deque
from enum import IntEnum

import torch

logger = logging.getLogger(__name__)


class CacheIndex(IntEnum):
    K = 0
    V = 1


class CausVidState:
    """Pipeline persistent state across forward()/micro-step calls.

    KV cache layout: one ``[2, num_slots, kv_size, num_heads, head_dim]`` tensor
    per layer (dim 0 indexed by ``CacheIndex``, dim 1 by slot). The non-stream
    ``forward()`` path uses ``num_slots == 1``; the stream-batch path uses
    ``num_slots > 1`` so chunks at different denoising steps live in distinct
    slots.
    """

    def __init__(self) -> None:
        self.is_initialized = False
        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state."""
        self.kv_cache: list | None = None
        self.crossattn_cache: list[dict[str, bool | torch.Tensor | None]] | None = None
        self.local_end_index: list | None = None
        self.global_end_index: list | None = None
        self.evict_queues: list[list[deque[int]]] | None = None

        self.is_initialized: bool = False
        self.num_slots: int = 1
        self.batch_size: int | None = None
        self.device: torch.device | None = None
        self.num_layers: int | None = None
        self.num_heads: int | None = None
        self.head_dim: int | None = None

        self.session_id: str | None = None
        self.last_decoded_latent: torch.Tensor | None = None
        self.current_lat_f: int = 0
        self.session_chunk_latent_frames: int | None = None
        self.session_sink_size: int | None = None
        self.session_local_attn_size: int | None = None

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
        gpu_layers: set[int] | None = None,
    ) -> None:
        """Allocate self-attn + cross-attn caches.

        ``self.kv_cache`` is ``list[Tensor]`` indexed by layer; each layer tensor
        is ``[2, num_slots, kv_size, num_heads, head_dim]`` (dim 0 = ``CacheIndex``,
        dim 1 = slot). ``local_end_index`` / ``global_end_index`` / ``evict_queues``
        are sized ``[layer][slot]`` (mutable int lists / deques).
        """
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_slots = num_slots
        self.device = device

        self.kv_cache = [
            torch.zeros(
                2, num_slots, kv_size, num_heads, head_dim, dtype=dtype,
                device=device if (gpu_layers is None or i in gpu_layers) else "cpu",
            )
            for i in range(num_layers)
        ]
        self.local_end_index = [[[0] for _ in range(num_slots)] for _ in range(num_layers)]
        self.global_end_index = [[[0] for _ in range(num_slots)] for _ in range(num_layers)]
        self.evict_queues = [[deque() for _ in range(num_slots)] for _ in range(num_layers)]

        self.crossattn_cache = [{"is_init": False, "k": None, "v": None} for _ in range(num_layers)]
        self.is_initialized = True

    def update_kv_cache(self, layer_index: int, updated_kv: torch.Tensor) -> None:
        assert self.kv_cache is not None, "KV caches not initialized, call create_kv_caches first"
        self.kv_cache[layer_index] = updated_kv.clone()

    def get_kv_cache(self) -> list:
        assert self.kv_cache is not None, "KV caches not initialized"
        return self.kv_cache

    def get_crossattn_cache(self) -> list[dict[str, bool | torch.Tensor | None]]:
        assert self.crossattn_cache is not None, "Cross-attn caches not initialized"
        return self.crossattn_cache

    def seed_all_slots_from(self, src_slot: int) -> None:
        """Replicate one slot's self-attn KV state into every other slot."""
        assert self.kv_cache is not None, "KV caches not initialized"
        if self.num_slots <= 1:
            return
        for layer in range(self.num_layers):
            cache = self.kv_cache[layer]
            src = cache[:, src_slot : src_slot + 1].clone()
            cache.copy_(src.expand(-1, self.num_slots, -1, -1, -1))
            src_local = self.local_end_index[layer][src_slot][0]
            src_global = self.global_end_index[layer][src_slot][0]
            src_evict = self.evict_queues[layer][src_slot]
            for slot in range(self.num_slots):
                if slot == src_slot:
                    continue
                self.local_end_index[layer][slot][0] = src_local
                self.global_end_index[layer][slot][0] = src_global
                self.evict_queues[layer][slot] = deque(src_evict)

    def advance(self, delta: int) -> None:
        self.current_lat_f += delta