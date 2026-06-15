# SPDX-License-Identifier: Apache-2.0
"""BDEKVCache — the engine-level KV cache orchestrator for one BDE model.

This is the *body* of BDE's KV management: it owns a vLLM ``KVCacheManager`` (a
single chunk-window group) and the per-request adapter lifecycle, and exposes the
per-chunk operations a rollout needs — allocate, slot mapping, commit, window
lookup, free. It lives in the model runner (worker / GPU side), co-located with
the model and the KV tensors; the DreamZero pipeline calls these methods during a
rollout. The main-process ``BDEEngine`` only selects the engine and is otherwise
thin.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

from vllm_omni.bde.kv_cache.adapter import BDERequestAdapter
from vllm_omni.bde.kv_cache.chunk_window import ChunkWindowSpec
from vllm_omni.bde.kv_cache.config import BDEKVConfig
from vllm_omni.bde.kv_cache.pool import build_kv_manager, compute_num_blocks
from vllm_omni.bde.kv_cache.slot_mapping import chunk_slot_mapping, resident_block_ids
from vllm_omni.bde.kv_cache.gather import allocate_kv_pool, pool_gather_window, pool_write_chunk

_log = init_logger(__name__)


class BDEKVCache:
    """Owns the paged KV pool + per-request lifecycle for a BDE model.

    Build once per loaded model (dimensions known); then per request:
    ``begin_request`` → per chunk (``allocate_chunk`` → ``chunk_write_slots`` →
    [model writes K/V] → ``commit_chunk``) → ``end_request``.
    """

    def __init__(
        self,
        config: BDEKVConfig,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        block_size: int,
        max_model_len: int,
        available_bytes: int,
        device: torch.device | None = None,
    ) -> None:
        if not config.enable:
            raise ValueError("BDEKVCache built with a disabled BDEKVConfig")
        if config.window_chunks is None:
            raise ValueError("Phase 1 requires a bounded window (window_chunks)")
        if config.chunk_size <= 0:
            raise ValueError("BDEKVConfig.chunk_size must be set (> 0)")

        self.config = config
        self.block_size = block_size
        self.spec = ChunkWindowSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            sliding_window=config.window_chunks * config.chunk_size,
            chunk_size=config.chunk_size,
            window_chunks=config.window_chunks,
            sink_chunks=config.sink_chunks,
            reset_at_boundary=config.reset_at_boundary,
        )
        # Each pool block spans all layers' K/V, so size against the per-layer
        # page size times the layer count.
        num_blocks = compute_num_blocks(
            available_bytes,
            config.gpu_memory_fraction,
            self.spec.page_size_bytes * num_layers,
        )
        layer_names = [f"bde.layer.{i}" for i in range(num_layers)]
        self.manager = build_kv_manager(self.spec, layer_names, num_blocks, max_model_len)
        self.num_blocks = num_blocks
        self.null_block_id = self.manager.block_pool.null_block.block_id
        self._adapters: dict[str, BDERequestAdapter] = {}

        # Allocate the per-layer paged K/V pools on the given device.
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self._k_pools: list[torch.Tensor] = []
        self._v_pools: list[torch.Tensor] = []
        if device is not None:
            self._k_pools, self._v_pools = allocate_kv_pool(
                num_blocks, block_size, num_layers, num_kv_heads, head_size, dtype, device
            )

    # -- request lifecycle ---------------------------------------------------

    def begin_request(
        self, request_id: str, *, prefill_prefix_tokens: int = 0
    ) -> BDERequestAdapter:
        adapter = BDERequestAdapter(
            request_id,
            chunk_size=self.spec.chunk_size,
            prefill_prefix_tokens=prefill_prefix_tokens,
        )
        self._adapters[request_id] = adapter
        _log.debug("BDE begin_request: req=%s prefill=%d", request_id, prefill_prefix_tokens)
        return adapter

    def end_request(self, adapter: BDERequestAdapter) -> None:
        _log.debug("BDE end_request: req=%s chunks=%d free=%d",
                    adapter.request_id, adapter.completed_chunks,
                    self.manager.block_pool.get_num_free_blocks())
        self.manager.free(adapter)
        self._adapters.pop(adapter.request_id, None)

    # -- per-chunk operations ------------------------------------------------

    def allocate_chunk(self, adapter: BDERequestAdapter) -> list[int]:
        """Allocate a chunk's blocks (evicting out-of-window blocks first).

        Returns the request's full block table (incl. null_block placeholders).
        """
        blocks = self.manager.allocate_slots(adapter, num_new_tokens=self.spec.chunk_size)
        if blocks is None:
            raise RuntimeError("BDE KV pool exhausted while allocating a chunk")
        table = self.block_table(adapter)
        resident = resident_block_ids(table, self.null_block_id)
        _log.debug("BDE allocate_chunk: req=%s chunk=%d table_len=%d resident=%d free=%d",
                    adapter.request_id, adapter.completed_chunks, len(table),
                    len(resident), self.manager.block_pool.get_num_free_blocks())
        return table

    def block_table(self, adapter: BDERequestAdapter) -> list[int]:
        return list(self.manager.get_block_ids(adapter.request_id)[0])

    def chunk_write_slots(self, adapter: BDERequestAdapter) -> torch.Tensor:
        """Slot mapping for the in-flight chunk — the K/V write target."""
        return chunk_slot_mapping(
            self.block_table(adapter),
            adapter.num_computed_tokens,
            self.spec.chunk_size,
            self.block_size,
        )

    def window_block_ids(self, adapter: BDERequestAdapter) -> list[int]:
        """Resident (non-null) blocks the read path gathers the window from."""
        return resident_block_ids(self.block_table(adapter), self.null_block_id)

    def commit_chunk(self, adapter: BDERequestAdapter) -> None:
        """Advance after the chunk's K/V is written. Once per chunk, not per step."""
        _log.debug("BDE commit: req=%s before=%d", adapter.request_id, adapter.completed_chunks)
        adapter.on_chunk_committed()
        _log.debug("BDE commit: req=%s after=%d", adapter.request_id, adapter.completed_chunks)

    # -- pool-backed K/V access (Step 4 — gather / write) --------------------

    def write_chunk_kv(
        self,
        layer_index: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        adapter: BDERequestAdapter,
    ) -> None:
        """Write one layer's committed-chunk K/V into the pool."""
        slots = self.chunk_write_slots(adapter)
        _log.debug("BDE write: req=%s layer=%d chunk=%d shapes=%s dev=%s",
                    adapter.request_id, layer_index, adapter.completed_chunks,
                    (tuple(new_k.shape), tuple(new_v.shape)), slots.device)
        pool_write_chunk(
            self._k_pools[layer_index],
            self._v_pools[layer_index],
            new_k,
            new_v,
            slots,
        )

    def gather_window(self, layer_index: int, adapter: BDERequestAdapter) -> torch.Tensor:
        """Gather the resident-window K/V for one layer.

        Returns a ``(2, 1, window, n_heads, head_dim)`` tensor — the format
        DreamZero's existing attention expects as its ``kv_cache`` argument.
        """
        window_ids = self.window_block_ids(adapter)
        window = pool_gather_window(
            self._k_pools[layer_index],
            self._v_pools[layer_index],
            window_ids,
            self.block_size,
            self.spec.sliding_window,
        )
        _log.debug("BDE gather: req=%s layer=%d blocks=%s window=%s dev=%s",
                    adapter.request_id, layer_index, window_ids,
                    tuple(window.shape), window.device)
        return window
