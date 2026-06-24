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

from collections.abc import Sequence

import torch
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
)
from vllm.v1.request import RequestStatus

from vllm_omni.experimental.bde.kv_cache.config import BDEKVConfig
from vllm_omni.experimental.bde.kv_cache.paged import (
    ChunkWindowSpec,
    allocate_kv_pool,
    build_window_slots,
    chunk_slot_mapping,
    pool_gather_window,
    pool_write_chunk,
    resident_block_ids,
)

_log = init_logger(__name__)


class BDERequestAdapter:
    """Duck-types the subset of ``vllm.v1.request.Request`` that the
    ``KVCacheManager`` reads (``allocate_slots`` / ``get_computed_blocks`` /
    ``free`` and the coordinator they call into).

    It is intentionally NOT a full ``Request``. The conformance test exercises a
    real ``KVCacheManager`` against this adapter so the surface cannot silently
    drift across vLLM versions.

    A BDE request advances one *chunk* at a time: ``allocate_slots`` is called
    once per chunk and ``num_computed_tokens`` advances only when a chunk is
    committed (:meth:`on_chunk_committed`), so the ``T`` denoise steps of a chunk
    reuse the same slots.
    """

    def __init__(
        self,
        request_id: str,
        *,
        chunk_size: int,
        prefill_prefix_tokens: int = 0,
    ) -> None:
        self.request_id = request_id
        self._chunk_size = chunk_size
        self._prefill = prefill_prefix_tokens
        self._completed_chunks = 0
        # Filled only when cross-request prefix reuse is enabled (Phase 3).
        self.block_hashes: list = []
        self.skip_reading_prefix_cache = True
        self.num_preemptions = 0
        # vLLM watermark gate reads this; map the request lifecycle onto it.
        self.status = RequestStatus.WAITING

    @property
    def num_computed_tokens(self) -> int:
        """Persistent KV already materialized (committed chunks + prefill)."""
        return self._prefill + self._completed_chunks * self._chunk_size

    @property
    def num_tokens(self) -> int:
        """Total tokens once the in-flight chunk is committed."""
        return self._prefill + (self._completed_chunks + 1) * self._chunk_size

    @property
    def num_prompt_tokens(self) -> int:
        """The prefill prefix length (read by ``cache_blocks`` when caching)."""
        return self._prefill

    @property
    def completed_chunks(self) -> int:
        return self._completed_chunks

    def on_chunk_committed(self) -> None:
        """Advance by one chunk. Call once per chunk, not per denoise step."""
        self._completed_chunks += 1


def compute_num_blocks(
    available_bytes: int,
    gpu_memory_fraction: float,
    page_size_bytes: int,
) -> int:
    """Number of KV blocks that fit in ``fraction`` of the memory budget."""
    if page_size_bytes <= 0:
        raise ValueError(f"page_size_bytes must be positive, got {page_size_bytes}")
    if not 0.0 < gpu_memory_fraction <= 1.0:
        raise ValueError(f"gpu_memory_fraction must be in (0, 1], got {gpu_memory_fraction}")
    budget = int(available_bytes * gpu_memory_fraction)
    return max(0, budget // page_size_bytes)


def build_kv_manager(
    spec: KVCacheSpec,
    layer_names: Sequence[str],
    num_blocks: int,
    max_model_len: int,
    *,
    enable_caching: bool = False,
) -> KVCacheManager:
    """Build a ``KVCacheManager`` with a single KV cache group for ``spec``.

    Args:
        spec: The KV cache spec for the group (e.g. a ``ChunkWindowSpec``).
        layer_names: Attention layers sharing this group's block table.
        num_blocks: Total physical blocks in the pool.
        max_model_len: Upper bound on a request's sequence length.
        enable_caching: Cross-request prefix caching (Phase 3); off in Phase 1.
    """
    layer_names = list(layer_names)
    group = KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
    tensors = [KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=layer_names)]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=[group],
    )
    return KVCacheManager(
        config,
        max_model_len=max_model_len,
        scheduler_block_size=spec.block_size,
        hash_block_size=spec.block_size,
        enable_caching=enable_caching,
    )


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
        cross_attn_length: int = 0,
        cross_attn_img_length: int = 0,
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
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = dtype
        self.cross_attn_length = cross_attn_length
        self.device = device or torch.device("cpu")
        self._adapters: dict[str, BDERequestAdapter] = {}

        # -- cross-attention pool (static, one-time-fill) -----------------------
        # Cross-attn KV is computed once from the text encoder and never changes.
        # Allocate a per-layer contiguous tensor for each branch: (2, text_len,
        # kv_heads, head_dim) where dim-0 = [pos, neg].  Allocated separately
        # from the self-attn block pool (static, small — ~400 MiB for
        # DreamZero I2V).  Both pools draw from the same GPU free memory but
        # the cross-attn pool is sized directly rather than via the block pool
        # budget (no eviction/chunk lifecycle needed).
        self.cross_attn_img_length = cross_attn_img_length
        self._cross_k: list[torch.Tensor] = []
        self._cross_v: list[torch.Tensor] = []
        # I2V image-token cross-attn pool. Like the text k/v, the image-token
        # k_img/v_img are session-invariant (the conditioning image doesn't change),
        # so they are cached once and read every denoise step — see
        # WanI2VCrossAttention (#4154 caches these model-side too). Empty for T2V.
        self._cross_k_img: list[torch.Tensor] = []
        self._cross_v_img: list[torch.Tensor] = []
        if device is not None and cross_attn_length > 0:
            cross_shape = (2, cross_attn_length, num_kv_heads, head_size)
            bytes_per_element = dtype.itemsize
            cross_bytes = (
                2  # K + V
                * 2  # pos + neg
                * cross_attn_length
                * num_kv_heads
                * head_size
                * bytes_per_element
                * num_layers
            )
            for _ in range(num_layers):
                self._cross_k.append(torch.empty(cross_shape, dtype=dtype, device=device))
                self._cross_v.append(torch.empty(cross_shape, dtype=dtype, device=device))
            _log.info(
                "BDE cross-attn pool: %d layers × (%d tok × %d heads × %d) = %.1f MiB",
                num_layers,
                cross_attn_length,
                num_kv_heads,
                head_size,
                cross_bytes / (1024 * 1024),
            )
            if cross_attn_img_length > 0:
                img_shape = (2, cross_attn_img_length, num_kv_heads, head_size)
                for _ in range(num_layers):
                    self._cross_k_img.append(torch.empty(img_shape, dtype=dtype, device=device))
                    self._cross_v_img.append(torch.empty(img_shape, dtype=dtype, device=device))
                _log.info(
                    "BDE cross-attn IMG pool: %d layers × %d img-tok (I2V)",
                    num_layers,
                    cross_attn_img_length,
                )

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
        # Floor: one forward needs the resident window plus the in-flight chunk for
        # BOTH CFG branches (pos/neg are independent adapters allocating from the
        # same pool), with a little eviction-transient headroom. The memory-fraction
        # heuristic can under-size this once block_size grows — e.g. frame-granular
        # paging at the true frame_seqlen makes each block larger and the pool
        # fewer-blocks — so guarantee the minimum the rollout cannot run without,
        # otherwise allocate_chunk hits an exhausted pool mid-forward.
        min_blocks = 2 * (config.window_chunks + 1) + 2
        if num_blocks < min_blocks:
            _log.warning(
                "BDE KV pool: memory-fraction sizing gave %d blocks; raising to the %d-block "
                "floor (2 CFG branches x (window_chunks=%d + 1) + 2 headroom)",
                num_blocks,
                min_blocks,
                config.window_chunks,
            )
            num_blocks = min_blocks
        layer_names = [f"bde.layer.{i}" for i in range(num_layers)]
        self.manager = build_kv_manager(self.spec, layer_names, num_blocks, max_model_len)
        self.num_blocks = num_blocks
        self.null_block_id = self.manager.block_pool.null_block.block_id

        # Allocate the per-layer paged K/V pools on the given device.
        self._k_pools: list[torch.Tensor] = []
        self._v_pools: list[torch.Tensor] = []
        if device is not None:
            self._k_pools, self._v_pools = allocate_kv_pool(
                num_blocks, block_size, num_layers, num_kv_heads, head_size, dtype, device
            )

    # -- cross-attention pool access -------------------------------------------
    # Cross-attn KV is static once populated — write once (from text encoder),
    # read many (every denoising step). Not managed through the paged block pool.

    def write_cross_kv(
        self,
        layer_idx: int,
        is_negative: bool,
        k: torch.Tensor,
        v: torch.Tensor,
        k_img: torch.Tensor | None = None,
        v_img: torch.Tensor | None = None,
    ) -> None:
        """Write one layer's cross-attn K/V into the pool.

        ``k`` / ``v``: ``(B, text_len, tp_num_heads, head_dim)``; only batch-0
        is copied (B=1 for inference). The ``is_negative`` flag selects the
        correct CFG branch slot. ``k_img`` / ``v_img`` (I2V image tokens,
        ``(B, 257, ...)``) are written when the image pool is allocated.
        """
        branch = 1 if is_negative else 0
        self._cross_k[layer_idx][branch].copy_(k[0])
        self._cross_v[layer_idx][branch].copy_(v[0])
        if k_img is not None and self._cross_k_img:
            self._cross_k_img[layer_idx][branch].copy_(k_img[0])
            self._cross_v_img[layer_idx][branch].copy_(v_img[0])

    def read_cross_kv(self, layer_idx: int, is_negative: bool) -> dict:
        """Return a pool-backed cross-attn cache dict for one layer.

        The dict matches the ``{"is_init": True, "k": Tensor, "v": Tensor}``
        convention the cross-attention module expects — it reads from the pool
        slice rather than from the lazy-initialised model-local dict. For I2V,
        ``k_img`` / ``v_img`` are added so the image-token cache (added by #4154)
        reads from the pool too.
        """
        branch = 1 if is_negative else 0
        k = self._cross_k[layer_idx][branch].unsqueeze(0)  # (1, L, heads, dim)
        v = self._cross_v[layer_idx][branch].unsqueeze(0)
        cache = {"is_init": True, "k": k, "v": v}
        if self._cross_k_img:
            cache["k_img"] = self._cross_k_img[layer_idx][branch].unsqueeze(0)
            cache["v_img"] = self._cross_v_img[layer_idx][branch].unsqueeze(0)
        return cache

    # -- request lifecycle ---------------------------------------------------

    def begin_request(self, request_id: str, *, prefill_prefix_tokens: int = 0) -> BDERequestAdapter:
        adapter = BDERequestAdapter(
            request_id,
            chunk_size=self.spec.chunk_size,
            prefill_prefix_tokens=prefill_prefix_tokens,
        )
        self._adapters[request_id] = adapter
        _log.debug("BDE begin_request: req=%s prefill=%d", request_id, prefill_prefix_tokens)
        return adapter

    def end_request(self, adapter: BDERequestAdapter) -> None:
        _log.debug(
            "BDE end_request: req=%s chunks=%d free=%d",
            adapter.request_id,
            adapter.completed_chunks,
            self.manager.block_pool.get_num_free_blocks(),
        )
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
        _log.debug(
            "BDE allocate_chunk: req=%s chunk=%d table_len=%d resident=%d free=%d",
            adapter.request_id,
            adapter.completed_chunks,
            len(table),
            len(resident),
            self.manager.block_pool.get_num_free_blocks(),
        )
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
        """Advance the adapter by one chunk after its K/V is written.

        This is the standalone per-chunk advance used by the unit lifecycle and the
        ``tests/bde`` suite. In the DreamZero production path the advance happens
        elsewhere: :meth:`BDEKVState.update_kv_cache` (gather.py) calls
        ``adapter.on_chunk_committed()`` directly as it allocates each frame-chunk,
        and the pipeline's ``_kv_commit`` -> :meth:`BDEKVState.commit_chunk` is only
        a no-op log. Call once per chunk, not per denoise step.
        """
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
        _log.debug(
            "BDE write: req=%s layer=%d chunk=%d shapes=%s dev=%s",
            adapter.request_id,
            layer_index,
            adapter.completed_chunks,
            (tuple(new_k.shape), tuple(new_v.shape)),
            slots.device,
        )
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
        _log.debug(
            "BDE gather: req=%s layer=%d blocks=%s window=%s dev=%s",
            adapter.request_id,
            layer_index,
            window_ids,
            tuple(window.shape),
            window.device,
        )
        return window

    def gather_window_all_layers(self, adapter: BDERequestAdapter) -> list[torch.Tensor]:
        """Gather the resident-window K/V for every layer in one shot.

        The window's block table is identical across layers within a forward, so
        the block-id lookup and the flat slot index are computed **once** and shared
        across all per-layer gathers — removing the per-layer Python work that the
        single-layer ``gather_window`` repeats. Returns one
        ``(2, 1, window, n_heads, head_dim)`` tensor per layer.
        """
        window_ids = self.window_block_ids(adapter)
        slots = build_window_slots(window_ids, self.block_size, self._k_pools[0].device)
        windows = [
            pool_gather_window(
                self._k_pools[i],
                self._v_pools[i],
                window_ids,
                self.block_size,
                self.spec.sliding_window,
                slots=slots,
            )
            for i in range(self.num_layers)
        ]
        _log.debug(
            "BDE gather-all: req=%s layers=%d blocks=%d window=%s",
            adapter.request_id,
            self.num_layers,
            len(window_ids),
            tuple(windows[0].shape),
        )
        return windows
