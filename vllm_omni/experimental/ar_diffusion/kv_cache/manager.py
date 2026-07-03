# SPDX-License-Identifier: Apache-2.0
"""ARDiffusionKVCache — the engine-level KV cache orchestrator for one AR-Diffusion model.

This is the *body* of AR-Diffusion's KV management: it owns a vLLM ``KVCacheManager`` (a
single chunk-window group) and the per-request adapter lifecycle, and exposes the
per-chunk operations a rollout needs — allocate, slot mapping, commit, window
lookup, free. It lives in the model runner (worker / GPU side), co-located with
the model and the KV tensors; the DreamZero pipeline calls these methods during a
rollout. The main-process ``ARDiffusionEngine`` only selects the engine and is otherwise
thin.
"""

from __future__ import annotations

import inspect
import os
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

from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import (
    ChunkWindowSpec,
    allocate_kv_pool_with_views,
    chunk_slot_mapping,
    pool_write_chunk,
    resident_block_ids,
)

_log = init_logger(__name__)


class ARDiffusionRequestAdapter:
    """Duck-types the subset of ``vllm.v1.request.Request`` that the
    ``KVCacheManager`` reads (``allocate_slots`` / ``get_computed_blocks`` /
    ``free`` and the coordinator they call into).

    It is intentionally NOT a full ``Request``. The conformance test exercises a
    real ``KVCacheManager`` against this adapter so the surface cannot silently
    drift across vLLM versions.

    A AR-Diffusion request advances one *chunk* at a time: ``allocate_slots`` is called
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
    kwargs = dict(max_model_len=max_model_len, hash_block_size=spec.block_size, enable_caching=enable_caching)
    params = inspect.signature(KVCacheManager).parameters
    if "scheduler_block_size" in params:
        kwargs["scheduler_block_size"] = spec.block_size
    if "max_num_batched_tokens" in params:
        kwargs["max_num_batched_tokens"] = max_model_len
    return KVCacheManager(config, **kwargs)


class ARDiffusionKVCache:
    """Owns the paged KV pool + per-request lifecycle for a AR-Diffusion model.

    Build once per loaded model (dimensions known); then per request:
    ``begin_request`` → per chunk (``allocate_chunk`` → ``chunk_write_slots`` →
    [model writes K/V] → ``commit_chunk``) → ``end_request``.
    """

    def __init__(
        self,
        config: ARDiffusionKVConfig,
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
        local_branches: int = 2,
        num_frame_per_block: int = 1,
    ) -> None:
        if not config.enable:
            raise ValueError("ARDiffusionKVCache built with a disabled ARDiffusionKVConfig")
        if config.window_chunks is None:
            raise ValueError("Phase 1 requires a bounded window (window_chunks)")
        if config.chunk_size <= 0:
            raise ValueError("ARDiffusionKVConfig.chunk_size must be set (> 0)")
        if local_branches not in (1, 2):
            raise ValueError(f"local_branches must be 1 or 2, got {local_branches}")

        self.config = config
        # How many CFG branches allocate from THIS rank's pool. Under
        # CFG-parallel x2 each rank executes exactly one branch (rank0 pos,
        # rank1 neg) and the other branch's lazy contexts never allocate, so
        # sizing for both would leave ~half the pool as idle capacity. A
        # single-process run (TP1 / offline) executes both branches -> 2.
        self.local_branches = local_branches
        self.num_frame_per_block = max(1, int(num_frame_per_block))
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = dtype
        self.cross_attn_length = cross_attn_length
        self.device = device or torch.device("cpu")
        self._adapters: dict[str, ARDiffusionRequestAdapter] = {}

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

        # Bytes the cross-attn pools consume directly (K+V, pos+neg, all layers).
        # Deducted from the self-attn paged-pool budget below so the two
        # allocations together stay within the GPU free-memory budget.
        def _cross_pool_bytes(length: int) -> int:
            return 2 * 2 * length * num_kv_heads * head_size * dtype.itemsize * num_layers

        cross_total_bytes = 0
        if device is not None and cross_attn_length > 0:
            cross_shape = (2, cross_attn_length, num_kv_heads, head_size)
            cross_bytes = _cross_pool_bytes(cross_attn_length)
            cross_total_bytes += cross_bytes
            for _ in range(num_layers):
                self._cross_k.append(torch.empty(cross_shape, dtype=dtype, device=device))
                self._cross_v.append(torch.empty(cross_shape, dtype=dtype, device=device))
            _log.info(
                "AR-Diffusion cross-attn pool: %d layers × (%d tok × %d heads × %d) = %.1f MiB",
                num_layers,
                cross_attn_length,
                num_kv_heads,
                head_size,
                cross_bytes / (1024 * 1024),
            )
            if cross_attn_img_length > 0:
                img_shape = (2, cross_attn_img_length, num_kv_heads, head_size)
                cross_total_bytes += _cross_pool_bytes(cross_attn_img_length)
                for _ in range(num_layers):
                    self._cross_k_img.append(torch.empty(img_shape, dtype=dtype, device=device))
                    self._cross_v_img.append(torch.empty(img_shape, dtype=dtype, device=device))
                _log.info(
                    "AR-Diffusion cross-attn IMG pool: %d layers × %d img-tok (I2V)",
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
        # Size the self-attn pool against the memory left after the cross-attn
        # pools (allocated above), so cross + self-attn stays within the budget.
        num_blocks = compute_num_blocks(
            max(0, available_bytes - cross_total_bytes),
            config.gpu_memory_fraction,
            self.spec.page_size_bytes * num_layers,
        )
        # Floor: one forward needs the resident window plus the in-flight chunk
        # (num_frame_per_block frame-blocks) for every branch THIS rank runs,
        # with a little eviction-transient headroom. The memory-fraction
        # heuristic can under-size this once block_size grows — e.g. frame-granular
        # paging at the true frame_seqlen makes each block larger and the pool
        # fewer-blocks — so guarantee the minimum the rollout cannot run without,
        # otherwise allocate_chunk hits an exhausted pool mid-forward.
        min_blocks = self.local_branches * (config.window_chunks + self.num_frame_per_block) + 2
        if num_blocks < min_blocks:
            _log.warning(
                "AR-Diffusion KV pool: memory-fraction sizing gave %d blocks; raising to the %d-block "
                "floor (%d local CFG branch(es) x (window_chunks=%d + num_frame_per_block=%d) + 2 headroom)",
                num_blocks,
                min_blocks,
                self.local_branches,
                config.window_chunks,
                self.num_frame_per_block,
            )
            num_blocks = min_blocks
        layer_names = [f"ar_diffusion.layer.{i}" for i in range(num_layers)]
        self.manager = build_kv_manager(self.spec, layer_names, num_blocks, max_model_len)
        self.managed_num_blocks = num_blocks
        self.num_blocks = num_blocks
        # Scratch blocks are outside KVCacheManager ownership. They hold current
        # denoise-step video KV when update_kv_cache=False and action/state KV in
        # both modes; they are reused every forward and never committed.
        # Default 4: measured high-water is num_frame_per_block video blocks (2)
        # + 1 action block per branch per forward, +1 margin.
        scratch_per_branch = int(os.environ.get("AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH", "4"))
        if scratch_per_branch <= 0:
            raise ValueError("AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH must be positive")
        self.scratch_blocks_per_branch = scratch_per_branch
        self.scratch_num_blocks = self.local_branches * scratch_per_branch
        self.num_blocks_total = self.managed_num_blocks + self.scratch_num_blocks
        self.null_block_id = self.manager.block_pool.null_block.block_id

        # Allocate the per-layer paged K/V pools on the given device.
        self._kv_pools: list[torch.Tensor] = []
        self._k_pools: list[torch.Tensor] = []
        self._v_pools: list[torch.Tensor] = []
        if device is not None:
            self._kv_pools, self._k_pools, self._v_pools = allocate_kv_pool_with_views(
                self.num_blocks_total,
                block_size,
                num_layers,
                num_kv_heads,
                head_size,
                dtype,
                device,
            )

    # -- cross-attention pool access -------------------------------------------
    # Cross-attn KV is static once populated — write once (from text encoder),
    # read many (every denoising step). Not managed through the paged block pool.

    def write_cross_kv(
        self,
        layer_idx: int,
        is_negative: bool,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        k_img: torch.Tensor | None = None,
        v_img: torch.Tensor | None = None,
    ) -> None:
        """Write one layer's cross-attn K/V into the pool.

        ``k`` / ``v``: ``(B, text_len, tp_num_heads, head_dim)``; only batch-0
        is copied (B=1 for inference). ``None`` skips the text half (window
        restarts keep the still-valid text K/V and rewrite only the image half).
        The ``is_negative`` flag selects the correct CFG branch slot. ``k_img`` /
        ``v_img`` (I2V image tokens, ``(B, 257, ...)``) are written when the
        image pool is allocated.
        """
        branch = 1 if is_negative else 0
        if k is not None:
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

    def begin_request(self, request_id: str, *, prefill_prefix_tokens: int = 0) -> ARDiffusionRequestAdapter:
        adapter = ARDiffusionRequestAdapter(
            request_id,
            chunk_size=self.spec.chunk_size,
            prefill_prefix_tokens=prefill_prefix_tokens,
        )
        self._adapters[request_id] = adapter
        _log.debug("AR-Diffusion begin_request: req=%s prefill=%d", request_id, prefill_prefix_tokens)
        return adapter

    def end_request(self, adapter: ARDiffusionRequestAdapter) -> None:
        _log.debug(
            "AR-Diffusion end_request: req=%s chunks=%d free=%d",
            adapter.request_id,
            adapter.completed_chunks,
            self.manager.block_pool.get_num_free_blocks(),
        )
        self.manager.free(adapter)
        self._adapters.pop(adapter.request_id, None)

    # -- per-chunk operations ------------------------------------------------

    def allocate_chunk(self, adapter: ARDiffusionRequestAdapter) -> list[int]:
        """Allocate a chunk's blocks (evicting out-of-window blocks first).

        Returns the request's full block table (incl. null_block placeholders).
        """
        blocks = self.manager.allocate_slots(adapter, num_new_tokens=self.spec.chunk_size)
        if blocks is None:
            raise RuntimeError("AR-Diffusion KV pool exhausted while allocating a chunk")
        table = self.block_table(adapter)
        resident = resident_block_ids(table, self.null_block_id)
        _log.debug(
            "AR-Diffusion allocate_chunk: req=%s chunk=%d table_len=%d resident=%d free=%d",
            adapter.request_id,
            adapter.completed_chunks,
            len(table),
            len(resident),
            self.manager.block_pool.get_num_free_blocks(),
        )
        return table

    def allocate_token_slots(self, adapter: ARDiffusionRequestAdapter, num_tokens: int) -> list[int]:
        """Allocate managed blocks for an in-flight video span without committing it."""
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        blocks = self.manager.allocate_slots(adapter, num_new_tokens=num_tokens)
        if blocks is None:
            raise RuntimeError("AR-Diffusion KV pool exhausted while allocating paged attention slots")
        return self.block_table(adapter)

    def block_table(self, adapter: ARDiffusionRequestAdapter) -> list[int]:
        return list(self.manager.get_block_ids(adapter.request_id)[0])

    def chunk_write_slots(self, adapter: ARDiffusionRequestAdapter) -> torch.Tensor:
        """Slot mapping for the in-flight chunk — the K/V write target."""
        return chunk_slot_mapping(
            self.block_table(adapter),
            adapter.num_computed_tokens,
            self.spec.chunk_size,
            self.block_size,
        )

    def scratch_block_ids(self, is_negative: bool, start: int, count: int) -> list[int]:
        """Return branch-local scratch block ids outside manager ownership."""
        if count < 0 or start < 0:
            raise ValueError(f"scratch start/count must be non-negative, got start={start}, count={count}")
        if start + count > self.scratch_blocks_per_branch:
            raise RuntimeError(
                "AR-Diffusion paged attention scratch blocks exhausted: "
                f"need [{start}, {start + count}) of {self.scratch_blocks_per_branch}. "
                "Increase AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH."
            )
        # With one local branch (CFG-parallel x2) the rank's only branch maps to
        # slot 0 whichever CFG side it is; the other branch never allocates here.
        branch_offset = self.scratch_blocks_per_branch if (is_negative and self.local_branches == 2) else 0
        base = self.managed_num_blocks + branch_offset + start
        return list(range(base, base + count))

    def key_cache(self, layer_idx: int) -> torch.Tensor:
        return self._kv_pools[layer_idx][0]

    def value_cache(self, layer_idx: int) -> torch.Tensor:
        return self._kv_pools[layer_idx][1]

    def window_block_ids(self, adapter: ARDiffusionRequestAdapter) -> list[int]:
        """Resident (non-null) managed blocks visible to paged attention."""
        return resident_block_ids(self.block_table(adapter), self.null_block_id)

    def commit_chunk(self, adapter: ARDiffusionRequestAdapter) -> None:
        """Advance the adapter by one chunk after its K/V is written.

        This standalone primitive is used by low-level manager tests. In the
        DreamZero paged-attention path, :meth:`ARDiffusionKVState.commit_paged_context`
        advances the adapter only after the forward succeeds. Call once per
        committed chunk, not per denoise step.
        """
        _log.debug("AR-Diffusion commit: req=%s before=%d", adapter.request_id, adapter.completed_chunks)
        adapter.on_chunk_committed()
        _log.debug("AR-Diffusion commit: req=%s after=%d", adapter.request_id, adapter.completed_chunks)

    # -- pool-backed K/V access --------------------------------------------

    def write_chunk_kv(
        self,
        layer_index: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        adapter: ARDiffusionRequestAdapter,
    ) -> None:
        """Write one layer's committed-chunk K/V into the pool."""
        slots = self.chunk_write_slots(adapter)
        _log.debug(
            "AR-Diffusion write: req=%s layer=%d chunk=%d shapes=%s dev=%s",
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
