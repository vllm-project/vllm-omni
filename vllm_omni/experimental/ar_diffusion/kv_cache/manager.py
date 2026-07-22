# SPDX-License-Identifier: Apache-2.0
"""Engine-level KV cache orchestration for AR-Diffusion models.

This is the *body* of AR-Diffusion's KV management: it owns a vLLM ``KVCacheManager`` (a
single chunk-window group) and the per-request adapter lifecycle, and exposes the
per-chunk operations a rollout needs — allocate, slot mapping, commit, window
lookup, free. It lives in the model runner (worker / GPU side), co-located with
the model and the KV tensors. The main-process ``ARDiffusionEngine`` only
selects the engine and is otherwise thin.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Collection, Iterable, Sequence

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

from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
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

    An AR-Diffusion request advances one *chunk* at a time: ``allocate_slots`` is called
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
    """Own the paged KV pool and KV-branch-local storage for one model.

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
        kv_branches: tuple[ARDiffusionKVBranchSpec, ...],
        session_capacity: int,
        cross_attention_lengths: dict[str, int] | None = None,
        device: torch.device | None = None,
        frames_per_block: int = 1,
        max_scratch_tokens_per_branch: int = 0,
    ) -> None:
        if not config.enable:
            raise ValueError("ARDiffusionKVCache built with a disabled ARDiffusionKVConfig")
        if config.window_chunks is None:
            raise ValueError("Phase 1 requires a bounded window (window_chunks)")
        if config.chunk_size <= 0:
            raise ValueError("ARDiffusionKVConfig.chunk_size must be set (> 0)")
        if not kv_branches:
            raise ValueError("ARDiffusionKVCache requires at least one KV branch")
        if session_capacity <= 0:
            raise ValueError(f"session_capacity must be positive, got {session_capacity}")
        kv_branch_names = [kv_branch.name for kv_branch in kv_branches]
        if len(kv_branch_names) != len(set(kv_branch_names)):
            raise ValueError(f"ARDiffusionKVCache KV branch names must be unique, got {kv_branch_names}")
        local_indices = {kv_branch.local_index for kv_branch in kv_branches}
        if local_indices != set(range(max(local_indices) + 1)):
            raise ValueError(
                "ARDiffusionKVCache KV branch local_index values must be contiguous from zero, "
                f"got {sorted(local_indices)}"
            )

        self.config = config
        self.kv_branches = kv_branches
        self.session_capacity = session_capacity
        self._kv_branch_local_indices = {kv_branch.name: kv_branch.local_index for kv_branch in kv_branches}
        self.num_local_kv_branches = max(local_indices) + 1
        if frames_per_block <= 0:
            raise ValueError(f"frames_per_block must be positive, got {frames_per_block}")
        if max_scratch_tokens_per_branch < 0:
            raise ValueError(f"max_scratch_tokens_per_branch must be non-negative, got {max_scratch_tokens_per_branch}")
        self.frames_per_block = int(frames_per_block)
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = dtype
        self.cross_attention_lengths = dict(cross_attention_lengths or {})
        invalid_cross = {name: length for name, length in self.cross_attention_lengths.items() if length <= 0}
        if invalid_cross:
            raise ValueError(f"cross_attention_lengths must be positive, got {invalid_cross}")
        self.device = device or torch.device("cpu")
        self._allocate_tensors = device is not None
        self._adapters: dict[str, ARDiffusionRequestAdapter] = {}

        # Named cross-attention K/V is allocated lazily per session and released
        # with that session. Reserve its worst-case capacity when sizing the
        # paged self-attention pool so the two stores share one memory budget.
        self._cross_sessions: dict[
            str,
            dict[str, dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]]],
        ] = {}

        def _cross_pool_bytes(length: int) -> int:
            return int(2 * len(self.kv_branches) * length * num_kv_heads * head_size * dtype.itemsize * num_layers)

        cross_bytes_per_session = sum(_cross_pool_bytes(length) for length in self.cross_attention_lengths.values())
        cross_total_bytes = cross_bytes_per_session * session_capacity
        if cross_total_bytes:
            _log.info(
                "AR-Diffusion cross-attn reservation: %.1f MiB/session × %d sessions = %.1f MiB",
                cross_bytes_per_session / (1024 * 1024),
                session_capacity,
                cross_total_bytes / (1024 * 1024),
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
        # Size the self-attn pool against memory reserved for the maximum number
        # of lazily allocated cross-attention sessions.
        num_blocks = compute_num_blocks(
            max(0, available_bytes - cross_total_bytes),
            config.gpu_memory_fraction,
            self.spec.page_size_bytes * num_layers,
        )
        # Floor: one forward needs the resident window plus the in-flight chunk
        # (frames_per_block frame-blocks) for every KV branch THIS rank runs,
        # with a little eviction-transient headroom. The memory-fraction
        # heuristic can under-size this once block_size grows — e.g. frame-granular
        # paging at the true frame_seqlen makes each block larger and the pool
        # fewer-blocks — so guarantee the minimum the rollout cannot run without,
        # otherwise allocate_chunk hits an exhausted pool mid-forward.
        resident_blocks = config.sink_chunks + config.window_chunks
        min_blocks = self.num_local_kv_branches * (resident_blocks + self.frames_per_block) + 2
        if num_blocks < min_blocks:
            _log.warning(
                "AR-Diffusion KV pool: memory-fraction sizing gave %d blocks; raising to the %d-block "
                "floor (%d local KV branch(es) x (sink_chunks=%d + window_chunks=%d "
                "+ frames_per_block=%d) + 2 headroom)",
                num_blocks,
                min_blocks,
                self.num_local_kv_branches,
                config.sink_chunks,
                config.window_chunks,
                self.frames_per_block,
            )
            num_blocks = min_blocks
        layer_names = [f"ar_diffusion.layer.{i}" for i in range(num_layers)]
        self.manager = build_kv_manager(self.spec, layer_names, num_blocks, max_model_len)
        self.managed_num_blocks = num_blocks
        self.num_blocks = num_blocks
        # Scratch blocks are outside KVCacheManager ownership. A non-committing
        # forward needs one block per current frame plus space for any
        # model-declared action/state tokens that coexist with video KV.
        declared_scratch_blocks = (max_scratch_tokens_per_branch + block_size - 1) // block_size
        minimum_scratch_blocks = self.frames_per_block + declared_scratch_blocks
        override = os.environ.get("AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH")
        override_blocks = int(override) if override is not None else 0
        if override_blocks < 0:
            raise ValueError("AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH must be non-negative")
        scratch_per_kv_branch = max(minimum_scratch_blocks, override_blocks)
        self.scratch_blocks_per_kv_branch = scratch_per_kv_branch
        self.scratch_num_blocks = self.num_local_kv_branches * scratch_per_kv_branch
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

    def _kv_branch_index(self, kv_branch: str) -> int:
        try:
            return self._kv_branch_local_indices[kv_branch]
        except KeyError as exc:
            expected = tuple(self._kv_branch_local_indices)
            raise KeyError(f"Unknown AR-Diffusion KV branch {kv_branch!r}; expected {expected}") from exc

    def _cross_attention_pool(
        self,
        session_id: str,
        cache_name: str,
        kv_branch: str,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if cache_name not in self.cross_attention_lengths:
            raise KeyError(
                f"Unknown AR-Diffusion cross-attention cache {cache_name!r}; "
                f"expected {tuple(self.cross_attention_lengths)}"
            )
        self._kv_branch_index(kv_branch)
        session = self._cross_sessions.get(session_id)
        if session is None:
            raise RuntimeError(
                f"AR-Diffusion cross-attention cache {cache_name!r} for session {session_id!r} "
                "was read before it was populated"
            )
        pool = session.get(cache_name, {}).get(kv_branch)
        if pool is None:
            raise RuntimeError(
                f"AR-Diffusion cross-attention cache {cache_name!r} for session {session_id!r} "
                f"and KV branch {kv_branch!r} was read before it was populated"
            )
        return pool

    def is_cross_attention_populated(self, session_id: str, cache_name: str, kv_branch: str) -> bool:
        """Whether a complete logical-branch cache has been published."""
        if cache_name not in self.cross_attention_lengths:
            raise KeyError(
                f"Unknown AR-Diffusion cross-attention cache {cache_name!r}; "
                f"expected {tuple(self.cross_attention_lengths)}"
            )
        self._kv_branch_index(kv_branch)
        session = self._cross_sessions.get(session_id)
        return session is not None and kv_branch in session.get(cache_name, {})

    def populate_cross_attention(
        self,
        session_id: str,
        cache_name: str,
        kv_branch: str,
        layer_kv: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Atomically populate one logical branch of a named cross-attention cache.

        ``layer_kv`` must yield exactly one ``(k, v)`` pair per model layer.
        Inputs have shape ``(B, length, local_kv_heads, head_size)``; batch zero
        is copied because AR-Diffusion currently supports one sequence per
        forward. The new cache is published only after every layer is copied,
        so failed projection/copy work cannot expose partially initialized KV.
        """
        try:
            length = self.cross_attention_lengths[cache_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown AR-Diffusion cross-attention cache {cache_name!r}; "
                f"expected {tuple(self.cross_attention_lengths)}"
            ) from exc
        self._kv_branch_index(kv_branch)
        session = self._cross_sessions.get(session_id)
        if session is None and len(self._cross_sessions) >= self.session_capacity:
            raise RuntimeError(
                "AR-Diffusion cross-attention session capacity exhausted; "
                "the runner must evict a session before allocating another"
            )
        if not self._allocate_tensors:
            raise RuntimeError("AR-Diffusion cross-attention tensors require a configured pool device")

        shape = (length, self.num_kv_heads, self.head_size)
        expected_input_shape = (1, *shape)
        k_pool = [torch.empty(shape, dtype=self.dtype, device=self.device) for _ in range(self.num_layers)]
        v_pool = [torch.empty(shape, dtype=self.dtype, device=self.device) for _ in range(self.num_layers)]
        populated_layers = 0
        for layer_idx, (k, v) in enumerate(layer_kv):
            if layer_idx >= self.num_layers:
                raise ValueError(
                    f"AR-Diffusion cross-attention cache {cache_name!r} expected {self.num_layers} layers, "
                    f"got more than {self.num_layers}"
                )
            if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                raise ValueError(
                    f"AR-Diffusion cross-attention cache {cache_name!r} layer {layer_idx} "
                    f"must yield torch.Tensor k/v, got {type(k).__name__} and {type(v).__name__}"
                )
            if tuple(k.shape) != expected_input_shape or tuple(v.shape) != expected_input_shape:
                raise ValueError(
                    f"AR-Diffusion cross-attention cache {cache_name!r} layer {layer_idx} expected "
                    f"k/v shape {expected_input_shape}, got {tuple(k.shape)} and {tuple(v.shape)}"
                )
            k_pool[layer_idx].copy_(k[0])
            v_pool[layer_idx].copy_(v[0])
            populated_layers += 1
        if populated_layers != self.num_layers:
            raise ValueError(
                f"AR-Diffusion cross-attention cache {cache_name!r} expected {self.num_layers} layers, "
                f"got {populated_layers}"
            )

        if session is None:
            session = {}
            self._cross_sessions[session_id] = session
        session.setdefault(cache_name, {})[kv_branch] = (k_pool, v_pool)

    def read_cross_attention_kv(
        self,
        session_id: str,
        cache_name: str,
        layer_idx: int,
        kv_branch: str,
    ) -> dict[str, torch.Tensor | bool]:
        """Return a model-facing K/V dict for one named cross-attention pool."""
        k_pool, v_pool = self._cross_attention_pool(session_id, cache_name, kv_branch)
        return {
            "is_init": True,
            "k": k_pool[layer_idx].unsqueeze(0),
            "v": v_pool[layer_idx].unsqueeze(0),
        }

    def retain_cross_attention(self, session_id: str, cache_names: Collection[str]) -> None:
        """Release named cross-attention caches not retained by an internal reset."""
        session = self._cross_sessions.get(session_id)
        if session is None:
            return
        keep = set(cache_names)
        for cache_name in tuple(session):
            if cache_name not in keep:
                del session[cache_name]
        if not session:
            self._cross_sessions.pop(session_id, None)

    def release_cross_attention(self, session_id: str) -> None:
        """Release every named cross-attention allocation for one session."""
        self._cross_sessions.pop(session_id, None)

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

    def scratch_block_ids(self, kv_branch: str, start: int, count: int) -> list[int]:
        """Return KV-branch-local scratch block ids outside manager ownership."""
        if count < 0 or start < 0:
            raise ValueError(f"scratch start/count must be non-negative, got start={start}, count={count}")
        if start + count > self.scratch_blocks_per_kv_branch:
            raise RuntimeError(
                "AR-Diffusion paged attention scratch blocks exhausted: "
                f"need [{start}, {start + count}) of {self.scratch_blocks_per_kv_branch}. "
                "Declare max_scratch_tokens_per_branch in the pipeline capability "
                "or increase AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH."
            )
        kv_branch_offset = self.scratch_blocks_per_kv_branch * self._kv_branch_index(kv_branch)
        base = self.managed_num_blocks + kv_branch_offset + start
        return list(range(base, base + count))

    def key_cache(self, layer_idx: int) -> torch.Tensor:
        return self._kv_pools[layer_idx][0]

    def value_cache(self, layer_idx: int) -> torch.Tensor:
        return self._kv_pools[layer_idx][1]

    def window_block_ids(self, adapter: ARDiffusionRequestAdapter) -> list[int]:
        """Resident (non-null) managed blocks visible to paged attention."""
        return [int(block_id) for block_id in resident_block_ids(self.block_table(adapter), self.null_block_id)]

    def commit_chunk(self, adapter: ARDiffusionRequestAdapter) -> None:
        """Advance the adapter by one chunk after its K/V is written.

        This standalone primitive is used by low-level manager tests. The
        paged-attention path uses :meth:`ARDiffusionKVState.commit_paged_context`,
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
