# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Noisy PP versioned KV: (req, chunk, step, block) cells, last-use eviction, column P2P."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import (
    PRIOR_HANDOFF_SLOT,
    Cell,
    CellPlan,
    CellSchedule,
    Inflight,
    Ordering,
    RequestKVTransfer,
    union_transfers,
    union_wait_ready,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.lifecycle import KVBackend, KVCacheLifecycle
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import allocate_kv_pool_with_views
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ARDiffusionPagedLayerInputs,
    _layer_idx_tensor,
)

VersionKey = tuple[str, int, int, int]  # req, chunk, step, block


def _version_key(req: str, version: Cell) -> VersionKey:
    return (req, version[0], version[1], version[2])


def _layer_of(key: VersionKey, *, num_layers: int, layer_offset: int = 0) -> int:
    """Absolute block → local pool layer; remote versions use ``block % num_layers``."""
    block = key[3]
    local = block - layer_offset
    if not 0 <= local < num_layers:
        local = block % num_layers
    return local


@dataclass(frozen=True)
class ARDiffusionNoisyKVSpec:
    num_layers: int
    num_kv_heads: int
    head_size: int
    block_size: int
    max_chunk_tokens: int
    max_history_chunks: int
    # Versions per chunk a session keeps across requests: one for the diagonal
    # orderings (the clean KV of each chunk), one per denoise step plus the
    # chunk-0 sink for Same-Step, whose consumer reads the version matching its
    # own step. Defaults to the diagonal case.
    max_history_versions: int | None = None

    @property
    def history_versions(self) -> int:
        """Retained versions per layer one rank may hold at once."""
        return int(self.max_history_versions or self.max_history_chunks)

    def __post_init__(self) -> None:
        if self.max_history_versions is not None and self.max_history_versions <= 0:
            raise ValueError(f"max_history_versions must be positive, got {self.max_history_versions}")
        for name in (
            "num_layers",
            "num_kv_heads",
            "head_size",
            "block_size",
            "max_chunk_tokens",
            "max_history_chunks",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.max_chunk_tokens % self.block_size != 0:
            raise ValueError("max_chunk_tokens must be a multiple of block_size")


class VersionPool:
    """Per-layer version slots. ``capacity`` is slots per layer (independent free stacks)."""

    def __init__(
        self,
        *,
        capacity: int,
        chunk_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_offset: int = 0,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.capacity = capacity
        self.chunk_blocks = chunk_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.layer_offset = layer_offset
        self.device = torch.device(device)
        kv_pools, k_pools, v_pools = allocate_kv_pool_with_views(
            num_blocks=capacity * chunk_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_size,
            dtype=dtype,
            device=device,
        )
        self.kv_pools = kv_pools
        self.k_pools = k_pools
        self.v_pools = v_pools
        self.free: list[list[int]] = [list(range(capacity - 1, -1, -1)) for _ in range(num_layers)]
        self.keys: dict[VersionKey, int] = {}

    def alloc(self, key: VersionKey) -> int:
        if key in self.keys:
            return self.keys[key]
        layer = self.local_layer(key)
        free = self.free[layer]
        if not free:
            raise RuntimeError(
                f"VersionPool exhausted on layer={layer} (per-layer capacity={self.capacity}, "
                f"num_layers={self.num_layers})"
            )
        slot = free.pop()
        self.keys[key] = slot
        return slot

    def release(self, key: VersionKey) -> None:
        slot = self.keys.pop(key, None)
        if slot is not None:
            self.free[self.local_layer(key)].append(slot)

    def slot_of(self, key: VersionKey) -> int:
        return self.keys[key]

    def has(self, key: VersionKey) -> bool:
        return key in self.keys

    @property
    def has_free(self) -> bool:
        return any(bool(stack) for stack in self.free)

    def block_ids(self, slot: int) -> list[int]:
        start = slot * self.chunk_blocks
        return list(range(start, start + self.chunk_blocks))

    def local_layer(self, key: VersionKey) -> int:
        return _layer_of(key, num_layers=self.num_layers, layer_offset=self.layer_offset)

    def kv_block_views(self, key: VersionKey, num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
        """K/V views ``[num_blocks, block_size, H, D]`` for one version."""
        slot = self.slot_of(key)
        layer = self.local_layer(key)
        start = slot * self.chunk_blocks
        end = start + num_blocks
        return self.kv_pools[layer][0][start:end], self.kv_pools[layer][1][start:end]

    def tensor_dict(self, key: VersionKey, num_blocks: int) -> dict[str, torch.Tensor]:
        k, v = self.kv_block_views(key, num_blocks)
        return {
            "k": k.contiguous(),
            "v": v.contiguous(),
            "layer": torch.tensor([self.local_layer(key)], dtype=torch.int32),
        }

    def copy_into(self, key: VersionKey, payload: dict[str, torch.Tensor]) -> None:
        slot = self.slot_of(key)
        layer = int(payload["layer"].reshape(-1)[0].item()) if "layer" in payload else self.local_layer(key)
        start = slot * self.chunk_blocks
        end = start + payload["k"].shape[0]
        self.kv_pools[layer][0][start:end].copy_(payload["k"])
        self.kv_pools[layer][1][start:end].copy_(payload["v"])


@dataclass
class NoisyLayerContext:
    is_ar_diffusion_paged_context: ClassVar[bool] = True
    layer_idx: int
    key_pool: torch.Tensor
    value_pool: torch.Tensor
    block_size: int
    video_slots: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    max_seq_len: int
    seq_len: int
    # Cached paged-attn view (avoids realloc on the hot path).
    _inputs: ARDiffusionPagedLayerInputs | None = None
    _empty_action_slots: ClassVar[dict[tuple[str, int | None], torch.Tensor]] = {}

    def _action_slots(self) -> torch.Tensor:
        dev = self.video_slots.device
        key = (dev.type, dev.index)
        buf = NoisyLayerContext._empty_action_slots.get(key)
        if buf is None or buf.device != dev:
            buf = self.video_slots.new_empty(0, dtype=torch.long)
            NoisyLayerContext._empty_action_slots[key] = buf
        return buf

    def to_layer_inputs(self) -> ARDiffusionPagedLayerInputs:
        if self._inputs is not None:
            return self._inputs
        self._inputs = ARDiffusionPagedLayerInputs(
            layer_idx=_layer_idx_tensor(self.layer_idx),
            key_pool=self.key_pool,
            value_pool=self.value_pool,
            block_size=self.block_size,
            seq_len=self.seq_len,
            video_slots=self.video_slots,
            action_slots=self._action_slots(),
            block_table=self.block_table,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            max_query_len=self.max_query_len,
            max_seq_len=self.max_seq_len,
        )
        return self._inputs


def _payload_bytes(payload: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for key, tensor in payload.items() if key in ("k", "v"))


def _wait_handles(handles: list[Any] | None) -> None:
    for handle in handles or ():
        if handle is not None:
            handle.wait()


def kv_comm_stream_enabled() -> bool:
    """Dedicated CUDA stream + deferred send wait (disabled; default transport only)."""
    return False


def kv_use_tensor_dict() -> bool:
    """Legacy Gloo-metadata KV path (disabled; hot path only)."""
    return False


def _pg_for_tensor(pp_group: Any, tensor: torch.Tensor) -> Any:
    return pp_group.cpu_group if tensor.is_cpu else pp_group.device_group


def _global_rank(pp_group: Any, group_rank: int) -> int:
    return int(pp_group.ranks[group_rank])


class NoisyKVTransport:
    def __init__(self, pool: VersionPool) -> None:
        self.pool = pool
        self._pending_recv: dict[VersionKey, tuple[list[Any], dict[str, torch.Tensor] | None, int]] = {}
        self._send_handles: list[Any] = []
        self._send_keepalives: list[torch.Tensor] = []
        self._comm_stream: torch.cuda.Stream | None = None
        self.bytes_sent = 0
        self.bytes_received = 0

    @property
    def uses_comm_stream(self) -> bool:
        return kv_comm_stream_enabled() and not kv_use_tensor_dict()

    def _ensure_comm_stream(self) -> torch.cuda.Stream | None:
        if not self.uses_comm_stream or self.pool.device.type != "cuda":
            return None
        if self._comm_stream is None:
            self._comm_stream = torch.cuda.Stream(device=self.pool.device)
        return self._comm_stream

    def exchange(
        self,
        transfers: tuple[RequestKVTransfer, ...],
        *,
        rank: int,
        pp_group: Any | None,
        chunk_tokens_by_request: dict[str, int],
    ) -> list[Any]:
        """Post transfers. Returns send handles, or ``[]`` when deferred to ``wait_sends``."""
        self._send_handles = []
        self._send_keepalives = []
        if pp_group is None or getattr(pp_group, "world_size", 1) <= 1:
            return []
        if kv_use_tensor_dict():
            return self._exchange_tensor_dict(
                transfers, rank=rank, pp_group=pp_group, chunk_tokens_by_request=chunk_tokens_by_request
            )
        defer = self.uses_comm_stream
        self._exchange_fixed_nccl(
            transfers,
            rank=rank,
            pp_group=pp_group,
            chunk_tokens_by_request=chunk_tokens_by_request,
            use_stream=defer,
        )
        if defer:
            return []
        # Caller owns waiting these handles (cell_executor / tests). Clear so
        # wait_sends/end_request do not wait again — gloo double-wait hangs.
        handles = list(self._send_handles)
        self._send_handles = []
        return handles

    def _exchange_tensor_dict(
        self,
        transfers: tuple[RequestKVTransfer, ...],
        *,
        rank: int,
        pp_group: Any,
        chunk_tokens_by_request: dict[str, int],
    ) -> list[Any]:
        n_send = n_recv = 0
        for xfer in transfers:
            key = _version_key(xfer.req, xfer.version)
            if xfer.src == rank:
                num_blocks = chunk_tokens_by_request[xfer.req] // self.pool.block_size
                payload = self.pool.tensor_dict(key, num_blocks)
                self.bytes_sent += _payload_bytes(payload)
                self._send_handles.extend(pp_group.isend_tensor_dict(payload, dst=xfer.dst))
                n_send += 1
            elif xfer.dst == rank:
                self.pool.alloc(key)
                payload, recv_handles, _ = pp_group.irecv_tensor_dict(src=xfer.src)
                nbytes = _payload_bytes(payload)
                merged = list(self._pending_recv.get(key, ([], None, 0))[0] or [])
                merged.extend(recv_handles)
                self._pending_recv[key] = (merged, payload, nbytes)
                n_recv += 1
        handles = list(self._send_handles)
        self._send_handles = []
        return handles

    def _exchange_fixed_nccl(
        self,
        transfers: tuple[RequestKVTransfer, ...],
        *,
        rank: int,
        pp_group: Any,
        chunk_tokens_by_request: dict[str, int],
        use_stream: bool,
    ) -> None:
        """Fixed-shape K/V P2P via one ``batch_isend_irecv``."""
        stream = self._ensure_comm_stream() if use_stream else None
        n_send = n_recv = 0
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(self.pool.device))

        ops: list[dist.P2POp] = []
        send_kept: list[torch.Tensor] = []
        recv_jobs: list[tuple[VersionKey, dict[str, torch.Tensor] | None, int]] = []

        for xfer in transfers:
            key = _version_key(xfer.req, xfer.version)
            num_blocks = chunk_tokens_by_request[xfer.req] // self.pool.block_size
            if xfer.src == rank:
                k_view, v_view = self.pool.kv_block_views(key, num_blocks)
                k = k_view if k_view.is_contiguous() else k_view.contiguous()
                v = v_view if v_view.is_contiguous() else v_view.contiguous()
                nbytes = k.numel() * k.element_size() + v.numel() * v.element_size()
                self.bytes_sent += nbytes
                dst_global = _global_rank(pp_group, xfer.dst)
                for tensor in (k, v):
                    group = _pg_for_tensor(pp_group, tensor)
                    ops.append(dist.P2POp(dist.isend, tensor, dst_global, group))
                    if tensor.is_cuda:
                        tensor.record_stream(stream or torch.cuda.current_stream(tensor.device))
                    send_kept.append(tensor)
                n_send += 1
            elif xfer.dst == rank:
                self.pool.alloc(key)
                k_view, v_view = self.pool.kv_block_views(key, num_blocks)
                if k_view.is_contiguous() and v_view.is_contiguous():
                    k_buf, v_buf = k_view, v_view
                    staging: dict[str, torch.Tensor] | None = None
                else:
                    k_buf, v_buf = k_view.contiguous(), v_view.contiguous()
                    staging = {
                        "k": k_buf,
                        "v": v_buf,
                        "layer": torch.tensor([self.pool.local_layer(key)], dtype=torch.int32),
                    }
                nbytes = k_buf.numel() * k_buf.element_size() + v_buf.numel() * v_buf.element_size()
                src_global = _global_rank(pp_group, xfer.src)
                for tensor in (k_buf, v_buf):
                    group = _pg_for_tensor(pp_group, tensor)
                    ops.append(dist.P2POp(dist.irecv, tensor, src_global, group))
                    if tensor.is_cuda and stream is not None:
                        tensor.record_stream(stream)
                recv_jobs.append((key, staging, nbytes))
                n_recv += 1

        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx:
            works = list(dist.batch_isend_irecv(ops)) if ops else []
        if n_send:
            self._send_handles.extend(works)
            self._send_keepalives.extend(send_kept)
        for key, staging, nbytes in recv_jobs:
            merged = list(self._pending_recv.get(key, ([], None, 0))[0] or [])
            if not merged:
                merged = list(works)
            else:
                merged.extend(works)
            self._pending_recv[key] = (merged, staging, nbytes)

    def wait_sends(self) -> None:
        """Host-wait deferred outbound KV (comm-stream mode)."""
        if self._send_handles:
            _wait_handles(self._send_handles)
        self._send_handles = []
        self._send_keepalives = []

    def await_ready(self, wanted: frozenset[VersionKey]) -> int:
        if not self._pending_recv:
            return 0
        ready = [key for key in self._pending_recv if key in wanted]
        for key in ready:
            handles, payload, nbytes = self._pending_recv.pop(key)
            _wait_handles(handles)
            self.bytes_received += nbytes
            if payload is not None:
                self.pool.copy_into(key, payload)
            if self._comm_stream is not None and self.pool.device.type == "cuda":
                torch.cuda.current_stream(self.pool.device).wait_stream(self._comm_stream)
        return len(self._pending_recv)

    def drain_all(self) -> None:
        self.wait_sends()
        for key in list(self._pending_recv):
            handles, payload, nbytes = self._pending_recv.pop(key)
            _wait_handles(handles)
            self.bytes_received += nbytes
            if payload is not None:
                self.pool.copy_into(key, payload)
        if self._comm_stream is not None and self.pool.device.type == "cuda":
            torch.cuda.current_stream(self.pool.device).wait_stream(self._comm_stream)

    def discard(self, key: VersionKey) -> None:
        pending = self._pending_recv.get(key)
        if pending is not None:
            handles, payload, nbytes = pending
            _wait_handles(handles)
            self._pending_recv.pop(key)
            self.bytes_received += nbytes
            del payload


class NoisyKVCache:
    def __init__(
        self,
        spec: ARDiffusionNoisyKVSpec,
        *,
        dtype: torch.dtype,
        device: torch.device,
        layer_groups: int,
        max_batch_size: int,
        step_replicas: int = 1,
        gpu_memory_fraction: float = 1.0,
        available_bytes: int | None = None,
        layer_offset: int = 0,
        blocks_per_rank: int = 1,
    ) -> None:
        if not 0.0 < float(gpu_memory_fraction) <= 1.0:
            raise ValueError(f"gpu_memory_fraction must be in (0, 1], got {gpu_memory_fraction}")
        self.spec = spec
        self.layer_groups = layer_groups
        self.max_batch_size = max(1, max_batch_size)
        self.blocks_per_rank = max(1, blocks_per_rank)
        chunk_blocks = spec.max_chunk_tokens // spec.block_size
        # Capacity is per layer. Same-Step needs at most H+2 resident versions:
        # one fixed clean sink, up to H recent chunks, and the current write;
        # chunk 0 is deduplicated while inside the recent window. The G/step
        # margins below also cover pipeline overlap and transfers.
        per_req = max(spec.max_history_chunks + 2 + layer_groups, spec.max_history_chunks + step_replicas)
        per_req = max(per_req, spec.history_versions + 2 + layer_groups, spec.history_versions + step_replicas)
        desired = self.max_batch_size * per_req
        bytes_per_version = (
            2
            * chunk_blocks
            * spec.block_size
            * spec.num_kv_heads
            * spec.head_size
            * torch.empty((), dtype=dtype).element_size()
        )
        bytes_all_layers = bytes_per_version * max(1, spec.num_layers)
        if available_bytes is not None:
            budget = int(available_bytes * float(gpu_memory_fraction))
            if bytes_all_layers <= 0:
                raise RuntimeError("invalid NoisyKV geometry: bytes_all_layers <= 0")
            max_by_budget = budget // bytes_all_layers
            if max_by_budget < 1:
                raise RuntimeError(
                    f"NoisyKV budget too small for one per-layer version slot: "
                    f"budget={budget} bytes_all_layers={bytes_all_layers}"
                )
            if max_by_budget < desired:
                raise RuntimeError(
                    f"NoisyKV budget fits {max_by_budget} versions but configured schedule requires {desired}: "
                    f"budget={budget} required_bytes={desired * bytes_all_layers}"
                )
        self.capacity = desired
        self.pool = VersionPool(
            capacity=self.capacity,
            chunk_blocks=chunk_blocks,
            block_size=spec.block_size,
            num_layers=spec.num_layers,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=dtype,
            device=device,
            layer_offset=layer_offset,
        )
        self.transport = NoisyKVTransport(self.pool)
        self.resident_peak = 0
        self.bytes_per_version = bytes_per_version
        self.reserved_bytes = self.capacity * bytes_all_layers
        # Pool-scoped history survives per-invocation NoisyKVState instances.
        self._session_structural_keys: dict[str, tuple[object, ...]] = {}
        self._retained_history: dict[str, set[Cell]] = {}
        self._active_states: dict[str, NoisyKVState] = {}

    def release_session(self, session_id: str) -> None:
        """Finish any active invocation, then release one session's history.

        The cache owns the active-state registry; callers should not inspect it
        or construct a temporary state merely to release a session.
        """
        state = self._active_states.get(session_id)
        if state is None:
            state = NoisyKVState(self)
        else:
            state.finish_request(session_id)
        state.release_session(session_id)

    def bind_session_structure(self, kv_id: str, key: tuple[object, ...]) -> None:
        """Record or check the schedule fingerprint for a persistent session pool key."""
        existing = self._session_structural_keys.get(kv_id)
        if existing is None:
            self._session_structural_keys[kv_id] = key
            return
        if existing != key:
            raise ValueError(
                f"session {kv_id!r} schedule structural_key mismatch: "
                f"retained={existing!r} request={key!r}; "
                "release_noisy_kv and restart, or keep D/S/ordering/H/blocks fixed"
            )

    def clear_session_structure(self, kv_id: str) -> None:
        self._session_structural_keys.pop(kv_id, None)


def clean_history_cells(schedule: CellSchedule, *, num_blocks: int) -> set[Cell]:
    """History cells to retain across requests for ``schedule``.

    Keeps the trailing ``H`` window ``[chunk_end-H, chunk_end)``. The diagonal
    orderings read the clean KV of each retained chunk, so one version per chunk
    is enough; a Same-Step consumer reads the version matching its own denoise
    step, so every denoise step of the window has to survive. Same-Step
    additionally keeps absolute chunk-0 clean sink cells for the whole session —
    later plans still demand ``(0, D, *)`` even after the trailing window has
    moved past chunk 0.
    """
    if num_blocks < 1:
        return set()
    t_clean = schedule.num_denoise_steps
    cells: set[Cell] = set()
    h = schedule.kv_history_chunks
    if h >= 1:
        end = schedule.chunk_end
        start = max(0, end - h)
        steps = range(schedule.num_denoise_steps) if schedule.ordering is Ordering.SAME_STEP else (t_clean,)
        cells |= {(chunk, step, block) for chunk in range(start, end) for step in steps for block in range(num_blocks)}
    if schedule.ordering is Ordering.SAME_STEP:
        cells |= {(0, t_clean, block) for block in range(num_blocks)}
    return cells


def prior_history_in_pool(
    cache: NoisyKVCache,
    kv_id: str,
    *,
    chunk_base: int,
) -> frozenset[Cell]:
    """Resident cells for ``kv_id`` with ``chunk < chunk_base`` (outer session supplies base)."""
    if chunk_base <= 0:
        return frozenset()
    return frozenset(
        (chunk, step, block) for sid, chunk, step, block in cache.pool.keys if sid == kv_id and chunk < chunk_base
    )


class NoisyKVState(KVBackend[tuple[str, Cell], list[NoisyLayerContext]], KVCacheLifecycle[str]):
    """Per-invocation bridge used by the chunk executor over a shared pool."""

    def __init__(self, cache: NoisyKVCache) -> None:
        self.cache = cache
        self._plans: dict[str, CellPlan] = {}
        self._chunk_tokens: dict[str, int] = {}
        self._last_use_global: dict[str, dict[Cell, int]] = {}
        self._retain = cache._retained_history
        self._local_tasks: dict[str, frozenset[Cell]] = {}
        self._completion_global: dict[str, dict[Cell, int]] = {}
        self._prepared_tasks: set[tuple[str, Cell]] = set()
        self.bytes_sent = 0
        self.bytes_received = 0
        self.transfers_deduped = 0
        self._inflight: tuple[Inflight, ...] = ()
        self._rank = 0
        self._pp_group: Any | None = None
        self._write_slots_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._query_start_cache: dict[int, torch.Tensor] = {}
        self._block_table_dev: torch.Tensor | None = None
        self._block_table_cpu: torch.Tensor | None = None
        self._block_table_h2d_event: Any | None = None
        self._seq_lens_dev: torch.Tensor | None = None
        self._prepare_buf_width: int = -1

    @property
    def resident_versions(self) -> int:
        return len(self.cache.pool.keys)

    @property
    def uses_comm_stream(self) -> bool:
        return self.cache.transport.uses_comm_stream

    def bind_rank(self, rank: int, pp_group: Any | None) -> None:
        if self._plans and rank != self._rank:
            raise RuntimeError("Cannot change Noisy KV rank while requests are active")
        self._rank = rank
        self._pp_group = pp_group

    def wait_sends(self) -> None:
        self.cache.transport.wait_sends()

    def protect(self, req: str, cells: Iterable[Cell]) -> None:
        """Keep ``cells`` for ``req`` out of last-use eviction and end_request release."""
        if self.cache._active_states.get(req) is not self:
            raise ValueError(f"Noisy KV owner {req!r} has no active request in this state")
        bucket = self._retain.setdefault(req, set())
        bucket.update(cells)

    def begin_request(
        self,
        req: str,
        plan: CellPlan,
        *,
        chunk_tokens: int,
        t0: int = 0,
        retain: Collection[Cell] | None = None,
    ) -> None:
        if chunk_tokens <= 0 or chunk_tokens % self.cache.spec.block_size != 0:
            raise ValueError("chunk_tokens must be a positive multiple of block_size")
        if chunk_tokens > self.cache.spec.max_chunk_tokens:
            raise ValueError("chunk_tokens exceeds max_chunk_tokens")
        if req in self.cache._active_states:
            raise RuntimeError(f"Noisy KV owner {req!r} already has an active request")
        if not 0 <= self._rank < plan.world:
            raise ValueError(f"rank {self._rank} out of range for world {plan.world}")
        self._plans[req] = plan
        self.cache._active_states[req] = self
        self._chunk_tokens[req] = chunk_tokens
        self._local_tasks[req] = frozenset(
            task for slot in range(plan.num_slots) if (task := plan.task(slot, self._rank)) is not None
        )
        mapped = {version: t0 + slot for version, slot in plan.last_use(self._rank).items()}
        self._last_use_global[req] = mapped
        self._completion_global[req] = {cell: t0 + plan.completion_tick(cell) for cell in self._local_tasks[req]}
        # An explicit policy replaces the previous call's keep set. Prior
        # sources remain alive through the plan's last-use table, not retention.
        if retain is not None:
            self._retain[req] = set(retain)

    def chunk_tokens(self, req: str) -> int:
        """Public read for mainline ``SupportsNoisyPPKV`` (admit token-uniformity)."""
        return self._chunk_tokens[req]

    def set_inflight(self, inflight: tuple[Inflight, ...]) -> None:
        self._inflight = inflight

    def retain_history(self, req: str, versions: Collection[Cell]) -> None:
        """Retain resident versions selected by the model's history policy."""
        if req not in self._plans:
            raise ValueError(f"Noisy KV owner {req!r} has no active request")
        missing = {_version_key(req, version) for version in versions}.difference(self.cache.pool.keys)
        if missing:
            raise ValueError(f"Cannot retain nonresident Noisy KV versions: {sorted(missing)}")
        self.protect(req, versions)

    def finish_request(self, owner: str) -> None:
        """Finish computation and retire temporary KV, preserving selected history."""
        self.end_request(owner)

    def release_history(self, owner: str) -> None:
        """Release retained storage only after this owner's computation ends."""
        if owner in self.cache._active_states:
            raise RuntimeError("Finish the Noisy KV request before releasing its history")
        self.end_request(owner, retain=())
        self.cache.clear_session_structure(owner)

    def _is_retained(self, req: str, version: Cell) -> bool:
        return version in self._retain.get(req, ())

    def end_request(self, req: str, *, retain: Collection[Cell] | None = None) -> None:
        """Finish one admitted request.

        ``retain is None`` keeps the existing protect set (normal plan
        completion). ``retain=()`` clears retained versions for ``req``.
        """
        active = self.cache._active_states.get(req)
        if active is not None and active is not self:
            raise RuntimeError(f"Noisy KV owner {req!r} is active in another state")
        self.cache.transport.wait_sends()
        keep_cells = set(retain) if retain is not None else self._retain.get(req, set())
        keep = {_version_key(req, cell) for cell in keep_cells}
        drop = [key for key in self.cache.pool.keys if key[0] == req and key not in keep]
        for key in drop:
            self.cache.transport.discard(key)
        for key in drop:
            self.cache.pool.release(key)
        self._plans.pop(req, None)
        self._chunk_tokens.pop(req, None)
        self._last_use_global.pop(req, None)
        self._local_tasks.pop(req, None)
        self._completion_global.pop(req, None)
        self._prepared_tasks.difference_update(tuple(item for item in self._prepared_tasks if item[0] == req))
        self.cache._active_states.pop(req, None)
        if keep_cells:
            self._retain[req] = set(keep_cells)
        else:
            self._retain.pop(req, None)

    def release_session(self, session_id: str) -> None:
        """Drop every pool version and bookkeeping for one finished session."""
        if session_id in self.cache._active_states:
            raise RuntimeError("Finish the Noisy KV request before releasing its history")
        self.end_request(session_id, retain=())
        leftover = [key for key in self.cache.pool.keys if key[0] == session_id]
        for key in leftover:
            self.cache.transport.discard(key)
            self.cache.pool.release(key)
        self._retain.pop(session_id, None)
        self.cache.clear_session_structure(session_id)

    def reset_all(self) -> None:
        if any(state is not self for state in self.cache._active_states.values()):
            raise RuntimeError("Cannot reset a Noisy KV cache with requests active in another state")
        self.cache.transport.wait_sends()
        for req in list(set(self._plans) | set(self._retain)):
            self.end_request(req, retain=())
        leftover = list(self.cache.pool.keys)
        for key in leftover:
            self.cache.transport.discard(key)
            self.cache.pool.release(key)
        self._plans.clear()
        self._local_tasks.clear()
        self._completion_global.clear()
        self._prepared_tasks.clear()
        self._chunk_tokens.clear()
        self._last_use_global.clear()
        self._retain.clear()
        self.cache._session_structural_keys.clear()
        self._inflight = ()
        self._clear_prepare_caches()

    def _clear_prepare_caches(self) -> None:
        self._write_slots_cache.clear()
        self._query_start_cache.clear()
        self._block_table_dev = None
        self._block_table_cpu = None
        self._block_table_h2d_event = None
        self._seq_lens_dev = None
        self._prepare_buf_width = -1

    def _ensure_prepare_bufs(self, width: int, device: torch.device) -> None:
        if (
            self._block_table_dev is not None
            and self._prepare_buf_width == width
            and self._block_table_dev.device == device
        ):
            return
        pin = device.type == "cuda"
        self._block_table_cpu = torch.zeros(1, width, dtype=torch.int32, pin_memory=pin)
        self._block_table_dev = torch.zeros(1, width, dtype=torch.int32, device=device)
        self._seq_lens_dev = torch.zeros(1, dtype=torch.int32, device=device)
        self._prepare_buf_width = width
        self._block_table_h2d_event = None
        self._write_slots_cache.clear()
        self._query_start_cache.clear()

    def _wait_block_table_h2d(self) -> None:
        """Keep pinned ``_block_table_cpu`` live until the prior async H2D finishes."""
        event = self._block_table_h2d_event
        if event is not None:
            event.synchronize()

    def _record_block_table_h2d(self, device: torch.device) -> None:
        if device.type != "cuda":
            self._block_table_h2d_event = None
            return
        if self._block_table_h2d_event is None:
            self._block_table_h2d_event = torch.cuda.Event()
        self._block_table_h2d_event.record()

    def _cached_query_start(self, chunk_tokens: int, device: torch.device) -> torch.Tensor:
        hit = self._query_start_cache.get(chunk_tokens)
        if hit is not None and hit.device == device:
            return hit
        qsl = torch.tensor([0, chunk_tokens], dtype=torch.int32, device=device)
        self._query_start_cache[chunk_tokens] = qsl
        return qsl

    def _cached_write_slots(self, write_slot: int, chunk_tokens: int, device: torch.device) -> torch.Tensor:
        """Contiguous write-slot mapping for one VersionPool slot."""
        key = (write_slot, chunk_tokens)
        hit = self._write_slots_cache.get(key)
        if hit is not None and hit.device == device:
            return hit
        pool = self.cache.pool
        base = write_slot * pool.chunk_blocks * pool.block_size
        slots = torch.arange(chunk_tokens, dtype=torch.long, device=device) + base
        self._write_slots_cache[key] = slots
        return slots

    def prepare(self, tasks: Sequence[tuple[str, Cell]]) -> list[list[NoisyLayerContext]]:
        """Prepare one read/write layer view for each owned cell task."""
        if len(set(tasks)) != len(tasks):
            raise ValueError("Noisy KV prepare tasks must be unique")
        for req, task in tasks:
            plan = self._plans.get(req)
            if plan is None:
                raise ValueError(f"Noisy KV owner {req!r} has no active request")
            if task not in self._local_tasks.get(req, ()):
                raise ValueError(f"Noisy KV task {task} is not scheduled locally for owner {req!r}")
        spec = self.cache.spec
        pool = self.cache.pool
        h = spec.max_history_chunks
        sink = int(any(plan.schedule.same_step_has_sink for plan in self._plans.values()))
        width = (h + 1 + sink) * pool.chunk_blocks
        max_seq_len = (h + 1 + sink) * spec.max_chunk_tokens
        max_query_len = self.cache.max_batch_size * spec.max_chunk_tokens
        batch: list[list[NoisyLayerContext]] = []
        device = pool.k_pools[0].device
        self._ensure_prepare_bufs(width, device)
        assert self._block_table_cpu is not None and self._block_table_dev is not None
        assert self._seq_lens_dev is not None
        share_bufs = len(tasks) <= 1
        for req, task in tasks:
            chunk_tokens = self._chunk_tokens[req]
            num_chunk_blocks = chunk_tokens // pool.block_size
            plan = self._plans[req]
            srcs = plan.sources(task, self._rank)
            write_key = _version_key(req, task)
            blocks: list[int] = []
            for src in srcs:
                src_key = _version_key(req, src.version)
                if not pool.has(src_key):
                    raise RuntimeError(f"I5: source {src_key} missing at prepare")
                blocks.extend(pool.block_ids(pool.slot_of(src_key))[:num_chunk_blocks])
            write_slot = pool.alloc(write_key)
            blocks.extend(pool.block_ids(write_slot)[:num_chunk_blocks])
            if len(blocks) > width:
                raise RuntimeError(f"block_table width {width} too small for {len(blocks)} blocks")
            # Do not overwrite pinned CPU source while a prior non_blocking H2D is in flight.
            self._wait_block_table_h2d()
            self._block_table_cpu[0, : len(blocks)] = torch.as_tensor(blocks, dtype=torch.int32)
            if len(blocks) < width:
                self._block_table_cpu[0, len(blocks) :].zero_()
            seq_len = (len(srcs) + 1) * chunk_tokens
            self._block_table_dev.copy_(self._block_table_cpu, non_blocking=device.type == "cuda")
            self._record_block_table_h2d(device)
            self._seq_lens_dev.fill_(seq_len)
            query_start_loc = self._cached_query_start(chunk_tokens, device)
            block_table = self._block_table_dev if share_bufs else self._block_table_dev.clone()
            seq_lens = self._seq_lens_dev if share_bufs else self._seq_lens_dev.clone()
            video_slots = self._cached_write_slots(write_slot, chunk_tokens, device)
            local_layer = pool.local_layer(write_key)
            batch.append(
                [
                    NoisyLayerContext(
                        layer_idx=local_layer,
                        key_pool=pool.k_pools[local_layer],
                        value_pool=pool.v_pools[local_layer],
                        block_size=pool.block_size,
                        video_slots=video_slots,
                        block_table=block_table,
                        query_start_loc=query_start_loc,
                        seq_lens=seq_lens,
                        max_query_len=max_query_len,
                        max_seq_len=max_seq_len,
                        seq_len=seq_len,
                    )
                ]
            )
        self._prepared_tasks.update((req, task) for req, task in tasks)
        self.cache.resident_peak = max(self.cache.resident_peak, self.resident_versions)
        return batch

    def publish(self, tasks: Sequence[tuple[str, Cell]]) -> None:
        """Confirm writes for tasks prepared by this invocation."""
        if len(set(tasks)) != len(tasks):
            raise ValueError("Noisy KV publish tasks must be unique")
        for req, task in tasks:
            if (req, task) not in self._prepared_tasks:
                raise RuntimeError(f"Noisy KV task {task} was not prepared for owner {req!r}")
            if not self.cache.pool.has(_version_key(req, task)):
                raise RuntimeError(f"Noisy KV task {task} has no reserved write cell for owner {req!r}")
        self._prepared_tasks.difference_update((req, task) for req, task in tasks)

    def exchange(self, slot: int) -> list[Any]:
        transfers = union_transfers(self._inflight, slot)
        self.transfers_deduped = len(transfers)
        handles = self.cache.transport.exchange(
            transfers, rank=self._rank, pp_group=self._pp_group, chunk_tokens_by_request=self._chunk_tokens
        )
        self.bytes_sent = self.cache.transport.bytes_sent
        self.bytes_received = self.cache.transport.bytes_received
        return handles

    def exchange_and_await_prior_handoffs(self, global_slot: int) -> list[Any]:
        """Exchange prior-history KV for requests whose first cell is ``global_slot``.

        Must run before ``prepare`` of that first cell: the normal per-slot
        ``exchange`` happens after forward, which is too late for continuation
        history that is only resident on the owner rank.
        """
        transfers: list[RequestKVTransfer] = []
        wanted: set[VersionKey] = set()
        for item in self._inflight:
            if global_slot != item.t0:
                continue
            for xfer in item.plan.transfers(PRIOR_HANDOFF_SLOT):
                transfers.append(RequestKVTransfer(req=item.req, version=xfer.version, src=xfer.src, dst=xfer.dst))
                if xfer.dst == self._rank:
                    wanted.add(_version_key(item.req, xfer.version))
        if not transfers:
            return []
        ordered = tuple(sorted(transfers, key=lambda x: (x.src, x.dst, x.req, x.version)))
        handles = self.cache.transport.exchange(
            ordered, rank=self._rank, pp_group=self._pp_group, chunk_tokens_by_request=self._chunk_tokens
        )
        self.bytes_sent = self.cache.transport.bytes_sent
        self.bytes_received = self.cache.transport.bytes_received
        for handle in handles:
            if handle is not None:
                handle.wait()
        if self.uses_comm_stream:
            self.cache.transport.wait_sends()
        if wanted:
            self.cache.transport.await_ready(frozenset(wanted))
            self.bytes_received = self.cache.transport.bytes_received
        return handles

    def evict(self, slot: int) -> None:
        if self.uses_comm_stream:
            self.cache.transport.wait_sends()
        drop: list[VersionKey] = []
        for req, used in self._last_use_global.items():
            for version, last in used.items():
                completion = self._completion_global.get(req, {}).get(version, last)
                if max(last, completion) <= slot and not self._is_retained(req, version):
                    drop.append(_version_key(req, version))
        for key in drop:
            self.cache.transport.discard(key)
            self.cache.pool.release(key)
        self.cache.resident_peak = max(self.cache.resident_peak, self.resident_versions)

    def await_ready(self, slot: int) -> int:
        wanted = {_version_key(req, version) for req, version in union_wait_ready(self._inflight, slot, self._rank)}
        pending = self.cache.transport.await_ready(frozenset(wanted))
        self.bytes_sent = self.cache.transport.bytes_sent
        self.bytes_received = self.cache.transport.bytes_received
        return pending

    def await_task_sources(self, tasks: tuple[tuple[str, Cell], ...]) -> int:
        """Wait remote history KV before attention reads it (first or later consume)."""
        wanted: set[VersionKey] = set()
        for req, task in tasks:
            plan = self._plans.get(req)
            if plan is None:
                continue
            for src in plan.sources(task, self._rank):
                if src.owner != self._rank:
                    wanted.add(_version_key(req, src.version))
        pending = self.cache.transport.await_ready(frozenset(wanted))
        self.bytes_sent = self.cache.transport.bytes_sent
        self.bytes_received = self.cache.transport.bytes_received
        return pending

    def drain(self) -> None:
        self.cache.transport.drain_all()
        self.bytes_received = self.cache.transport.bytes_received
