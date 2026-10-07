# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Noisy KV: (req, chunk, step) slots, last-use eviction, column P2P."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm_omni.experimental.ar_diffusion.chunk_schedule import (
    ChunkPlan,
    ChunkStep,
    Inflight,
    RequestKVTransfer,
    union_transfers,
    union_wait_ready,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import allocate_kv_pool_with_views, compute_slot_mapping
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import ARDiffusionPagedLayerInputs

VersionKey = tuple[str, int, int]  # req, chunk, step


def _version_key(req: str, version: ChunkStep) -> VersionKey:
    return (req, version[0], version[1])


@dataclass(frozen=True)
class ARDiffusionNoisyKVSpec:
    num_layers: int
    num_kv_heads: int
    head_size: int
    block_size: int
    max_chunk_tokens: int
    max_history_chunks: int

    def __post_init__(self) -> None:
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
    """Fixed-length version slots on top of ``allocate_kv_pool_with_views``."""

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
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.chunk_blocks = chunk_blocks
        self.block_size = block_size
        self.num_layers = num_layers
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
        self.free: list[int] = list(range(capacity - 1, -1, -1))
        self.keys: dict[VersionKey, int] = {}

    def alloc(self, key: VersionKey) -> int:
        if key in self.keys:
            return self.keys[key]
        if not self.free:
            raise RuntimeError(f"VersionPool exhausted (capacity={self.capacity})")
        slot = self.free.pop()
        self.keys[key] = slot
        return slot

    def release(self, key: VersionKey) -> None:
        slot = self.keys.pop(key, None)
        if slot is not None:
            self.free.append(slot)

    def slot_of(self, key: VersionKey) -> int:
        return self.keys[key]

    def has(self, key: VersionKey) -> bool:
        return key in self.keys

    def block_ids(self, slot: int) -> list[int]:
        start = slot * self.chunk_blocks
        return list(range(start, start + self.chunk_blocks))

    def tensor_dict(self, slot: int, num_blocks: int) -> dict[str, torch.Tensor]:
        payload: dict[str, torch.Tensor] = {}
        start = slot * self.chunk_blocks
        end = start + num_blocks
        for layer in range(self.num_layers):
            payload[f"k.{layer}"] = self.kv_pools[layer][0][start:end].contiguous()
            payload[f"v.{layer}"] = self.kv_pools[layer][1][start:end].contiguous()
        return payload

    def copy_into(self, slot: int, payload: dict[str, torch.Tensor]) -> None:
        start = slot * self.chunk_blocks
        end = start + payload["k.0"].shape[0]
        for layer in range(self.num_layers):
            self.kv_pools[layer][0][start:end].copy_(payload[f"k.{layer}"])
            self.kv_pools[layer][1][start:end].copy_(payload[f"v.{layer}"])


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

    def to_layer_inputs(self) -> ARDiffusionPagedLayerInputs:
        return ARDiffusionPagedLayerInputs(
            layer_idx=torch.tensor(self.layer_idx, dtype=torch.int64),
            key_pool=self.key_pool,
            value_pool=self.value_pool,
            block_size=self.block_size,
            seq_len=self.seq_len,
            video_slots=self.video_slots,
            action_slots=self.video_slots.new_empty(0, dtype=torch.long),
            block_table=self.block_table,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            max_query_len=self.max_query_len,
            max_seq_len=self.max_seq_len,
        )


def _payload_bytes(payload: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in payload.values())


def _wait_handles(handles: list[Any] | None) -> None:
    for handle in handles or ():
        if handle is not None:
            handle.wait()


class NoisyKVTransport:
    def __init__(self, pool: VersionPool) -> None:
        self.pool = pool
        self._pending_recv: dict[VersionKey, tuple[Any, dict[str, torch.Tensor]]] = {}
        self._send_handles: list[Any] = []
        self.bytes_sent = 0
        self.bytes_received = 0

    def exchange(
        self,
        transfers: tuple[RequestKVTransfer, ...],
        *,
        rank: int,
        pp_group: Any | None,
        chunk_tokens_by_request: dict[str, int],
    ) -> list[Any]:
        """Post this slot's transfers. Returns only the local *send* handles.

        Receive handles stay in ``_pending_recv`` and are waited on demand by
        ``await_ready``; a rank never blocks on inbound data it does not read
        next slot.
        """
        self._send_handles = []
        if pp_group is None or getattr(pp_group, "world_size", 1) <= 1:
            return []
        for xfer in transfers:
            key = _version_key(xfer.req, xfer.version)
            if xfer.src == rank:
                slot = self.pool.slot_of(key)
                num_blocks = chunk_tokens_by_request[xfer.req] // self.pool.block_size
                payload = self.pool.tensor_dict(slot, num_blocks)
                self.bytes_sent += _payload_bytes(payload)
                self._send_handles.extend(pp_group.isend_tensor_dict(payload, dst=xfer.dst))
            elif xfer.dst == rank:
                slot = self.pool.alloc(key)
                payload, recv_handles, _ = pp_group.irecv_tensor_dict(src=xfer.src)
                merged = list(self._pending_recv.get(key, (None, payload))[0] or [])
                merged.extend(recv_handles)
                self._pending_recv[key] = (merged, payload)
        return list(self._send_handles)

    def await_ready(self, wanted: frozenset[VersionKey]) -> int:
        """Wait and land only the inbound versions in ``wanted``.

        Returns the number of versions that remained pending (kept for a later
        slot because they are not consumed yet).
        """
        if not self._pending_recv:
            return 0
        ready = [key for key in self._pending_recv if key in wanted]
        for key in ready:
            handles, payload = self._pending_recv.pop(key)
            _wait_handles(handles)
            self.bytes_received += _payload_bytes(payload)
            self.pool.copy_into(self.pool.slot_of(key), payload)
        return len(self._pending_recv)

    def drain_all(self) -> None:
        """Wait and land every outstanding inbound version (teardown only)."""
        for key in list(self._pending_recv):
            handles, payload = self._pending_recv.pop(key)
            _wait_handles(handles)
            self.bytes_received += _payload_bytes(payload)
            self.pool.copy_into(self.pool.slot_of(key), payload)

    def discard(self, key: VersionKey) -> None:
        """Complete an unused inbound transfer before its slot is reclaimed."""
        pending = self._pending_recv.pop(key, None)
        if pending is not None:
            handles, payload = pending
            _wait_handles(handles)
            self.bytes_received += _payload_bytes(payload)


class NoisyKVCache:
    def __init__(
        self,
        spec: ARDiffusionNoisyKVSpec,
        *,
        dtype: torch.dtype,
        device: torch.device,
        layer_groups: int,
        max_batch_size: int,
        stages: int = 1,
        gpu_memory_fraction: float = 1.0,
        available_bytes: int | None = None,
    ) -> None:
        if not 0.0 < float(gpu_memory_fraction) <= 1.0:
            raise ValueError(f"gpu_memory_fraction must be in (0, 1], got {gpu_memory_fraction}")
        self.spec = spec
        self.layer_groups = layer_groups
        self.max_batch_size = max(1, max_batch_size)
        chunk_blocks = spec.max_chunk_tokens // spec.block_size
        # Latest-KV overlap also retains the stage frontier during transfer allocation.
        per_req = max(spec.max_history_chunks + 2 + layer_groups, spec.max_history_chunks + stages)
        desired = self.max_batch_size * per_req
        bytes_per_version = (
            2
            * spec.num_layers
            * chunk_blocks
            * spec.block_size
            * spec.num_kv_heads
            * spec.head_size
            * torch.empty((), dtype=dtype).element_size()
        )
        if available_bytes is not None:
            budget = int(available_bytes * float(gpu_memory_fraction))
            if bytes_per_version <= 0:
                raise RuntimeError("invalid NoisyKV geometry: bytes_per_version <= 0")
            max_by_budget = budget // bytes_per_version
            if max_by_budget < 1:
                raise RuntimeError(
                    f"NoisyKV budget too small for one version slot: "
                    f"budget={budget} bytes_per_version={bytes_per_version}"
                )
            if max_by_budget < desired:
                # Shrink capacity to fit free HBM rather than OOM after DiT load.
                self.capacity = max_by_budget
            else:
                self.capacity = desired
        else:
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
        )
        self.transport = NoisyKVTransport(self.pool)
        self.resident_peak = 0
        self.bytes_per_version = bytes_per_version
        self.reserved_bytes = self.capacity * bytes_per_version


class NoisyKVState:
    """Per-request (and in-flight) bridge used by the stage executor."""

    def __init__(self, cache: NoisyKVCache) -> None:
        self.cache = cache
        self._plans: dict[str, ChunkPlan] = {}
        self._chunk_tokens: dict[str, int] = {}
        self._last_use_global: dict[str, dict[ChunkStep, int]] = {}
        self.bytes_sent = 0
        self.bytes_received = 0
        self.transfers_deduped = 0
        self._inflight: tuple[Inflight, ...] = ()
        self._rank = 0
        self._pp_group: Any | None = None

    @property
    def resident_versions(self) -> int:
        return len(self.cache.pool.keys)

    def bind_rank(self, rank: int, pp_group: Any | None) -> None:
        self._rank = rank
        self._pp_group = pp_group

    def begin_request(self, req: str, plan: ChunkPlan, *, chunk_tokens: int, t0: int = 0) -> None:
        if chunk_tokens <= 0 or chunk_tokens % self.cache.spec.block_size != 0:
            raise ValueError("chunk_tokens must be a positive multiple of block_size")
        if chunk_tokens > self.cache.spec.max_chunk_tokens:
            raise ValueError("chunk_tokens exceeds max_chunk_tokens")
        self._plans[req] = plan
        self._chunk_tokens[req] = chunk_tokens
        mapped = {version: t0 + slot for version, slot in plan.last_use(self._rank).items()}
        self._last_use_global[req] = mapped

    def set_inflight(self, inflight: tuple[Inflight, ...]) -> None:
        self._inflight = inflight

    def end_request(self, req: str) -> None:
        drop = [key for key in self.cache.pool.keys if key[0] == req]
        for key in drop:
            self.cache.transport.discard(key)
        for key in drop:
            self.cache.pool.release(key)
        self._plans.pop(req, None)
        self._chunk_tokens.pop(req, None)
        self._last_use_global.pop(req, None)

    def reset_all(self) -> None:
        """Fail-closed: drop every version and local request bookkeeping."""
        for req in list(self._plans):
            self.end_request(req)
        # Catch versions that were allocated without a planned begin_request.
        leftover = list(self.cache.pool.keys)
        for key in leftover:
            self.cache.transport.discard(key)
            self.cache.pool.release(key)
        self._plans.clear()
        self._chunk_tokens.clear()
        self._last_use_global.clear()
        self._inflight = ()

    def prepare(self, tasks: tuple[tuple[str, ChunkStep], ...]) -> list[list[NoisyLayerContext]]:
        spec = self.cache.spec
        pool = self.cache.pool
        h = spec.max_history_chunks
        width = (h + 1) * pool.chunk_blocks
        max_seq_len = (h + 1) * spec.max_chunk_tokens
        max_query_len = self.cache.max_batch_size * spec.max_chunk_tokens
        batch: list[list[NoisyLayerContext]] = []
        for req, task in tasks:
            chunk_tokens = self._chunk_tokens[req]
            num_chunk_blocks = chunk_tokens // pool.block_size
            plan = self._plans[req]
            srcs = plan.sources(task, self._rank)
            write_key = _version_key(req, task)
            write_slot = pool.alloc(write_key)
            write_blocks = pool.block_ids(write_slot)[:num_chunk_blocks]
            blocks: list[int] = []
            for src in srcs:
                src_key = _version_key(req, src.version)
                if not pool.has(src_key):
                    raise RuntimeError(f"I5: source {src_key} missing at prepare")
                blocks.extend(pool.block_ids(pool.slot_of(src_key))[:num_chunk_blocks])
            blocks.extend(write_blocks)
            padded = blocks + [0] * (width - len(blocks))
            seq_len = (len(srcs) + 1) * chunk_tokens
            device = pool.k_pools[0].device
            block_table = torch.tensor([padded], dtype=torch.int32, device=device)
            query_start_loc = torch.tensor([0, chunk_tokens], dtype=torch.int32, device=device)
            seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
            video_slots = compute_slot_mapping(
                write_blocks, torch.arange(chunk_tokens, dtype=torch.long), pool.block_size
            ).to(device=device)
            layer_ctxs = [
                NoisyLayerContext(
                    layer_idx=layer,
                    key_pool=pool.k_pools[layer],
                    value_pool=pool.v_pools[layer],
                    block_size=pool.block_size,
                    video_slots=video_slots,
                    block_table=block_table,
                    query_start_loc=query_start_loc,
                    seq_lens=seq_lens,
                    max_query_len=max_query_len,
                    max_seq_len=max_seq_len,
                    seq_len=seq_len,
                )
                for layer in range(spec.num_layers)
            ]
            batch.append(layer_ctxs)
        self.cache.resident_peak = max(self.cache.resident_peak, self.resident_versions)
        return batch

    def publish(self, tasks: tuple[tuple[str, ChunkStep], ...]) -> None:
        del tasks

    def exchange(self, slot: int) -> list[Any]:
        transfers = union_transfers(self._inflight, slot)
        self.transfers_deduped = len(transfers)
        handles = self.cache.transport.exchange(
            transfers, rank=self._rank, pp_group=self._pp_group, chunk_tokens_by_request=self._chunk_tokens
        )
        self.bytes_sent = self.cache.transport.bytes_sent
        self.bytes_received = self.cache.transport.bytes_received
        return handles

    def evict(self, slot: int) -> None:
        """Release versions whose last use was ``slot``.

        Only safe once this slot's sends are done (I9): a slot still holding an
        unfinished outbound copy is skipped and reclaimed on a later slot.
        """
        drop: list[VersionKey] = []
        for req, used in self._last_use_global.items():
            for version, last in used.items():
                if last == slot:
                    drop.append(_version_key(req, version))
        for key in drop:
            self.cache.transport.discard(key)
            self.cache.pool.release(key)
        self.cache.resident_peak = max(self.cache.resident_peak, self.resident_versions)

    def await_ready(self, slot: int) -> int:
        """Wait only the inbound versions this rank reads at ``slot + 1`` (I9②)."""
        wanted = {_version_key(req, version) for req, version in union_wait_ready(self._inflight, slot, self._rank)}
        pending = self.cache.transport.await_ready(frozenset(wanted))
        self.bytes_sent = self.cache.transport.bytes_sent
        self.bytes_received = self.cache.transport.bytes_received
        return pending

    def drain(self) -> None:
        """Teardown: land every inbound version still in flight."""
        self.cache.transport.drain_all()
        self.bytes_received = self.cache.transport.bytes_received
