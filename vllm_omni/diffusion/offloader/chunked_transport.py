# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Layout and lifecycle contracts for chunked weight transport.

This module deliberately has no platform-stream dependency.  Manifest and
packing correctness can therefore be tested on CPU before a backend submits
H2D copies or collectives.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return (value + divisor - 1) // divisor


def round_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def dtype_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


class ParameterOwner(str, Enum):
    FSDP_MANAGED = "fsdp_managed"
    CHUNKED_FS_OFFLOAD = "chunked_fs_offload"
    RESIDENT = "resident"
    UNSUPPORTED_RESIDENT = "unsupported_resident"


class WeightLayout(str, Enum):
    CHUNK_MAJOR = "chunk_major"
    WHOLE_BLOCK = "whole_block"


class PinFailurePolicy(str, Enum):
    FAIL = "fail"
    WHOLE_BLOCK_FALLBACK = "whole_block_fallback"


class SlotPhase(str, Enum):
    REUSABLE = "reusable"
    SUBMITTED = "submitted"
    READY = "ready"
    IN_USE = "in_use"


@dataclass(frozen=True)
class TensorMeta:
    name: str
    offset: int
    numel: int
    shape: tuple[int, ...]
    is_buffer: bool = False
    owner: ParameterOwner = ParameterOwner.CHUNKED_FS_OFFLOAD
    # Physical stride of the source tensor; ``numel`` counts *storage*
    # elements so non-contiguous views (e.g. online FP8 Cutlass transposed
    # weights) round-trip through the flat transport buffer without changing
    # their physical layout.  None means a legacy contiguous manifest entry.
    stride: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ChunkMeta:
    chunk_id: int
    cpu_offset: int
    full_offset: int
    valid_numel: int
    padded_numel: int
    local_numel: int


@dataclass(frozen=True)
class DTypeManifest:
    dtype: torch.dtype
    tensors: tuple[TensorMeta, ...]
    chunks: tuple[ChunkMeta, ...]
    total_numel: int
    padded_numel: int
    local_numel: int
    local_chunk_numel: int
    alignment_numel: int

    @property
    def pinned_bytes(self) -> int:
        return self.local_numel * dtype_element_size(self.dtype)


@dataclass(frozen=True)
class PartManifest:
    block_id: int
    part_id: str
    weight_shard_size: int
    weight_shard_rank: int
    chunk_size_bytes: int
    alignment_bytes: int
    layout: WeightLayout
    dtypes: tuple[DTypeManifest, ...]
    digest: str

    @property
    def pinned_bytes(self) -> int:
        return sum(dtype_manifest.pinned_bytes for dtype_manifest in self.dtypes)

    @property
    def chunk_count(self) -> int:
        return sum(len(dtype_manifest.chunks) for dtype_manifest in self.dtypes)


@dataclass
class PinBudget:
    """Exact pinned-memory reservation gate shared by one engine."""

    limit_bytes: int | None
    required_bytes: int = 0
    reserved_bytes: int = 0
    allocations: dict[str, int] = field(default_factory=dict)

    def plan(self, key: str, size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError(f"pin size must be non-negative, got {size_bytes}")
        if key in self.allocations:
            raise ValueError(f"duplicate pin budget key: {key}")
        next_required = self.required_bytes + size_bytes
        if self.limit_bytes is not None and next_required > self.limit_bytes:
            raise MemoryError(
                f"pinned Host budget exceeded: required={next_required} limit={self.limit_bytes} key={key}"
            )
        self.allocations[key] = size_bytes
        self.required_bytes = next_required

    def reserve(self, key: str) -> None:
        try:
            size_bytes = self.allocations[key]
        except KeyError as exc:
            raise KeyError(f"pin allocation was not planned: {key}") from exc
        self.reserved_bytes += size_bytes
        if self.reserved_bytes > self.required_bytes:
            raise RuntimeError("pinned Host reservation exceeds the planned budget")


TensorSpec = tuple[str, torch.Tensor, bool]
PinnedAllocator = Callable[[int, torch.dtype], torch.Tensor]


def is_chunk_transport_supported(tensor: torch.Tensor) -> bool:
    return tensor.ndim > 0 and (tensor.is_floating_point() or tensor.is_complex())


def _full_chunk_numel(
    dtype: torch.dtype,
    weight_shard_size: int,
    chunk_size_bytes: int,
    alignment_bytes: int,
) -> tuple[int, int]:
    if weight_shard_size <= 0:
        raise ValueError(f"weight_shard_size must be positive, got {weight_shard_size}")
    if chunk_size_bytes <= 0:
        raise ValueError(f"chunk_size_bytes must be positive, got {chunk_size_bytes}")
    if alignment_bytes <= 0:
        raise ValueError(f"alignment_bytes must be positive, got {alignment_bytes}")
    element_size = dtype_element_size(dtype)
    alignment_numel = ceil_div(alignment_bytes, element_size)
    collective_alignment = weight_shard_size * alignment_numel
    requested_numel = chunk_size_bytes // element_size
    full_chunk_numel = requested_numel - requested_numel % collective_alignment
    if full_chunk_numel == 0:
        raise ValueError(
            "chunk size is smaller than one aligned collective unit: "
            f"chunk_size_bytes={chunk_size_bytes}, dtype={dtype}, "
            f"weight_shard_size={weight_shard_size}, alignment_bytes={alignment_bytes}"
        )
    return full_chunk_numel, alignment_numel


def build_part_manifest(
    tensor_specs: Sequence[TensorSpec],
    *,
    block_id: int,
    part_id: str,
    weight_shard_size: int,
    weight_shard_rank: int,
    chunk_size_bytes: int,
    alignment_bytes: int = 256,
    layout: WeightLayout = WeightLayout.CHUNK_MAJOR,
) -> PartManifest:
    if not 0 <= weight_shard_rank < weight_shard_size:
        raise ValueError(f"weight_shard_rank={weight_shard_rank} is outside [0, {weight_shard_size})")

    grouped: OrderedDict[torch.dtype, list[TensorSpec]] = OrderedDict()
    for name, tensor, is_buffer in tensor_specs:
        if not is_chunk_transport_supported(tensor):
            continue
        grouped.setdefault(tensor.dtype, []).append((name, tensor, is_buffer))

    dtype_manifests: list[DTypeManifest] = []
    for dtype, dtype_specs in grouped.items():
        offset = 0
        tensor_metas: list[TensorMeta] = []
        for name, tensor, is_buffer in dtype_specs:
            stride = tensor.stride()
            storage_numel = (
                0
                if tensor.numel() == 0
                else 1 + sum((size - 1) * axis_stride for size, axis_stride in zip(tensor.shape, stride))
            )
            tensor_metas.append(
                TensorMeta(
                    name=name,
                    offset=offset,
                    numel=storage_numel,
                    shape=tuple(tensor.shape),
                    is_buffer=is_buffer,
                    stride=tuple(stride),
                )
            )
            offset += storage_numel

        total_numel = offset
        chunks: list[ChunkMeta] = []
        cpu_offset = 0
        if layout is WeightLayout.CHUNK_MAJOR:
            full_chunk_numel, alignment_numel = _full_chunk_numel(
                dtype,
                weight_shard_size,
                chunk_size_bytes,
                alignment_bytes,
            )
            for chunk_id, full_offset in enumerate(range(0, total_numel, full_chunk_numel)):
                valid_numel = min(full_chunk_numel, total_numel - full_offset)
                padded_numel = round_up(
                    valid_numel,
                    weight_shard_size * alignment_numel,
                )
                local_numel = padded_numel // weight_shard_size
                chunks.append(
                    ChunkMeta(
                        chunk_id=chunk_id,
                        cpu_offset=cpu_offset,
                        full_offset=full_offset,
                        valid_numel=valid_numel,
                        padded_numel=padded_numel,
                        local_numel=local_numel,
                    )
                )
                cpu_offset += local_numel
        else:
            alignment_numel = 1
            local_numel = ceil_div(total_numel, weight_shard_size)
            padded_numel = local_numel * weight_shard_size
            chunks.append(
                ChunkMeta(
                    chunk_id=0,
                    cpu_offset=0,
                    full_offset=0,
                    valid_numel=total_numel,
                    padded_numel=padded_numel,
                    local_numel=local_numel,
                )
            )
            cpu_offset = local_numel

        padded_total = chunks[-1].full_offset + chunks[-1].padded_numel if chunks else 0
        dtype_manifests.append(
            DTypeManifest(
                dtype=dtype,
                tensors=tuple(tensor_metas),
                chunks=tuple(chunks),
                total_numel=total_numel,
                padded_numel=padded_total,
                local_numel=cpu_offset,
                local_chunk_numel=max((chunk.local_numel for chunk in chunks), default=0),
                alignment_numel=alignment_numel,
            )
        )

    digest_payload = {
        "block_id": block_id,
        "part_id": part_id,
        "weight_shard_size": weight_shard_size,
        "chunk_size_bytes": chunk_size_bytes,
        "alignment_bytes": alignment_bytes,
        "layout": layout.value,
        "dtypes": [
            {
                "dtype": str(dtype_manifest.dtype),
                "tensors": [
                    {
                        "name": tensor.name,
                        "offset": tensor.offset,
                        "numel": tensor.numel,
                        "shape": tensor.shape,
                        "is_buffer": tensor.is_buffer,
                        "owner": tensor.owner.value,
                        "stride": tensor.stride,
                    }
                    for tensor in dtype_manifest.tensors
                ],
                "chunks": [chunk.__dict__ for chunk in dtype_manifest.chunks],
            }
            for dtype_manifest in dtype_manifests
        ],
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return PartManifest(
        block_id=block_id,
        part_id=part_id,
        weight_shard_size=weight_shard_size,
        weight_shard_rank=weight_shard_rank,
        chunk_size_bytes=chunk_size_bytes,
        alignment_bytes=alignment_bytes,
        layout=layout,
        dtypes=tuple(dtype_manifests),
        digest=digest,
    )


def _default_pinned_allocator(numel: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)


def pack_local_shard(
    tensor_specs: Sequence[TensorSpec],
    manifest: PartManifest,
    *,
    allocator: PinnedAllocator | None = None,
) -> dict[torch.dtype, torch.Tensor]:
    """Pack final tensors directly into the manifest's local Host layout."""
    allocator = allocator or _default_pinned_allocator
    sources = {name: tensor for name, tensor, _ in tensor_specs}
    packed: dict[torch.dtype, torch.Tensor] = {}

    for dtype_manifest in manifest.dtypes:
        local = allocator(dtype_manifest.local_numel, dtype_manifest.dtype)
        if local.device.type != "cpu":
            raise ValueError(f"pinned shard allocator returned non-CPU tensor: {local.device}")
        local.zero_()

        if manifest.layout is WeightLayout.WHOLE_BLOCK:
            chunk = dtype_manifest.chunks[0]
            shard_begin = manifest.weight_shard_rank * chunk.local_numel
            shard_end = min(shard_begin + chunk.local_numel, dtype_manifest.total_numel)
            _copy_flat_range(
                local,
                dst_offset=0,
                source_begin=shard_begin,
                source_end=shard_end,
                tensor_metas=dtype_manifest.tensors,
                sources=sources,
            )
        else:
            for chunk in dtype_manifest.chunks:
                source_begin = chunk.full_offset + manifest.weight_shard_rank * chunk.local_numel
                source_end = min(
                    source_begin + chunk.local_numel,
                    chunk.full_offset + chunk.valid_numel,
                )
                _copy_flat_range(
                    local,
                    dst_offset=chunk.cpu_offset,
                    source_begin=source_begin,
                    source_end=source_end,
                    tensor_metas=dtype_manifest.tensors,
                    sources=sources,
                )
        packed[dtype_manifest.dtype] = local

    return packed


def _flat_physical(source: torch.Tensor, storage_numel: int) -> torch.Tensor:
    """Flatten *source* in physical storage order.

    Contiguous tensors flatten logically (which is also their physical
    order).  Non-contiguous views — online FP8 stores Cutlass weights as
    transposed views (e.g. stride=(1, K)) — are copied through a physical
    layout so the flat transport buffer round-trips with the original
    stride; flattening them logically and rebuilding with ``.view()`` would
    change the layout and make scaled_mm reject the weight.
    """
    if source.is_contiguous():
        return source.reshape(-1)
    flat = torch.empty(storage_numel, dtype=source.dtype, device=source.device)
    torch.as_strided(flat, size=source.shape, stride=source.stride()).copy_(source)
    return flat


def _copy_flat_range(
    destination: torch.Tensor,
    *,
    dst_offset: int,
    source_begin: int,
    source_end: int,
    tensor_metas: Iterable[TensorMeta],
    sources: dict[str, torch.Tensor],
) -> None:
    if source_end <= source_begin:
        return
    for tensor_meta in tensor_metas:
        tensor_begin = tensor_meta.offset
        tensor_end = tensor_begin + tensor_meta.numel
        overlap_begin = max(source_begin, tensor_begin)
        overlap_end = min(source_end, tensor_end)
        if overlap_begin >= overlap_end:
            continue
        source = _flat_physical(sources[tensor_meta.name], tensor_meta.numel)
        source_offset = overlap_begin - tensor_begin
        count = overlap_end - overlap_begin
        destination_offset = dst_offset + overlap_begin - source_begin
        destination[destination_offset : destination_offset + count].copy_(
            source[source_offset : source_offset + count]
        )


def reconstruct_full_flat(
    local_shards: Sequence[torch.Tensor],
    dtype_manifest: DTypeManifest,
    *,
    layout: WeightLayout = WeightLayout.CHUNK_MAJOR,
) -> torch.Tensor:
    """CPU reference reconstruction for synchronous parity tests."""
    if not local_shards:
        raise ValueError("at least one local shard is required")
    full = torch.empty(dtype_manifest.total_numel, dtype=dtype_manifest.dtype)
    if layout is WeightLayout.WHOLE_BLOCK:
        gathered = torch.cat(local_shards)
        full.copy_(gathered[: dtype_manifest.total_numel])
        return full

    for chunk in dtype_manifest.chunks:
        gathered = torch.cat([local[chunk.cpu_offset : chunk.cpu_offset + chunk.local_numel] for local in local_shards])
        full[chunk.full_offset : chunk.full_offset + chunk.valid_numel].copy_(gathered[: chunk.valid_numel])
    return full


@dataclass(frozen=True)
class TransferTicket:
    request_generation: int
    forward_generation: int
    block_id: int
    part_id: str
    output_slot: int
    chunk_count: int
    ready_event: Any | None = field(default=None, compare=False)
    last_collective_key: tuple[Any, ...] | None = None

    @property
    def owner_key(self) -> tuple[int, int, int, str]:
        return (
            self.request_generation,
            self.forward_generation,
            self.block_id,
            self.part_id,
        )


@dataclass
class OutputSlotState:
    phase: SlotPhase = SlotPhase.REUSABLE
    ticket: TransferTicket | None = None
    last_use_event: Any | None = None
    fallback_reason: str | None = None


class ChunkTransportState:
    """Own generation and conflict checks for persistent output slots."""

    def __init__(self, block_id: int, slot_count: int = 2) -> None:
        self.block_id = block_id
        self._forward_generation = 0
        self._slots = [OutputSlotState() for _ in range(slot_count)]
        self._closed = False

    @property
    def slots(self) -> tuple[OutputSlotState, ...]:
        return tuple(self._slots)

    def begin(
        self,
        *,
        output_slot: int,
        chunk_count: int,
        request_generation: int = 0,
        part_id: str = "block",
        ready_event: Any | None = None,
        last_collective_key: tuple[Any, ...] | None = None,
    ) -> TransferTicket:
        if self._closed:
            raise RuntimeError("chunk transport state is closed")
        slot = self._slots[output_slot]
        if slot.phase is not SlotPhase.REUSABLE:
            current = slot.ticket
            if (
                current is not None
                and current.request_generation == request_generation
                and current.block_id == self.block_id
                and current.part_id == part_id
                and current.chunk_count == chunk_count
                and current.last_collective_key == last_collective_key
            ):
                return current
            raise RuntimeError(
                f"output slot {output_slot} is owned by {current.owner_key if current else slot.phase.value}"
            )

        self._forward_generation += 1
        ticket = TransferTicket(
            request_generation=request_generation,
            forward_generation=self._forward_generation,
            block_id=self.block_id,
            part_id=part_id,
            output_slot=output_slot,
            chunk_count=chunk_count,
            ready_event=ready_event,
            last_collective_key=last_collective_key,
        )
        slot.phase = SlotPhase.SUBMITTED
        slot.ticket = ticket
        return ticket

    def mark_ready(self, ticket: TransferTicket) -> None:
        slot = self._require_current(ticket)
        if slot.phase is not SlotPhase.SUBMITTED:
            raise RuntimeError(f"cannot mark {slot.phase.value} slot ready")
        slot.phase = SlotPhase.READY

    def mark_in_use(self, ticket: TransferTicket) -> None:
        slot = self._require_current(ticket)
        if slot.phase not in (SlotPhase.READY, SlotPhase.IN_USE):
            raise RuntimeError(f"cannot attach consumer to {slot.phase.value} slot")
        slot.phase = SlotPhase.IN_USE

    def release(self, ticket: TransferTicket, last_use_event: Any | None) -> None:
        slot = self._require_current(ticket)
        if slot.phase not in (SlotPhase.READY, SlotPhase.IN_USE):
            raise RuntimeError(f"cannot release {slot.phase.value} slot")
        slot.last_use_event = last_use_event
        slot.ticket = None
        slot.phase = SlotPhase.REUSABLE

    def is_current(self, ticket: TransferTicket) -> bool:
        return self._slots[ticket.output_slot].ticket == ticket

    def reset(self) -> None:
        if any(slot.phase in (SlotPhase.SUBMITTED, SlotPhase.IN_USE) for slot in self._slots):
            raise RuntimeError("cannot reset chunk transport with in-flight slots")
        self._slots = [OutputSlotState() for _ in self._slots]
        self._forward_generation = 0
        self._closed = False

    def close(self) -> None:
        if any(slot.phase in (SlotPhase.SUBMITTED, SlotPhase.IN_USE) for slot in self._slots):
            raise RuntimeError("cannot close chunk transport with in-flight slots")
        self._closed = True

    def _require_current(self, ticket: TransferTicket) -> OutputSlotState:
        slot = self._slots[ticket.output_slot]
        if slot.ticket != ticket:
            raise RuntimeError(f"stale transfer ticket for output slot {ticket.output_slot}")
        return slot


@dataclass
class TransportCounters:
    submissions: int = 0
    submitted_chunks: int = 0
    consumer_attaches: int = 0
    releases: int = 0
    resets: int = 0


class ChunkedWeightTransport:
    """Reference lifecycle contract shared by platform-specific submitters.

    The backend owns stream operations; this class owns prepared Host storage,
    generation tickets, consumer attachment, last-use release, and bounded
    reset/close behavior.
    """

    def __init__(self, block_id: int, slot_count: int = 2) -> None:
        self.state = ChunkTransportState(block_id, slot_count=slot_count)
        self.manifest: PartManifest | None = None
        self.host_shards: dict[torch.dtype, torch.Tensor] = {}
        self.counters = TransportCounters()

    def prepare(
        self,
        manifest: PartManifest,
        host_shards: dict[torch.dtype, torch.Tensor],
    ) -> None:
        if manifest.block_id != self.state.block_id:
            raise ValueError(
                f"manifest block_id={manifest.block_id} does not match transport block_id={self.state.block_id}"
            )
        expected = {dtype_manifest.dtype: dtype_manifest.local_numel for dtype_manifest in manifest.dtypes}
        actual = {dtype: shard.numel() for dtype, shard in host_shards.items()}
        if actual != expected:
            raise ValueError(f"Host shard sizes do not match manifest: expected={expected}, actual={actual}")
        self.manifest = manifest
        self.host_shards = host_shards

    def begin_submission(
        self,
        *,
        output_slot: int,
        request_generation: int,
        ready_event: Any,
        part_id: str = "block",
        last_collective_key: tuple[Any, ...] | None = None,
    ) -> TransferTicket:
        if self.manifest is None:
            raise RuntimeError("chunk transport is not prepared")
        previous = self.state.slots[output_slot].ticket
        ticket = self.state.begin(
            output_slot=output_slot,
            chunk_count=self.manifest.chunk_count,
            request_generation=request_generation,
            part_id=part_id,
            ready_event=ready_event,
            last_collective_key=last_collective_key,
        )
        if ticket is not previous:
            self.counters.submissions += 1
            self.counters.submitted_chunks += ticket.chunk_count
        return ticket

    def mark_ready(self, ticket: TransferTicket) -> None:
        self.state.mark_ready(ticket)

    def attach_ready(
        self,
        ticket: TransferTicket,
        wait_event: Callable[[Any], None],
    ) -> None:
        self.state.mark_in_use(ticket)
        wait_event(ticket.ready_event)
        self.counters.consumer_attaches += 1

    def record_last_use(self, ticket: TransferTicket, last_use_event: Any) -> None:
        self.state.release(ticket, last_use_event)
        self.counters.releases += 1

    def reset(self) -> None:
        self.state.reset()
        self.counters.resets += 1

    def reset_counters(self) -> None:
        """Start a fresh accounting window without changing transport state."""
        self.counters = TransportCounters()

    def close(self) -> None:
        self.state.close()
        self.host_shards = {}
