# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunk-major Host packing and AllGather slice metadata."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

import torch


def dtype_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


class WeightLayout(str, Enum):
    CHUNK_MAJOR = "chunk_major"
    WHOLE_BLOCK = "whole_block"


@dataclass(frozen=True)
class TensorMeta:
    name: str
    offset: int
    numel: int
    shape: tuple[int, ...]
    is_buffer: bool = False
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
    def chunk_count(self) -> int:
        return sum(len(item.chunks) for item in self.dtypes)


TensorSpec = tuple[str, torch.Tensor, bool]
PinnedAllocator = Callable[[int, torch.dtype], torch.Tensor]


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return (value + divisor - 1) // divisor


def round_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def is_chunk_transport_supported(tensor: torch.Tensor) -> bool:
    return tensor.ndim > 0


def _full_chunk_numel(
    dtype: torch.dtype, weight_shard_size: int, chunk_size_bytes: int, alignment_bytes: int
) -> tuple[int, int]:
    if min(weight_shard_size, chunk_size_bytes, alignment_bytes) <= 0:
        raise ValueError("weight_shard_size, chunk_size_bytes, alignment_bytes must be positive")
    element_size = dtype_element_size(dtype)
    alignment_numel = ceil_div(alignment_bytes, element_size)
    collective_alignment = weight_shard_size * alignment_numel
    requested_numel = chunk_size_bytes // element_size
    full_chunk_numel = requested_numel - requested_numel % collective_alignment
    if full_chunk_numel == 0:
        raise ValueError(
            f"chunk size is smaller than one aligned collective unit: chunk_size_bytes={chunk_size_bytes}, "
            f"dtype={dtype}, weight_shard_size={weight_shard_size}, alignment_bytes={alignment_bytes}"
        )
    return full_chunk_numel, alignment_numel


def _storage_numel(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return 0
    return 1 + sum((size - 1) * st for size, st in zip(tensor.shape, tensor.stride()))


def _flat_physical(source: torch.Tensor, storage_numel: int) -> torch.Tensor:
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
        tensor_begin, tensor_end = tensor_meta.offset, tensor_meta.offset + tensor_meta.numel
        overlap_begin, overlap_end = max(source_begin, tensor_begin), min(source_end, tensor_end)
        if overlap_begin >= overlap_end:
            continue
        source = _flat_physical(sources[tensor_meta.name], tensor_meta.numel)
        count = overlap_end - overlap_begin
        dst = dst_offset + overlap_begin - source_begin
        destination[dst : dst + count].copy_(
            source[overlap_begin - tensor_begin : overlap_begin - tensor_begin + count]
        )


def pack_local_shard(
    tensor_specs: Sequence[TensorSpec],
    manifest: PartManifest,
    *,
    allocator: PinnedAllocator | None = None,
) -> dict[torch.dtype, torch.Tensor]:
    def _cpu_empty(numel: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(numel, dtype=dtype, device="cpu")

    allocator = allocator or _cpu_empty
    sources = {name: tensor for name, tensor, _ in tensor_specs}
    packed: dict[torch.dtype, torch.Tensor] = {}
    rank = manifest.weight_shard_rank
    for dm in manifest.dtypes:
        local = allocator(dm.local_numel, dm.dtype)
        if local.device.type != "cpu":
            raise ValueError(f"shard allocator returned non-CPU tensor: {local.device}")
        local.zero_()
        if manifest.layout is WeightLayout.WHOLE_BLOCK:
            chunk = dm.chunks[0]
            begin = rank * chunk.local_numel
            _copy_flat_range(
                local,
                dst_offset=0,
                source_begin=begin,
                source_end=min(begin + chunk.local_numel, dm.total_numel),
                tensor_metas=dm.tensors,
                sources=sources,
            )
        else:
            for chunk in dm.chunks:
                begin = chunk.full_offset + rank * chunk.local_numel
                _copy_flat_range(
                    local,
                    dst_offset=chunk.cpu_offset,
                    source_begin=begin,
                    source_end=min(begin + chunk.local_numel, chunk.full_offset + chunk.valid_numel),
                    tensor_metas=dm.tensors,
                    sources=sources,
                )
        packed[dm.dtype] = local
    return packed


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
        if is_chunk_transport_supported(tensor):
            grouped.setdefault(tensor.dtype, []).append((name, tensor, is_buffer))
    dtype_manifests: list[DTypeManifest] = []
    for dtype, dtype_specs in grouped.items():
        offset = 0
        tensor_metas: list[TensorMeta] = []
        for name, tensor, is_buffer in dtype_specs:
            numel = _storage_numel(tensor)
            tensor_metas.append(
                TensorMeta(name, offset, numel, tuple(tensor.shape), is_buffer, stride=tuple(tensor.stride()))
            )
            offset += numel
        total_numel, chunks, cpu_offset = offset, [], 0
        if layout is WeightLayout.CHUNK_MAJOR:
            full_chunk_numel, alignment_numel = _full_chunk_numel(
                dtype, weight_shard_size, chunk_size_bytes, alignment_bytes
            )
            for chunk_id, full_offset in enumerate(range(0, total_numel, full_chunk_numel)):
                valid_numel = min(full_chunk_numel, total_numel - full_offset)
                padded_numel = round_up(valid_numel, weight_shard_size * alignment_numel)
                local_numel = padded_numel // weight_shard_size
                chunks.append(ChunkMeta(chunk_id, cpu_offset, full_offset, valid_numel, padded_numel, local_numel))
                cpu_offset += local_numel
        else:
            alignment_numel = 1
            local_numel = ceil_div(total_numel, weight_shard_size)
            chunks.append(ChunkMeta(0, 0, 0, total_numel, local_numel * weight_shard_size, local_numel))
            cpu_offset = local_numel
        padded_total = chunks[-1].full_offset + chunks[-1].padded_numel if chunks else 0
        dtype_manifests.append(
            DTypeManifest(
                dtype,
                tuple(tensor_metas),
                tuple(chunks),
                total_numel,
                padded_total,
                cpu_offset,
                max((c.local_numel for c in chunks), default=0),
                alignment_numel,
            )
        )
    payload = {
        "block_id": block_id,
        "part_id": part_id,
        "weight_shard_size": weight_shard_size,
        "chunk_size_bytes": chunk_size_bytes,
        "alignment_bytes": alignment_bytes,
        "layout": layout.value,
        "dtypes": [
            {
                "dtype": str(dm.dtype),
                "tensors": [
                    {
                        "name": t.name,
                        "offset": t.offset,
                        "numel": t.numel,
                        "shape": t.shape,
                        "is_buffer": t.is_buffer,
                        "owner": "chunked_fs_offload",
                        "stride": t.stride,
                    }
                    for t in dm.tensors
                ],
                "chunks": [c.__dict__ for c in dm.chunks],
            }
            for dm in dtype_manifests
        ],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return PartManifest(
        block_id,
        part_id,
        weight_shard_size,
        weight_shard_rank,
        chunk_size_bytes,
        alignment_bytes,
        layout,
        tuple(dtype_manifests),
        digest,
    )
