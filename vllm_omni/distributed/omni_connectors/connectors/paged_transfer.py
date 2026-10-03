# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Internal page views over KV allocations owned by the native workers."""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import msgspec
import torch

Region = tuple[int, int, int]


def coalesce_page_regions(
    local: tuple[Region, ...], remote: tuple[Region, ...], regions_per_layer: int
) -> tuple[list[Region], list[Region]]:
    """Merge paired contiguous pages without crossing a layer's allocation."""
    local_runs: list[Region] = []
    remote_runs: list[Region] = []
    for index, (destination, source) in enumerate(zip(local, remote, strict=True)):
        if index % regions_per_layer and all(
            previous[0] + previous[1] == current[0] and previous[2] == current[2]
            for previous, current in ((local_runs[-1], destination), (remote_runs[-1], source))
        ):
            for runs, region in ((local_runs, destination), (remote_runs, source)):
                address, size, device = runs[-1]
                runs[-1] = (address, size + region[1], device)
        else:
            local_runs.append(destination)
            remote_runs.append(source)
    return local_runs, remote_runs


class PageReadyEvent(Protocol):
    def synchronize(self) -> None: ...


class PageGeometry(msgspec.Struct, frozen=True):
    namespace: str
    layout: str
    layers: tuple[str, ...]
    dtype: str
    device_type: str
    page_axes: tuple[str, ...]
    page_shape: tuple[int, ...]
    page_strides: tuple[int, ...]


class PageOffer(msgspec.Struct, frozen=True):
    schema_version: int
    kind: str
    generation: str
    pool_epoch: str
    geometry: PageGeometry
    num_tokens: int
    num_blocks: int
    expected_readers: int
    regions: tuple[Region, ...]
    agent_metadata: bytes
    sender_host: str
    sender_zmq_port: int
    claim_id: str = ""


@dataclass(eq=False)
class KVPagePool:
    """A borrowed pool; registering it never allocates or packs KV tensors."""

    caches: dict[str, torch.Tensor]
    namespace: str
    layout: str
    epoch: str = field(default_factory=lambda: uuid.uuid4().hex)
    registrations: list[object] = field(default_factory=list)
    exports: set[str] = field(default_factory=set)
    reads: set[str] = field(default_factory=set)
    reservations: dict[str, tuple[int, tuple[int, ...]]] = field(default_factory=dict)
    geometry: PageGeometry = field(init=False)
    block_axis: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.caches or not self.namespace or not self.layout:
            raise ValueError("A KV page pool requires caches, a namespace and a resolved layout")
        tensor = next(iter(self.caches.values()))
        if tensor.ndim == 4:
            # vLLM 0.30 native views: [blocks, heads, tokens, packed content].
            self.block_axis, start = 0, 1
            axes = ("heads", "tokens", "content")
        elif tensor.ndim == 5 and tensor.shape[0] == 2:
            # Legacy backend views keep K/V as separate dense page regions.
            self.block_axis, start = 1, 2
            axes = ("tokens", "heads", "channels")
        else:
            raise ValueError("Page transfer requires native 4D packed or 5D separate K/V cache views")
        shape, strides = tuple(tensor.shape[start:]), tuple(tensor.stride()[start:])
        stride = 1
        for axis in sorted(range(3), key=strides.__getitem__):
            if strides[axis] != stride:
                raise ValueError("Each K/V page must have a dense physical region")
            stride *= shape[axis]
        for cache in self.caches.values():
            if (
                cache.ndim != tensor.ndim
                or cache.shape[:start] != tensor.shape[:start]
                or tuple(cache.shape[start:]) != shape
                or tuple(cache.stride()[start:]) != strides
                or cache.dtype != tensor.dtype
                or cache.device != tensor.device
            ):
                raise ValueError("Page transfer requires one uniform KV cache group")
        self.geometry = PageGeometry(
            self.namespace, self.layout, tuple(self.caches), str(tensor.dtype), tensor.device.type, axes, shape, strides
        )

    @property
    def block_size(self) -> int:
        return self.geometry.page_shape[self.geometry.page_axes.index("tokens")]

    def registration_regions(self) -> list[Region]:
        storage_regions = {
            (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes(), tensor.device.index or 0)
            for tensor in self.caches.values()
        }
        return sorted(storage_regions)

    def regions(self, block_ids: Sequence[int]) -> tuple[Region, ...]:
        num_blocks = next(iter(self.caches.values())).shape[self.block_axis]
        if len(set(block_ids)) != len(block_ids) or any(not 0 <= block < num_blocks for block in block_ids):
            raise ValueError("KV block IDs must be distinct and inside their registered pool")
        regions = []
        size = math.prod(self.geometry.page_shape) * next(iter(self.caches.values())).element_size()
        for tensor in self.caches.values():
            address, device = tensor.data_ptr(), tensor.device.index or 0
            block_stride = tensor.stride(self.block_axis) * tensor.element_size()
            kv_offsets = (0, tensor.stride(0) * tensor.element_size()) if tensor.ndim == 5 else (0,)
            for block in block_ids:
                for kv_offset in kv_offsets:
                    regions.append((address + block * block_stride + kv_offset, size, device))
        return tuple(regions)

    def validate_offer(self, offer: PageOffer, block_ids: Sequence[int]) -> tuple[Region, ...]:
        if offer.schema_version != 1 or offer.kind != "pages" or offer.geometry != self.geometry:
            raise ValueError(
                f"Incompatible NIXL KV page geometry: source={offer.geometry}, destination={self.geometry}"
            )
        if offer.num_blocks != len(block_ids) or not 0 < offer.num_tokens <= len(block_ids) * self.block_size:
            raise ValueError("NIXL KV transfer must fit whole reserved destination blocks")
        regions = self.regions(block_ids)
        if len(regions) != len(offer.regions) or any(
            local[1] != remote[1] for local, remote in zip(regions, offer.regions, strict=True)
        ):
            raise ValueError("NIXL KV page region lengths do not match")
        return regions


@dataclass(frozen=True)
class ReservedKVPages:
    request_id: str
    allocation_generation: int
    pool: KVPagePool
    block_ids: tuple[int, ...]


class PageTransferCapability(Protocol):
    def register_page_pool(self, pool: KVPagePool) -> None: ...

    def export_pages(
        self,
        key: str,
        pool: KVPagePool,
        block_ids: Sequence[int],
        num_tokens: int,
        *,
        expected_readers: int,
        ready: PageReadyEvent,
        generation: str | None = None,
    ) -> PageOffer: ...

    def claim_pages(
        self,
        key: str,
        host: str,
        port: int,
        *,
        generation: str,
        claim_id: str,
        page_claim_ids: tuple[str, ...],
        geometry: PageGeometry,
    ) -> PageOffer | None: ...

    def cancel_page_claim(self, key: str, offer: PageOffer) -> bool: ...

    def read_into(self, key: str, offer: PageOffer, target: ReservedKVPages) -> str: ...

    def poll_page_read(self, read_id: str) -> bool: ...

    def retire_pages(self, target: ReservedKVPages) -> None: ...

    def unregister_page_pool(self, pool: KVPagePool) -> None: ...
