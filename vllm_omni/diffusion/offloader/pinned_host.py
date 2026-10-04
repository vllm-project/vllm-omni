# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Copy current accelerator weights into shared pinned host storage.

The ROCm host allocator rounds each allocation to a power of two. Packing
physical storages avoids retaining that padding for every individual weight
while two model components are offloaded at the same time.
"""

from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn

from .tensor_utils import is_dtensor, set_tensor_storage

_ALIGNMENT = 256
_SLAB_BYTES = 4 * 1024 * 1024 * 1024


def plan_pinned_slabs(sizes: list[int], *, slab_bytes: int = _SLAB_BYTES) -> tuple[list[int], list[tuple[int, int]]]:
    """Return power-of-two allocation sizes and aligned storage placements.

    Large storages get their own larger slab. If packing would increase the
    allocator's total backing, keep separate allocations instead.
    """
    if slab_bytes < _ALIGNMENT or slab_bytes & (slab_bytes - 1):
        raise ValueError("Slab capacity must be a power of two at least 256 bytes")
    if any(size <= 0 for size in sizes):
        raise ValueError("Only nonempty physical storage spans can be packed")
    used: list[int] = []
    capacities: list[int] = []
    placements = [(-1, -1)] * len(sizes)
    for index in sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True):
        aligned = (sizes[index] + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
        candidates = [i for i, capacity in enumerate(capacities) if used[i] + aligned <= capacity]
        if candidates:
            slab = min(candidates, key=lambda i: capacities[i] - used[i] - aligned)
        else:
            slab = len(used)
            used.append(0)
            capacities.append(max(slab_bytes, 1 << (aligned - 1).bit_length()))
        placements[index] = (slab, used[slab])
        used[slab] += aligned
    allocations = [1 << (size - 1).bit_length() for size in used]
    separate = [1 << (size - 1).bit_length() for size in sizes]
    if sum(allocations) > sum(separate):
        return separate, [(index, 0) for index in range(len(sizes))]
    return allocations, placements


@dataclass(frozen=True)
class _Binding:
    target: torch.Tensor
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int


def release_cached_pinned_memory() -> None:
    """Release inactive host slabs after movement, where Torch exposes the API.

    Live allocations and copies protected by allocator events stay retained.
    GPU empty_cache only releases device allocations, not these host slabs.
    """
    release = getattr(torch._C, "_host_emptyCache", None)
    if callable(release):
        release()


def move_module_to_pinned_cpu(module: nn.Module, *, non_blocking: bool) -> bool | None:
    """Move dense local CUDA/HIP tensors; return None for the ordinary path.

    Physical storage grouping preserves aliases, offsets, dtypes and strides.
    Copy current GPU values every time; no immutable CPU master is retained.
    Rebind only after all D2H copies complete, including on transfer failure.
    """
    targets: list[torch.Tensor] = []
    seen: set[int] = set()
    for tensor in chain(module.parameters(), module.buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        if (
            is_dtensor(tensor)
            or tensor.layout != torch.strided
            or tensor.is_meta
            or tensor.is_conj()
            or tensor.is_neg()
        ):
            return None
        if tensor.device.type == "cpu":
            continue
        if tensor.device.type != "cuda" or tensor.untyped_storage().nbytes() == 0:
            return None
        targets.append(tensor)
    if not targets:
        return False

    grouped: dict[tuple[int | None, int, int], tuple[torch.Tensor, list[_Binding]]] = {}
    for target in targets:
        source = target.detach()
        storage = source.untyped_storage()
        key = (source.device.index, storage.data_ptr(), storage.nbytes())
        if key not in grouped:
            grouped[key] = (source, [])
        grouped[key][1].append(
            _Binding(target, source.dtype, tuple(source.shape), tuple(source.stride()), source.storage_offset())
        )
    groups = list(grouped.values())
    sizes = [source.untyped_storage().nbytes() for source, _ in groups]
    allocations, placements = plan_pinned_slabs(sizes)
    slabs = [torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True) for size in allocations]
    views: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        for (source, bindings), size, (slab, offset) in zip(groups, sizes, placements, strict=True):
            raw = torch.empty(0, dtype=torch.uint8, device=source.device).set_(
                source.untyped_storage(), 0, (size,), (1,)
            )
            slabs[slab].narrow(0, offset, size).copy_(raw, non_blocking=non_blocking)
            for binding in bindings:
                element_size = torch.empty(0, dtype=binding.dtype).element_size()
                view = torch.empty(0, dtype=binding.dtype, device="cpu").set_(
                    slabs[slab].untyped_storage(),
                    offset // element_size + binding.offset,
                    binding.shape,
                    binding.stride,
                )
                views.append((binding.target, view))
    finally:
        # Keep source and destination owners alive until outstanding copies
        # complete, even if one copy or view construction raises.
        synchronization_error: BaseException | None = None
        for device_index in {source.device.index for source, _ in groups}:
            try:
                torch.cuda.current_stream(device_index).synchronize()
            except BaseException as exc:
                synchronization_error = synchronization_error or exc
        if synchronization_error is not None:
            raise synchronization_error

    originals = [target.detach() for target, _ in views]
    committed = 0
    try:
        for target, view in views:
            set_tensor_storage(target, view)
            committed += 1
    except BaseException:
        for (target, _), original in zip(views[:committed], originals[:committed], strict=True):
            set_tensor_storage(target, original)
        raise
    return True
