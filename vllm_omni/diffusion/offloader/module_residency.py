# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""On-demand module staging backed by immutable pinned CPU storage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

from .tensor_utils import dtype_size, is_dtensor, set_tensor_storage

logger = init_logger(__name__)

_PINNED_ALIGNMENT_BYTES = 256
_PINNED_SLAB_BYTES = 4 * 1024 * 1024 * 1024


class BoundedAllocatorCache:
    """Retain reusable allocator blocks without monopolizing device memory.

    Component offload normally calls ``empty_cache`` after every stage. That
    makes the next stage return to the device allocator even though PyTorch's
    cached blocks are immediately reusable. This policy keeps the cache while
    both of these bounds hold:

    * cached-but-unallocated memory is at most 25% of device capacity; and
    * at least 5% of device capacity is physically free.

    Missing memory telemetry is handled conservatively by releasing the cache.
    Failure paths can force release; normal executor shutdown keeps its own
    unconditional device-cache cleanup because this policy is not global.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        max_cached_fraction: float = 0.25,
        min_free_fraction: float = 0.05,
    ) -> None:
        if not 0.0 <= max_cached_fraction <= 1.0:
            raise ValueError(f"max_cached_fraction must be in [0, 1], got {max_cached_fraction}")
        if not 0.0 <= min_free_fraction <= 1.0:
            raise ValueError(f"min_free_fraction must be in [0, 1], got {min_free_fraction}")
        self.device = device
        self.max_cached_fraction = max_cached_fraction
        self.min_free_fraction = min_free_fraction

    def _should_release(self) -> bool:
        if self.device.type == "npu":
            # The NPU allocator does not implement the generic DeviceAllocator interface.
            device_module = torch.get_device_module(self.device)
            reserved = int(device_module.memory_reserved(self.device))
            allocated = int(device_module.memory_allocated(self.device))
        else:
            reserved = int(torch.accelerator.memory_reserved(self.device))
            allocated = int(torch.accelerator.memory_allocated(self.device))
        free, total = current_omni_platform.get_device_memory(self.device)
        cached = max(0, reserved - allocated)
        return cached > int(total * self.max_cached_fraction) or free < int(total * self.min_free_fraction)

    def release_if_needed(self, *, force: bool = False) -> bool:
        """Release cached blocks when a bound is crossed or release is forced."""
        if not force:
            try:
                if not self._should_release():
                    return False
            except Exception as exc:
                # Preserve the pre-retention behavior on platforms that do not
                # expose allocator telemetry.
                logger.debug("Allocator cache telemetry unavailable; releasing cache: %s", exc)
        current_omni_platform.empty_cache()
        return True


@dataclass(frozen=True)
class _TensorBinding:
    target: torch.Tensor
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


@dataclass
class _SourceStorageGroup:
    source: torch.Tensor
    bindings: list[_TensorBinding]


@dataclass
class _StorageGroup:
    master: torch.Tensor
    bindings: list[_TensorBinding]


@dataclass(frozen=True)
class _PinnedSlabPlan:
    allocations: tuple[int, ...]
    placements: tuple[tuple[int, int], ...]
    separate_backing_bytes: int

    @property
    def packed_backing_bytes(self) -> int:
        return sum(self.allocations)


class _PinnedSlabAllocationError(RuntimeError):
    """A packed pinned allocation failed before any module storage changed."""


def _power_of_two_backing(size: int) -> int:
    if size <= 0:
        raise ValueError("Pinned storage sizes must be positive")
    return 1 << (size - 1).bit_length()


def _plan_pinned_slabs(
    sizes: Iterable[int],
    *,
    slab_bytes: int = _PINNED_SLAB_BYTES,
) -> _PinnedSlabPlan:
    """Place physical storages into aligned slabs without increasing backing.

    ROCm's pinned-host allocator rounds allocations to power-of-two backing.
    Packing physical storages into a small number of slabs avoids paying that
    rounding cost for every parameter while retaining a fail-closed fallback
    to one allocation per storage whenever packing would be worse.
    """
    storage_sizes = tuple(sizes)
    if slab_bytes < _PINNED_ALIGNMENT_BYTES or slab_bytes & (slab_bytes - 1):
        raise ValueError("Pinned slab capacity must be a power of two at least 256 bytes")
    if any(size <= 0 for size in storage_sizes):
        raise ValueError("Pinned storage sizes must be positive")
    if not storage_sizes:
        return _PinnedSlabPlan((), (), 0)

    used: list[int] = []
    capacities: list[int] = []
    placements = [(-1, -1)] * len(storage_sizes)
    for index in sorted(range(len(storage_sizes)), key=storage_sizes.__getitem__, reverse=True):
        aligned = (
            (storage_sizes[index] + _PINNED_ALIGNMENT_BYTES - 1) // _PINNED_ALIGNMENT_BYTES * _PINNED_ALIGNMENT_BYTES
        )
        candidates = [
            slab_index for slab_index, capacity in enumerate(capacities) if used[slab_index] + aligned <= capacity
        ]
        if candidates:
            slab_index = min(
                candidates,
                key=lambda candidate: capacities[candidate] - used[candidate] - aligned,
            )
        else:
            slab_index = len(used)
            used.append(0)
            capacities.append(max(slab_bytes, _power_of_two_backing(aligned)))
        placements[index] = (slab_index, used[slab_index])
        used[slab_index] += aligned

    allocations = tuple(_power_of_two_backing(size) for size in used)
    separate = tuple(_power_of_two_backing(size) for size in storage_sizes)
    separate_backing_bytes = sum(separate)
    if sum(allocations) > separate_backing_bytes:
        return _PinnedSlabPlan(
            allocations=separate,
            placements=tuple((index, 0) for index in range(len(storage_sizes))),
            separate_backing_bytes=separate_backing_bytes,
        )
    return _PinnedSlabPlan(
        allocations=allocations,
        placements=tuple(placements),
        separate_backing_bytes=separate_backing_bytes,
    )


def _unique_module_tensors(modules: tuple[nn.Module, ...]) -> list[torch.Tensor]:
    targets: list[torch.Tensor] = []
    seen_targets: set[int] = set()
    for target in chain.from_iterable(chain(module.parameters(), module.buffers()) for module in modules):
        if id(target) not in seen_targets:
            seen_targets.add(id(target))
            targets.append(target)
    return targets


def _local_tensor(target: torch.Tensor) -> torch.Tensor:
    return target.to_local() if is_dtensor(target) else target


def _collect_source_groups(modules: tuple[nn.Module, ...]) -> list[_SourceStorageGroup]:
    grouped: dict[tuple[object, ...], _SourceStorageGroup] = {}
    for target in _unique_module_tensors(modules):
        local = _local_tensor(target)
        if local.is_meta:
            raise ValueError("PinnedModuleStager cannot snapshot a meta tensor")
        storage = local.untyped_storage()
        if storage.nbytes() == 0:
            storage_key: tuple[object, ...] = ("empty", id(target))
        else:
            storage_key = (
                local.device.type,
                local.device.index,
                storage.data_ptr(),
                storage.nbytes(),
            )
        binding = _TensorBinding(
            target=target,
            dtype=local.dtype,
            shape=tuple(local.shape),
            stride=tuple(local.stride()),
            storage_offset=local.storage_offset(),
        )
        if storage_key not in grouped:
            grouped[storage_key] = _SourceStorageGroup(source=local, bindings=[])
        grouped[storage_key].bindings.append(binding)
    return list(grouped.values())


def _physical_storage_view(source: torch.Tensor) -> torch.Tensor:
    storage = source.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=source.device).set_(
        storage,
        0,
        (storage.nbytes(),),
        (1,),
    )


def _synchronize_source_devices(source_devices: Iterable[torch.device]) -> None:
    devices = sorted(
        source_devices,
        key=lambda device: (device.type, device.index if device.index is not None else -1),
    )
    if not devices:
        return

    previous_device = current_omni_platform.current_device()
    try:
        for source_device in devices:
            current_omni_platform.set_device(source_device)
            current_omni_platform.current_stream().synchronize()
    finally:
        current_omni_platform.set_device(torch.device(devices[0].type, previous_device))


def _copy_groups_to_cpu(
    source_groups: list[_SourceStorageGroup],
    *,
    pin_memory: bool,
    pack_pinned: bool,
    non_blocking: bool,
) -> list[_StorageGroup]:
    if pack_pinned and any(group.source.device.type not in {"cpu", "cuda"} for group in source_groups):
        pack_pinned = False
    if not pin_memory or not pack_pinned:
        groups: list[_StorageGroup] = []
        for group in source_groups:
            storage_view = _physical_storage_view(group.source)
            master = storage_view.detach() if storage_view.device.type == "cpu" else storage_view.to("cpu")
            if pin_memory and not master.is_pinned():
                master = master.pin_memory()
            groups.append(_StorageGroup(master=master, bindings=group.bindings))
        return groups

    nonempty = [(index, group) for index, group in enumerate(source_groups) if group.source.untyped_storage().nbytes()]
    plan = _plan_pinned_slabs(group.source.untyped_storage().nbytes() for _, group in nonempty)
    try:
        slabs = [torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True) for size in plan.allocations]
    except (RuntimeError, MemoryError) as exc:
        raise _PinnedSlabAllocationError("Packed pinned-host slab allocation failed") from exc
    masters: list[torch.Tensor | None] = [None] * len(source_groups)
    source_devices = {group.source.device for _, group in nonempty if group.source.device.type != "cpu"}
    needs_sync = non_blocking and bool(source_devices)
    try:
        for (group_index, group), (slab_index, offset) in zip(
            nonempty,
            plan.placements,
            strict=True,
        ):
            size = group.source.untyped_storage().nbytes()
            master = slabs[slab_index].narrow(0, offset, size)
            master.copy_(
                _physical_storage_view(group.source),
                non_blocking=non_blocking,
            )
            masters[group_index] = master
    finally:
        # Keep every source and slab alive until all queued copies finish,
        # including when allocation, copying, or view construction fails.
        if needs_sync:
            _synchronize_source_devices(source_devices)

    for index, master in enumerate(masters):
        if master is None:
            masters[index] = torch.empty(0, dtype=torch.uint8, device="cpu", pin_memory=True)

    if plan.packed_backing_bytes < plan.separate_backing_bytes:
        logger.debug(
            "Packed %d physical storages into %d pinned slabs: %d bytes instead of %d bytes of allocator backing",
            len(nonempty),
            len(plan.allocations),
            plan.packed_backing_bytes,
            plan.separate_backing_bytes,
        )
    return [
        _StorageGroup(master=master, bindings=group.bindings)
        for master, group in zip(masters, source_groups, strict=True)
        if master is not None
    ]


def _view_from_backing(backing: torch.Tensor, binding: _TensorBinding) -> torch.Tensor:
    element_size = dtype_size(binding.dtype)
    byte_offset = backing.storage_offset() * dtype_size(backing.dtype) + binding.storage_offset * element_size
    if byte_offset % element_size:
        raise ValueError("Pinned backing offset is not aligned for the tensor dtype")
    return torch.empty(0, dtype=binding.dtype, device=backing.device).set_(
        backing.untyped_storage(),
        byte_offset // element_size,
        binding.shape,
        binding.stride,
    )


def _set_storage_preserving_inference(target: torch.Tensor, value: torch.Tensor) -> None:
    with torch.inference_mode(_local_tensor(target).is_inference()):
        set_tensor_storage(target, value)


def _view_preserving_inference(backing: torch.Tensor, binding: _TensorBinding) -> torch.Tensor:
    # Creating a view while inference mode is enabled makes that view an
    # inference tensor. Build it under the target's mode so rebinding a normal
    # Parameter keeps its version counter available for cache invalidation.
    with torch.inference_mode(_local_tensor(binding.target).is_inference()):
        return _view_from_backing(backing, binding)


def _move_module_to_packed_pinned_cpu(
    module: nn.Module,
    *,
    non_blocking: bool,
) -> bool | None:
    """Move one dense local ROCm module into packed pinned host backing.

    Returns ``None`` when the ordinary tensor path must handle an unsupported
    layout or mixed residency. The module is not mutated until every physical
    storage has been copied and all destination views have been constructed.
    """
    targets = _unique_module_tensors((module,))
    if not targets:
        return False

    accelerator_targets = 0
    device_indices: set[int | None] = set()
    for target in targets:
        if (
            is_dtensor(target)
            or target.layout != torch.strided
            or target.is_meta
            or target.is_conj()
            or target.is_neg()
            or target.untyped_storage().nbytes() == 0
        ):
            return None
        if target.device.type == "cpu":
            continue
        if target.device.type != "cuda":
            return None
        accelerator_targets += 1
        device_indices.add(target.device.index)
    if accelerator_targets == 0:
        return False
    if accelerator_targets != len(targets) or len(device_indices) != 1:
        return None

    source_groups = _collect_source_groups((module,))
    try:
        groups = _copy_groups_to_cpu(
            source_groups,
            pin_memory=True,
            pack_pinned=True,
            non_blocking=non_blocking,
        )
    except _PinnedSlabAllocationError:
        logger.debug("Packed pinned-host allocation failed; using ordinary tensor movement", exc_info=True)
        return None
    replacements = [
        (binding.target, _view_preserving_inference(group.master, binding))
        for group in groups
        for binding in group.bindings
    ]
    originals = [target.data for target, _ in replacements]
    committed = 0
    try:
        for target, replacement in replacements:
            _set_storage_preserving_inference(target, replacement)
            committed += 1
    except BaseException:
        rollback_error: BaseException | None = None
        for (target, _), original in zip(
            replacements[:committed],
            originals[:committed],
            strict=True,
        ):
            try:
                _set_storage_preserving_inference(target, original)
            except BaseException as exc:
                rollback_error = rollback_error or exc
        if rollback_error is not None:
            raise RuntimeError("Packed pinned-host rebinding failed and rollback was incomplete") from rollback_error
        raise
    return True


class PinnedModuleStager:
    """Stage immutable module groups without copying device weights to CPU.

    ``nn.Module.to("cpu")`` performs a device-to-host copy for every parameter
    after each forward. In inference the weights are immutable, so retain one
    pinned CPU master instead. ``load`` materializes device storage from that
    master; ``offload`` only rebinds Parameters and buffers to the master.

    A module iterable is treated as one staging group. It uses one copy stream
    and one reusable completion event. Tensors sharing storage keep their
    shapes, strides, offsets, dtypes, and aliases across every transition.

    With ``retain_device_storage`` the device storages survive ``offload``:
    ``load`` then only rebinds to the same fixed storages (weights are
    immutable, so no H2D copy repeats). The stable storage addresses keep
    CUDA-graph-captured weight pointers valid across offload/load swaps, at
    the cost of keeping the device storage resident while offloaded.
    """

    def __init__(
        self,
        module: nn.Module | Iterable[nn.Module],
        device: torch.device,
        *,
        pin_memory: bool = True,
        copy_stream: Any | None = None,
        cache_retention: BoundedAllocatorCache | None = None,
        retain_device_storage: bool = False,
    ) -> None:
        modules = (module,) if isinstance(module, nn.Module) else tuple(module)
        if not modules or not all(isinstance(item, nn.Module) for item in modules):
            raise ValueError("PinnedModuleStager requires at least one nn.Module")

        self.device = device
        self.copy_stream = copy_stream if copy_stream is not None else current_omni_platform.Stream()
        self._ready_event = current_omni_platform.Event()
        self.cache_retention = cache_retention
        self.retain_device_storage = retain_device_storage
        self.loaded = False
        self._groups = self._snapshot_groups(
            modules,
            pin_memory=pin_memory,
            pack_pinned=pin_memory and current_omni_platform.is_rocm(),
        )
        self._device_storages: list[torch.Tensor] = []
        self._restore_masters()

    @staticmethod
    def _snapshot_groups(
        modules: tuple[nn.Module, ...],
        *,
        pin_memory: bool,
        pack_pinned: bool = False,
    ) -> list[_StorageGroup]:
        source_groups = _collect_source_groups(modules)
        try:
            return _copy_groups_to_cpu(
                source_groups,
                pin_memory=pin_memory,
                pack_pinned=pack_pinned,
                non_blocking=pack_pinned,
            )
        except _PinnedSlabAllocationError:
            logger.debug("Packed pinned-host allocation failed; using separate pinned masters", exc_info=True)
            return _copy_groups_to_cpu(
                source_groups,
                pin_memory=pin_memory,
                pack_pinned=False,
                non_blocking=False,
            )

    @staticmethod
    def _view(backing: torch.Tensor, binding: _TensorBinding) -> torch.Tensor:
        return _view_from_backing(backing, binding)

    def _bind(self, storages: list[torch.Tensor]) -> None:
        # Staging commonly runs inside inference_mode. Assigning an inference
        # tensor to Parameter.data would turn the parameter into an inference
        # tensor too, silently disabling version increments on later edits.
        # Preserve each target's original semantics. An inference parameter
        # cannot acquire a version counter by rebinding its storage either.
        for storage, group in zip(storages, self._groups):
            for binding in group.bindings:
                _set_storage_preserving_inference(
                    binding.target,
                    _view_preserving_inference(storage, binding),
                )

    def _restore_masters(self) -> None:
        self._bind([group.master for group in self._groups])

    def set_cache_retention(self, cache_retention: BoundedAllocatorCache | None) -> None:
        self.cache_retention = cache_retention

    def _release_cache(self, *, force: bool = False) -> None:
        if self.cache_retention is None:
            current_omni_platform.empty_cache()
        else:
            self.cache_retention.release_if_needed(force=force)

    def _load_once(self) -> None:
        device_storages = [torch.empty_like(group.master, device=self.device) for group in self._groups]
        compute_stream = current_omni_platform.current_stream()
        self.copy_stream.wait_stream(compute_stream)
        with current_omni_platform.stream(self.copy_stream):
            for device_storage, group in zip(device_storages, self._groups):
                device_storage.copy_(group.master, non_blocking=group.master.is_pinned())
            self._ready_event.record(self.copy_stream)

        self._bind(device_storages)
        compute_stream.wait_event(self._ready_event)
        self._device_storages = device_storages
        self.loaded = True

    def _cleanup_failed_load(self) -> None:
        try:
            current_omni_platform.synchronize()
        except Exception:
            logger.debug("Device synchronization failed while cleaning up module staging", exc_info=True)
        self._restore_masters()
        self._device_storages.clear()
        self.loaded = False
        self._release_cache(force=True)

    def load(self) -> None:
        if self.loaded:
            return
        if self._device_storages:
            # Storage retained from a previous load still holds the immutable
            # weights; rebinding suffices, no H2D copy repeats.
            self._bind(self._device_storages)
            self.loaded = True
            return
        try:
            self._load_once()
        except torch.OutOfMemoryError:
            # A retained cache is normally reusable by this process, but an
            # explicit flush gives external memory pressure one bounded retry.
            self._cleanup_failed_load()
            try:
                self._load_once()
            except BaseException:
                self._cleanup_failed_load()
                raise
        except BaseException:
            self._cleanup_failed_load()
            raise

    def offload(self, *, release_cache: bool = True) -> None:
        """Restore CPU masters, optionally skipping unconditional cache purges.

        An explicit bounded cache policy still checks its limits when release
        is disabled. Offload failures always force cache release.
        """
        if not self.loaded:
            return

        # The module has completed on the compute stream. Synchronize once at
        # the stage boundary, then discard device storage without any D2H copy.
        try:
            current_omni_platform.synchronize()
            self._restore_masters()
        except BaseException:
            self._device_storages.clear()
            self.loaded = False
            self._release_cache(force=True)
            raise
        if not self.retain_device_storage:
            self._device_storages.clear()
        self.loaded = False
        if release_cache or self.cache_retention is not None:
            self._release_cache()


__all__ = ["BoundedAllocatorCache", "PinnedModuleStager"]
