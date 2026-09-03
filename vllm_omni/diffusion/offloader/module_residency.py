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

from .tensor_utils import is_dtensor, set_tensor_storage

logger = init_logger(__name__)


# Margin over the largest cached-but-unallocated value observed at a component
# boundary. The observed peak already includes the weight-reuse payload and
# the transient footprint of the running workload, so this only covers
# check-to-check variation.
_OBSERVED_PEAK_MARGIN = 1.25
# Hard ceiling for the learned budget so a spurious observation can never
# monopolize a small device; the #6526 L20X retention plateau (~23.8 GB on a
# ~140 GB card) stayed well below this fraction.
_MAX_BUDGET_CAPACITY_FRACTION = 0.25


class BoundedAllocatorCache:
    """Retain reusable allocator blocks without monopolizing device memory.

    Component offload normally calls ``empty_cache`` after every stage. That
    makes the next stage return to the device allocator even though PyTorch's
    cached blocks are immediately reusable. This policy keeps the cache while
    both of these bounds hold:

    * cached-but-unallocated memory stays within the learned budget — the
      largest cached value observed at any component boundary times a margin,
      capped at a quarter of device capacity; and
    * at least 5% of device capacity is physically free.

    The budget needs no user-chosen fraction, no multiplier constant, and no
    warmup flag. It starts at zero, so the first boundaries of a cold start
    release once per observation step while the peak climbs; afterwards the
    budget tracks the observed peak and boundaries retain steadily. A
    workload change whose cached footprint grows by more than the margin pays
    one further release while the new peak lands. Detaching the last attached
    ``PinnedModuleStager`` resets the observed peak, so one workload never
    permanently influences another. Missing memory telemetry is handled
    conservatively by releasing the cache. Failure paths can force release;
    normal executor shutdown keeps its own unconditional device-cache cleanup
    because this policy is not global.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        min_free_fraction: float = 0.05,
    ) -> None:
        if not 0.0 <= min_free_fraction <= 1.0:
            raise ValueError(f"min_free_fraction must be in [0, 1], got {min_free_fraction}")
        self.device = device
        self.min_free_fraction = min_free_fraction
        self._staged_bytes = 0
        self._observed_cached_peak = 0

    def grow_retention_budget(self, staged_bytes: int) -> None:
        """Record an attached stager's staged byte total."""
        if staged_bytes < 0:
            raise ValueError(f"staged_bytes must be >= 0, got {staged_bytes}")
        self._staged_bytes += int(staged_bytes)

    def shrink_retention_budget(self, staged_bytes: int) -> None:
        """Record a stager detach; the last detach resets the observed peak."""
        if staged_bytes < 0:
            raise ValueError(f"staged_bytes must be >= 0, got {staged_bytes}")
        self._staged_bytes = max(0, self._staged_bytes - int(staged_bytes))
        if self._staged_bytes == 0:
            self._observed_cached_peak = 0

    @property
    def observed_cached_peak(self) -> int:
        """Largest cached-but-unallocated value seen at a boundary check."""
        return self._observed_cached_peak

    @property
    def budget_bytes(self) -> int:
        """Reusable-cache ceiling: the observed peak times its margin."""
        return int(self._observed_cached_peak * _OBSERVED_PEAK_MARGIN)

    def _should_release(self) -> bool:
        reserved = int(torch.accelerator.memory_reserved(self.device))
        allocated = int(torch.accelerator.memory_allocated(self.device))
        free, total = current_omni_platform.get_device_memory(self.device)
        cached = max(0, reserved - allocated)
        capacity_cap = int(total * _MAX_BUDGET_CAPACITY_FRACTION)
        budget = min(self.budget_bytes, capacity_cap)
        # Decide against the pre-observation budget, then record the peak so a
        # one-off over-budget release also teaches the budget for later checks.
        release = cached > budget or free < int(total * self.min_free_fraction)
        if cached > self._observed_cached_peak:
            self._observed_cached_peak = cached
        if release:
            reason = "cached-over-budget" if cached > budget else "low-free-memory"
            logger.debug(
                "Releasing retained allocator cache (%s): cached=%d budget=%d staged=%d "
                "observed_peak=%d capacity_cap=%d free=%d total=%d",
                reason,
                cached,
                budget,
                self._staged_bytes,
                self._observed_cached_peak,
                capacity_cap,
                free,
                total,
            )
            return True
        logger.debug(
            "Retaining allocator cache: cached=%d budget=%d staged=%d observed_peak=%d "
            "capacity_cap=%d free=%d total=%d",
            cached,
            budget,
            self._staged_bytes,
            self._observed_cached_peak,
            capacity_cap,
            free,
            total,
        )
        return False

    def release_if_needed(self, *, force: bool = False) -> bool:
        """Release cached blocks when a bound is crossed or release is forced."""
        if not force:
            try:
                if not self._should_release():
                    return False
            except Exception as exc:
                # Preserve the pre-retention behavior on platforms that do not
                # expose allocator telemetry through torch.accelerator.
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
class _StorageGroup:
    master: torch.Tensor
    bindings: list[_TensorBinding]


class PinnedModuleStager:
    """Stage immutable module groups without copying device weights to CPU.

    ``nn.Module.to("cpu")`` performs a device-to-host copy for every parameter
    after each forward. In inference the weights are immutable, so retain one
    pinned CPU master instead. ``load`` materializes device storage from that
    master; ``offload`` only rebinds Parameters and buffers to the master.

    A module iterable is treated as one staging group. It uses one copy stream
    and one reusable completion event. Tensors sharing storage keep their
    shapes, strides, offsets, dtypes, and aliases across every transition.
    """

    def __init__(
        self,
        module: nn.Module | Iterable[nn.Module],
        device: torch.device,
        *,
        pin_memory: bool = True,
        copy_stream: Any | None = None,
        cache_retention: BoundedAllocatorCache | None = None,
    ) -> None:
        modules = (module,) if isinstance(module, nn.Module) else tuple(module)
        if not modules or not all(isinstance(item, nn.Module) for item in modules):
            raise ValueError("PinnedModuleStager requires at least one nn.Module")

        self.device = device
        self.copy_stream = copy_stream if copy_stream is not None else current_omni_platform.Stream()
        self._ready_event = current_omni_platform.Event()
        self.cache_retention = None
        self.loaded = False
        self._groups = self._snapshot_groups(modules, pin_memory=pin_memory)
        self._device_storages: list[torch.Tensor] = []
        self._restore_masters()
        self.set_cache_retention(cache_retention)

    @staticmethod
    def _local_tensor(target: torch.Tensor) -> torch.Tensor:
        return target.to_local() if is_dtensor(target) else target

    @classmethod
    def _snapshot_groups(
        cls,
        modules: tuple[nn.Module, ...],
        *,
        pin_memory: bool,
    ) -> list[_StorageGroup]:
        targets: list[torch.Tensor] = []
        seen_targets: set[int] = set()
        for target in chain.from_iterable(chain(item.parameters(), item.buffers()) for item in modules):
            if id(target) not in seen_targets:
                seen_targets.add(id(target))
                targets.append(target)

        grouped: dict[tuple[Any, ...], tuple[torch.Tensor, list[_TensorBinding]]] = {}
        for target in targets:
            local = cls._local_tensor(target)
            if local.is_meta:
                raise ValueError("PinnedModuleStager cannot snapshot a meta tensor")
            storage = local.untyped_storage()
            if storage.nbytes() == 0:
                storage_key: tuple[Any, ...] = ("empty", id(target))
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
                grouped[storage_key] = (local, [])
            grouped[storage_key][1].append(binding)

        groups: list[_StorageGroup] = []
        for source, bindings in grouped.values():
            storage = source.untyped_storage()
            storage_view = torch.empty(0, dtype=torch.uint8, device=source.device).set_(
                storage,
                0,
                (storage.nbytes(),),
                (1,),
            )
            master = storage_view.detach() if storage_view.device.type == "cpu" else storage_view.to("cpu")
            if pin_memory and not master.is_pinned():
                master = master.pin_memory()
            groups.append(_StorageGroup(master=master, bindings=bindings))
        return groups

    @staticmethod
    def _view(backing: torch.Tensor, binding: _TensorBinding) -> torch.Tensor:
        return torch.empty(0, dtype=binding.dtype, device=backing.device).set_(
            backing.untyped_storage(),
            binding.storage_offset,
            binding.shape,
            binding.stride,
        )

    def _bind(self, storages: list[torch.Tensor]) -> None:
        for storage, group in zip(storages, self._groups):
            for binding in group.bindings:
                set_tensor_storage(binding.target, self._view(storage, binding))

    def _restore_masters(self) -> None:
        self._bind([group.master for group in self._groups])

    @property
    def staged_bytes(self) -> int:
        """Total unique staged storage bytes (masters are uint8 byte views)."""
        return sum(group.master.numel() for group in self._groups)

    def set_cache_retention(self, cache_retention: BoundedAllocatorCache | None) -> None:
        if cache_retention is self.cache_retention:
            return
        if self.cache_retention is not None:
            self.cache_retention.shrink_retention_budget(self.staged_bytes)
        self.cache_retention = cache_retention
        if cache_retention is not None:
            cache_retention.grow_retention_budget(self.staged_bytes)

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
        try:
            self._load_once()
        except torch.OutOfMemoryError:
            # A retained cache is normally reusable by this process, but an
            # explicit flush gives external memory pressure one bounded retry.
            logger.info(
                "Staging out of memory (%d staged bytes); releasing retained cache and retrying once",
                self.staged_bytes,
            )
            self._cleanup_failed_load()
            try:
                self._load_once()
            except BaseException:
                self._cleanup_failed_load()
                raise
        except BaseException:
            self._cleanup_failed_load()
            raise

    def offload(self) -> None:
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
        self._device_storages.clear()
        self.loaded = False
        self._release_cache()


__all__ = ["BoundedAllocatorCache", "PinnedModuleStager"]
