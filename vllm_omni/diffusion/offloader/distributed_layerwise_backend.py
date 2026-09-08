# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Distributed Layerwise Offload backend with double-buffered H2D.

This module implements the RFC-1 "Distributed Layerwise Offload" mechanism that:

* Optionally shards model weights across DP ranks and reconstructs each block
  with AllGather, or streams a complete rank-local block without a collective.
* Can retain compatible checkpoint tensors as node-shared mmap sources instead
  of creating a persistent private host copy in every rank.
* Uses a fixed double-buffer scheme that keeps only two layers' worth of
  weights on each device at any time.
* Pipelines H2D transfers and, when enabled, AllGather communications on
  dedicated streams, overlapping them with computation.
* Is hardware-agnostic, supporting both NVIDIA GPU (CUDA) and Ascend NPU
  (CANN) platforms via vLLM-Omni's platform abstraction layer.
"""

from __future__ import annotations

import threading
import time
import weakref
from itertools import chain
from typing import Any

import torch
import torch.distributed
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.diffusion.model_loader.host_weight_plan import (
    HostWeightPlan,
)
from vllm_omni.host_weight_runtime import HostWeightLease
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig, run_cleanup_steps
from .component_utils import (
    clear_encoder_layerwise_state,
    get_encoder_block_groups,
    iter_streamable_dits,
    move_non_block_state_to_device,
    prepare_component,
    prepare_pipeline_components,
    set_encoder_layerwise_state,
)
from .config import DIT_COMPONENT, TEXT_ENCODER_COMPONENT
from .host_registration import (
    HostRegistration,
    HostRegistrationCleanupError,
    HostRegistrationError,
    register_host_mappings,
)
from .module_collector import ModuleDiscovery
from .offload_plan import (
    OffloadPlan,
    get_offload_plan,
)
from .tensor_utils import (
    clear_block_storage,
    clear_tensor_storage,
    describe_tensor_storage,
    flatten_physical_storage,
    group_named_tensors_by_dtype,
    is_materialized_tensor,
    make_offload_placeholder,
    materialization_probe,
    module_materialization_probe,
    restore_tensor_storage,
    set_tensor_storage,
    tensor_storage_metadata,
)
from .tensor_utils import (
    dtype_size as _dtype_size,
)

logger = init_logger(__name__)

# A backend normally owns both objects below. This process-lifetime safety
# owner prevents HostWeightLease.__del__ from unmapping storage when cleanup
# failure unwinds startup and the backend itself becomes unreachable. A clean
# retry removes the pair before closing the lease.
_ACTIVE_HWR_REGISTRATIONS: list[tuple[HostRegistration, HostWeightLease]] = []
_ACTIVE_HWR_REGISTRATIONS_LOCK = threading.Lock()


def _retain_active_hwr_registration(
    registration: HostRegistration,
    lease: HostWeightLease,
) -> None:
    with _ACTIVE_HWR_REGISTRATIONS_LOCK:
        if not any(
            candidate is registration and candidate_lease is lease
            for candidate, candidate_lease in _ACTIVE_HWR_REGISTRATIONS
        ):
            _ACTIVE_HWR_REGISTRATIONS.append((registration, lease))


def _forget_active_hwr_registration(
    registration: HostRegistration,
    lease: HostWeightLease,
) -> None:
    with _ACTIVE_HWR_REGISTRATIONS_LOCK:
        _ACTIVE_HWR_REGISTRATIONS[:] = [
            (candidate, candidate_lease)
            for candidate, candidate_lease in _ACTIVE_HWR_REGISTRATIONS
            if candidate is not registration or candidate_lease is not lease
        ]


# Threshold (in MB) for deciding whether a non-block DiT submodule should
# use layerwise offload (streaming hooks) or be moved to GPU as a resident
# module.  Submodules larger than this are offloaded to save HBM; smaller
# ones stay resident for lower latency.
_ON_DEMAND_THRESHOLD_MB = 1024


class DistributedLayerwiseOffloadHook(ModelHook):
    """Hook for distributed layerwise offloading with fixed double-buffer.

    Each rank stores only a shard of each block's weights on host memory.
    Two device slots alternate: one holds current weights, one holds next weights.
    H2D and AllGather run asynchronously on dedicated streams, overlapped
    with computation.

    Supports both NVIDIA GPU (CUDA) and Ascend NPU (CANN) platforms.
    """

    _HOOK_NAME = "distributed_layerwise_offload"

    def __init__(
        self,
        next_block: nn.Module,
        device: torch.device,
        dp_group: torch.distributed.ProcessGroup | None,
        dp_size: int,
        rank: int,
        copy_stream: Any | None = None,
        comm_stream: Any | None = None,
        pin_memory: bool = True,
        shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] | None = None,
        rank_local_mmap: bool = False,
        tensor_transforms: dict[int, Any] | None = None,
        materialization_probe_tensor: torch.Tensor | None = None,
    ):
        assert isinstance(next_block, nn.Module), "transformer block must be type `torch.nn.Module`"
        if type(dp_size) is not int or dp_size < 1:
            raise ValueError(f"dp_size must be a positive integer, got {dp_size!r}")
        if type(rank) is not int or not 0 <= rank < dp_size:
            raise ValueError(f"rank must satisfy 0 <= rank < dp_size, got rank={rank!r}, dp_size={dp_size}")
        if dp_size > 1 and dp_group is None:
            raise ValueError("dp_group is required when dp_size is greater than one")
        if rank_local_mmap and dp_size != 1:
            raise ValueError("rank_local_mmap requires dp_size=1")

        self.next_block = next_block
        self.device = device
        self.dp_group = dp_group
        self.dp_size = dp_size
        self.rank = rank
        self.pin_memory = pin_memory
        self.rank_local_mmap = rank_local_mmap
        self.registered_mmap = False
        self.tensor_transforms = tensor_transforms or {}
        self._materialization_probe = materialization_probe_tensor

        self.copy_stream = copy_stream or current_omni_platform.Stream()
        self.comm_stream = comm_stream or current_omni_platform.Stream()

        # Double buffers: either shared (from backend) or self-allocated (lazy)
        if shared_buffers is not None:
            self.gpu_buffers: list[dict[torch.dtype, torch.Tensor] | None] = shared_buffers
            self._owns_buffers = False
        else:
            self.gpu_buffers = [None, None]
            self._owns_buffers = True
        self.ready_events: list[Any | None] = [None, None]

        # Sharded host weights for the next block, keyed by dtype
        self.cpu_shards: dict[torch.dtype, torch.Tensor] = {}
        # File-backed source tensors for rank-local mmap.  Unlike cpu_shards,
        # these remain immutable views of the checkpoint and are never pinned
        # or flattened into a model-sized private allocation.
        self.cpu_sources: dict[torch.dtype, list[dict[str, Any]]] = {}
        self.metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

        # Rank-local mmap uses two host staging slots shared by every hook in
        # this worker.  They are assigned by the backend after all block sizes
        # are known, mirroring the shared device-buffer allocation.
        self.cpu_staging_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]
        self.cpu_staging_events: list[Any | None] = [None, None]

        # Current slot index (0 or 1).  Updated dynamically by the previous
        # hook's prefetch_layer call via _prefetched_slot.  This ensures
        # correct slot tracking for ALL block counts (including odd N).
        self.current_slot = 0
        self._prefetched_slot: int | None = None

        # Backward link to previous hook for fallback (cache-dit skip)
        self._prev_hook: DistributedLayerwiseOffloadHook | None = None

        # Marks the first hook in a shared-buffer group.  When multiple DiT
        # groups share the same 2 GPU buffers, another group may have
        # overwritten this group's slot between forwards.  The first block
        # must sync-prefetch on entry to ensure it loads the correct
        # weights, even if is_materialized sees a non-empty tensor left by
        # the other group.
        self._is_group_first: bool = False

        # Group ID for per-slot contamination tracking.  Shared across all
        # hooks in the same group.  When a hook prefetches into a slot, it
        # stamps the slot with its group ID.  On group entry, the first
        # hook checks whether the slot was last written by its own group
        # (tail hook's async prefetch) — if so, skip the sync-prefetch.
        self._group_id: int = -1
        self._shared_slot_group: list[int] | None = None  # [-1, -1], shared

        # Parameters/buffers of the current and next blocks
        self.block_parameters: dict[str, nn.Parameter] = {}
        self.block_buffers: dict[str, torch.Tensor] = {}
        self.next_block_parameters: dict[str, nn.Parameter] = {}
        self.next_block_buffers: dict[str, torch.Tensor] = {}

        # Per-block synchronization primitive: set after H2D copy completes.
        self._prefetch_done: Any | None = None

        # Shared shard (AllGather input) buffers — assigned by backend.
        self.gpu_shard_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]

        self._cached_repoint: tuple[tuple[Any, ...], ...] = ()

    # ------------------------------------------------------------------ #
    #  DTensor helpers (shared with LayerwiseOffloadHook)                 #
    # ------------------------------------------------------------------ #

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        module = super().initialize_hook(module)

        self.block_parameters = dict(module.named_parameters())
        self.block_buffers = dict(module.named_buffers())
        if self._materialization_probe is None:
            self._materialization_probe = materialization_probe(self.block_parameters, self.block_buffers)

        self.next_block_parameters = dict(self.next_block.named_parameters())
        self.next_block_buffers = dict(self.next_block.named_buffers())

        if self.rank_local_mmap:
            self.cpu_sources, self.metadata = self._collect_mmap_sources(
                self.next_block_parameters,
                self.next_block_buffers,
                self.tensor_transforms,
            )
        else:
            # Shard next block's weights and store local shard in pinned CPU memory.
            self.cpu_shards, self.metadata = self._shard_and_pin(
                self.next_block_parameters,
                self.next_block_buffers,
                self.dp_size,
                self.rank,
                self.pin_memory,
                self.tensor_transforms,
            )

        # Allocate device buffers only if not using shared buffers from backend
        if self._owns_buffers:
            self._allocate_device_buffers()

        # Cache parameter re-pointing metadata to avoid per-layer dict lookups.
        self._cached_repoint = tuple(
            (
                self.next_block_parameters[m["name"]]
                if m["name"] in self.next_block_parameters
                else self.next_block_buffers[m["name"]],
                dtype,
                m["offset"],
                m["numel"],
                m["shape"],
                m["stride"],
            )
            for dtype, metas in self.metadata.items()
            for m in metas
        )

        # Pre-compute AG output sizes (avoid sum() per layer).
        self._ag_output_sizes: dict[torch.dtype, int] = {}
        if self.dp_size > 1:
            for dtype in self.metadata:
                shard_numel = self.cpu_shards[dtype].numel()
                self._ag_output_sizes[dtype] = shard_numel * self.dp_size

        # Commit the storage mutation only after every allocation and metadata
        # calculation above has succeeded. Until here, hook setup is retryable.
        clear_tensor_storage(chain(self.next_block_parameters.values(), self.next_block_buffers.values()))

        return module

    @staticmethod
    def _collect_mmap_sources(
        params: dict[str, nn.Parameter],
        bufs: dict[str, torch.Tensor],
        tensor_transforms: dict[int, Any] | None = None,
    ) -> tuple[dict[torch.dtype, list[dict[str, Any]]], dict[torch.dtype, list[dict[str, Any]]]]:
        """Retain file-backed tensors and describe their runtime layout.

        The returned sources preserve safetensors mmap storage.  Runtime-layout
        adapters are applied only while packing a bounded staging slot, so no
        full-model anonymous CPU copy is retained by a worker.
        """
        cpu_sources: dict[torch.dtype, list[dict[str, Any]]] = {}
        metadata: dict[torch.dtype, list[dict[str, Any]]] = {}
        offsets: dict[torch.dtype, int] = {}
        for name, target in chain(params.items(), bufs.items()):
            source = target.to_local() if hasattr(target, "to_local") else target
            if source.device.type != "cpu":
                raise ValueError(
                    f"Rank-local mmap storage requires CPU checkpoint views, but {name!r} is on {source.device}."
                )

            dtype = source.dtype
            offset = offsets.get(dtype, 0)
            transform = (tensor_transforms or {}).get(id(target))
            runtime_source = transform(source) if callable(transform) else source
            if runtime_source.dtype != dtype or runtime_source.shape != source.shape:
                raise ValueError(
                    "mmap weight transform changed tensor metadata for "
                    f"{name!r}: expected dtype={dtype}, shape={tuple(source.shape)}, "
                    f"got dtype={runtime_source.dtype}, shape={tuple(runtime_source.shape)}"
                )
            stride = runtime_source.stride()
            storage_numel = (
                0
                if runtime_source.numel() == 0
                else 1 + sum((size - 1) * axis_stride for size, axis_stride in zip(runtime_source.shape, stride))
            )

            cpu_sources.setdefault(dtype, []).append(
                {
                    "name": name,
                    "tensor": source.detach(),
                    "transform": transform,
                }
            )
            metadata.setdefault(dtype, []).append(
                {
                    "name": name,
                    "offset": offset,
                    "numel": storage_numel,
                    "shape": runtime_source.shape,
                    "stride": stride,
                }
            )
            offsets[dtype] = offset + storage_numel

        return cpu_sources, metadata

    @staticmethod
    def _shard_and_pin(
        params: dict[str, nn.Parameter],
        bufs: dict[str, torch.Tensor],
        dp_size: int,
        rank: int,
        pin_memory: bool,
        tensor_transforms: dict[int, Any] | None = None,
    ) -> tuple[dict[torch.dtype, torch.Tensor], dict[torch.dtype, list[dict[str, Any]]]]:
        """Flatten params+buffers by dtype, split into DP shards, store local shard.

        Each rank stores only ``1/dp_size`` of the total weights. The full
        tensor is reconstructed at runtime via AllGather.
        """
        dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}
        cpu_shards: dict[torch.dtype, torch.Tensor] = {}

        for dtype, named_weights in group_named_tensors_by_dtype(params, bufs).items():
            # Apply loader-declared layout conversions block by block while
            # preserving the physical tensor layout used by the kernels.
            specs = describe_tensor_storage(named_weights, tensor_transforms)
            total_numel = sum(spec.storage_numel for spec in specs)

            # Equal-sized shards (ceil division) for all_gather_into_tensor
            shard_size = (total_numel + dp_size - 1) // dp_size  # ceil
            shard_start = rank * shard_size
            shard_end = min(shard_start + shard_size, total_numel)

            # Allocate ONLY the shard (1/dp_size), zero-padded to ceil.
            # Avoids materialising the full block on CPU.
            shard = torch.zeros(
                shard_size,
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
            )

            current_offset = 0
            for spec in specs:
                # Offsets remain relative to the FULL flattened buffer
                # (needed for correct AllGather reconstruction).
                dtype_metadata.setdefault(dtype, []).append(tensor_storage_metadata(spec, current_offset))

                # Copy ONLY the portion within [shard_start, shard_end)
                overlap_start = max(current_offset, shard_start)
                overlap_end = min(current_offset + spec.storage_numel, shard_end)
                if overlap_start < overlap_end:
                    flat_storage = flatten_physical_storage(spec.value, spec.storage_numel)
                    src_start = overlap_start - current_offset
                    src_end = overlap_end - current_offset
                    dst_start = overlap_start - shard_start
                    dst_end = overlap_end - shard_start
                    shard[dst_start:dst_end].copy_(flat_storage[src_start:src_end])

                current_offset += spec.storage_numel

            cpu_shards[dtype] = shard

        return cpu_shards, dtype_metadata

    def _allocate_device_buffers(self) -> None:
        """Pre-allocate exactly two device buffers (one per slot)."""
        for slot in range(2):
            gpu_weights: dict[torch.dtype, torch.Tensor] = {}
            for dtype, metas in self.metadata.items():
                total_numel = sum(m["numel"] for m in metas)
                # AllGather output = dp_size * shard_size (padded)
                padded = total_numel
                if self.dp_size > 1:
                    shard_sz = (total_numel + self.dp_size - 1) // self.dp_size
                    padded = shard_sz * self.dp_size
                gpu_weights[dtype] = torch.empty(padded, dtype=dtype, device=self.device)
            self.gpu_buffers[slot] = gpu_weights

    @property
    def is_materialized(self) -> bool:
        """Check whether this block's parameters hold real data on device."""
        return self._materialization_probe is None or is_materialized_tensor(self._materialization_probe)

    # ------------------------------------------------------------------ #
    #  Prefetch: H2D + AllGather (overlapped on dedicated streams)      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_mmap_source(
        source_info: dict[str, Any],
        meta: dict[str, Any],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        source = source_info["tensor"]
        transform = source_info["transform"]
        if callable(transform):
            source = transform(source)
        if source.dtype != dtype or source.shape != meta["shape"] or source.stride() != meta["stride"]:
            raise ValueError(
                "mmap weight transform changed tensor layout for "
                f"{source_info['name']!r}: expected dtype={dtype}, "
                f"shape={tuple(meta['shape'])}, stride={meta['stride']}; "
                f"got dtype={source.dtype}, shape={tuple(source.shape)}, "
                f"stride={source.stride()}"
            )
        return source

    @staticmethod
    def _pack_mmap_sources(
        cpu_sources: dict[torch.dtype, list[dict[str, Any]]],
        metadata: dict[torch.dtype, list[dict[str, Any]]],
        slot_buffers: dict[torch.dtype, torch.Tensor],
    ) -> dict[torch.dtype, torch.Tensor]:
        staged: dict[torch.dtype, torch.Tensor] = {}
        for dtype, metas in metadata.items():
            total_numel = sum(meta["numel"] for meta in metas)
            destination = slot_buffers[dtype][:total_numel]
            sources = cpu_sources[dtype]
            for source_info, meta in zip(sources, metas, strict=True):
                source = DistributedLayerwiseOffloadHook._resolve_mmap_source(source_info, meta, dtype)
                start = meta["offset"]
                physical_storage = destination[start : start + meta["numel"]]
                if source.is_contiguous():
                    physical_storage.copy_(source.flatten())
                else:
                    torch.as_strided(
                        physical_storage,
                        size=source.shape,
                        stride=source.stride(),
                    ).copy_(source)
            staged[dtype] = destination
        return staged

    def _stage_mmap_sources(self, slot: int) -> dict[torch.dtype, torch.Tensor]:
        """Pack this block's mmap views into one bounded host staging slot."""
        previous_copy = self.cpu_staging_events[slot]
        if previous_copy is not None:
            synchronize = getattr(previous_copy, "synchronize", None)
            if callable(synchronize):
                synchronize()
            else:
                # Platform events normally expose synchronize().  Retain a
                # correctness fallback for test and non-CUDA platform shims.
                current_omni_platform.synchronize()
            self.cpu_staging_events[slot] = None

        slot_buffers = self.cpu_staging_buffers[slot]
        if slot_buffers is None:
            raise RuntimeError(f"cpu_staging_buffers[{slot}] was not allocated")
        return self._pack_mmap_sources(self.cpu_sources, self.metadata, slot_buffers)

    @staticmethod
    def _copy_mmap_sources_to_device(
        cpu_sources: dict[torch.dtype, list[dict[str, Any]]],
        metadata: dict[torch.dtype, list[dict[str, Any]]],
        gpu_buffers: dict[torch.dtype, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> None:
        """Copy registered source views directly into flattened device buffers."""
        for dtype, metas in metadata.items():
            destination = gpu_buffers[dtype]
            sources = cpu_sources[dtype]
            for source_info, meta in zip(sources, metas, strict=True):
                source = DistributedLayerwiseOffloadHook._resolve_mmap_source(source_info, meta, dtype)
                start = meta["offset"]
                physical_storage = destination[start : start + meta["numel"]]
                async_copy = non_blocking and source.is_pinned()
                if source.is_contiguous():
                    physical_storage.copy_(source.flatten(), non_blocking=async_copy)
                else:
                    torch.as_strided(
                        physical_storage,
                        size=source.shape,
                        stride=source.stride(),
                    ).copy_(source, non_blocking=async_copy)

    @torch.compiler.disable
    def prefetch_layer(self, slot: int, non_blocking: bool = True) -> None:
        """Prepare next block's weights into the shared device buffer for *slot*.

        Uses the pre-allocated ``self.gpu_buffers[slot]`` instead of
        allocating fresh tensors every layer.  This enforces the fixed
        double-buffer memory bound (exactly 2 blocks on device).
        """
        self.copy_stream.wait_stream(current_omni_platform.current_stream())

        evt = current_omni_platform.Event()
        gpu_weights = self.gpu_buffers[slot]
        assert gpu_weights is not None, f"gpu_buffers[{slot}] not allocated"

        if self.dp_size <= 1 or self.dp_group is None:
            if self.rank_local_mmap and self.registered_mmap:
                with current_omni_platform.stream(self.copy_stream):
                    self._copy_mmap_sources_to_device(
                        self.cpu_sources,
                        self.metadata,
                        gpu_weights,
                        non_blocking=non_blocking,
                    )
                    evt.record(self.copy_stream)
            else:
                cpu_weights = self._stage_mmap_sources(slot) if self.rank_local_mmap else self.cpu_shards
                with current_omni_platform.stream(self.copy_stream):
                    for dtype, cpu_shard in cpu_weights.items():
                        gw = gpu_weights[dtype]
                        async_copy = non_blocking and cpu_shard.is_pinned()
                        gw[: cpu_shard.numel()].copy_(cpu_shard, non_blocking=async_copy)
                    evt.record(self.copy_stream)
                if self.rank_local_mmap:
                    # The CPU slot may be overwritten only after this H2D copy has
                    # finished.  The shared event protects reuse by another hook.
                    self.cpu_staging_events[slot] = evt
        else:
            gpu_shards: dict[torch.dtype, torch.Tensor] = {}
            shard_bufs = self.gpu_shard_buffers[slot]
            assert shard_bufs is not None, f"gpu_shard_buffers[{slot}] not allocated"
            with current_omni_platform.stream(self.copy_stream):
                for dtype, cpu_shard in self.cpu_shards.items():
                    gpu_shard = shard_bufs[dtype][: cpu_shard.numel()]
                    gpu_shard.copy_(cpu_shard, non_blocking=non_blocking)
                    gpu_shards[dtype] = gpu_shard

            self.comm_stream.wait_stream(self.copy_stream)
            with current_omni_platform.stream(self.comm_stream):
                for dtype, local_shard in gpu_shards.items():
                    # Slice the shared (max-sized) output buffer down to this
                    # block's actual AllGather output size. The buffers are
                    # sized to the *largest* block across all groups, so for any
                    # smaller block the full buffer would violate the
                    # all_gather_into_tensor contract
                    # (output.numel() == world_size * input.numel()).
                    # Repoint offsets are relative to the block's flattened
                    # buffer, so a prefix slice is safe.
                    gw = gpu_weights[dtype][: self._ag_output_sizes[dtype]]
                    torch.distributed.all_gather_into_tensor(
                        gw,
                        local_shard,
                        group=self.dp_group,
                    )
                evt.record(self.comm_stream)

        self.ready_events[slot] = evt
        self._prefetch_done = evt
        self._prefetched_slot = slot

        # Stamp the slot with this hook's group ID so that group-first
        # hooks can detect whether another group has overwritten the slot.
        if self._shared_slot_group is not None:
            self._shared_slot_group[slot] = self._group_id

        # Re-point using cached metadata (avoids per-layer dict lookups).
        for target, dtype, offset, numel, shape, stride in self._cached_repoint:
            set_tensor_storage(
                target,
                torch.as_strided(
                    gpu_weights[dtype][offset : offset + numel],
                    size=shape,
                    stride=stride,
                ),
            )

    def _wait_for_weights(self, slot: int) -> None:
        """Wait until the slot holding this block is ready for compute.

        The ready event for this slot was set by the *previous* hook's
        prefetch_layer (which prefetched THIS block's weights into the
        shared buffer).  This hook's own ready_events[slot] is None because
        it never prefetched into this slot itself.  Fall back to the
        previous hook's event to ensure the compute stream waits for the
        AllGather (or H2D) to complete before reading the buffer.
        """
        evt = self.ready_events[slot]
        if evt is None and self._prev_hook is not None:
            evt = self._prev_hook.ready_events[slot]
        if evt is not None:
            current_omni_platform.current_stream().wait_event(evt)

    # ------------------------------------------------------------------ #
    #  Offload: free device memory for current block                     #
    # ------------------------------------------------------------------ #

    @torch.compiler.disable
    def offload_layer(self) -> None:
        """Free GPU memory for current block by replacing tensors with placeholders."""
        clear_block_storage(self.block_parameters, self.block_buffers, self._prefetch_done)
        self._prefetch_done = None

    @torch.compiler.disable
    def restore_next_block_to_cpu(self) -> None:
        """Restore hook-owned master weights before removing the hook.

        Every circular hook owns the host backing for its ``next_block``.
        Dropping that hook while the module points at rotating device buffers
        or placeholders would make a later enable shard invalid tensors.
        """

        if self.rank_local_mmap:
            for dtype, metas in self.metadata.items():
                for source_info, meta in zip(self.cpu_sources[dtype], metas, strict=True):
                    source = self._resolve_mmap_source(source_info, meta, dtype)
                    target = (
                        self.next_block_parameters[meta["name"]]
                        if meta["name"] in self.next_block_parameters
                        else self.next_block_buffers[meta["name"]]
                    )
                    restore_tensor_storage(target, source, device="cpu")
            return

        if self.dp_size <= 1:
            for dtype, metas in self.metadata.items():
                flat = self.cpu_shards[dtype]
                for meta in metas:
                    value = torch.as_strided(
                        flat[meta["offset"] : meta["offset"] + meta["numel"]],
                        size=meta["shape"],
                        stride=meta["stride"],
                    )
                    target = (
                        self.next_block_parameters[meta["name"]]
                        if meta["name"] in self.next_block_parameters
                        else self.next_block_buffers[meta["name"]]
                    )
                    restore_tensor_storage(target, value, device="cpu")
            return

        if self.dp_group is None:
            raise RuntimeError("Cannot restore distributed offload weights without the DLO process group")
        self.prefetch_layer(0, non_blocking=False)
        current_omni_platform.synchronize()
        for target in chain(self.next_block_parameters.values(), self.next_block_buffers.values()):
            local = target.to_local() if hasattr(target, "to_local") else target
            restore_tensor_storage(target, local, device="cpu")

    # ------------------------------------------------------------------ #
    #  ModelHook interface                                                #
    # ------------------------------------------------------------------ #

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        # Dynamic slot tracking: read current_slot from the previous hook's
        # _prefetched_slot (the slot where this block's data was actually
        # loaded).  This corrects the initial i%2 assignment for odd block
        # counts, where the circular tail→head prefetch would otherwise
        # collide with the head's read slot.
        if self._prev_hook is not None and self._prev_hook._prefetched_slot is not None:
            self.current_slot = self._prev_hook._prefetched_slot

        # Group-first hook: check whether the shared buffer slot was
        # overwritten by another group since our last forward.  If the
        # slot still contains our own data (from the tail hook's async
        # prefetch), skip the sync-prefetch and just wait for the event.
        if self._is_group_first and self._prev_hook is not None:
            slot_contaminated = True
            if self._shared_slot_group is not None:
                slot_contaminated = self._shared_slot_group[self.current_slot] != self._group_id
            if slot_contaminated:
                # Another group (or no group) wrote to our slot — re-fetch
                self._prev_hook.prefetch_layer(self.current_slot, non_blocking=False)
            # Always wait for data to be ready (handles both sync and async paths)
            self._prev_hook._wait_for_weights(self.current_slot)
        elif not self.is_materialized and self._prev_hook is not None:
            # Previous hook was skipped (e.g. by cache-dit), sync-prefetch
            self._prev_hook.prefetch_layer(self.current_slot, non_blocking=False)
            self._prev_hook._wait_for_weights(self.current_slot)

        # Prefetch next layer into the other slot (overlapped with compute).
        # No explicit get_weights() here — offload_layer() in the previous
        # hook's post_forward already enqueued wait_event for the current
        # layer's H2D/AllGather, which is sufficient to ensure compute_stream
        # waits for weights before reading them.  The redundant wait_event
        # caused cascading NPU stream synchronization stalls (~10ms/layer).
        next_slot = 1 - self.current_slot
        self.prefetch_layer(next_slot, non_blocking=True)

        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        self.offload_layer()
        return output


# ---------------------------------------------------------------------- #
#  Module-level helpers                                                   #
# ---------------------------------------------------------------------- #


def apply_distributed_block_hook(
    module: nn.Module,
    next_block: nn.Module,
    device: torch.device,
    dp_group: torch.distributed.ProcessGroup | None,
    dp_size: int,
    rank: int,
    copy_stream: Any | None = None,
    comm_stream: Any | None = None,
    pin_memory: bool = True,
    shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] | None = None,
    rank_local_mmap: bool = False,
    tensor_transforms: dict[int, Any] | None = None,
    materialization_probe_tensor: torch.Tensor | None = None,
) -> DistributedLayerwiseOffloadHook:
    """Register a DistributedLayerwiseOffloadHook on *module*."""
    registry = HookRegistry.get_or_create(module)
    hook = DistributedLayerwiseOffloadHook(
        next_block=next_block,
        device=device,
        dp_group=dp_group,
        dp_size=dp_size,
        rank=rank,
        copy_stream=copy_stream,
        comm_stream=comm_stream,
        pin_memory=pin_memory,
        shared_buffers=shared_buffers,
        rank_local_mmap=rank_local_mmap,
        tensor_transforms=tensor_transforms,
        materialization_probe_tensor=materialization_probe_tensor,
    )
    registry.register_hook(DistributedLayerwiseOffloadHook._HOOK_NAME, hook)
    return hook


def remove_distributed_block_hook(module: nn.Module) -> None:
    """Remove the distributed layerwise offload hook from *module*."""
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    if registry is not None:
        registry.remove_hook(DistributedLayerwiseOffloadHook._HOOK_NAME)
        logger.debug("Removed distributed offload hook from %s", module.__class__.__name__)


class PinnedResidentLayerGroup:
    """Keep selected layers available for stage-scoped device residency.


    TODO(offload): Extract this alongside PinnedModuleStager after the
    distributed shard-and-pin operation becomes a shared storage primitive.
    It currently remains here because it depends on DLO's local-shard layout.
    Unlike ``module.to(device)``/``module.to("cpu")``, this group retains a
    pinned CPU master copy (or mmap source plus bounded staging) and never
    copies generated device weights back to host. Entering the denoise stage
    performs one asynchronous H2D pass;
    leaving it only restores zero-sized placeholders and releases the device
    buffers.  This lets the following VAE stage reuse the same HBM.

    The resident path is intentionally local-shard only.  With tensor
    parallelism, the regular model loader has already produced each rank's TP
    shard, so no DP AllGather is required or desirable here.
    """

    def __init__(
        self,
        blocks: list[nn.Module],
        device: torch.device,
        copy_stream: Any,
        pin_memory: bool,
        rank_local_mmap: bool = False,
        defer_staging: bool = False,
        tensor_transforms: dict[int, Any] | None = None,
    ) -> None:
        self.device = device
        self.copy_stream = copy_stream
        self.loaded = False
        self.rank_local_mmap = rank_local_mmap
        self.registered_mmap = False
        self.pin_memory = pin_memory
        self._states: list[dict[str, Any]] = []
        self._gpu_buffers: list[dict[torch.dtype, torch.Tensor]] = []

        for block in blocks:
            params = dict(block.named_parameters())
            bufs = dict(block.named_buffers())
            targets: dict[str, torch.Tensor] = {**params, **bufs}
            if rank_local_mmap:
                cpu_sources, metadata = DistributedLayerwiseOffloadHook._collect_mmap_sources(
                    params,
                    bufs,
                    tensor_transforms,
                )
                cpu_shards = {}
            else:
                cpu_shards, metadata = DistributedLayerwiseOffloadHook._shard_and_pin(
                    params,
                    bufs,
                    dp_size=1,
                    rank=0,
                    pin_memory=pin_memory,
                    tensor_transforms=tensor_transforms,
                )
                cpu_sources = {}
            self._states.append(
                {
                    "targets": targets,
                    "cpu_shards": cpu_shards,
                    "cpu_sources": cpu_sources,
                    "metadata": metadata,
                }
            )

        self._cpu_staging_buffers: list[dict[torch.dtype, torch.Tensor]] = []
        self._cpu_staging_events: list[Any | None] = [None, None]
        if rank_local_mmap and not defer_staging:
            max_sizes: dict[torch.dtype, int] = {}
            for state in self._states:
                for dtype, metas in state["metadata"].items():
                    total = sum(meta["numel"] for meta in metas)
                    max_sizes[dtype] = max(max_sizes.get(dtype, 0), total)
            for _ in range(2):
                buffers = {}
                for dtype, total in max_sizes.items():
                    buffer = torch.empty(
                        total,
                        dtype=dtype,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    buffers[dtype] = buffer
                self._cpu_staging_buffers.append(buffers)

        # All host masters and optional staging buffers are now ready. Clear
        # model storage as one final constructor commit so an earlier failure
        # leaves every resident block intact.
        clear_tensor_storage(target for state in self._states for target in state["targets"].values())

    def load(self) -> None:
        if self.loaded:
            return

        # Allocate on the compute stream so the caching allocator can reuse
        # blocks released by the encoder stage.  Allocating on the copy stream
        # would create a separate stream-local pool and inflate peak HBM.
        gpu_buffers: list[dict[torch.dtype, torch.Tensor]] = []
        for state in self._states:
            block_buffers: dict[torch.dtype, torch.Tensor] = {}
            for dtype, metas in state["metadata"].items():
                total = sum(meta["numel"] for meta in metas)
                block_buffers[dtype] = torch.empty(total, dtype=dtype, device=self.device)
            gpu_buffers.append(block_buffers)

        self.copy_stream.wait_stream(current_omni_platform.current_stream())
        ready = current_omni_platform.Event()
        with current_omni_platform.stream(self.copy_stream):
            for index, (state, block_buffers) in enumerate(zip(self._states, gpu_buffers)):
                if self.rank_local_mmap and self.registered_mmap:
                    DistributedLayerwiseOffloadHook._copy_mmap_sources_to_device(
                        state["cpu_sources"],
                        state["metadata"],
                        block_buffers,
                        non_blocking=True,
                    )
                    continue
                if self.rank_local_mmap:
                    slot = index % 2
                    previous_copy = self._cpu_staging_events[slot]
                    if previous_copy is not None:
                        synchronize = getattr(previous_copy, "synchronize", None)
                        if callable(synchronize):
                            synchronize()
                        else:
                            current_omni_platform.synchronize()
                    cpu_weights = DistributedLayerwiseOffloadHook._pack_mmap_sources(
                        state["cpu_sources"],
                        state["metadata"],
                        self._cpu_staging_buffers[slot],
                    )
                else:
                    cpu_weights = state["cpu_shards"]

                for dtype, cpu_weight in cpu_weights.items():
                    block_buffers[dtype].copy_(
                        cpu_weight,
                        non_blocking=cpu_weight.is_pinned(),
                    )
                if self.rank_local_mmap:
                    slot_ready = current_omni_platform.Event()
                    slot_ready.record(self.copy_stream)
                    self._cpu_staging_events[slot] = slot_ready
            ready.record(self.copy_stream)

        for state, block_buffers in zip(self._states, gpu_buffers):
            targets = state["targets"]
            for dtype, metas in state["metadata"].items():
                gpu_buffer = block_buffers[dtype]
                for meta in metas:
                    set_tensor_storage(
                        targets[meta["name"]],
                        torch.as_strided(
                            gpu_buffer[meta["offset"] : meta["offset"] + meta["numel"]],
                            size=meta["shape"],
                            stride=meta["stride"],
                        ),
                    )

        current_omni_platform.current_stream().wait_event(ready)
        self._gpu_buffers = gpu_buffers
        self.loaded = True

    def offload(self) -> None:
        if not self.loaded:
            return

        # Denoise kernels consume these buffers on the current compute stream.
        # Synchronize once at the stage boundary; no D2H copy is necessary
        # because the pinned CPU master weights were never overwritten.
        current_omni_platform.synchronize()
        for state in self._states:
            for target in state["targets"].values():
                set_tensor_storage(target, make_offload_placeholder(target))
        self._gpu_buffers.clear()
        self.loaded = False

    def restore_to_cpu(self) -> None:
        """Materialize the persistent host masters back into the module.

        Stage-scoped ``offload()`` deliberately leaves placeholders in the
        module while this group owns the CPU backing. ``disable()`` discards
        the group, so it must first restore ordinary CPU tensors to make a
        later enable cycle safe.
        """
        self.offload()

        for state in self._states:
            targets = state["targets"]
            for dtype, metas in state["metadata"].items():
                if self.rank_local_mmap:
                    for source_info, meta in zip(state["cpu_sources"][dtype], metas, strict=True):
                        source = DistributedLayerwiseOffloadHook._resolve_mmap_source(source_info, meta, dtype)
                        restore_tensor_storage(targets[meta["name"]], source, device="cpu")
                    continue

                flat = state["cpu_shards"][dtype]
                for meta in metas:
                    source = torch.as_strided(
                        flat[meta["offset"] : meta["offset"] + meta["numel"]],
                        size=meta["shape"],
                        stride=meta["stride"],
                    )
                    restore_tensor_storage(targets[meta["name"]], source, device="cpu")


# ---------------------------------------------------------------------- #
#  Backend                                                                #
# ---------------------------------------------------------------------- #


class DistributedLayerwiseOffloadBackend(OffloadBackend):
    """Distributed layer-wise (block-level) offloading backend.

    Supports both GPU (CUDA) and NPU (CANN) platforms.
    Device type is determined by the device passed to enable().

    Each rank stores only a shard of each block's weights on host memory.
    Two device slots alternate: one holds current weights, one holds next
    weights. H2D and AllGather run asynchronously on dedicated streams,
    overlapped with computation.
    """

    def __init__(
        self,
        config: OffloadConfig,
        device: torch.device,
        host_weight_plan: HostWeightPlan | None = None,
    ):
        super().__init__(config, device)

        self.copy_stream = current_omni_platform.Stream()
        self.comm_stream = current_omni_platform.Stream()
        self.dp_group: torch.distributed.ProcessGroup | None = None
        self.dp_size = config.dp_size
        self.rank = 0
        self._blocks: list[list[nn.Module]] = []
        self._all_hook_groups: list[list[DistributedLayerwiseOffloadHook]] = []
        self._resident_blocks: list[nn.Module] = []
        self._resident_layer_group: PinnedResidentLayerGroup | None = None
        self._residency_pipeline_ref: weakref.ReferenceType[nn.Module] | None = None
        self._encoder_modules: list[nn.Module] = []
        self._staged_components: list[nn.Module] = []
        self._using_mmap = False
        self._using_rank_local_mmap = False
        self._using_registered_mmap = False
        self.host_weight_plan = host_weight_plan
        self._host_weight_lease: HostWeightLease | None = None
        self._host_registration: HostRegistration | None = None
        self._mmap_transforms_by_tensor_id: dict[int, Any] = {}
        self._poisoned_reason: str | None = None

    def load_resident_layers(self) -> None:
        """Load the model-declared leading blocks for the denoise stage."""
        if self._resident_layer_group is not None:
            self._resident_layer_group.load()

    def offload_resident_layers(self) -> None:
        """Release leading blocks before VAE decode to bound peak HBM."""
        if self._resident_layer_group is not None:
            self._resident_layer_group.offload()

    def _clear_residency_controller(self) -> None:
        pipeline_ref = self._residency_pipeline_ref
        self._residency_pipeline_ref = None
        pipeline = None if pipeline_ref is None else pipeline_ref()
        if pipeline is not None and getattr(pipeline, "_dlo_residency_controller", None) is self:
            pipeline._dlo_residency_controller = None

    def _rank_local_source_tensors(
        self,
        hooks: list[DistributedLayerwiseOffloadHook],
    ) -> tuple[torch.Tensor, ...]:
        """Return the exact CPU sources that direct H2D would consume."""
        tensors: list[torch.Tensor] = []
        seen: set[int] = set()

        def collect(cpu_sources: dict[torch.dtype, list[dict[str, Any]]]) -> None:
            for sources in cpu_sources.values():
                for source in sources:
                    tensor = source["tensor"]
                    if id(tensor) not in seen:
                        seen.add(id(tensor))
                        tensors.append(tensor)

        for hook in hooks:
            collect(hook.cpu_sources)
        if self._resident_layer_group is not None:
            for state in self._resident_layer_group._states:
                collect(state["cpu_sources"])
        return tuple(tensors)

    def _try_register_hwr_mmap(self, source_tensors: tuple[torch.Tensor, ...]) -> bool:
        """Register the complete final-layout lease under pinned-memory policy."""
        lease = self._host_weight_lease
        if lease is None:
            return False
        if not self.config.pin_cpu_memory:
            logger.info("HWR mmap registration disabled by pin_cpu_memory=False; using bounded host staging")
            return False
        if not lease.mapped_regions or not source_tensors:
            logger.warning("HWR mmap registration found no mapped sources; using bounded host staging")
            return False

        limit_gib = self.config.dlo_host_registration_limit_gib
        max_bytes = int(limit_gib * 1024**3) if limit_gib > 0 else None
        started = time.perf_counter()
        try:
            registration = register_host_mappings(
                lease.mapped_regions,
                device=self.device,
                max_bytes=max_bytes,
            )
            try:
                unpinned = [tensor for tensor in source_tensors if tensor.numel() and not tensor.is_pinned()]
            except Exception as exc:
                errors = registration.close()
                if errors:
                    self._host_registration = registration
                    _retain_active_hwr_registration(registration, lease)
                    raise HostRegistrationCleanupError(
                        "CUDA registration succeeded but pinned-source verification failed, "
                        f"and rollback failed: {errors[:3]}"
                    ) from exc
                raise HostRegistrationError(f"cannot verify registered HWR sources: {exc}") from exc
            if unpinned:
                errors = registration.close()
                if errors:
                    self._host_registration = registration
                    _retain_active_hwr_registration(registration, lease)
                    raise HostRegistrationCleanupError(
                        "CUDA registration succeeded but PyTorch rejected mapped sources, "
                        f"and rollback failed: {errors[:3]}"
                    )
                raise HostRegistrationError(
                    "CUDA registration succeeded but PyTorch did not recognize "
                    f"{len(unpinned)} mapped source(s) as pinned"
                )
        except HostRegistrationCleanupError as exc:
            # Falling back could close a lease while the platform still owns
            # one of its mappings. Fail startup and retain ownership for retry.
            active_registration = exc.active_registration
            if active_registration is not None:
                self._host_registration = active_registration
                _retain_active_hwr_registration(active_registration, lease)
            logger.exception("HWR mmap registration rollback failed")
            raise
        except HostRegistrationError as exc:
            logger.warning("HWR registered direct H2D unavailable (%s); using bounded host staging", exc)
            return False

        self._host_registration = registration
        _retain_active_hwr_registration(registration, lease)
        logger.info(
            "Registered %.2f GiB of HWR mmap in %d range(s) for direct H2D in %.3f s",
            registration.total_bytes / 1024**3,
            registration.region_count,
            time.perf_counter() - started,
        )
        return True

    def _configure_hwr_transfer(self, hooks: list[DistributedLayerwiseOffloadHook]) -> None:
        """Select registered direct H2D or bounded staging once per backend."""
        plan = self.host_weight_plan
        if (
            (not hooks and self._resident_layer_group is None)
            or not self._using_rank_local_mmap
            or plan is None
            or plan.backing_kind != "host_weight_runtime"
        ):
            return

        source_tensors = self._rank_local_source_tensors(hooks)
        self._using_registered_mmap = self._try_register_hwr_mmap(source_tensors)
        for hook in hooks:
            hook.registered_mmap = self._using_registered_mmap
        if self._resident_layer_group is not None:
            self._resident_layer_group.registered_mmap = self._using_registered_mmap
            if self._using_registered_mmap:
                self._resident_layer_group._cpu_staging_buffers.clear()

    def _release_registered_mmap(self) -> None:
        """Release every platform registration before closing the HWR lease."""
        registration = self._host_registration
        if registration is None:
            return
        errors = registration.close()
        if errors:
            lease = self._host_weight_lease
            if lease is not None and not lease.closed:
                _retain_active_hwr_registration(registration, lease)
            logger.error("HWR mmap unregistration failed; retaining lease mappings for retry: %s", errors[:3])
            raise HostRegistrationCleanupError(f"failed to unregister {len(errors)} HWR mmap range(s)")
        lease = self._host_weight_lease
        if lease is not None:
            _forget_active_hwr_registration(registration, lease)
        self._host_registration = None
        logger.info("Unregistered HWR mmap ranges")

    def _load_weights_via_mmap(
        self,
        pipeline: nn.Module,
        modules,
        plan: HostWeightPlan,
    ) -> None:
        """Load DiT checkpoint tensors as file-backed safetensors views.

        When the transformer is created on meta device, this method
        replaces meta params with mmap views of the checkpoint files.
        The views point to OS page cache shared across ranks. AllGather mode
        copies only the rank's 1/dp_size shard to a persistent private buffer;
        rank-local mode retains the views and packs one block at a time into
        bounded staging storage.

        Non-DiT modules (VAE, encoders) are NOT affected — they were
        created on CPU with real weights via from_pretrained.
        """
        from safetensors import safe_open

        logger.info(
            "Loading DiT weights via mmap (meta -> shared page cache): %d tensors",
            len(plan.bindings),
        )

        # --- Convert DiT modules to meta device ---
        # The transformer was created normally (with random weights and
        # correct non-persistent buffers from __init__).  We convert it
        # to meta device to release the random weights (they're useless),
        # then load real weights via mmap.
        #
        # Non-persistent buffers (e.g. RoPE inv_freq, timestep freqs) are
        # NOT in the checkpoint — they are computed from formulas in
        # __init__.  Save them before meta conversion and restore after
        # mmap loading, so we don't need model-specific buffer rebuild code.
        #
        # The loader proved topology, coverage, source metadata, and adapter
        # compatibility before it skipped ordinary weight materialization.
        # This method only realizes that exact plan; it does not select or
        # rediscover a checkpoint layout.

        #
        # Important: when DiT modules are nested (e.g. transformer contains
        # transformer.language_model), to_empty("meta") on the parent
        # converts the child's buffers too.  So we must save ALL buffers
        # from ALL DiT modules BEFORE any to_empty call.
        saved_buffers: dict[int, dict[str, torch.Tensor]] = {}
        for dit_module in modules.dits:
            bufs: dict[str, torch.Tensor] = {}
            for name, buf in dit_module.named_buffers():
                # Check if this buffer is non-persistent on its OWNING module
                owner = dit_module
                parts = name.split(".")
                for part in parts[:-1]:
                    owner = getattr(owner, part)
                buf_name = parts[-1]
                is_non_persistent = buf_name in owner._non_persistent_buffers_set
                # Save if non-persistent OR already meta (shouldn't happen
                # before to_empty, but be safe)
                if is_non_persistent:
                    bufs[name] = buf.detach().clone()
            saved_buffers[id(dit_module)] = bufs

        # Now convert all DiT modules to meta (after saving all buffers).
        # Skip modules that are already meta (happens when a parent DiT
        # module contains a child DiT module — to_empty on the parent
        # already converted the child).
        for dit_module in modules.dits:
            if any(p.is_meta for p in dit_module.parameters()):
                logger.info(
                    "%s already on meta device (skipping to_empty, %d buffers saved)",
                    dit_module.__class__.__name__,
                    len(saved_buffers.get(id(dit_module), {})),
                )
                continue
            dit_module.to_empty(device="meta")
            logger.info(
                "Converted %s to meta device (%d non-persistent buffers saved)",
                dit_module.__class__.__name__,
                len(saved_buffers.get(id(dit_module), {})),
            )

        # Cache open file handles
        file_cache: dict[str, Any] = {}
        loaded_names: set[str] = set()

        # Realize the loader's exact runtime-name bindings.  In particular,
        # do not reconstruct names from DLO block discovery: storage planning
        # and transfer topology are separate contracts.
        for runtime_name, binding in plan.bindings.items():
            parent_path, _, leaf_name = runtime_name.rpartition(".")
            try:
                parent = pipeline.get_submodule(parent_path)
            except AttributeError as exc:
                raise RuntimeError(f"Host-weight plan target module {parent_path!r} no longer exists") from exc

            target = parent._parameters.get(leaf_name)
            is_parameter = target is not None
            if target is None:
                target = parent._buffers.get(leaf_name)
            if target is None:
                raise RuntimeError(f"Host-weight plan target tensor {runtime_name!r} no longer exists")

            if binding.file_path not in file_cache:
                file_cache[binding.file_path] = safe_open(
                    binding.file_path,
                    framework="pt",
                    device="cpu",
                )
            tensor = file_cache[binding.file_path].get_tensor(binding.checkpoint_key)
            if is_parameter:
                replacement = torch.nn.Parameter(tensor, requires_grad=target.requires_grad)
                parent._parameters[leaf_name] = replacement
            else:
                replacement = tensor
                parent._buffers[leaf_name] = replacement
            loaded_names.add(runtime_name)

        logger.info("Realized %d loader-planned tensors as mmap views", len(loaded_names))

        # Keep file handles open — _shard_and_pin will read from the mmap views.
        # They will be released after _shard_and_pin completes (when params are
        # replaced with offload placeholders).
        self._mmap_file_cache = file_cache
        # The regular loader runs model-specific post-load transforms after
        # assigning checkpoint tensors. The mmap path bypasses that loader, so
        # preserve the same lifecycle for transforms such as Cosmos3's fp32
        # timestep embedder.
        for dit_name, dit_module in zip(modules.dit_names, modules.dits):
            # Restore non-persistent buffers before post-load hooks and strict
            # validation. They are constructor-derived and intentionally have
            # no checkpoint binding.
            bufs = saved_buffers.get(id(dit_module), {})
            for name, buf in bufs.items():
                parent_path, _, leaf_name = name.rpartition(".")
                parent = dit_module.get_submodule(parent_path)
                restore_device = torch.device("cpu") if self._using_rank_local_mmap else self.device
                parent._buffers[leaf_name] = buf.to(restore_device)

            post_load_weights = getattr(dit_module, "post_load_weights", None)
            if callable(post_load_weights):
                post_load_weights()

            # Call validate_loaded_weights if the model defines it.
            # This preserves the sound-weight and action-weight validation
            # that AutoWeightsLoader.load_weights() triggers in the regular
            # load path (e.g. Cosmos3 checks for missing audio/action weights).
            validate = getattr(dit_module, "validate_loaded_weights", None)
            if callable(validate):
                local_loaded_names = {
                    name.removeprefix(f"{dit_name}.") for name in loaded_names if name.startswith(f"{dit_name}.")
                }
                validate(local_loaded_names)

        # Post-load hooks may rebind parameters (for example while casting a
        # submodule), so associate deferred bounded transforms with the final
        # runtime tensor objects rather than the initial mmap replacements.
        self._mmap_transforms_by_tensor_id.clear()
        for runtime_name, binding in plan.bindings.items():
            if binding.transform is None:
                continue
            parent_path, _, leaf_name = runtime_name.rpartition(".")
            parent = pipeline.get_submodule(parent_path)
            target = parent._parameters.get(leaf_name)
            if target is None:
                target = parent._buffers.get(leaf_name)
            if target is None:
                raise RuntimeError(
                    f"Host-weight transform target {runtime_name!r} no longer exists after post-load processing"
                )
            self._mmap_transforms_by_tensor_id[id(target)] = binding.transform

        remaining_meta: list[str] = []
        for dit_name, dit_module in zip(modules.dit_names, modules.dits):
            remaining_meta.extend(
                f"{dit_name}.{name}" for name, tensor in dit_module.named_parameters() if tensor.is_meta
            )
            for name, tensor in dit_module.named_buffers():
                parent_path, _, leaf_name = name.rpartition(".")
                owner = dit_module.get_submodule(parent_path)
                if leaf_name not in owner._non_persistent_buffers_set and tensor.is_meta:
                    remaining_meta.append(f"{dit_name}.{name}")
        if remaining_meta:
            raise RuntimeError(
                "The prevalidated host-weight plan left "
                f"{len(remaining_meta)} DiT tensors on the meta device "
                f"(first 5: {remaining_meta[:5]})."
            )

    def _init_dp_group(self) -> None:
        """Reuse the process group initialized by parallel_state.

        When DP > 1, uses the DP group (handles strided groups with SP/CFG/PP).
        When DP = 1 but SP > 1 (and dp_size was set to sp_size in OffloadConfig),
        uses the SP group so weights are sharded across SP ranks.
        """
        if self.dp_size <= 1:
            logger.info("Distributed layerwise offload: dp_size=1, running without AllGather")
            self.dp_group = None
            return

        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialized. "
                "Distributed layerwise offload with dp_size > 1 requires "
                "an initialized process group."
            )

        from vllm_omni.diffusion.distributed.parallel_state import (
            get_data_parallel_world_size,
            get_dp_group,
        )

        # Determine which parallel group to use for sharding.
        # When data_parallel_size > 1, use the DP group.
        # When data_parallel_size == 1 but dp_size > 1 (set from sp_size in
        # OffloadConfig), use the SP group for weight sharding.
        dp_world = get_data_parallel_world_size()
        if dp_world > 1:
            coord = get_dp_group()
        else:
            from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

            coord = get_sp_group()
            logger.info(
                "Distributed layerwise offload: DP=1, using SP group (world_size=%d) for weight sharding",
                coord.world_size,
            )

        self.dp_group = coord.device_group
        self.rank = coord.rank_in_group
        self.dp_size = coord.world_size

        logger.info(
            "Distributed layerwise offload: dp_size=%d, rank_in_group=%d, global_rank=%d, group_ranks=%s",
            self.dp_size,
            self.rank,
            coord.rank,
            coord.ranks,
        )

    def _component_transport(
        self,
        component: str,
    ) -> tuple[torch.distributed.ProcessGroup | None, int, int]:
        if self.config.uses_allgather(component):
            return self.dp_group, self.dp_size, self.rank
        return None, 1, 0

    def _has_multirank_allgather(self) -> bool:
        components = self.config.components or frozenset({DIT_COMPONENT})
        return self.dp_size > 1 and any(self.config.uses_allgather(component) for component in components)

    def _install_hook_group(
        self,
        blocks: list[nn.Module] | nn.ModuleList,
        component: str,
        *,
        use_dit_mmap: bool = False,
    ) -> list[DistributedLayerwiseOffloadHook]:
        """Install one circular block ring with the shared slot protocol."""
        block_list = list(blocks)
        if len(block_list) <= 1:
            raise ValueError("A distributed layerwise hook group requires at least two blocks")

        group, group_size, group_rank = self._component_transport(component)
        hooks: list[DistributedLayerwiseOffloadHook] = []
        self._all_hook_groups.append(hooks)
        self._blocks.append(block_list)
        probes = {id(block): module_materialization_probe(block) for block in block_list}
        for block, next_block in zip(
            chain((block_list[-1],), block_list[:-1]),
            block_list,
            strict=True,
        ):
            hooks.append(
                apply_distributed_block_hook(
                    block,
                    next_block,
                    self.device,
                    group,
                    group_size,
                    group_rank,
                    self.copy_stream,
                    self.comm_stream,
                    self.config.pin_cpu_memory,
                    shared_buffers=[None, None],
                    rank_local_mmap=self._using_rank_local_mmap if use_dit_mmap else False,
                    tensor_transforms=self._mmap_transforms_by_tensor_id if use_dit_mmap else None,
                    materialization_probe_tensor=probes[id(block)],
                )
            )

        # hooks = [last -> first, block0 -> block1, ...]. Alternating slots
        # keep a prefetch from overwriting the current block for any ring size.
        for index, hook in enumerate(hooks):
            hook._prev_hook = hooks[index - 1]
            hook.current_slot = index % 2
        hooks[1]._is_group_first = True
        return hooks

    def _try_layerwise_offload_encoder(
        self,
        module: nn.Module,
        name: str,
        plan: OffloadPlan | None,
        stage_on_demand: bool = False,
    ) -> bool:
        """Apply DLO hooks to a plan-declared text encoder.

        Rank-local transfer is always safe because it preserves the tensors
        produced by the model loader. Multi-rank AllGather is enabled only
        when the model declares that those tensors are replicated across the
        selected DLO group; encoder TP groups contain different shards and
        are deliberately not used for this transport.
        """
        if plan is None:
            return False

        group_size = self._component_transport(TEXT_ENCODER_COMPONENT)[1]
        if group_size > 1 and name not in plan.encoder_dlo_weight_replication:
            raise ValueError(
                f"Text encoder {name!r} cannot use DLO AllGather across the DiT offload group: "
                "its loader-produced weights are not declared replicated across that group. "
                "Set layer_options.text_encoder.weight_transfer='rank-local' in diffusion_offload_config "
                "for encoder-TP or rank-specific layouts."
            )
        block_groups = get_encoder_block_groups(
            module,
            name,
            plan,
            strict=self.config.components is not None or group_size > 1,
        )
        if not block_groups:
            return False

        encoder_hooks: list[DistributedLayerwiseOffloadHook] = []
        for blocks in block_groups:
            encoder_hooks.extend(self._install_hook_group(blocks, TEXT_ENCODER_COMPONENT))
        # Track the module before placement, which may fail, so outer rollback
        # also removes its partially-installed block hooks and marker state.
        self._encoder_modules.append(module)
        if not stage_on_demand:
            move_non_block_state_to_device(module, block_groups, self.device)
        set_encoder_layerwise_state(
            module,
            encoder_hooks,
            block_groups,
        )
        logger.info(
            "Enabled %s DLO transfer for text encoder %s (%d blocks across %d stacks, group_size=%d)",
            self.config.transfer_for(TEXT_ENCODER_COMPONENT).value,
            name,
            sum(len(blocks) for blocks in block_groups),
            len(block_groups),
            group_size,
        )
        return True

    def _try_layerwise_offload_submodule(self, module: nn.Module, name: str, plan: OffloadPlan | None = None) -> bool:
        """Try to apply layerwise offload to a large submodule's blocks.
        Resolution order:
        1. OffloadPlan.offload_submodules (declarative, if plan is provided)
        2. Heuristic search for common block-list attributes

        Returns True if layerwise offload was applied, False otherwise.
        """
        from operator import attrgetter

        blocks = None
        blocks_attr = None

        # 1. Check OffloadPlan first (declarative — no guessing)
        if plan is not None and name in plan.offload_submodules:
            attr_name = plan.offload_submodules[name]
            try:
                candidate = attrgetter(attr_name)(module)
                if isinstance(candidate, nn.ModuleList) and len(candidate) > 1:
                    blocks = candidate
                    blocks_attr = attr_name
            except AttributeError:
                logger.warning(
                    "OffloadPlan declared block attr '%s' for submodule '%s' "
                    "but attribute not found — falling back to heuristic",
                    attr_name,
                    name,
                )

        # 2. Fallback: heuristic search
        if blocks is None:
            for attr_name in ("layers", "blocks", "h", "model.layers"):
                try:
                    candidate = attrgetter(attr_name)(module)
                except AttributeError:
                    continue
                if isinstance(candidate, nn.ModuleList) and len(candidate) > 1:
                    blocks = candidate
                    blocks_attr = attr_name
                    break

        if blocks is None:
            return False

        logger.info(
            "Distributed layerwise offload for submodule '%s.%s' (%d blocks, %.0f MB total, group_size=%d)",
            name,
            blocks_attr,
            len(blocks),
            sum(p.nelement() * p.element_size() for p in module.parameters()) / 1048576,
            self._component_transport(DIT_COMPONENT)[1],
        )

        # Move non-block parts of the submodule to GPU (small: embeddings, norms)
        for child_name, child in module.named_children():
            if child_name != blocks_attr:
                child.to(self.device)

        self._install_hook_group(blocks, DIT_COMPONENT, use_dit_mmap=True)
        return True

    def _prepare_dit_non_block_modules(
        self,
        dit_module: nn.Module,
        blocks_attr_names: list[str],
        all_dit_modules: set[int],
        plan: OffloadPlan | None,
    ) -> None:
        """Place or hook the DiT parts that are outside its repeated blocks.

        This must run even when every repeated block is resident.  Otherwise
        an all-resident stage skips placement for modules such as H3's token
        refiner and enters the forward pass with CPU or meta tensors.
        """
        for name, module in dit_module.named_children():
            if name in blocks_attr_names:
                logger.debug("Skipped blocks module %s", name)
                continue

            module_mb = (
                sum(
                    param.nelement() * param.element_size() if not getattr(param, "is_meta", False) else 0
                    for param in module.parameters()
                )
                / 1048576
            )
            explicitly_planned = plan is not None and name in plan.offload_submodules
            if explicitly_planned or module_mb > _ON_DEMAND_THRESHOLD_MB:
                if id(module) in all_dit_modules:
                    logger.info("Submodule '%s' is already a DiT module, skipping layerwise offload", name)
                elif self._try_layerwise_offload_submodule(module, name, plan):
                    pass
                else:
                    prepare_component(
                        module,
                        name,
                        device=self.device,
                        stage_on_demand=True,
                        blockwise=False,
                        staged_components=self._staged_components,
                    )
                continue

            try:
                module.to(self.device)
            except (NotImplementedError, RuntimeError):
                # Non-persistent buffers such as RoPE frequencies do not
                # exist in the checkpoint and must be reconstructed.
                has_meta_buffer = any(getattr(buffer, "is_meta", False) for buffer in module.buffers(recurse=True))
                if not has_meta_buffer:
                    raise
                saved_params = {
                    param_name: param.data.clone()
                    for param_name, param in module.named_parameters()
                    if not getattr(param, "is_meta", False)
                }
                module.to_empty(device=self.device)
                for submodule in module.modules():
                    if hasattr(submodule, "reset_parameters"):
                        submodule.reset_parameters()
                for param_name, param in module.named_parameters():
                    if param_name in saved_params:
                        param.data.copy_(saved_params[param_name])
                module.to(self.device)

        for param in dit_module._parameters.values():
            if param is not None and not getattr(param, "is_meta", False):
                param.data = param.data.to(self.device, non_blocking=True)
        for buffer in dit_module._buffers.values():
            if buffer is not None:
                buffer.data = buffer.data.to(self.device, non_blocking=True)

    def enable(self, pipeline: nn.Module) -> None:
        """Enable DLO and make partial startup failures transactional."""
        if self._poisoned_reason is not None:
            raise RuntimeError(self._poisoned_reason)

        # A rank-local rollback can reconstruct ordinary tensors from its host
        # masters. A multi-rank AllGather rollback is unsafe: another rank may
        # never enter the matching restoration collective after startup fails.
        restore_allgather_weights = not self._has_multirank_allgather()
        try:
            self._enable(pipeline)
        except BaseException:
            try:
                self._disable(restore_allgather_weights=restore_allgather_weights)
            except BaseException:
                logger.exception("DistributedLayerwiseOffloadBackend cleanup failed while handling an enable failure")
            raise

    def _enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("DistributedLayerwiseOffloadBackend already enabled")
            return

        # Initialize DP group (if not already done by early init)
        if self.dp_group is None and self._has_multirank_allgather():
            self._init_dp_group()

        modules = ModuleDiscovery.discover(pipeline)
        if not modules.dits and self.config.offloads(DIT_COMPONENT):
            if self.host_weight_plan is not None:
                raise RuntimeError(
                    "DLO received a loader-owned host-weight plan, but no DiT modules were discovered to consume it"
                )
            message = "No DiT/transformer modules found for selected DiT offload"
            if self.config.components is not None:
                raise ValueError(message)
            logger.warning(message)

        # Retrieve optional declarative OffloadPlan from the pipeline.
        # When present, replaces heuristic block discovery.
        plan = get_offload_plan(pipeline)

        if self.config.dlo_resident_layers:
            resident_paths = frozenset() if plan is None else plan.resident_dit_paths
            if not resident_paths.intersection(modules.dit_names):
                message = (
                    f"resident_layers={self.config.dlo_resident_layers} was requested, but this model declares "
                    "no matching resident_dit_paths"
                )
                if self.config.components is not None:
                    raise ValueError(message)
                logger.warning("%s; all blocks will be streamed.", message)

        # Storage selection belongs to the loader.  DLO consumes the exact
        # prevalidated plan that caused the loader to skip materialization;
        # without a plan, all weights must already come from the ordinary
        # loader.  The transfer protocol is selected independently below.
        host_weight_plan = self.host_weight_plan
        self._using_mmap = host_weight_plan is not None
        # A one-rank AllGather transport is rank-local in practice. Preserve
        # the mmap source as the host master instead of eagerly closing it.
        self._using_rank_local_mmap = self._using_mmap and self._component_transport(DIT_COMPONENT)[1] <= 1
        if host_weight_plan is not None:
            if host_weight_plan.backing_kind == "host_weight_runtime":
                carrier = host_weight_plan.lease_carrier
                if carrier is None:
                    raise RuntimeError("DLO received a Host Weight Runtime plan without a lease carrier")
                self._host_weight_lease = carrier.take()
                if self._host_weight_lease.closed:
                    raise RuntimeError("DLO received a closed Host Weight Runtime lease")
                # The final-layout restorer has already rebound the model to
                # immutable host tensors.  Treat those tensors as mmap-like
                # sources; transport setup below selects registered direct
                # H2D or the bounded two-slot staging fallback.
                self._using_rank_local_mmap = True
                logger.info(
                    "DLO consuming final-layout Host Weight Runtime lease %s",
                    self._host_weight_lease.provenance.resolution_id,
                )
            elif host_weight_plan.backing_kind == "checkpoint_mmap":
                self._load_weights_via_mmap(
                    pipeline,
                    modules,
                    host_weight_plan,
                )
            else:
                raise ValueError(f"Unsupported DLO host-weight backing: {host_weight_plan.backing_kind}")
            if self._using_rank_local_mmap:
                logger.info(
                    "DLO rank-local host storage enabled: source pages are "
                    "node-shared; transfer setup will select registered direct H2D or bounded host staging"
                )
        else:
            remaining_meta = [
                name
                for dit_name, dit_module in zip(modules.dit_names, modules.dits)
                for name, tensor in chain(
                    dit_module.named_parameters(),
                    dit_module.named_buffers(),
                )
                if getattr(tensor, "is_meta", False)
            ]
            if remaining_meta:
                raise RuntimeError(
                    f"DLO received meta tensors without a loader-owned host-weight plan (first 5: {remaining_meta[:5]})"
                )
            logger.info("DLO is using host tensors materialized by the ordinary loader")

        # Apply each selected encoder transfer while keeping explicit VAEs and
        # unselected components resident.
        prepare_pipeline_components(
            modules,
            self.config,
            plan,
            device=self.device,
            staged_components=self._staged_components,
            enable_encoder_blocks=self._try_layerwise_offload_encoder,
        )

        if self.config.offloads(DIT_COMPONENT):
            logger.info("Applying distributed layer-wise offloading on %s", modules.dit_names)

        # Collect all DiT module objects to detect submodules that are
        # already handled as a separate DiT module (avoids duplicate hooks).
        all_dit_modules = set(id(m) for m in modules.dits)

        # Apply hooks for each DiT module
        for dit_name, dit_module, blocks_attr_names, blocks in iter_streamable_dits(
            modules, self.config, self.device, plan
        ):
            resident_count = 0
            if plan is not None and dit_name in plan.resident_dit_paths:
                resident_count = min(self.config.dlo_resident_layers, len(blocks))
            if resident_count:
                resident_blocks = blocks[:resident_count]
                self._resident_blocks.extend(resident_blocks)
                blocks = blocks[resident_count:]
                logger.info(
                    "Keeping %d leading blocks resident on %s; streaming %d tail blocks",
                    resident_count,
                    dit_name,
                    len(blocks),
                )

            self._prepare_dit_non_block_modules(
                dit_module,
                blocks_attr_names,
                all_dit_modules,
                plan,
            )

            num_blocks = len(blocks)
            if num_blocks == 0:
                logger.info("All blocks for %s are resident; no streaming hooks required", dit_name)
                continue
            if num_blocks <= 1:
                if self.config.components is not None:
                    raise ValueError(
                        f"Selected DiT {dit_name!r} leaves only one streaming block after "
                        f"resident_layers={resident_count}; choose a resident count that "
                        "leaves zero or at least two streaming blocks"
                    )
                logger.warning(
                    "#Streaming target layers (blocks) <= 1. Keeping the final block resident on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                self._resident_blocks.extend(blocks)
                continue

            self._install_hook_group(blocks, DIT_COMPONENT, use_dit_mmap=True)
        if self._resident_blocks:
            self._resident_layer_group = PinnedResidentLayerGroup(
                self._resident_blocks,
                self.device,
                self.copy_stream,
                self.config.pin_cpu_memory,
                rank_local_mmap=self._using_rank_local_mmap,
                defer_staging=bool(self._all_hook_groups),
                tensor_transforms=self._mmap_transforms_by_tensor_id,
            )
            pipeline._dlo_residency_controller = self
            self._residency_pipeline_ref = weakref.ref(pipeline)

        all_hooks = [hook for group in self._all_hook_groups for hook in group]
        self._configure_hwr_transfer(all_hooks)

        if not self._all_hook_groups:
            self.enabled = bool(self._resident_blocks or self._encoder_modules or self._staged_components)
            if self._using_mmap and not self.enabled:
                self._release_mmap_handles()
            if not self.enabled and not self.config.offloads(DIT_COMPONENT):
                raise ValueError(
                    "None of the selected distributed layerwise offload components have "
                    "a model-declared streamable or on-demand plan"
                )
            return

        # Unified allocation: 2 shared output buffers + 2 shared shard buffers
        # sized to the max block across ALL module groups (gen_layers +
        # language_model).  Groups execute sequentially, so 2 buffers suffice.
        unified_buffers = self._allocate_shared_buffers(all_hooks)
        allgather_hooks = [hook for hook in all_hooks if hook.dp_size > 1]
        mmap_hooks = [hook for hook in all_hooks if hook.rank_local_mmap]
        unified_shard_buffers = self._allocate_shared_shard_buffers(allgather_hooks) if allgather_hooks else None
        unified_cpu_staging = None
        cpu_staging_events = None
        if self._using_rank_local_mmap and not self._using_registered_mmap:
            unified_cpu_staging = self._allocate_shared_cpu_staging_buffers(
                mmap_hooks,
                self._resident_layer_group,
            )
            cpu_staging_events = [None, None]
            if self._resident_layer_group is not None:
                # Resident and streamed layers execute in the same stage and
                # reuse the same host slots. Events serialize slot reuse.
                self._resident_layer_group._cpu_staging_buffers = [
                    buffers for buffers in unified_cpu_staging if buffers is not None
                ]
                self._resident_layer_group._cpu_staging_events = cpu_staging_events

        # Shared slot-group tracker: _shared_slot_group[slot] = group_id
        # that last wrote to that slot.  Group-first hooks use this to
        # skip sync-prefetch when the slot still contains their own data.
        shared_slot_group = [-1, -1]

        for group_idx, group in enumerate(self._all_hook_groups):
            for hook in group:
                hook.gpu_buffers = unified_buffers
                hook._owns_buffers = False
                if unified_shard_buffers is not None and hook.dp_size > 1:
                    hook.gpu_shard_buffers = unified_shard_buffers
                if hook.rank_local_mmap and unified_cpu_staging is not None and cpu_staging_events is not None:
                    hook.cpu_staging_buffers = unified_cpu_staging
                    hook.cpu_staging_events = cpu_staging_events
                hook._group_id = group_idx
                hook._shared_slot_group = shared_slot_group

        # Defer every group's first prefetch until its first forward. DP ranks
        # may intentionally own different rank-local groups (for example only
        # rank 0 owns the text encoder) while sharing DiT AllGather groups.
        # Entering a collective for each rank's locally first group here would
        # give the ranks different collective orders and deadlock startup. The
        # group-first hook treats the initially unowned slot as contaminated
        # and performs the required synchronous prefetch on first use.

        total_blocks = sum(len(b) for b in self._blocks)
        transfer_summary = ", ".join(
            f"{component}: {self.config.transfer_for(component).value}"
            for component in sorted(self.config.components or {DIT_COMPONENT})
        )
        logger.info(
            f"Distributed layer-wise offloading enabled on {total_blocks} blocks "
            f"across {len(self._all_hook_groups)} group(s), "
            f"transfers={{{transfer_summary}}}, "
            f"unified shared_buffers=2"
        )

        self.enabled = True

        if self._using_mmap and not self._using_rank_local_mmap:
            # AllGather mode copied each rank's persistent shard, so the source
            # mappings are no longer needed. Rank-local mode retains them as
            # the node-shared host master until disable().
            self._release_mmap_handles()

        self._cleanup_after_loading()

    def _release_mmap_handles(self) -> None:
        """Release source handles and the transport-owned HWR lease."""
        if self._host_registration is not None:
            raise HostRegistrationCleanupError("cannot close HWR mappings while host registration is still active")
        self._mmap_transforms_by_tensor_id.clear()
        if hasattr(self, "_mmap_file_cache"):
            self._mmap_file_cache.clear()
            del self._mmap_file_cache
            logger.info("Released safetensors mmap file handles")
        lease = self._host_weight_lease
        self._host_weight_lease = None
        if lease is not None and not lease.closed:
            lease.close()
            logger.info("Released Host Weight Runtime lease %s", lease.provenance.resolution_id)
        if self.host_weight_plan is not None and self.host_weight_plan.lease_carrier is not None:
            self.host_weight_plan.lease_carrier.close()

    def _cleanup_after_loading(self) -> None:
        """Synchronize and release freed device/CPU memory after sharding."""
        current_omni_platform.synchronize()
        current_omni_platform.empty_cache()
        import ctypes as _ctypes
        import gc as _gc

        _gc.collect()
        try:
            _ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def _disable(self, *, restore_allgather_weights: bool) -> None:
        has_open_lease = self._host_weight_lease is not None and not self._host_weight_lease.closed
        has_registration = self._host_registration is not None
        has_carrier = (
            self.host_weight_plan is not None
            and self.host_weight_plan.lease_carrier is not None
            and not self.host_weight_plan.lease_carrier.closed
        )
        has_partial_hooks = bool(
            self._blocks
            or self._all_hook_groups
            or self._encoder_modules
            or self._staged_components
            or self._resident_layer_group is not None
            or self._residency_pipeline_ref is not None
        )
        if (
            not self.enabled
            and not hasattr(self, "_mmap_file_cache")
            and not has_open_lease
            and not has_registration
            and not has_carrier
            and not has_partial_hooks
        ):
            return

        self._clear_residency_controller()

        # A hook can leave the circular tail prefetch queued after the final
        # forward. Drain every transport before releasing hook-owned host or
        # device buffers, including the ordinary rank-local path.
        sync_error = run_cleanup_steps([("synchronizing pending DLO transfers", current_omni_platform.synchronize)])

        unique_hooks: list[DistributedLayerwiseOffloadHook] = []
        seen_blocks: set[int] = set()
        for hook in chain.from_iterable(self._all_hook_groups):
            block_id = id(hook.next_block)
            if block_id not in seen_blocks:
                seen_blocks.add(block_id)
                unique_hooks.append(hook)

        allgather_hooks = [hook for hook in unique_hooks if hook.dp_size > 1]
        rank_local_hooks = [hook for hook in unique_hooks if hook.dp_size <= 1]
        skipped_allgather = bool(allgather_hooks) and not restore_allgather_weights
        if skipped_allgather:
            # Startup rollback cannot safely enter a collective that a failed
            # peer may never reach. Those blocks cannot be reconstructed, so
            # make accidental reuse explicit instead of accepting zero weights.
            self._poisoned_reason = (
                "Distributed layerwise offload startup skipped AllGather weight restoration; "
                "recreate the backend and reload the pipeline before retrying"
            )

        collective_error = None
        if restore_allgather_weights:
            # Run collective-bearing restores before any rank-local operation:
            # a local failure must never keep this rank out of a later AllGather.
            collective_error = run_cleanup_steps(
                ("restoring an AllGather block", hook.restore_next_block_to_cpu) for hook in allgather_hooks
            )
        rank_local_error = run_cleanup_steps(
            ("restoring a rank-local block", hook.restore_next_block_to_cpu) for hook in rank_local_hooks
        )
        removal_error = run_cleanup_steps(
            (
                "removing a distributed block hook",
                lambda block=block: remove_distributed_block_hook(block),
            )
            for blocks in self._blocks
            for block in blocks
        )
        encoder_error = run_cleanup_steps(
            (
                "clearing distributed encoder state",
                lambda module=module: clear_encoder_layerwise_state(module),
            )
            for module in self._encoder_modules
        )

        resident_error = None
        if self._resident_layer_group is not None:
            # Resident layers are always rank-local (the public parser rejects
            # resident DiT layers combined with AllGather).
            resident_error = run_cleanup_steps(
                [("restoring resident DLO blocks", self._resident_layer_group.restore_to_cpu)]
            )

        lifecycle_error = next(
            (
                error
                for error in (
                    sync_error,
                    collective_error,
                    rank_local_error,
                    removal_error,
                    encoder_error,
                    resident_error,
                )
                if error is not None
            ),
            None,
        )

        # Unregistration is independent once streams have drained, but the
        # lease/file mappings must stay alive if restoration or hook removal
        # needs a retry.
        registration_error = None
        if sync_error is None:
            registration_error = run_cleanup_steps(
                [("releasing registered HWR mappings", self._release_registered_mmap)]
            )
        cleanup_error = lifecycle_error or registration_error
        if cleanup_error is not None:
            raise cleanup_error

        release_error = run_cleanup_steps([("releasing DLO mmap handles", self._release_mmap_handles)])

        self._blocks.clear()
        self._all_hook_groups.clear()
        self._resident_blocks.clear()
        self._resident_layer_group = None
        self._encoder_modules.clear()
        self._staged_components.clear()
        # Loader plans are single-use (HWR carriers in particular). A clean
        # disable can rebuild from restored CPU tensors; a poisoned backend is
        # guarded above and must be recreated.
        self.host_weight_plan = None
        self._using_mmap = False
        self._using_rank_local_mmap = False
        self._using_registered_mmap = False
        self.enabled = False
        logger.info("Distributed layer-wise offloading disabled")
        if release_error is not None:
            raise release_error

    def disable(self) -> None:
        self._disable(restore_allgather_weights=True)

    @staticmethod
    def _allocate_shared_buffers(
        hooks: list[DistributedLayerwiseOffloadHook],
    ) -> list[dict[torch.dtype, torch.Tensor] | None]:
        """Allocate exactly 2 shared device buffers sized to the largest block.

        All hooks share these 2 buffers. At any time, slot 0 holds the current
        layer's weights and slot 1 holds the next layer's weights (or vice
        versa). This ensures only 2 layers' worth of weights reside on device,
        regardless of the total number of blocks.
        """
        max_sizes: dict[torch.dtype, int] = {}
        for hook in hooks:
            dp = hook.dp_size
            for dtype, metas in hook.metadata.items():
                total = sum(m["numel"] for m in metas)
                # AllGather output = dp * ceil(total/dp) (padded for equal shards)
                if dp > 1:
                    total = ((total + dp - 1) // dp) * dp
                if dtype not in max_sizes or total > max_sizes[dtype]:
                    max_sizes[dtype] = total

        device = hooks[0].device
        shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]
        for slot in range(2):
            gpu_weights: dict[torch.dtype, torch.Tensor] = {}
            for dtype, total_numel in max_sizes.items():
                gpu_weights[dtype] = torch.empty(total_numel, dtype=dtype, device=device)
            shared_buffers[slot] = gpu_weights

        logger.info(
            "Allocated 2 shared device buffers (max block size: %s)",
            {str(k): f"{v * _dtype_size(k) / 1024 / 1024:.1f}MB" for k, v in max_sizes.items()},
        )
        return shared_buffers

    @staticmethod
    def _allocate_shared_cpu_staging_buffers(
        hooks: list[DistributedLayerwiseOffloadHook],
        resident_group: PinnedResidentLayerGroup | None = None,
    ) -> list[dict[torch.dtype, torch.Tensor] | None]:
        """Allocate two bounded host slots for rank-local mmap -> device copies."""
        max_sizes: dict[torch.dtype, int] = {}
        for hook in hooks:
            for dtype, metas in hook.metadata.items():
                total = sum(meta["numel"] for meta in metas)
                max_sizes[dtype] = max(max_sizes.get(dtype, 0), total)
        if resident_group is not None:
            for state in resident_group._states:
                for dtype, metas in state["metadata"].items():
                    total = sum(meta["numel"] for meta in metas)
                    max_sizes[dtype] = max(max_sizes.get(dtype, 0), total)

        pin_memory = hooks[0].pin_memory if hooks else bool(resident_group and resident_group.pin_memory)
        shared_staging: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]
        for slot in range(2):
            buffers: dict[torch.dtype, torch.Tensor] = {}
            for dtype, total_numel in max_sizes.items():
                buffer = torch.empty(
                    total_numel,
                    dtype=dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                buffers[dtype] = buffer
            shared_staging[slot] = buffers

        logger.info(
            "Allocated 2 shared host staging buffers for rank-local mmap (max block size: %s, pinned=%s)",
            {str(k): f"{v * _dtype_size(k) / 1024 / 1024:.1f}MB" for k, v in max_sizes.items()},
            pin_memory,
        )
        return shared_staging

    @staticmethod
    def _allocate_shared_shard_buffers(
        hooks: list[DistributedLayerwiseOffloadHook],
    ) -> list[dict[torch.dtype, torch.Tensor] | None]:
        """Allocate 2 shared shard (AllGather input) buffers sized to the
        largest per-rank shard.

        All hooks share these 2 input buffers — the same device address is
        reused for every layer's AllGather so HCCL can reuse its internal
        communication buffers (preventing rank-dependent HBM imbalance).
        Cost: 2 × max_shard_size (~184 MB) instead of 2 × N_hooks × shard.
        """
        max_shard_sizes: dict[torch.dtype, int] = {}
        for hook in hooks:
            for dtype, shard in hook.cpu_shards.items():
                numel = shard.numel()
                if dtype not in max_shard_sizes or numel > max_shard_sizes[dtype]:
                    max_shard_sizes[dtype] = numel

        device = hooks[0].device
        shared_shard_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]
        for slot in range(2):
            shard_bufs: dict[torch.dtype, torch.Tensor] = {}
            for dtype, numel in max_shard_sizes.items():
                shard_bufs[dtype] = torch.empty(numel, dtype=dtype, device=device)
            shared_shard_buffers[slot] = shard_bufs

        logger.info(
            "Allocated 2 shared shard buffers (max shard size: %s)",
            {str(k): f"{v * _dtype_size(k) / 1024 / 1024:.1f}MB" for k, v in max_shard_sizes.items()},
        )
        return shared_shard_buffers
