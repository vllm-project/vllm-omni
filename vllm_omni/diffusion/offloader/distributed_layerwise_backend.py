# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed Layerwise Offload backend with chunked H2D + AllGather overlap.

This module implements the RFC-1 "Distributed Layerwise Offload" mechanism that:

* Optionally shards model weights across the weight-shard (fully-shard) ranks
  and reconstructs each block with chunked AllGather, or streams a complete
  rank-local block without a collective.
* Consumes the loader-owned host-weight plan (checkpoint mmap views) when the
  loader skipped ordinary weight materialization.  The sharded AllGather path
  (weight_shard_size > 1) packs each block into pinned (or pageable fallback)
  private Host shards; the rank-local path (weight_shard_size == 1) instead
  retains the checkpoint mmap views as the node-shared host backing and
  stages one block at a time through two bounded pinned staging slots.
* Uses a fixed double-buffer scheme that keeps only two layers' worth of
  weights on each device at any time.
* Pipelines chunked H2D transfers and, when enabled, AllGather communications
  on dedicated streams, overlapping them with computation.
* Is hardware-agnostic, supporting both NVIDIA GPU (CUDA) and Ascend NPU
  (CANN) platforms via vLLM-Omni's platform abstraction layer.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
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

from .base import OffloadBackend, OffloadConfig
from .block_discovery import ChunkOwnedBlock, ChunkOwnership, get_blocks_from_dit
from .chunked_transport import (
    ChunkedWeightTransport,
    PartManifest,
    PinBudget,
    PinFailurePolicy,
    TransferTicket,
    TransportBackendKind,
    WeightLayout,
    build_part_manifest,
    is_chunk_transport_supported,
    pack_local_shard,
)
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
    dtype_size as _dtype_size,
)
from .tensor_utils import (
    is_materialized_tensor,
    make_offload_placeholder,
    set_tensor_storage,
)
from .weight_transport_backend import (
    ChunkCompletion,
    ChunkEvents,
    TransportCapability,
    TransportSelection,
    TransportStreams,
    WeightTransportBackend,
    create_transport_backend,
    select_transport,
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

# Trace markers consumed by benchmarks/diffusion/extract_dlo_timeline.py.
# H2D/AllGather ranges execute on the submit worker; ``submit`` is emitted on
# the profiled main thread so every asynchronous submission remains visible
# even when PyTorch's thread-local record_function state does not propagate.
_TRACE_ENV = "VLLM_OMNI_DLO_TRACE"


def _trace_marker(kind: str, block_id: int, chunk_id: int | None = None) -> str:
    if chunk_id is None:
        return f"dlo.{kind}.block_{block_id}"
    return f"dlo.{kind}.block_{block_id}.chunk_{chunk_id}"


def _prefetch_barrier() -> None:
    """Sentinel queued on the single submit executor to drain prior work."""


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
        weight_shard_group: torch.distributed.ProcessGroup | None,
        weight_shard_size: int,
        weight_shard_rank: int,
        copy_stream: Any | None = None,
        comm_stream: Any | None = None,
        shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] | None = None,
        shared_chunk_slot_events: list[Any | None] | None = None,
        shared_output_slot_events: list[Any | None] | None = None,
        shared_slot_owners: list[DistributedLayerwiseOffloadHook | None] | None = None,
        prepared_host_part: dict[str, Any] | None = None,
        h2d_done_events: list[Any] | None = None,
        transport_done_events: list[Any] | None = None,
        output_ready_events: list[Any] | None = None,
        last_use_events: list[Any | None] | None = None,
        prefetch_executor: concurrent.futures.Executor | None = None,
        rank_local_mmap: bool = False,
        pin_memory: bool = True,
        tensor_transforms: dict[int, Any] | None = None,
        data_transport_backend: WeightTransportBackend | None = None,
    ):
        assert isinstance(next_block, nn.Module), "transformer block must be type `torch.nn.Module`"

        self.next_block = next_block
        self.device = device
        self.weight_shard_group = weight_shard_group
        self.weight_shard_size = weight_shard_size
        self.weight_shard_rank = weight_shard_rank

        self.data_transport_backend = data_transport_backend
        self.registered_mmap = False

        self.copy_stream = copy_stream or current_omni_platform.Stream()
        self.comm_stream = comm_stream or current_omni_platform.Stream()

        # Host storage mode.  The sharded AllGather path receives a
        # backend-prepared Host part (chunk manifest + packed private shard).
        # The rank-local path (weight_shard_size == 1) collects host storage
        # itself in initialize_hook: retained checkpoint mmap views when the
        # loader supplied a checkpoint_mmap plan, otherwise a private pinned
        # copy via _shard_and_pin (the standard loader path).
        self.rank_local = prepared_host_part is None
        if not self.rank_local and data_transport_backend is None:
            # The sharded path drives every transfer through the configured
            # transport backend; rank-local hooks never touch it.
            raise RuntimeError("data_transport_backend is required")
        self.rank_local_mmap = rank_local_mmap
        self.pin_memory = pin_memory
        self.tensor_transforms = tensor_transforms or {}
        if self.rank_local and weight_shard_size > 1:
            raise RuntimeError("rank-local host storage requires weight_shard_size == 1")
        if not self.rank_local and rank_local_mmap:
            raise RuntimeError("rank_local_mmap is only valid without a prepared Host part")

        if not self.rank_local:
            # Backend-prepared Host storage for the block this hook transports.
            self.cpu_shards: dict[torch.dtype, torch.Tensor] = prepared_host_part["cpu_shards"]
            self.metadata: dict[torch.dtype, list[dict[str, Any]]] = prepared_host_part["metadata"]
            self.manifest: PartManifest | None = prepared_host_part["manifest"]
            self.fallback_reason: str | None = prepared_host_part.get("fallback_reason")

            # The manifest is the canonical source of the block id.  The backend
            # always populates it; the block_id and the memory-address fallback
            # below are unreachable in production and would produce wrong ids.
            self.block_id: int = self.manifest.block_id
            self.transport: ChunkedWeightTransport | None = ChunkedWeightTransport(self.block_id, slot_count=2)
            self.transport.prepare(self.manifest, self.cpu_shards)
            self.transport_state = self.transport.state
        else:
            # Populated by initialize_hook (mmap sources or a private copy).
            self.cpu_shards = {}
            self.metadata = {}
            self.manifest = None
            self.fallback_reason = None
            # The backend overwrites block_id with the ownership id right
            # after construction; the id() fallback only affects trace
            # markers in standalone (test) constructions.
            self.block_id = id(next_block)
            self.transport = None
            self.transport_state = None

        # Block id used for the 'compute' marker, i.e. the module this hook is
        # attached to rather than the block it transports.  The backend
        # overwrites it right after construction.
        self.compute_block_id = self.block_id

        # File-backed source tensors for rank-local mmap.  Unlike cpu_shards,
        # these remain immutable views of the checkpoint and are never pinned
        # or flattened into a model-sized private allocation.
        self.cpu_sources: dict[torch.dtype, list[dict[str, Any]]] = {}
        # Rank-local mmap uses two host staging slots shared by every hook in
        # this worker.  They are assigned by the backend after all block sizes
        # are known, mirroring the shared device-buffer allocation.
        self.cpu_staging_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]
        self.cpu_staging_events: list[Any | None] = [None, None]

        # Double buffers: either shared (from backend) or self-allocated (lazy)
        if shared_buffers is not None:
            self.gpu_buffers: list[dict[torch.dtype, torch.Tensor] | None] = shared_buffers
            self._owns_buffers = False
        else:
            self.gpu_buffers = [None, None]
            self._owns_buffers = True
        # Local chunk (AllGather input) buffers, indexed by *input* slot.
        self.gpu_shard_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]

        self.ready_events: list[Any | None] = [None, None]
        self.ready_tickets: list[TransferTicket | None] = [None, None]

        # Pre-allocated per-output-slot completion events.  Allocating them
        # once keeps prefetch_layer free of Event construction.
        self._output_ready_events: list[Any] = output_ready_events or [current_omni_platform.Event() for _ in range(2)]
        # Per local-chunk input slot: H2D completion and AllGather completion.
        self._h2d_done_events: list[Any] = h2d_done_events or [current_omni_platform.Event() for _ in range(2)]
        self._transport_done_events: list[Any] = transport_done_events or [
            current_omni_platform.Event() for _ in range(2)
        ]
        # Recorded on the main thread immediately before async submit.  The
        # worker waits on this event instead of calling wait_stream() against
        # a compute stream that the main thread is concurrently appending to.
        self._submit_dependency_event = current_omni_platform.Event()
        # Shared across the hook group: last-use event per output slot, and
        # the event proving the prior AllGather finished reading an input slot.
        self._output_slot_events: list[Any | None] = (
            shared_output_slot_events if shared_output_slot_events is not None else [None, None]
        )
        self._chunk_slot_events: list[Any | None] = (
            shared_chunk_slot_events if shared_chunk_slot_events is not None else [None, None]
        )
        self._last_use_events: list[Any | None] = last_use_events if last_use_events is not None else [None, None]
        self._shared_slot_owners: list[DistributedLayerwiseOffloadHook | None] = (
            shared_slot_owners if shared_slot_owners is not None else [None, None]
        )

        self._request_generation = 0

        # Current slot index (0 or 1).  Updated dynamically by the previous
        # hook's prefetch_layer call via _prefetched_slot.  This ensures
        # correct slot tracking for ALL block counts (including odd N).
        self.current_slot = 0
        self._prefetched_slot: int | None = None

        # Async submit: the group's single-worker executor enqueues stream
        # work for the next block while this block's compute is launched.
        self._prefetch_executor = prefetch_executor
        self._prefetch_future: concurrent.futures.Future | None = None

        self.trace_enabled = os.environ.get(_TRACE_ENV, "0") not in ("", "0", "false", "False")
        self._compute_trace: Any | None = None

        # Backward link to previous hook for fallback (cache-dit skip)
        self._prev_hook: DistributedLayerwiseOffloadHook | None = None

        # Marks the first hook in a shared-buffer group.  When multiple DiT
        # groups share the same 2 GPU buffers, another group may have
        # overwritten this group's slot between forwards.  The first block
        # must sync-prefetch on entry to ensure it loads the correct
        # weights, even if is_materialized sees a non-empty tensor left by
        # the other group.
        self._is_group_first: bool = False
        # Marks the hook that transports the group's first block, i.e. the one
        # whose submit crosses the group boundary.
        self._is_group_tail: bool = False

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
        self._transported_names = {meta["name"] for metas in self.metadata.values() for meta in metas}

        # Per-block synchronization primitive: set after H2D copy completes.
        self._prefetch_done: Any | None = None

        # Pending async AllGather work (prevent GC before completion).
        self._pending_work: Any | None = None
        self._cached_repoint: list | None = None

    # ------------------------------------------------------------------ #
    #  DTensor helpers (shared with LayerwiseOffloadHook)                 #
    # ------------------------------------------------------------------ #

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        module = super().initialize_hook(module)

        self.block_parameters = dict(module.named_parameters())
        self.block_buffers = dict(module.named_buffers())

        self.next_block_parameters = dict(self.next_block.named_parameters())
        self.next_block_buffers = dict(self.next_block.named_buffers())

        if self.rank_local:
            if self.rank_local_mmap:
                # Retain the checkpoint mmap views as the host backing; the
                # block is staged through the shared bounded slots at prefetch.
                self.cpu_sources, self.metadata = self._collect_mmap_sources(
                    self.next_block_parameters,
                    self.next_block_buffers,
                    self.tensor_transforms,
                )
            else:
                # Ordinary loader tensors: private host copy, H2D only.
                self.cpu_shards, self.metadata = self._shard_and_pin(
                    self.next_block_parameters,
                    self.next_block_buffers,
                    dp_size=1,
                    rank=0,
                    pin_memory=self.pin_memory,
                    tensor_transforms=self.tensor_transforms,
                )
        # else: host storage was packed by the backend (_prepare_host_storage)
        # before this hook was constructed; nothing to shard here.
        self._transported_names = {meta["name"] for metas in self.metadata.values() for meta in metas}

        # Allocate device buffers only if not using shared buffers from backend
        if self._owns_buffers:
            self._allocate_device_buffers()

        # Cache parameter re-pointing metadata to avoid per-layer dict lookups.
        self._cached_repoint = []
        for _slot in range(2):
            repoint = []
            for dtype, metas in self.metadata.items():
                for m in metas:
                    target = (
                        self.next_block_parameters[m["name"]]
                        if m["name"] in self.next_block_parameters
                        else self.next_block_buffers[m["name"]]
                    )
                    repoint.append(
                        (
                            target,
                            dtype,
                            m["offset"],
                            m["numel"],
                            m["shape"],
                            m.get("stride"),
                        )
                    )
            self._cached_repoint.append(repoint)

        return module

    @staticmethod
    def _collect_mmap_sources(
        params: dict[str, nn.Parameter],
        bufs: dict[str, torch.Tensor],
        tensor_transforms: dict[int, Any] | None = None,
    ) -> tuple[dict[torch.dtype, list[dict[str, Any]]], dict[torch.dtype, list[dict[str, Any]]]]:
        """Retain file-backed tensors and replace module storage with placeholders.

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

            # The detached source above keeps the mmap storage alive while the
            # module parameter/buffer is rebound to the rotating device slot.
            set_tensor_storage(target, make_offload_placeholder(target))

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
        dtype_grouped: dict[torch.dtype, dict[str, torch.Tensor]] = {}
        dtype_metadata: dict[torch.dtype, list[dict[str, Any]]] = {}

        for name, param_or_buf in chain(params.items(), bufs.items()):
            dtype = param_or_buf.dtype
            if dtype not in dtype_grouped:
                dtype_grouped[dtype] = {}
            dtype_grouped[dtype][name] = param_or_buf

        cpu_shards: dict[torch.dtype, torch.Tensor] = {}

        for dtype, name2weights in dtype_grouped.items():
            # Resolve local tensors (handle DTensor via to_local)
            weights_with_local = []
            for name, t in name2weights.items():
                local_t = t.to_local() if hasattr(t, "to_local") else t
                mmap_transform = (tensor_transforms or {}).get(id(t))
                if callable(mmap_transform):
                    # Some checkpoints use a layout that is converted by the
                    # regular weight loader (for example MiniMax-H3 grouped
                    # QKV).  Apply that conversion one block at a time while
                    # copying the rank-local CPU shard.  Keeping the raw
                    # parameter as an mmap view avoids a private full-model
                    # copy in every worker.
                    local_t = mmap_transform(local_t)
                stride = local_t.stride()
                storage_numel = (
                    0
                    if local_t.numel() == 0
                    else 1 + sum((size - 1) * axis_stride for size, axis_stride in zip(local_t.shape, stride))
                )
                weights_with_local.append((name, t, local_t, storage_numel, stride))

            total_numel = sum(storage_numel for _, _, _, storage_numel, _ in weights_with_local)

            # Equal-sized shards (ceil division) for all_gather_into_tensor
            shard_size = (total_numel + dp_size - 1) // dp_size  # ceil
            shard_start = rank * shard_size
            shard_end = min(shard_start + shard_size, total_numel)

            # Allocate ONLY the shard (1/dp_size), zero-padded to ceil.
            # Avoids materialising the full block on CPU.
            shard = torch.zeros(shard_size, dtype=dtype, device="cpu")

            current_offset = 0
            for (
                name,
                original_tensor,
                local_tensor,
                storage_numel,
                stride,
            ) in weights_with_local:
                if dtype not in dtype_metadata:
                    dtype_metadata[dtype] = []
                # Offsets remain relative to the FULL flattened buffer
                # (needed for correct AllGather reconstruction).
                dtype_metadata[dtype].append(
                    {
                        "name": name,
                        "offset": current_offset,
                        "numel": storage_numel,
                        "shape": local_tensor.shape,
                        "stride": stride,
                    }
                )

                # Copy ONLY the portion within [shard_start, shard_end)
                overlap_start = max(current_offset, shard_start)
                overlap_end = min(current_offset + storage_numel, shard_end)
                if overlap_start < overlap_end:
                    if local_tensor.is_contiguous():
                        flat_storage = local_tensor.flatten()
                    else:
                        # Online FP8 stores Cutlass weights as transposed views
                        # (e.g. stride=(1, K)). Flattening such a tensor in
                        # logical order and later rebuilding it with .view()
                        # changes its layout and makes scaled_mm reject it.
                        # Pack the physical storage order and preserve the
                        # original stride for zero-copy reconstruction.
                        flat_storage = torch.zeros(
                            storage_numel,
                            dtype=dtype,
                            device=local_tensor.device,
                        )
                        physical_view = torch.as_strided(
                            flat_storage,
                            size=local_tensor.shape,
                            stride=stride,
                        )
                        physical_view.copy_(local_tensor)
                    src_start = overlap_start - current_offset
                    src_end = overlap_end - current_offset
                    dst_start = overlap_start - shard_start
                    dst_end = overlap_end - shard_start
                    shard[dst_start:dst_end].copy_(flat_storage[src_start:src_end])

                # Replace original tensor with placeholder (frees CPU storage)
                set_tensor_storage(
                    original_tensor,
                    make_offload_placeholder(original_tensor),
                )
                current_offset += storage_numel

            if pin_memory:
                shard = shard.pin_memory()

            cpu_shards[dtype] = shard

        return cpu_shards, dtype_metadata

    def _allocate_device_buffers(self) -> None:
        """Pre-allocate exactly two device buffers (one per slot).

        In prepared mode the manifest's ``padded_numel`` is the AllGather
        output size, i.e. the sum of every chunk's padded extent, so the same
        buffer serves both the chunk-major and the whole-block fallback
        layout.  In rank-local mode there is no collective, so the buffers
        are sized to the block's exact flattened extent.
        """
        for slot in range(2):
            gpu_weights: dict[torch.dtype, torch.Tensor] = {}
            if self.manifest is not None:
                for dtype_manifest in self.manifest.dtypes:
                    gpu_weights[dtype_manifest.dtype] = torch.empty(
                        dtype_manifest.padded_numel,
                        dtype=dtype_manifest.dtype,
                        device=self.device,
                    )
            else:
                for dtype, metas in self.metadata.items():
                    total_numel = sum(m["numel"] for m in metas)
                    gpu_weights[dtype] = torch.empty(
                        total_numel,
                        dtype=dtype,
                        device=self.device,
                    )
            self.gpu_buffers[slot] = gpu_weights

    @property
    def is_materialized(self) -> bool:
        """Check whether every transported tensor for this block is materialized."""
        producer = self._prev_hook
        names = producer._transported_names if producer is not None else set(self.block_parameters)
        for name in names:
            target = self.block_parameters.get(name, self.block_buffers.get(name))
            if target is not None and not is_materialized_tensor(target):
                return False
        return True

    # ------------------------------------------------------------------ #
    #  Tracing                                                            #
    # ------------------------------------------------------------------ #

    def _trace_range(self, kind: str, chunk_id: int | None = None, block_id: int | None = None) -> Any:
        """Return a profiler range whose name the DLO timeline analyzer parses.

        No-op unless ``VLLM_OMNI_DLO_TRACE`` is set.  The marker text must stay
        in the ``dlo.<kind>.block_<id>[.chunk_<id>]`` shape consumed by
        ``benchmarks/diffusion/extract_dlo_timeline.py``.
        """
        if not self.trace_enabled:
            return contextlib.nullcontext()
        marker_block = self.block_id if block_id is None else block_id
        return torch.profiler.record_function(_trace_marker(kind, marker_block, chunk_id))

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
    def _submit_prefetch(
        self,
        slot: int,
        non_blocking: bool = True,
        submit_after_event: Any | None = None,
    ) -> bool:
        """Enqueue the chunked H2D + AllGather pipeline for *slot*.

        This only enqueues stream work and touches no Python-level Parameter
        storage, so it is safe to run on the group's prefetch worker thread.
        ``_apply_repoint`` performs the Parameter re-pointing on the main
        thread.  Returns True when a new submission was made.

        ``submit_after_event`` is recorded on the MAIN thread before the
        current block's compute kernels are launched.  Waiting on that fixed
        boundary lets the next H2D overlap current compute without mutating
        the compute stream from this worker thread.
        """
        if self.fallback_reason is not None:
            # Pageable Host memory: an async copy would race with the packer.
            non_blocking = False

        if submit_after_event is None:
            raise RuntimeError("prefetch submit requires a main-thread dependency event")
        self.copy_stream.wait_event(submit_after_event)

        gpu_weights = self.gpu_buffers[slot]
        assert gpu_weights is not None, f"gpu_buffers[{slot}] not allocated"

        previous_ticket = self.ready_tickets[slot]
        if previous_ticket is not None and self.transport_state.is_current(previous_ticket):
            # Idempotent re-prefetch: the slot already owns a live submission.
            return False

        previous_owner = self._shared_slot_owners[slot]
        if previous_owner is not None and previous_owner is not self:
            overwritten_ticket = previous_owner.ready_tickets[slot]
            if overwritten_ticket is not None and previous_owner.transport_state.is_current(overwritten_ticket):
                previous_owner.transport.record_last_use(
                    overwritten_ticket,
                    self._output_slot_events[slot],
                )
                previous_owner.ready_tickets[slot] = None
                previous_owner.ready_events[slot] = None

        evt = self._output_ready_events[slot]
        ticket = None
        if not self.rank_local:
            ticket = self.transport.begin_submission(
                output_slot=slot,
                request_generation=self._request_generation,
                ready_event=evt,
                part_id=self.manifest.part_id,
                last_collective_key=(
                    self._request_generation,
                    self.block_id,
                    slot,
                    self.manifest.digest[:16],
                ),
            )
            self.ready_tickets[slot] = ticket

        if self.rank_local:
            # Rank-local: no collective, one whole-block H2D.  Rank-local mmap
            # first stages the block through the bounded host slot; the plain
            # rank-local path copies its private host shard directly.
            last_use = self._output_slot_events[slot]
            if last_use is not None:
                self.copy_stream.wait_event(last_use)
            with current_omni_platform.stream(self.copy_stream):
                with self._trace_range("h2d"):
                    if self.rank_local_mmap and self.registered_mmap:
                        self._copy_mmap_sources_to_device(
                            self.cpu_sources,
                            self.metadata,
                            gpu_weights,
                            non_blocking=non_blocking,
                        )
                    else:
                        cpu_weights = self._stage_mmap_sources(slot) if self.rank_local_mmap else self.cpu_shards
                        for dtype, cpu_shard in cpu_weights.items():
                            gw = gpu_weights[dtype]
                            async_copy = non_blocking and cpu_shard.is_pinned()
                            gw[: cpu_shard.numel()].copy_(cpu_shard, non_blocking=async_copy)
                evt.record(self.copy_stream)
            if self.rank_local_mmap:
                # The CPU staging slot may be overwritten only after this H2D
                # copy has finished.  The shared event serializes slot reuse
                # by the next hook that stages into the same slot.
                self.cpu_staging_events[slot] = evt
        elif self.weight_shard_size <= 1 or self.weight_shard_group is None:
            last_use = self._output_slot_events[slot]
            if last_use is not None:
                self.copy_stream.wait_event(last_use)
            with current_omni_platform.stream(self.copy_stream):
                with self._trace_range("h2d"):
                    for dtype, cpu_shard in self.cpu_shards.items():
                        gw = gpu_weights[dtype]
                        gw[: cpu_shard.numel()].copy_(cpu_shard, non_blocking=non_blocking)
                evt.record(self.copy_stream)
        else:
            if non_blocking:
                for dtype, cpu_source in self.cpu_shards.items():
                    if cpu_source.numel() and not cpu_source.is_pinned():
                        raise RuntimeError(
                            f"chunk H2D requires pinned Host source for block {self.block_id} "
                            f"dtype={dtype}; tensor is not pinned. Async overlap is unsafe "
                            "with pageable source — set dlo_pin_failure_policy=whole_block_fallback "
                            "to allow pageable fallback, or ensure pin_cpu_memory=True."
                        )

            streams = TransportStreams(copy=self.copy_stream, communication=self.comm_stream)
            self.data_transport_backend.begin_part(streams, self._output_slot_events[slot])
            completions: list[ChunkCompletion] = []
            shard_bufs = self.gpu_shard_buffers

            chunk_specs = [
                (dtype_manifest.dtype, chunk)
                for dtype_manifest in self.manifest.dtypes
                for chunk in dtype_manifest.chunks
            ]
            for transfer_index, (dtype, chunk) in enumerate(chunk_specs):
                input_slot = transfer_index % 2
                cpu_source = self.cpu_shards[dtype]
                source = cpu_source[chunk.cpu_offset : chunk.cpu_offset + chunk.local_numel]

                local_input = None
                if self.data_transport_backend.requires_local_input:
                    if len(shard_bufs) != 2 or shard_bufs[0] is None or shard_bufs[1] is None:
                        raise RuntimeError("selected transport backend requires two local chunk input buffers")
                    local_input = shard_bufs[input_slot][dtype][: chunk.local_numel]

                completion = self.data_transport_backend.submit_chunk(
                    source=source,
                    local_input=local_input,
                    full_output=gpu_weights[dtype][chunk.full_offset : chunk.full_offset + chunk.padded_numel],
                    chunk_meta=chunk,
                    streams=streams,
                    events=ChunkEvents(
                        h2d_done=self._h2d_done_events[input_slot],
                        transport_done=self._transport_done_events[input_slot],
                        input_reusable=self._chunk_slot_events[input_slot],
                    ),
                    group=self.weight_shard_group,
                    generation=self._request_generation,
                    non_blocking=non_blocking,
                    trace=lambda kind, chunk_id=chunk.chunk_id: self._trace_range(kind, chunk_id),
                )
                completions.append(completion)
                if self.data_transport_backend.requires_local_input:
                    self._chunk_slot_events[input_slot] = completion.event

            self.data_transport_backend.finalize_part(
                completions,
                ready_event=evt,
                streams=streams,
            )

        self.ready_events[slot] = evt
        self._prefetch_done = evt
        if ticket is not None:
            self.transport.mark_ready(ticket)
        self._prefetched_slot = slot
        if not self.rank_local:
            self._shared_slot_owners[slot] = self

        # Stamp the slot with this hook's group ID so that group-first
        # hooks can detect whether another group has overwritten the slot.
        if self._shared_slot_group is not None:
            self._shared_slot_group[slot] = self._group_id
        return True

    @torch.compiler.disable
    def _apply_repoint(self, slot: int) -> None:
        """Point the next block's Parameters at the slot's device buffer.

        This mutates Python-level Parameter storage, so it must run on the
        MAIN thread before the consuming forward reads the parameters.  It does
        not need to run before the stream work is enqueued.
        """
        gpu_weights = self.gpu_buffers[slot]
        if gpu_weights is None or self._cached_repoint is None:
            return
        # Re-point using cached metadata (avoids per-layer dict lookups).
        # The original stride is preserved so non-contiguous weights (e.g.
        # online FP8 Cutlass transposed views) round-trip through the flat
        # transport buffer without changing their physical layout.
        for target, dtype, offset, numel, shape, stride in self._cached_repoint[slot]:
            flat = gpu_weights[dtype][offset : offset + numel]
            set_tensor_storage(
                target,
                torch.as_strided(flat, size=shape, stride=stride) if stride is not None else flat.view(shape),
            )

    @torch.compiler.disable
    def prefetch_layer(self, slot: int, non_blocking: bool = True) -> None:
        """Synchronous prefetch: submit the transfers and re-point immediately.

        Kept for the synchronous fallback call sites (group-first
        contamination, cache-dit skip, and the enable() bootstrap).  The
        asynchronous path in ``pre_forward`` splits this into
        ``_submit_prefetch`` (off-thread) plus ``_apply_repoint`` (main thread,
        drained in ``get_weights``).
        """
        # Never let a main-thread submit interleave with a worker submit: every
        # FS rank must issue the identical collective sequence.
        self._drain_prefetch(apply_repoint=True)
        compute_stream = current_omni_platform.current_stream()
        self._submit_dependency_event.record(compute_stream)
        with self._trace_range("submit"):
            if self._prefetch_executor is not None:
                submitted = self._prefetch_executor.submit(
                    self._submit_prefetch,
                    slot,
                    non_blocking,
                    self._submit_dependency_event,
                ).result()
            else:
                submitted = self._submit_prefetch(
                    slot,
                    non_blocking,
                    submit_after_event=self._submit_dependency_event,
                )
        if submitted:
            self._apply_repoint(slot)

    def _drain_prefetch(self, apply_repoint: bool = True) -> None:
        """Wait for this hook's outstanding async submit, then re-point.

        Any exception raised inside the worker surfaces here via
        ``Future.result()`` — it is never swallowed.
        """
        future = self._prefetch_future
        if future is None:
            return
        self._prefetch_future = None
        submitted = future.result()
        if apply_repoint and self._prefetched_slot is not None and submitted is not False:
            self._apply_repoint(self._prefetched_slot)

    def get_weights(self, slot: int) -> dict[torch.dtype, torch.Tensor] | None:
        """Attach the producer's ready event and return full weights for *slot*.

        The ready event for this slot was set by the *previous* hook's
        prefetch (which prefetched THIS block's weights into the shared
        buffer).  This hook's own ready_events[slot] may be None because it
        never prefetched into this slot itself, so fall back to the previous
        hook's event.

        Before waiting, drain the producer's async submit and run its
        Parameter re-pointing on the main thread.

        A missing or non-current ticket means the slot this compute is about to
        read is not owned by the transfer that filled it.  Waiting on the raw
        event anyway would reintroduce exactly the read-after-overwrite the
        ticket state machine exists to prevent, so both cases fail fast.
        """
        for producer in (self._prev_hook, self):
            if producer is not None:
                producer._drain_prefetch(apply_repoint=True)

        owner = self
        evt = self.ready_events[slot]
        if evt is None and self._prev_hook is not None:
            owner = self._prev_hook
            evt = owner.ready_events[slot]

        if self.rank_local:
            # No ticket state machine in rank-local mode: the ready event
            # alone proves the whole-block H2D has been enqueued.
            if evt is not None:
                current_omni_platform.current_stream().wait_event(evt)
            return self.gpu_buffers[slot]

        if evt is None:
            raise RuntimeError(
                f"block {self.block_id} has no ready event for output slot {slot}: "
                "compute would read weights before any transfer published them"
            )

        ticket = owner.ready_tickets[slot]
        if ticket is None or not owner.transport_state.is_current(ticket):
            raise RuntimeError(
                f"block {self.block_id} cannot consume output slot {slot}: the producing "
                f"ticket is {'missing' if ticket is None else 'stale'} because another "
                "transfer took the slot; refusing to wait on an event that no longer "
                "proves this block's weights are ready"
            )
        owner.transport.attach_ready(
            ticket,
            lambda event: current_omni_platform.current_stream().wait_event(event),
        )
        return self.gpu_buffers[slot]

    # ------------------------------------------------------------------ #
    #  Offload: free device memory for current block                     #
    # ------------------------------------------------------------------ #

    @torch.compiler.disable
    def offload_layer(self) -> None:
        """Free GPU memory for current block by replacing tensors with placeholders."""
        slot = self.current_slot
        compute_stream = current_omni_platform.current_stream()

        # Publish the last-use event for this output slot so the next producer
        # waits for our compute to finish reading before overwriting it.
        evt = self._last_use_events[slot]
        if evt is None:
            evt = current_omni_platform.Event()
            self._last_use_events[slot] = evt
        evt.record(compute_stream)
        self._output_slot_events[slot] = evt

        prev = self._prev_hook
        if prev is not None:
            ticket = prev.ready_tickets[slot]
            if ticket is not None and prev.transport_state.is_current(ticket):
                prev.transport.record_last_use(ticket, evt)
                # Retire the ticket and its ready event as a pair: get_weights
                # falls back to this hook's ready event and then validates the
                # ticket, so a leftover event without a ticket would trip the
                # fail-fast check on the next ring reuse (review: #6374).
                prev.ready_tickets[slot] = None
                prev.ready_events[slot] = None
                if self._shared_slot_owners[slot] is prev:
                    self._shared_slot_owners[slot] = None

        self._prefetch_done = None

        transported_names = prev._transported_names if prev is not None else set(self.block_parameters)
        for name in transported_names:
            target = self.block_parameters.get(name, self.block_buffers.get(name))
            if target is not None:
                set_tensor_storage(target, make_offload_placeholder(target))

    # ------------------------------------------------------------------ #
    #  Request lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def set_request_generation(self, generation: int) -> None:
        if generation < self._request_generation:
            raise RuntimeError(f"request generation moved backwards: {generation} < {self._request_generation}")
        self._request_generation = generation

    def drain_request(self) -> None:
        """Release every live slot so the next request starts from a clean state."""
        self._drain_prefetch(apply_repoint=False)

        if self._compute_trace is not None:
            self._compute_trace.__exit__(None, None, None)
            self._compute_trace = None

        for slot in range(2):
            ticket = self.ready_tickets[slot]
            if ticket is None or not self.transport_state.is_current(ticket):
                continue
            # A tail prefetch issued by the final block has no consumer in
            # this request.  Retire it at producer-ready without wiring that
            # unused transfer into the default compute stream; doing so would
            # also block async output D2H behind weight transport that cannot
            # affect the completed request output.
            self.transport.record_last_use(ticket, self.ready_events[slot])
            self.ready_tickets[slot] = None
            self.ready_events[slot] = None
            if self._shared_slot_owners[slot] is self:
                self._shared_slot_owners[slot] = None

        for name in self._transported_names:
            target = self.next_block_parameters.get(name, self.next_block_buffers.get(name))
            if target is not None:
                set_tensor_storage(target, make_offload_placeholder(target))

        self._prefetched_slot = None
        if self.transport is not None:
            self.transport.reset()
        else:
            # Rank-local mode has no tickets; drop the stale ready events so
            # the next request cannot mistake them for a live transfer.
            self.ready_events = [None, None]

    # ------------------------------------------------------------------ #
    #  ModelHook interface                                                #
    # ------------------------------------------------------------------ #

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        # Drain the producer before reading its selected slot.  With async
        # submit, _prefetched_slot is published by the worker and may still
        # contain the previous forward's value until Future.result() returns.
        if self._prev_hook is not None:
            self._prev_hook._drain_prefetch(apply_repoint=True)
            if self._prev_hook._prefetched_slot is not None:
                self.current_slot = self._prev_hook._prefetched_slot

        compute_stream = current_omni_platform.current_stream()

        # Group-first hook: check whether the shared buffer slot was
        # overwritten by another group since our last forward.  If the
        # slot still contains our own data (from the tail hook's async
        # prefetch), skip the sync-prefetch and just wait for the event.
        if self._is_group_first and self._prev_hook is not None:
            if self._prefetch_executor is not None:
                # Another DiT group may have an outstanding tail submit on
                # the shared single-worker executor.  Drain the global queue
                # before reading the shared slot owner or issuing a sync
                # repair, so every FS rank preserves one collective order.
                self._prefetch_executor.submit(_prefetch_barrier).result()
            slot_contaminated = True
            if self._shared_slot_group is not None:
                slot_contaminated = self._shared_slot_group[self.current_slot] != self._group_id
            ticket = self._prev_hook.ready_tickets[self.current_slot]
            has_current_ticket = ticket is not None and self._prev_hook.transport_state.is_current(ticket)
            if slot_contaminated:
                if has_current_ticket:
                    # Another group overwrote the buffer while our ticket was
                    # still live.  Retire it so the re-prefetch really submits
                    # instead of hitting the idempotent guard.
                    self._prev_hook.transport.record_last_use(ticket, self._output_slot_events[self.current_slot])
                    self._prev_hook.ready_tickets[self.current_slot] = None
                # Another group (or no group) wrote to our slot — re-fetch
                self._prev_hook.prefetch_layer(self.current_slot, non_blocking=False)
        elif not self.is_materialized and self._prev_hook is not None:
            # Previous hook was skipped (e.g. by cache-dit).  Only re-submit
            # when no valid async prefetch is already in flight for this slot.
            ticket = self._prev_hook.ready_tickets[self.current_slot]
            has_current_ticket = ticket is not None and self._prev_hook.transport_state.is_current(ticket)
            if not has_current_ticket:
                self._prev_hook.prefetch_layer(self.current_slot, non_blocking=False)

        # Every block attaches the producer's ready event, including the case
        # where draining above already made the Parameter view materialized.
        self.get_weights(self.current_slot)

        # Submit the next layer's transfers off-thread and return immediately,
        # so this block's compute kernels are launched in parallel with the
        # ~2 x chunk_count host-side launches of the next block's submit.
        next_slot = 1 - self.current_slot
        self._drain_prefetch(apply_repoint=True)
        self._submit_dependency_event.record(compute_stream)
        with self._trace_range("submit"):
            if self._prefetch_executor is not None:
                self._prefetch_future = self._prefetch_executor.submit(
                    self._submit_prefetch, next_slot, True, self._submit_dependency_event
                )
            else:
                submitted = self._submit_prefetch(
                    next_slot,
                    True,
                    self._submit_dependency_event,
                )
                if submitted:
                    self._apply_repoint(next_slot)

        if self._compute_trace is not None:
            raise RuntimeError("compute trace range was not closed")
        if self.trace_enabled:
            self._compute_trace = self._trace_range("compute", block_id=self.compute_block_id)
            self._compute_trace.__enter__()

        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        if self._compute_trace is not None:
            self._compute_trace.__exit__(None, None, None)
            self._compute_trace = None
        self.offload_layer()
        return output


# ---------------------------------------------------------------------- #
#  Module-level helpers                                                   #
# ---------------------------------------------------------------------- #


def apply_distributed_block_hook(
    module: nn.Module,
    next_block: nn.Module,
    device: torch.device,
    weight_shard_group: torch.distributed.ProcessGroup | None,
    weight_shard_size: int,
    weight_shard_rank: int,
    copy_stream: Any | None = None,
    comm_stream: Any | None = None,
    shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] | None = None,
    shared_chunk_slot_events: list[Any | None] | None = None,
    shared_output_slot_events: list[Any | None] | None = None,
    shared_slot_owners: list[DistributedLayerwiseOffloadHook | None] | None = None,
    prepared_host_part: dict[str, Any] | None = None,
    h2d_done_events: list[Any] | None = None,
    transport_done_events: list[Any] | None = None,
    output_ready_events: list[Any] | None = None,
    last_use_events: list[Any | None] | None = None,
    prefetch_executor: concurrent.futures.Executor | None = None,
    rank_local_mmap: bool = False,
    pin_memory: bool = True,
    tensor_transforms: dict[int, Any] | None = None,
    data_transport_backend: WeightTransportBackend | None = None,
) -> DistributedLayerwiseOffloadHook:
    """Register a DistributedLayerwiseOffloadHook on *module*."""
    registry = HookRegistry.get_or_create(module)
    hook = DistributedLayerwiseOffloadHook(
        next_block=next_block,
        device=device,
        weight_shard_group=weight_shard_group,
        weight_shard_size=weight_shard_size,
        weight_shard_rank=weight_shard_rank,
        copy_stream=copy_stream,
        comm_stream=comm_stream,
        shared_buffers=shared_buffers,
        shared_chunk_slot_events=shared_chunk_slot_events,
        shared_output_slot_events=shared_output_slot_events,
        shared_slot_owners=shared_slot_owners,
        prepared_host_part=prepared_host_part,
        h2d_done_events=h2d_done_events,
        transport_done_events=transport_done_events,
        output_ready_events=output_ready_events,
        last_use_events=last_use_events,
        prefetch_executor=prefetch_executor,
        rank_local_mmap=rank_local_mmap,
        pin_memory=pin_memory,
        tensor_transforms=tensor_transforms,
        data_transport_backend=data_transport_backend,
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
                    buffer = torch.empty(total, dtype=dtype, device="cpu")
                    if pin_memory:
                        buffer = buffer.pin_memory()
                    buffers[dtype] = buffer
                self._cpu_staging_buffers.append(buffers)

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

        # Weight-shard group used by the chunked transport.
        self.weight_shard_group: torch.distributed.ProcessGroup | None = config.weight_shard_group
        self.weight_shard_cpu_group: Any | None = config.weight_shard_cpu_group
        # Use only the degree resolved by OffloadConfig.from_od_config().
        self.weight_shard_size: int = max(1, int(getattr(config, "weight_shard_size", 1) or 1))
        self.weight_shard_rank: int = int(getattr(config, "weight_shard_rank", 0) or 0)
        self.weight_shard_ranks: tuple[int, ...] = tuple(range(self.weight_shard_size))
        self.transport_capability: TransportCapability | None = None
        self.transport_selection: TransportSelection | None = None
        self.data_transport_backend: WeightTransportBackend | None = None

        self._blocks: list[list[nn.Module]] = []
        self._all_hook_groups: list[list[DistributedLayerwiseOffloadHook]] = []
        self._resident_blocks: list[nn.Module] = []
        self._resident_layer_group: PinnedResidentLayerGroup | None = None
        self._using_mmap = False
        self._using_rank_local_mmap = False
        self._using_registered_mmap = False
        self.host_weight_plan = host_weight_plan
        self._host_weight_lease: HostWeightLease | None = None
        self._host_registration: HostRegistration | None = None
        self._mmap_transforms_by_tensor_id: dict[int, Any] = {}

        # Chunked transport planning state.
        # Streaming block lists are collected during discovery and only turned
        # into hooks after the Host plan and Host storage exist.
        self._pending_block_groups: list[tuple[list[nn.Module], str]] = []
        self._chunk_ownership: ChunkOwnership | None = None
        self._block_ids: dict[int, int] = {}
        self._planned_manifests: dict[int, PartManifest] = {}
        self._prepared_host_parts: dict[int, dict[str, Any]] = {}
        self.pin_budget: PinBudget | None = None
        self._forced_fallback_reason: str | None = None

        # Shared (per-engine) event arrays handed to every hook.
        self._shared_h2d_done_events: list[Any] = []
        self._shared_transport_done_events: list[Any] = []
        self._shared_output_ready_events: list[Any] = []
        self._shared_last_use_events: list[Any | None] = [None, None]
        self._shared_chunk_slot_events: list[Any | None] = [None, None]
        self._shared_output_slot_events: list[Any | None] = [None, None]
        self._shared_slot_owners: list[DistributedLayerwiseOffloadHook | None] = [None, None]
        # Per-slot "which hook group wrote this slot last" tracker.  Allocated
        # here (not in enable()) because the request lifecycle can drain a
        # backend whose enable() returned early with no streaming groups.
        self._shared_slot_group: list[int] | None = None

        # Request lifecycle.
        self._request_generation = 0
        self._request_active = False

        # Async prefetch: single-worker thread pool for background prefetching.
        # max_workers MUST stay 1 — every FS rank has to submit the identical
        # (block, part, dtype, chunk) collective sequence, so concurrent
        # submits would reorder the collectives and hang or corrupt.
        self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dlo_prefetch"
        )
        self._prefetch_future: concurrent.futures.Future | None = None

    def load_resident_layers(self) -> None:
        """Load the model-declared leading blocks for the denoise stage."""
        if self._resident_layer_group is not None:
            self._resident_layer_group.load()

    def offload_resident_layers(self) -> None:
        """Release leading blocks before VAE decode to bound peak HBM."""
        if self._resident_layer_group is not None:
            self._resident_layer_group.offload()

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
        The views point to OS page cache shared across ranks.  The sharded
        AllGather path copies only each rank's local shard into private Host
        storage in ``_prepare_host_storage``, after which the views are
        released; rank-local mode retains the views as the host master and
        packs one block at a time into bounded staging storage.

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

    # ------------------------------------------------------------------ #
    #  Weight shard group                                                 #
    # ------------------------------------------------------------------ #

    def _init_weight_shard_group(self) -> None:
        """Resolve the DP/SP group that owns the chunked weight shards."""
        if self.weight_shard_size <= 1:
            logger.info("Distributed layerwise offload: weight_shard_size=1, running without AllGather")
            self.weight_shard_group = None
            self.weight_shard_cpu_group = None
            self.weight_shard_rank = 0
            self.weight_shard_ranks = (0,)
            self._publish_weight_shard_config()
            return

        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialized. "
                "Distributed layerwise offload with weight_shard_size > 1 requires "
                "an initialized process group."
            )

        # With DP > 1 the weight collective runs on the DP group; with DP = 1
        # but SP > 1 it runs on the SP group.
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_data_parallel_world_size,
            get_dp_group,
        )

        if get_data_parallel_world_size() > 1:
            coord = get_dp_group()
        else:
            from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

            coord = get_sp_group()
            logger.info(
                "Distributed layerwise offload: DP=1, using SP group (world_size=%d) for weight sharding",
                coord.world_size,
            )
        if coord.world_size != self.weight_shard_size:
            raise ValueError(
                "DLO weight shard degree does not match the resolved group: "
                f"config={self.weight_shard_size}, group={coord.world_size}"
            )
        # Weight AllGathers run from the prefetch worker while SP activation
        # collectives run from the compute thread. Give DLO a companion
        # communicator so their ordering cannot interleave on one HCCL group.
        self.weight_shard_group = torch.distributed.new_group(
            ranks=list(coord.ranks),
            backend=torch.distributed.get_backend(coord.device_group),
            use_local_synchronization=True,
        )
        self.weight_shard_cpu_group = coord.cpu_group
        self.weight_shard_rank = coord.rank_in_group
        self.weight_shard_ranks = tuple(coord.ranks)
        self._publish_weight_shard_config()
        logger.info(
            "Distributed layerwise offload: weight_shard_size=%d, rank_in_group=%d, group_ranks=%s",
            self.weight_shard_size,
            self.weight_shard_rank,
            coord.ranks,
        )

    def _publish_weight_shard_config(self) -> None:
        """Mirror the resolved FS identity onto the config for other layers."""
        self.config.weight_shard_size = self.weight_shard_size
        self.config.weight_shard_rank = self.weight_shard_rank
        self.config.weight_shard_group = self.weight_shard_group
        self.config.weight_shard_cpu_group = self.weight_shard_cpu_group

    def _probe_transport_capability(self) -> TransportCapability:
        """Probe only what the FS chunk-schedule backends rely on."""
        native_persistent = bool(
            self.device.type == "npu"
            and hasattr(torch, "npu")
            and hasattr(torch.npu, "NPUGraph")
            and hasattr(torch.npu, "graph")
        )
        return TransportCapability(
            world_size=self.weight_shard_size,
            rank=self.weight_shard_rank,
            global_ranks=self.weight_shard_ranks,
            native_persistent=native_persistent,
        )

    def _configure_transport_backend(self) -> None:
        self.transport_capability = self._probe_transport_capability()
        requested_backend = TransportBackendKind(
            getattr(self.config, "dlo_transport_backend", TransportBackendKind.AUTO.value)
        )
        self.transport_selection = select_transport(
            requested_backend,
            self.transport_capability,
        )
        self.data_transport_backend = create_transport_backend(
            self.transport_selection,
            self.transport_capability,
        )
        logger.info(
            "DLO transport backend: requested=%s effective=%s",
            requested_backend.value,
            self.transport_selection.effective_backend.value,
        )

    # ------------------------------------------------------------------ #
    #  Host plan and Host storage                                         #
    # ------------------------------------------------------------------ #

    @property
    def _pin_failure_policy(self) -> PinFailurePolicy:
        return PinFailurePolicy(getattr(self.config, "dlo_pin_failure_policy", "fail"))

    @property
    def _using_rank_local(self) -> bool:
        """Rank-local mode streams whole blocks without a weight collective.

        Chunked packing (manifests, private shards, tickets) applies only to
        the sharded AllGather path; with ``weight_shard_size == 1`` each hook
        collects its own host storage instead.
        """
        return self.weight_shard_size <= 1

    @property
    def _alignment_bytes(self) -> int:
        return int(getattr(self.config, "dlo_alignment_bytes", 256))

    def _block_tensor_specs(self, module: nn.Module) -> list[tuple[str, torch.Tensor, bool]]:
        """Collect the final, rank-local tensors a block contributes.

        DTensor parameters are resolved with ``to_local``; checkpoint layout
        adapters registered by the loader's host-weight plan are applied here
        so the manifest describes the tensors that actually reach the device.
        """
        specs: list[tuple[str, torch.Tensor, bool]] = []
        for name, tensor, is_buffer in chain(
            ((n, t, False) for n, t in module.named_parameters()),
            ((n, t, True) for n, t in module.named_buffers()),
        ):
            local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
            transform = self._mmap_transforms_by_tensor_id.get(id(tensor))
            if transform is None:
                # Legacy attribute path kept for tests that install the
                # transform directly on the tensor.
                transform = getattr(tensor, "mmap_weight_transform", None)
                if not getattr(tensor, "mmap_weight_transform_pending", False):
                    transform = None
            if callable(transform):
                transformed = transform(local)
                # A layout transform may only change the view (stride/order),
                # never the element count or the dtype family — otherwise the
                # manifest metadata would silently misdescribe the tensor and
                # the transport would move corrupted bytes (review: #6374).
                if transformed.numel() != local.numel():
                    raise RuntimeError(
                        f"mmap weight transform for {name!r} changed numel "
                        f"({local.numel()} -> {transformed.numel()}); the manifest "
                        "would describe incorrect transport metadata"
                    )
                if transformed.is_floating_point() != local.is_floating_point():
                    raise RuntimeError(
                        f"mmap weight transform for {name!r} changed the dtype family "
                        f"({local.dtype} -> {transformed.dtype}); the manifest "
                        "would describe incorrect transport metadata"
                    )
                local = transformed
            specs.append((name, local, is_buffer))
        return specs

    def _plan_chunk_manifests(self, ownership: ChunkOwnership) -> None:
        """Build every final layout and enforce the pin cap before allocation."""
        self.pin_budget = PinBudget(limit_bytes=self.config.dlo_pin_budget_bytes)
        policy = self._pin_failure_policy
        self._planned_manifests = {}
        self._forced_fallback_reason = None

        for block in ownership.blocks:
            specs = [spec for spec in self._block_tensor_specs(block.module) if is_chunk_transport_supported(spec[1])]
            manifest = build_part_manifest(
                specs,
                block_id=self._block_ids[id(block.module)],
                part_id="block",
                weight_shard_size=self.weight_shard_size,
                weight_shard_rank=self.weight_shard_rank,
                chunk_size_bytes=self.config.chunk_size_bytes,
                alignment_bytes=self._alignment_bytes,
                layout=WeightLayout.CHUNK_MAJOR,
            )
            try:
                self.pin_budget.plan(block.path, manifest.pinned_bytes)
            except MemoryError as exc:
                if policy is PinFailurePolicy.FAIL:
                    raise MemoryError(
                        "pinned Host budget exceeded before allocation: "
                        f"required={self.pin_budget.required_bytes + manifest.pinned_bytes} "
                        f"limit={self.config.dlo_pin_budget_bytes}"
                    ) from exc
                # Keep the key so _prepare_host_storage can still reserve it;
                # the pageable fallback consumes no pinned budget.
                self._forced_fallback_reason = "pinned Host budget exceeded before allocation"
                self.pin_budget.plan(block.path, 0)
            self._planned_manifests[id(block.module)] = manifest

        if self._forced_fallback_reason is not None:
            # The budget fallback is engine-wide. Discard any pinned bytes
            # planned before the first oversized block so accounting matches
            # the pageable whole-block storage that will actually be built.
            self.pin_budget = PinBudget(limit_bytes=self.config.dlo_pin_budget_bytes)
            for block in ownership.blocks:
                self.pin_budget.plan(block.path, 0)

        logger.info(
            "DLO Host plan: blocks=%d required_pinned=%.3f GiB budget=%s policy=%s fallback=%s",
            len(ownership.blocks),
            self.pin_budget.required_bytes / 1024**3,
            self.config.dlo_pin_budget_bytes,
            policy.value,
            self._forced_fallback_reason,
        )

    def _prepare_host_storage(self, ownership: ChunkOwnership) -> None:
        """Pack Host shards with one FS-wide fallback decision per block."""
        policy = self._pin_failure_policy
        pin_requested = bool(self.config.pin_cpu_memory)

        # Validate the pin_cpu_memory + policy combination once before
        # allocating anything.  With policy=fail the user has explicitly
        # opted out of silent degradation, so combining it with
        # pin_cpu_memory=False is a misconfiguration that should be caught
        # early rather than producing a surprise whole-block run.
        if self.config.dlo_use_allgather and not pin_requested and self._forced_fallback_reason is None:
            if policy is PinFailurePolicy.FAIL:
                raise ValueError(
                    "pin_cpu_memory=False is incompatible with "
                    "dlo_pin_failure_policy=fail: chunk transport requires "
                    "pinned Host memory for async H2D overlap.  Either set "
                    "pin_cpu_memory=True, or change dlo_pin_failure_policy "
                    "to 'whole_block_fallback' to accept pageable (non-"
                    "overlapping) whole-block transport."
                )
            # whole_block_fallback: treat as an engine-wide forced fallback so
            # the per-block loop uses the same pageable path for every block.
            self._forced_fallback_reason = "pin_cpu_memory=False"

        pinned_bytes = 0
        fallback_blocks = 0

        for block in ownership.blocks:
            module = block.module
            specs = self._block_tensor_specs(module)
            chunk_specs = [spec for spec in specs if is_chunk_transport_supported(spec[1])]
            manifest = self._planned_manifests[id(module)]

            fallback_reason = self._forced_fallback_reason

            failed = 0
            detail = fallback_reason
            cpu_shards: dict[torch.dtype, torch.Tensor] | None = None
            if fallback_reason is None:
                try:
                    cpu_shards = pack_local_shard(chunk_specs, manifest)
                except (MemoryError, RuntimeError, OSError) as exc:
                    # The pinned allocator can also fail with RuntimeError/OS
                    # errors (lock limit, driver); treat every allocation
                    # failure uniformly through the FS vote (review: #6374).
                    failed = 1
                    detail = f"pinned Host allocation failed for {block.path}: {exc}"

            # A pageable fallback changes the collective sequence, so every FS
            # rank must take the same decision even if only one rank failed.
            agreed = self._agree_pin_failure(failed)
            if agreed:
                if policy is PinFailurePolicy.FAIL:
                    if detail is None:
                        detail = "another FS rank failed"
                    logger.error("DLO pinned Host allocation failed: %s", detail)
                    # Do not leave a partially prepared backend behind: drop
                    # every shard packed so far (freeing pinned Host memory)
                    # and close the mmap handles (review: #6374).
                    self._abort_prepared_host_storage()
                    raise RuntimeError("pinned allocation failed on at least one FS rank")
                fallback_reason = detail or "another FS rank failed"
                cpu_shards = None

            if cpu_shards is None:
                manifest = build_part_manifest(
                    chunk_specs,
                    block_id=self._block_ids[id(module)],
                    part_id="block",
                    weight_shard_size=self.weight_shard_size,
                    weight_shard_rank=self.weight_shard_rank,
                    chunk_size_bytes=self.config.chunk_size_bytes,
                    alignment_bytes=self._alignment_bytes,
                    layout=WeightLayout.WHOLE_BLOCK,
                )
                cpu_shards = pack_local_shard(
                    chunk_specs,
                    manifest,
                    allocator=lambda numel, dtype: torch.empty(numel, dtype=dtype, device="cpu"),
                )
                fallback_blocks += 1
            else:
                pinned_bytes += manifest.pinned_bytes

            self.pin_budget.reserve(block.path)
            self._planned_manifests[id(module)] = manifest

            metadata: dict[torch.dtype, list[dict[str, Any]]] = {}
            for dtype_manifest in manifest.dtypes:
                metadata[dtype_manifest.dtype] = [
                    {
                        "name": tensor.name,
                        "offset": tensor.offset,
                        "numel": tensor.numel,
                        "shape": torch.Size(tensor.shape),
                        "stride": tensor.stride,
                    }
                    for tensor in dtype_manifest.tensors
                ]

            # Host storage now owns the weights: drop the original storage so
            # the mmap views and CPU copies can be released.
            chunked_names = {tensor.name for dtype_manifest in manifest.dtypes for tensor in dtype_manifest.tensors}
            sources = {name: tensor for name, tensor, _ in specs}
            for name in chunked_names:
                set_tensor_storage(sources[name], make_offload_placeholder(sources[name]))
            # Tensors the chunk transport cannot carry (0-dim, integer dtypes)
            # stay resident on device.
            for name, tensor, _is_buffer in specs:
                if name not in chunked_names and is_materialized_tensor(tensor):
                    set_tensor_storage(tensor, tensor.detach().to(self.device))

            self._prepared_host_parts[id(module)] = {
                "cpu_shards": cpu_shards,
                "metadata": metadata,
                "manifest": manifest,
                "fallback_reason": fallback_reason,
            }

        logger.info(
            "DLO Host storage ready: pinned=%.3f GiB fallback_blocks=%d/%d",
            pinned_bytes / 1024**3,
            fallback_blocks,
            len(ownership.blocks),
        )

    def _abort_prepared_host_storage(self) -> None:
        """Best-effort cleanup after a failed _prepare_host_storage.

        Drops every shard packed so far (freeing pinned Host memory) and
        closes the mmap handles, so the backend is not left in a partially
        initialized state (review: #6374).
        """
        self._prepared_host_parts.clear()
        self._planned_manifests = {}
        self.pin_budget = None
        self._release_mmap_handles()

    def _agree_pin_failure(self, failed: int) -> bool:
        """Return the FS-wide OR of per-rank pinned allocation failure."""
        group = self.config.weight_shard_cpu_group or self.weight_shard_cpu_group
        if self.weight_shard_size <= 1 or group is None or not torch.distributed.is_initialized():
            return bool(failed)
        flag = torch.tensor([failed], dtype=torch.int32)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX, group=group)
        return bool(flag.item())

    def _validate_manifest_consistency(self, ownership: ChunkOwnership) -> None:
        """Fail before HCCL if any FS rank planned a different schedule."""
        if self.weight_shard_size <= 1:
            return
        group = self.config.weight_shard_cpu_group or self.weight_shard_cpu_group
        if group is None or not torch.distributed.is_initialized():
            raise RuntimeError("DLO manifest validation requires the FS CPU process group")

        signature = []
        for block in ownership.blocks:
            manifest = self._prepared_host_parts[id(block.module)]["manifest"]
            signature.append(
                (
                    manifest.block_id,
                    block.path,
                    manifest.digest,
                    manifest.layout.value,
                    tuple(
                        (
                            str(dtype_manifest.dtype),
                            len(dtype_manifest.chunks),
                            dtype_manifest.total_numel,
                            dtype_manifest.padded_numel,
                            dtype_manifest.local_numel,
                        )
                        for dtype_manifest in manifest.dtypes
                    ),
                )
            )

        gathered: list[Any] = [None] * self.weight_shard_size
        torch.distributed.all_gather_object(gathered, signature, group=group)
        mismatched = [rank for rank, remote in enumerate(gathered) if remote != signature]
        if mismatched:
            raise RuntimeError(
                "DLO manifest/schedule mismatch across the FS group before the first "
                f"weight collective; mismatched_ranks={mismatched}"
            )
        logger.info("Validated identical DLO manifest schedule across %d FS ranks", self.weight_shard_size)

    # ------------------------------------------------------------------ #
    #  Request lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def begin_request(self) -> None:
        """Open one request's transport generation across every hook."""
        if self._request_active:
            raise RuntimeError("DLO request re-entry is not supported")
        self._request_active = True
        for group in self._all_hook_groups:
            for hook in group:
                hook.set_request_generation(self._request_generation)

    def end_request(self) -> None:
        """Drain in-flight transfers and retire this request's generation."""
        if not self._request_active:
            raise RuntimeError("DLO request lifecycle ended without a matching begin")
        self._drain_transport()
        if self.data_transport_backend is not None:
            self.data_transport_backend.reset_generation(self._request_generation)
        self._request_generation += 1
        self._request_active = False

    @contextmanager
    def request_context(self) -> Iterator[None]:
        self.begin_request()
        try:
            yield
        finally:
            self.end_request()

    def _drain_transport(self) -> None:
        # A hook whose drain raises (e.g. from future.result()) must not stop
        # the remaining hooks from draining — otherwise their futures, pinned
        # buffers and prefetch threads leak. Drain everything, then re-raise
        # the first error (review: #6374).
        first_error: BaseException | None = None
        for group in self._all_hook_groups:
            for hook in group:
                try:
                    hook.drain_request()
                except BaseException as exc:  # noqa: BLE001
                    if first_error is None:
                        first_error = exc
        if self._shared_slot_group is not None:
            for slot in range(len(self._shared_slot_group)):
                self._shared_slot_group[slot] = -1
                self._shared_slot_owners[slot] = None
        if first_error is not None:
            raise first_error

    def get_transport_metrics(self) -> dict[str, Any]:
        """Report the exact Host/device staging cost of the chunked transport."""
        manifests = [part["manifest"] for part in self._prepared_host_parts.values()]
        fallback_parts = sum(
            1 for part in self._prepared_host_parts.values() if part.get("fallback_reason") is not None
        )

        logical_weight_bytes = 0
        max_padded: dict[torch.dtype, int] = {}
        max_local_chunk: dict[torch.dtype, int] = {}
        for manifest in manifests:
            for dtype_manifest in manifest.dtypes:
                element_size = _dtype_size(dtype_manifest.dtype)
                logical_weight_bytes += dtype_manifest.total_numel * element_size
                dtype = dtype_manifest.dtype
                max_padded[dtype] = max(max_padded.get(dtype, 0), dtype_manifest.padded_numel)
                max_local_chunk[dtype] = max(max_local_chunk.get(dtype, 0), dtype_manifest.local_chunk_numel)

        full_output_staging_bytes = 2 * sum(numel * _dtype_size(dtype) for dtype, numel in max_padded.items())
        local_input_staging_bytes = 0
        if self.data_transport_backend is not None and self.data_transport_backend.requires_local_input:
            local_input_staging_bytes = 2 * sum(numel * _dtype_size(dtype) for dtype, numel in max_local_chunk.items())

        submissions = 0
        submitted_chunks = 0
        consumer_attaches = 0
        releases = 0
        for group in self._all_hook_groups:
            for hook in group:
                if hook.transport is None:
                    # Rank-local hooks stream whole blocks without the
                    # chunked transport state machine.
                    continue
                counters = hook.transport.counters
                submissions += counters.submissions
                submitted_chunks += counters.submitted_chunks
                consumer_attaches += counters.consumer_attaches
                releases += counters.releases

        parts = len(self._prepared_host_parts)
        selection = self.transport_selection
        capability = self.transport_capability
        backend_counters = self.data_transport_backend.counters if self.data_transport_backend is not None else None
        return {
            "weight_shard_size": self.weight_shard_size,
            "parts": parts,
            "logical_weight_bytes": logical_weight_bytes,
            "pinned_required_bytes": self.pin_budget.required_bytes if self.pin_budget is not None else 0,
            "pinned_reserved_bytes": self.pin_budget.reserved_bytes if self.pin_budget is not None else 0,
            "full_output_staging_bytes": full_output_staging_bytes,
            "local_input_staging_bytes": local_input_staging_bytes,
            "transport_staging_bytes": full_output_staging_bytes + local_input_staging_bytes,
            # torch.distributed/HCCL does not expose its internal workspace
            # allocation through this backend. Keep it separate from the two
            # explicitly owned staging pools instead of double-counting them.
            "collective_workspace_bytes": None,
            "fallback_parts": fallback_parts,
            "fallback_ratio": (fallback_parts / parts) if parts else 0.0,
            "transport_backend_requested": selection.requested_backend.value if selection is not None else None,
            "transport_backend_effective": selection.effective_backend.value if selection is not None else None,
            "transport_native_persistent": capability.native_persistent if capability is not None else False,
            "backend_submitted_parts": backend_counters.submitted_parts if backend_counters is not None else 0,
            "backend_submitted_chunks": backend_counters.submitted_chunks if backend_counters is not None else 0,
            "backend_host_h2d_bytes": backend_counters.host_h2d_bytes if backend_counters is not None else 0,
            "backend_fabric_bytes": backend_counters.fabric_bytes if backend_counters is not None else 0,
            "backend_schedule_builds": backend_counters.schedule_builds if backend_counters is not None else 0,
            "backend_schedule_replays": backend_counters.schedule_replays if backend_counters is not None else 0,
            "backend_chunks": dict(backend_counters.backend_chunks) if backend_counters is not None else {},
            "request_generation": self._request_generation,
            "submissions": submissions,
            "submitted_chunks": submitted_chunks,
            "consumer_attaches": consumer_attaches,
            "releases": releases,
        }

    def reset_transport_counters(self) -> None:
        """Reset request activity counters at a benchmark boundary."""
        for group in self._all_hook_groups:
            for hook in group:
                if hook.transport is not None:
                    hook.transport.reset_counters()
        if self.data_transport_backend is not None:
            self.data_transport_backend.reset_counters()

    # ------------------------------------------------------------------ #
    #  Ownership, hook construction, shared events                        #
    # ------------------------------------------------------------------ #

    def _build_chunk_ownership(self) -> ChunkOwnership:
        """Enumerate every streamed block and assign stable small block ids.

        The trace analyzer selects a window of *consecutive* block ids, so ids
        are consecutive integers in forward order rather than ``id(module)``.
        """
        ownership = ChunkOwnership()
        next_block_id = 0
        for blocks, group_path in self._pending_block_groups:
            for index, block in enumerate(blocks):
                key = id(block)
                if key in self._block_ids:
                    continue
                self._block_ids[key] = next_block_id
                ownership.block_ids[key] = next_block_id
                next_block_id += 1
                ownership.blocks.append(ChunkOwnedBlock(module=block, path=f"{group_path}.{index}"))
        self._chunk_ownership = ownership
        return ownership

    def _resolve_chunk_ownership(self, pipeline: nn.Module) -> ChunkOwnership:
        """Build ownership from the blocks discovered during ``enable()``."""
        return self._build_chunk_ownership()

    def _create_block_hook(
        self,
        module: nn.Module,
        next_block: nn.Module,
        shared_buffers: list[dict[torch.dtype, torch.Tensor] | None],
    ) -> DistributedLayerwiseOffloadHook:
        """Register one hook that transports *next_block* into a shared slot."""
        # Rank-local hooks collect their own host storage in initialize_hook;
        # the chunked AllGather path consumes the backend-prepared Host part.
        if not self._using_rank_local and self.data_transport_backend is None:
            raise RuntimeError("DLO data transport backend is not configured")
        prepared = None if self._using_rank_local else self._prepared_host_parts[id(next_block)]
        hook = apply_distributed_block_hook(
            module,
            next_block,
            self.device,
            self.weight_shard_group,
            self.weight_shard_size,
            self.weight_shard_rank,
            copy_stream=self.copy_stream,
            comm_stream=self.comm_stream,
            shared_buffers=shared_buffers,
            shared_chunk_slot_events=self._shared_chunk_slot_events,
            shared_output_slot_events=self._shared_output_slot_events,
            shared_slot_owners=self._shared_slot_owners,
            prepared_host_part=prepared,
            h2d_done_events=self._shared_h2d_done_events,
            transport_done_events=self._shared_transport_done_events,
            output_ready_events=self._shared_output_ready_events,
            last_use_events=self._shared_last_use_events,
            prefetch_executor=self._prefetch_executor,
            rank_local_mmap=self._using_rank_local_mmap,
            pin_memory=self.config.pin_cpu_memory,
            tensor_transforms=self._mmap_transforms_by_tensor_id,
            data_transport_backend=self.data_transport_backend,
        )
        if self._using_rank_local:
            # No manifest in rank-local mode; assign the ownership id directly.
            hook.block_id = self._block_ids[id(next_block)]
        # The 'compute' trace marker names the module being computed, not the
        # block whose weights this hook transports.
        hook.compute_block_id = self._block_ids[id(module)]
        hook.set_request_generation(self._request_generation)
        return hook

    def _create_hook_groups(
        self,
        shared_buffers: list[dict[torch.dtype, torch.Tensor] | None],
        shared_slot_group: list[int],
        unified_chunk_buffers: list[dict[torch.dtype, torch.Tensor] | None] | None = None,
    ) -> None:
        """Create the hooks in a circular sliding window.

        The last block prefetches the first block and block ``i`` prefetches
        block ``i + 1``.  All hooks share the 2 global device buffers.
        """
        for group_idx, (blocks, _group_path) in enumerate(self._pending_block_groups):
            num_blocks = len(blocks)
            block_hooks: list[DistributedLayerwiseOffloadHook] = [
                self._create_block_hook(blocks[-1], blocks[0], shared_buffers)
            ]
            for i, block in enumerate(blocks[:-1]):
                block_hooks.append(self._create_block_hook(block, blocks[(i + 1) % num_blocks], shared_buffers))

            # Wire backward references for cache-dit fallback
            for i in range(len(block_hooks)):
                block_hooks[i]._prev_hook = block_hooks[i - 1]

            # Assign slots in list order: block_hooks = [last_hook, block0, ..., blockN-2]
            # This ensures last_hook.current_slot != block0_hook.current_slot,
            # so the circular prefetch (last_hook -> block0) writes to a
            # different slot than block0 reads from.  Correct for ALL N.
            for i, hook in enumerate(block_hooks):
                hook.current_slot = i % 2
                hook._group_id = group_idx
                hook._shared_slot_group = shared_slot_group
                if unified_chunk_buffers is not None:
                    hook.gpu_shard_buffers = unified_chunk_buffers

            # Mark block0_hook (index 1) as group-first
            if len(block_hooks) > 1:
                block_hooks[1]._is_group_first = True
            block_hooks[0]._is_group_tail = True

            self._all_hook_groups.append(block_hooks)
            self._blocks.append(blocks)

    def _bootstrap_first_group(self) -> None:
        """Synchronously materialize block 0 before the first forward."""
        if not self._all_hook_groups:
            return
        group = self._all_hook_groups[0]
        if len(group) < 2:
            raise RuntimeError("DLO hook ring requires at least two blocks")
        first_producer = group[0]  # tail block -> block 0
        first_consumer = group[1]  # block 0 -> block 1
        first_slot = first_consumer.current_slot
        first_producer.prefetch_layer(slot=first_slot, non_blocking=False)
        first_consumer.get_weights(first_slot)

    def _allocate_shared_events(self) -> None:
        """Allocate the per-slot event arrays once and share them everywhere.

        Allocating here (instead of per hook, per forward) keeps the submit
        path free of Event construction.
        """
        if not self._shared_h2d_done_events:
            self._shared_h2d_done_events = [current_omni_platform.Event() for _ in range(2)]
        if not self._shared_transport_done_events:
            self._shared_transport_done_events = [current_omni_platform.Event() for _ in range(2)]
        if not self._shared_output_ready_events:
            self._shared_output_ready_events = [current_omni_platform.Event() for _ in range(2)]
        self._shared_last_use_events = [current_omni_platform.Event() for _ in range(2)]
        self._shared_chunk_slot_events = [None, None]
        self._shared_output_slot_events = [None, None]

    def _register_on_demand_hook(self, module: nn.Module, label: str, *, stage_on_demand: bool = False) -> None:
        """Prepare a pipeline-managed stage component or keep it resident.

        Components that expose an explicit stage lifecycle are initially
        offloaded and loaded by their pipeline only around encode/decode.
        Other models retain the conservative resident behavior because a
        generic post-forward hook can disrupt the DiT prefetch streams.
        """
        offload_to_cpu = getattr(module, "offload_to_cpu", None)
        if stage_on_demand and callable(offload_to_cpu):
            offload_to_cpu()
            logger.info("Prepared %s (%s) for pipeline-managed staged offload", label, module.__class__.__name__)
            return
        module.to(self.device)
        logger.info("Moved %s (%s) to GPU (resident)", label, module.__class__.__name__)

    def _try_layerwise_offload_encoder(self, module: nn.Module, name: str, plan: OffloadPlan | None) -> bool:
        """Stream plan-declared encoder blocks on each rank without AllGather."""
        if plan is None or name not in plan.encoder_block_attrs:
            return False
        if getattr(module, "_omni_layerwise_enabled", False):
            return True

        from operator import attrgetter

        from vllm_omni.diffusion.offloader.layerwise_backend import apply_block_hook

        hooks = []
        block_groups = []
        copy_stream = current_omni_platform.Stream()
        for block_path in plan.encoder_block_attrs[name]:
            try:
                blocks = attrgetter(block_path)(module)
            except AttributeError:
                logger.warning("Encoder offload path %s.%s was not found", name, block_path)
                continue
            if not isinstance(blocks, nn.ModuleList) or len(blocks) <= 1:
                logger.warning("Encoder offload path %s.%s is not a streamable block list", name, block_path)
                continue
            group_hooks = [
                apply_block_hook(blocks[-1], blocks[0], self.device, copy_stream, self.config.pin_cpu_memory)
            ]
            group_hooks.extend(
                apply_block_hook(block, blocks[index + 1], self.device, copy_stream, self.config.pin_cpu_memory)
                for index, block in enumerate(blocks[:-1])
            )
            for index, hook in enumerate(group_hooks):
                hook._prev_hook = group_hooks[index - 1]
            hooks.extend(group_hooks)
            block_groups.append(blocks)

        if not hooks:
            return False
        # The component lifecycle uses these generic attributes to keep only
        # non-block encoder state resident during the encode phase.
        module._omni_layerwise_hooks = hooks
        module._omni_layerwise_block_groups = block_groups
        module._omni_layerwise_enabled = True
        logger.info(
            "Enabled rank-local layerwise offload for encoder %s (%d blocks across %d stacks)",
            name,
            sum(len(blocks) for blocks in block_groups),
            len(block_groups),
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

        num_blocks = len(blocks)
        logger.info(
            "Distributed layerwise offload for submodule '%s.%s' (%d blocks, %.0f MB total, weight_shard_size=%d)",
            name,
            blocks_attr,
            num_blocks,
            sum(p.nelement() * p.element_size() for p in module.parameters()) / 1048576,
            self.weight_shard_size,
        )

        # Move non-block parts of the submodule to GPU (small: embeddings, norms)
        for child_name, child in module.named_children():
            if child_name != blocks_attr:
                child.to(self.device)

        # Hook creation is deferred: the manifest plan must clear the pin cap
        # for every block before any Host or device buffer is allocated.
        self._pending_block_groups.append((list(blocks), f"{name}.{blocks_attr}"))
        return True

    def _prepare_dit_non_block_modules(
        self,
        dit_module: nn.Module,
        dit_name: str,
        blocks_attr_names: list[str],
        all_dit_modules: set[int],
        plan: OffloadPlan | None,
    ) -> None:
        """Place or hook the DiT parts that are outside its repeated blocks.

        This must run even when every repeated block is resident.  Otherwise
        an all-resident stage skips placement for modules such as H3's token
        refiner and enters the forward pass with CPU or meta tensors.
        """
        _ON_DEMAND_THRESHOLD = _ON_DEMAND_THRESHOLD_MB
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
            if explicitly_planned or module_mb > _ON_DEMAND_THRESHOLD:
                if id(module) in all_dit_modules:
                    logger.info("Submodule '%s' is already a DiT module, skipping layerwise offload", name)
                elif self._try_layerwise_offload_submodule(module, name, plan):
                    pass
                else:
                    self._register_on_demand_hook(module, name)
                continue

            try:
                module.to(self.device)
            except (NotImplementedError, RuntimeError):
                # Non-persistent buffers such as RoPE frequencies do not
                # exist in the checkpoint and must be reconstructed.
                has_meta_buffer = any(getattr(buffer, "is_meta", False) for buffer in module.buffers(recurse=True))
                if has_meta_buffer:
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
                try:
                    module.to(self.device)
                except Exception:
                    logger.warning("Module %s still has meta params after mmap load", name)

        for param in dit_module._parameters.values():
            if param is not None and not getattr(param, "is_meta", False):
                param.data = param.data.to(self.device, non_blocking=True)
        for buffer in dit_module._buffers.values():
            if buffer is not None:
                buffer.data = buffer.data.to(self.device, non_blocking=True)

    def enable(self, pipeline: nn.Module) -> None:
        """Enable DLO and make partial startup failures transactional."""
        try:
            self._enable(pipeline)
        except BaseException:
            try:
                self.disable()
            except BaseException:
                logger.exception("DLO cleanup failed while handling an enable failure")
            raise

    def _enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("DistributedLayerwiseOffloadBackend already enabled")
            return

        self._on_demand_shard_infos: list[dict] = []
        self._on_demand_handles: list[Any] = []

        # Resolve the FS group that owns the weight shard dimension.
        if self.weight_shard_group is None:
            self._init_weight_shard_group()
        self._configure_transport_backend()

        modules = ModuleDiscovery.discover(pipeline)
        if not modules.dits:
            if self.host_weight_plan is not None:
                raise RuntimeError(
                    "DLO received a loader-owned host-weight plan, but no DiT modules were discovered to consume it"
                )
            logger.warning("No DiT/transformer modules found, skipping distributed layer-wise offloading")
            return

        # Retrieve optional declarative OffloadPlan from the pipeline.
        # When present, replaces heuristic block discovery.
        plan = get_offload_plan(pipeline)

        if self.config.dlo_resident_layers and (plan is None or not plan.resident_dit_paths):
            logger.warning(
                "dlo_resident_layers=%d was requested, but this model declares no "
                "resident_dit_paths; all blocks will be streamed.",
                self.config.dlo_resident_layers,
            )

        # Storage selection belongs to the loader.  DLO consumes the exact
        # prevalidated plan that caused the loader to skip materialization;
        # without a plan, all weights must already come from the ordinary
        # loader (or, under HSDP, from the HSDP loader — the plan builder
        # declines HSDP, so chunk-owned blocks arrive as ordinary tensors
        # here).  The chunked transfer protocol is selected independently
        # below.
        self._using_mmap = self.host_weight_plan is not None
        self._using_rank_local_mmap = self._using_mmap and self.weight_shard_size <= 1
        if self._using_mmap:
            if self.host_weight_plan.backing_kind == "host_weight_runtime":
                carrier = self.host_weight_plan.lease_carrier
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
            elif self.host_weight_plan.backing_kind == "checkpoint_mmap":
                self._load_weights_via_mmap(
                    pipeline,
                    modules,
                    self.host_weight_plan,
                )
            else:
                raise ValueError(f"Unsupported DLO host-weight backing: {self.host_weight_plan.backing_kind}")
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

        # Keep VAE/encoders on CPU; move to GPU on-demand via hooks.
        # This saves ~4.3 GB HBM per card (VAE 1.3 + encoder 1.1 + sound 1.9)
        # during the DiT forward pass.  They are only needed briefly for
        # text-encoding (before DiT) and VAE-decode (after DiT).
        for enc, enc_name in zip(modules.encoders, modules.encoder_names):
            self._try_layerwise_offload_encoder(enc, enc_name, plan)
            self._register_on_demand_hook(
                enc, "encoder", stage_on_demand=plan is not None and enc_name in plan.on_demand_component_paths
            )
        for vae, vae_name in zip(modules.vaes, modules.vae_names):
            self._register_on_demand_hook(
                vae,
                "vae",
                stage_on_demand=(plan is not None and vae_name in plan.on_demand_component_paths),
            )

        # Move resident modules to GPU (small modules needed every forward)
        for name, module in zip(modules.resident_names, modules.resident_modules):
            try:
                module.to(self.device)
            except Exception as exc:
                logger.debug("Failed to move resident module %s to GPU: %s", name, exc)

        logger.info("Applying distributed layer-wise offloading on %s", modules.dit_names)

        # Collect all DiT module objects to detect submodules that are
        # already handled as a separate DiT module (avoids duplicate hooks).
        all_dit_modules = set(id(m) for m in modules.dits)

        # Apply hooks for each DiT module
        for i, dit_module in enumerate(modules.dits):
            dit_name = modules.dit_names[i]
            logger.info(f"Applying hooks on {dit_name} ({dit_module.__class__.__name__})")

            blocks_attr_names, blocks = get_blocks_from_dit(dit_module)

            if not blocks:
                logger.warning(
                    "Target layers (blocks) not found. Skipping offloading on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                dit_module.to(self.device)
                continue

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
                dit_name,
                blocks_attr_names,
                all_dit_modules,
                plan,
            )

            num_blocks = len(blocks)
            if num_blocks == 0:
                logger.info("All blocks for %s are resident; no streaming hooks required", dit_name)
                continue
            if num_blocks <= 1:
                logger.warning(
                    "#Streaming target layers (blocks) <= 1. Keeping the final block resident on %s (%s)",
                    dit_name,
                    dit_module.__class__.__name__,
                )
                self._resident_blocks.extend(blocks)
                continue

            # Hook creation is deferred: in the chunked AllGather path the
            # Host plan must enforce the pin cap and pack every block's Host
            # shard before any hook exists; in the rank-local path hooks
            # collect their own host storage, but the shared device buffers
            # are sized after every block's metadata is known.
            self._pending_block_groups.append((list(blocks), dit_name))

        if self._resident_blocks:
            self._resident_layer_group = PinnedResidentLayerGroup(
                self._resident_blocks,
                self.device,
                self.copy_stream,
                self.config.pin_cpu_memory,
                rank_local_mmap=self._using_rank_local_mmap,
                # When streaming groups exist they share the engine-wide
                # staging slots assigned below; without them the resident
                # group allocates its own bounded slots.
                defer_staging=bool(self._pending_block_groups),
                tensor_transforms=self._mmap_transforms_by_tensor_id,
            )
            pipeline._dlo_residency_controller = self

        if not self._pending_block_groups:
            self.enabled = bool(self._resident_blocks)
            if self._using_mmap and not self.enabled:
                # Nothing retains the mmap views: no streaming hooks, and no
                # resident group holding rank-local mmap sources.
                self._release_mmap_handles()
            return

        # 1. Ownership + stable block ids (small consecutive ints so the trace
        #    analyzer sees contiguous block ranges).
        ownership = self._resolve_chunk_ownership(pipeline)

        # Shared slot-group tracker: _shared_slot_group[slot] = group_id
        # that last wrote to that slot.  Group-first hooks use this to
        # skip sync-prefetch when the slot still contains their own data.
        shared_slot_group = [-1, -1]
        self._shared_slot_group = shared_slot_group

        if self._using_rank_local:
            # Rank-local path: no chunked packing.  Each hook collects its own
            # host storage (retained checkpoint mmap views, or a private
            # pinned copy for ordinary loader tensors) when it is created.
            self._allocate_shared_events()
            self._create_hook_groups([None, None], shared_slot_group)

            all_hooks: list[DistributedLayerwiseOffloadHook] = [
                hook for group in self._all_hook_groups for hook in group
            ]
            self._configure_hwr_transfer(all_hooks)
            unified_buffers = self._allocate_shared_rank_local_buffers(all_hooks)
            unified_cpu_staging = None
            cpu_staging_events = None
            if self._using_rank_local_mmap and not self._using_registered_mmap:
                unified_cpu_staging = self._allocate_shared_cpu_staging_buffers(
                    all_hooks,
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
            for hook in all_hooks:
                hook.gpu_buffers = unified_buffers
                hook._owns_buffers = False
                if unified_cpu_staging is not None and cpu_staging_events is not None:
                    hook.cpu_staging_buffers = unified_cpu_staging
                    hook.cpu_staging_events = cpu_staging_events
        else:
            # 2. Plan every final layout and enforce the pin cap before allocation.
            self._plan_chunk_manifests(ownership)

            # 3. Pack Host shards; this replaces the original parameter storage.
            self._prepare_host_storage(ownership)
            self._validate_manifest_consistency(ownership)

            # 4. Unified allocation sized from the *prepared* manifests (which may
            #    have been rebuilt as WHOLE_BLOCK by the fallback path).
            prepared_manifests = [part["manifest"] for part in self._prepared_host_parts.values()]
            unified_buffers = self._allocate_shared_buffers(prepared_manifests)
            unified_chunk_buffers = None
            if self.data_transport_backend is not None and self.data_transport_backend.requires_local_input:
                unified_chunk_buffers = self._allocate_shared_chunk_buffers(prepared_manifests)
            self._allocate_shared_events()

            # 5. Create the hooks in a circular sliding window:
            #    last block prefetches first block, block i prefetches block (i+1).
            #    All hooks share 2 global device buffers.
            self._create_hook_groups(unified_buffers, shared_slot_group, unified_chunk_buffers)

        # Prefetch first block of the FIRST module group only.
        # Subsequent groups share the same 2 device buffers; prefetching
        # them now would overwrite the first group's data in the shared
        # buffer (both groups default to slot 0).  Instead, subsequent
        # groups' first blocks remain as meta placeholders, and their
        # pre_forward will sync-prefetch on-demand via the is_materialized
        # check — by which point the first group's forward has completed
        # and its buffer slots are free.
        self._bootstrap_first_group()

        total_blocks = sum(len(b) for b in self._blocks)
        logger.info(
            f"Distributed layer-wise offloading enabled on {total_blocks} blocks "
            f"across {len(self._all_hook_groups)} group(s), "
            f"weight_shard_size={self.weight_shard_size}, unified shared_buffers=2"
        )

        self.enabled = True

        if self._using_mmap and not self._using_rank_local_mmap:
            # _prepare_host_storage packed every planned tensor into private
            # Host shards and rebound the parameters to placeholders, so the
            # mmap views (and their file handles) are no longer needed.
            # Rank-local mmap mode retains the views as the node-shared host
            # master until disable().
            self._release_mmap_handles()

        # Assign GPU buffers to sharded on-demand modules (VAE/encoders).
        # Each module gets a dedicated input (shard-sized) and output
        # (full-sized) buffer.  VAE/encoders run before/after DiT, never
        # concurrently, so peak HBM = max(DiT, VAE) not sum.
        for si in self._on_demand_shard_infos:
            out_bufs: dict[torch.dtype, torch.Tensor] = {}
            in_bufs: dict[torch.dtype, torch.Tensor] = {}
            for dtype, shard in si["cpu_shards"].items():
                # AllGather output = weight_shard_size * shard_size
                full_size = shard.numel() * self.weight_shard_size if self.weight_shard_size > 1 else shard.numel()
                out_bufs[dtype] = torch.empty(full_size, dtype=dtype, device=self.device)
                in_bufs[dtype] = torch.empty(shard.shape, dtype=dtype, device=self.device)
            si["gpu_output"] = out_bufs
            si["gpu_input"] = in_bufs
            _mb = sum(t.nelement() * t.element_size() for t in out_bufs.values()) / 1048576
            logger.info("Allocated %.0f MB GPU buffer for sharded on-demand module", _mb)

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

    def disable(self) -> None:
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
            or getattr(self, "_on_demand_handles", ())
            or self._resident_layer_group is not None
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

        if self._using_rank_local_mmap or has_registration:
            current_omni_platform.synchronize()

        # Join outstanding submits and retire every bootstrap/tail ticket
        # before closing the transport state.  One hook's failure must not
        # skip the others (review: #6374); the first error is re-raised after
        # every hook has been drained and closed.
        first_error: BaseException | None = None
        for group in self._all_hook_groups:
            for hook in group:
                try:
                    hook.drain_request()
                except BaseException as exc:  # noqa: BLE001
                    if first_error is None:
                        first_error = exc

        # Close every hook's transport so pinned Host shards are released and
        # the slot state machine is locked against further use.
        for group in self._all_hook_groups:
            for hook in group:
                try:
                    if hook.transport is not None:
                        hook.transport.close()
                except BaseException as exc:  # noqa: BLE001
                    if first_error is None:
                        first_error = exc
                hook.cpu_shards = {}  # drop pinned-memory references
                hook.cpu_sources = {}  # drop retained mmap views

        if self.data_transport_backend is not None:
            self.data_transport_backend.close()
            self.data_transport_backend = None

        for blocks in self._blocks:
            for block in blocks:
                remove_distributed_block_hook(block)

        for h in getattr(self, "_on_demand_handles", []):
            h.remove()
        self._on_demand_handles = []
        self._on_demand_shard_infos = []

        self.offload_resident_layers()
        self._blocks.clear()
        self._all_hook_groups.clear()
        self._resident_blocks.clear()
        self._resident_layer_group = None
        self._release_registered_mmap()
        self._release_mmap_handles()
        # The backend-level prepared Host parts alias the same pinned shards
        # as the hooks' cpu_shards dropped above; clear them together with
        # the pin-budget accounting so disable actually releases the pinned
        # Host memory and leaves no stale metrics behind (review: #6374).
        self._prepared_host_parts.clear()
        self._planned_manifests = {}
        self.pin_budget = None
        self._using_mmap = False
        self._using_rank_local_mmap = False
        self._using_registered_mmap = False
        self.enabled = False

        # Drain the single-worker executor.  Pending submits were already
        # joined above; shutdown() here just releases the thread cleanly so
        # repeated enable/disable cycles don't accumulate threads.
        self._prefetch_executor.shutdown(wait=False)
        self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dlo_prefetch"
        )

        logger.info("Distributed layer-wise offloading disabled")
        if first_error is not None:
            raise first_error

    def _allocate_shared_buffers(
        self,
        manifests: list[PartManifest],
    ) -> list[dict[torch.dtype, torch.Tensor] | None]:
        """Allocate exactly 2 shared full-output buffers for the largest block.

        Both slots are sized to the maximum *padded* per-dtype extent across
        every group, because ``all_gather_into_tensor`` writes
        ``weight_shard_size * local_numel`` elements per chunk and the last
        chunk of a block is padded up to that multiple.
        """
        max_sizes: dict[torch.dtype, int] = {}
        for manifest in manifests:
            for dtype_manifest in manifest.dtypes:
                dtype = dtype_manifest.dtype
                max_sizes[dtype] = max(max_sizes.get(dtype, 0), dtype_manifest.padded_numel)

        shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]
        for slot in range(2):
            gpu_weights: dict[torch.dtype, torch.Tensor] = {}
            for dtype, total_numel in max_sizes.items():
                gpu_weights[dtype] = torch.empty(total_numel, dtype=dtype, device=self.device)
            shared_buffers[slot] = gpu_weights

        logger.info(
            "Allocated 2 shared device buffers (max block size: %s)",
            {str(k): f"{v * _dtype_size(k) / 1024 / 1024:.1f}MB" for k, v in max_sizes.items()},
        )
        return shared_buffers

    def _allocate_shared_rank_local_buffers(
        self,
        hooks: list[DistributedLayerwiseOffloadHook],
    ) -> list[dict[torch.dtype, torch.Tensor] | None]:
        """Allocate exactly 2 shared device buffers sized to the largest block.

        Rank-local mode has no AllGather, so the slots hold each block's exact
        flattened extent with no per-shard padding.
        """
        max_sizes: dict[torch.dtype, int] = {}
        for hook in hooks:
            for dtype, metas in hook.metadata.items():
                total = sum(m["numel"] for m in metas)
                max_sizes[dtype] = max(max_sizes.get(dtype, 0), total)

        shared_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]
        for slot in range(2):
            gpu_weights: dict[torch.dtype, torch.Tensor] = {}
            for dtype, total_numel in max_sizes.items():
                gpu_weights[dtype] = torch.empty(total_numel, dtype=dtype, device=self.device)
            shared_buffers[slot] = gpu_weights

        logger.info(
            "Allocated 2 shared rank-local device buffers (max block size: %s)",
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
                buffer = torch.empty(total_numel, dtype=dtype, device="cpu")
                if pin_memory:
                    buffer = buffer.pin_memory()
                buffers[dtype] = buffer
            shared_staging[slot] = buffers

        logger.info(
            "Allocated 2 shared host staging buffers for rank-local mmap (max block size: %s, pinned=%s)",
            {str(k): f"{v * _dtype_size(k) / 1024 / 1024:.1f}MB" for k, v in max_sizes.items()},
            pin_memory,
        )
        return shared_staging

    def _allocate_shared_chunk_buffers(
        self,
        manifests: list[PartManifest],
    ) -> list[dict[torch.dtype, torch.Tensor] | None]:
        """Allocate 2 shared local-chunk (AllGather input) buffers.

        These are indexed by *input* slot, not output slot: chunk ``i`` uses
        input slot ``i % 2`` so the H2D of chunk ``i+1`` overlaps the AllGather
        of chunk ``i``.  Reusing the same two device addresses for every block
        lets HCCL reuse its internal communication buffers.
        """
        max_chunk_sizes: dict[torch.dtype, int] = {}
        for manifest in manifests:
            for dtype_manifest in manifest.dtypes:
                dtype = dtype_manifest.dtype
                max_chunk_sizes[dtype] = max(max_chunk_sizes.get(dtype, 0), dtype_manifest.local_chunk_numel)

        shared_chunk_buffers: list[dict[torch.dtype, torch.Tensor] | None] = [None, None]
        for slot in range(2):
            chunk_bufs: dict[torch.dtype, torch.Tensor] = {}
            for dtype, numel in max_chunk_sizes.items():
                chunk_bufs[dtype] = torch.empty(numel, dtype=dtype, device=self.device)
            shared_chunk_buffers[slot] = chunk_bufs

        logger.info(
            "Allocated 2 shared local chunk buffers (max chunk size: %s)",
            {str(k): f"{v * _dtype_size(k) / 1024 / 1024:.1f}MB" for k, v in max_chunk_sizes.items()},
        )
        return shared_chunk_buffers
