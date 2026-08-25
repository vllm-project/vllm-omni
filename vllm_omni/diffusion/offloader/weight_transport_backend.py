# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pluggable data-plane backends for chunked diffusion weight transport."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from vllm_omni.platforms import current_omni_platform

from .chunked_transport import (
    ChunkMeta,
    PartManifest,
    TransportBackendKind,
)


@dataclass(frozen=True)
class TransportCapability:
    world_size: int
    rank: int
    global_ranks: tuple[int, ...]
    native_persistent: bool = False


@dataclass(frozen=True)
class SupportResult:
    supported: bool
    reason: str | None = None


@dataclass(frozen=True)
class TransportSelection:
    requested_backend: TransportBackendKind
    effective_backend: TransportBackendKind


@dataclass(frozen=True)
class TransportStreams:
    copy: Any
    communication: Any


@dataclass(frozen=True)
class ChunkEvents:
    h2d_done: Any
    transport_done: Any
    input_reusable: Any | None = None


@dataclass(frozen=True)
class ChunkCompletion:
    event: Any
    stream: Any


@dataclass
class BackendCounters:
    submitted_parts: int = 0
    submitted_chunks: int = 0
    host_h2d_bytes: int = 0
    fabric_bytes: int = 0
    schedule_builds: int = 0
    schedule_replays: int = 0
    backend_chunks: dict[str, int] = field(default_factory=dict)


TraceFactory = Callable[[str], AbstractContextManager[Any]]


class WeightTransportBackend(Protocol):
    kind: TransportBackendKind
    requires_local_input: bool
    counters: BackendCounters

    def supports(self, capability: TransportCapability, plan: PartManifest | None = None) -> SupportResult: ...

    def begin_part(self, streams: TransportStreams, prior_last_use: Any | None) -> None: ...

    def submit_chunk(
        self,
        *,
        source: torch.Tensor | None,
        local_input: torch.Tensor | None,
        full_output: torch.Tensor,
        chunk_meta: ChunkMeta,
        streams: TransportStreams,
        events: ChunkEvents,
        group: torch.distributed.ProcessGroup | None,
        generation: int,
        non_blocking: bool,
        trace: TraceFactory,
    ) -> ChunkCompletion: ...

    def finalize_part(
        self,
        completions: Sequence[ChunkCompletion],
        *,
        ready_event: Any,
        streams: TransportStreams,
    ) -> Any: ...

    def reset_generation(self, generation: int) -> None: ...

    def reset_counters(self) -> None: ...

    def close(self) -> None: ...


def _support_backend(
    backend: TransportBackendKind,
    capability: TransportCapability,
) -> SupportResult:
    if backend is TransportBackendKind.REFERENCE:
        return SupportResult(True)

    if backend is TransportBackendKind.GROUP_SCATTER_AG:
        if capability.world_size <= 1:
            return SupportResult(False, "group_scatter_ag requires an FS group larger than one rank")
        return SupportResult(True)

    if backend is TransportBackendKind.GROUP_PERSISTENT:
        if capability.world_size <= 1:
            return SupportResult(False, "group_persistent requires an FS group larger than one rank")
        if not capability.native_persistent:
            return SupportResult(False, "group_persistent requires native NPU graph support (torch.npu.NPUGraph)")
        return SupportResult(True)

    return SupportResult(False, f"unsupported backend: {backend.value}")


def select_transport(
    requested_backend: TransportBackendKind,
    capability: TransportCapability,
) -> TransportSelection:
    if requested_backend is TransportBackendKind.AUTO:
        # group_persistent stays opt-in: it only pays off once a reference
        # trace proves per-chunk launch overhead, so auto keeps the safe
        # chunked AllGather schedule.
        candidate = (
            TransportBackendKind.GROUP_SCATTER_AG if capability.world_size > 1 else TransportBackendKind.REFERENCE
        )
    else:
        candidate = requested_backend

    support = _support_backend(candidate, capability)
    if support.supported:
        return TransportSelection(
            requested_backend=requested_backend,
            effective_backend=candidate,
        )

    # An explicitly requested backend that this host cannot run is a hard
    # configuration error, never a silent fallback.
    raise ValueError(f"dlo_transport_backend={candidate.value} is unsupported on this host: {support.reason}")


class _BaseBackend:
    kind = TransportBackendKind.REFERENCE
    requires_local_input = True
    writes_output_on_copy = False

    def __init__(self, capability: TransportCapability) -> None:
        self.capability = capability
        self.counters = BackendCounters()
        self._generation = -1
        self._closed = False

    def supports(self, capability: TransportCapability, plan: PartManifest | None = None) -> SupportResult:
        del plan
        return _support_backend(self.kind, capability)

    def begin_part(self, streams: TransportStreams, prior_last_use: Any | None) -> None:
        if self._closed:
            raise RuntimeError("weight transport backend is closed")
        self.counters.submitted_parts += 1
        if prior_last_use is None:
            return
        streams.communication.wait_event(prior_last_use)
        if self.writes_output_on_copy:
            streams.copy.wait_event(prior_last_use)

    def _count_chunk(self, host_bytes: int, fabric_bytes: int) -> None:
        self.counters.submitted_chunks += 1
        self.counters.host_h2d_bytes += host_bytes
        self.counters.fabric_bytes += fabric_bytes
        key = self.kind.value
        self.counters.backend_chunks[key] = self.counters.backend_chunks.get(key, 0) + 1

    def finalize_part(
        self,
        completions: Sequence[ChunkCompletion],
        *,
        ready_event: Any,
        streams: TransportStreams,
    ) -> Any:
        stream = completions[-1].stream if completions else streams.communication
        with current_omni_platform.stream(stream):
            ready_event.record(stream)
        return ready_event

    def reset_generation(self, generation: int) -> None:
        if generation < self._generation:
            raise RuntimeError(f"transport generation moved backwards: {generation} < {self._generation}")
        self._generation = generation

    def reset_counters(self) -> None:
        self.counters = BackendCounters()

    def close(self) -> None:
        self._closed = True


class ReferenceBackend(_BaseBackend):
    kind = TransportBackendKind.REFERENCE
    _transport_trace_name = "all_gather"

    def __init__(self, capability: TransportCapability) -> None:
        super().__init__(capability)
        self.requires_local_input = capability.world_size > 1
        self.writes_output_on_copy = capability.world_size <= 1

    def submit_chunk(
        self,
        *,
        source: torch.Tensor | None,
        local_input: torch.Tensor | None,
        full_output: torch.Tensor,
        chunk_meta: ChunkMeta,
        streams: TransportStreams,
        events: ChunkEvents,
        group: torch.distributed.ProcessGroup | None,
        generation: int,
        non_blocking: bool,
        trace: TraceFactory,
    ) -> ChunkCompletion:
        del generation
        if source is None:
            raise RuntimeError("reference transport requires a local Host source")

        if self.capability.world_size <= 1:
            if source.numel() != chunk_meta.padded_numel:
                raise RuntimeError("single-rank reference transport requires one full padded Host chunk")
            with current_omni_platform.stream(streams.copy):
                with trace("h2d"):
                    full_output.copy_(source, non_blocking=non_blocking)
                events.h2d_done.record(streams.copy)
            self._count_chunk(source.numel() * source.element_size(), 0)
            return ChunkCompletion(events.h2d_done, streams.copy)

        if local_input is None or group is None:
            raise RuntimeError("reference FS transport requires a local input buffer and process group")
        return self._submit_fs_chunk(
            source=source,
            local_input=local_input,
            full_output=full_output,
            chunk_meta=chunk_meta,
            streams=streams,
            events=events,
            non_blocking=non_blocking,
            trace=trace,
            collective=lambda: torch.distributed.all_gather_into_tensor(full_output, local_input, group=group),
        )

    def _submit_fs_chunk(
        self,
        *,
        source: torch.Tensor,
        local_input: torch.Tensor,
        full_output: torch.Tensor,
        chunk_meta: ChunkMeta,
        streams: TransportStreams,
        events: ChunkEvents,
        non_blocking: bool,
        trace: TraceFactory,
        collective: Callable[[], None],
    ) -> ChunkCompletion:
        """Shared H2D(chunk) -> collective(chunk) schedule for FS-sharded input."""
        if events.input_reusable is not None:
            streams.copy.wait_event(events.input_reusable)
        with current_omni_platform.stream(streams.copy):
            with trace("h2d"):
                local_input.copy_(source, non_blocking=non_blocking)
            events.h2d_done.record(streams.copy)

        streams.communication.wait_event(events.h2d_done)
        with current_omni_platform.stream(streams.communication):
            with trace(self._transport_trace_name):
                collective()
            events.transport_done.record(streams.communication)

        element_size = full_output.element_size()
        fabric_bytes = 0
        if self.capability.rank == 0:
            fabric_bytes = chunk_meta.padded_numel * element_size * (self.capability.world_size - 1)
        self._count_chunk(source.numel() * source.element_size(), fabric_bytes)
        return ChunkCompletion(events.transport_done, streams.communication)


class GroupScatterAllGatherBackend(ReferenceBackend):
    """Safe H2D plus all-gather implementation for FS-sharded Host input.

    This is intentionally the reference scheduling contract.  Some HCCL
    versions do not support multiple outstanding all-gathers on one process
    group when ordering is represented only with caller-stream events.
    Keeping each collective's normal completion semantics prevents an input
    slot from being overwritten while HCCL still reads it.
    """

    kind = TransportBackendKind.GROUP_SCATTER_AG


class GroupPersistentBackend(ReferenceBackend):
    """Persistent launch state for stable FS chunk schedules.

    One stable H2D + HCCL chunk schedule is captured in an NPUGraph per
    double-buffer slot and replayed afterwards.  ``native_persistent`` is
    validated at selection time, so a capture failure is a hard error, not
    a silent runtime fallback.
    """

    kind = TransportBackendKind.GROUP_PERSISTENT
    _transport_trace_name = "persistent_replay"

    def __init__(self, capability: TransportCapability) -> None:
        super().__init__(capability)
        self._graphs: dict[tuple[Any, ...], Any] = {}
        self._graph_pool = None
        graph_pool_handle = getattr(torch.npu, "graph_pool_handle", None)
        if callable(graph_pool_handle):
            self._graph_pool = graph_pool_handle()

    def submit_chunk(
        self,
        *,
        source: torch.Tensor | None,
        local_input: torch.Tensor | None,
        full_output: torch.Tensor,
        chunk_meta: ChunkMeta,
        streams: TransportStreams,
        events: ChunkEvents,
        group: torch.distributed.ProcessGroup | None,
        generation: int,
        non_blocking: bool,
        trace: TraceFactory,
    ) -> ChunkCompletion:
        del generation
        if source is None or local_input is None or group is None:
            raise RuntimeError("group_persistent requires Host source, local input, and process group")

        key = (
            id(group),
            local_input.dtype,
            local_input.numel(),
            local_input.data_ptr(),
            full_output.numel(),
            full_output.data_ptr(),
            chunk_meta.padded_numel,
        )

        def collective() -> None:
            graph = self._graphs.get(key)
            if graph is None:
                self.counters.schedule_builds += 1
                # All ranks must enter capture in the same order. Prime the
                # HCCL kernel eagerly first, then capture the stable operation
                # using the shared NPU graph pool.
                torch.distributed.barrier(group=group)
                torch.distributed.all_gather_into_tensor(full_output, local_input, group=group)
                torch.npu.synchronize()
                torch.distributed.barrier(group=group)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                graph_kwargs = {"stream": streams.communication}
                if self._graph_pool is not None:
                    graph_kwargs["pool"] = self._graph_pool
                with torch.npu.graph(graph, **graph_kwargs):
                    torch.distributed.all_gather_into_tensor(full_output, local_input, group=group)
                torch.npu.synchronize()
                self._graphs[key] = graph
            else:
                self.counters.schedule_replays += 1
            graph.replay()

        return self._submit_fs_chunk(
            source=source,
            local_input=local_input,
            full_output=full_output,
            chunk_meta=chunk_meta,
            streams=streams,
            events=events,
            non_blocking=non_blocking,
            trace=trace,
            collective=collective,
        )

    def close(self) -> None:
        self._graphs.clear()
        super().close()


def create_transport_backend(
    selection: TransportSelection,
    capability: TransportCapability,
) -> WeightTransportBackend:
    backend_type: type[_BaseBackend]
    if selection.effective_backend is TransportBackendKind.REFERENCE:
        backend_type = ReferenceBackend
    elif selection.effective_backend is TransportBackendKind.GROUP_SCATTER_AG:
        backend_type = GroupScatterAllGatherBackend
    elif selection.effective_backend is TransportBackendKind.GROUP_PERSISTENT:
        backend_type = GroupPersistentBackend
    else:
        raise RuntimeError(
            f"effective transport backend {selection.effective_backend.value} has no validated implementation"
        )
    return backend_type(capability)
