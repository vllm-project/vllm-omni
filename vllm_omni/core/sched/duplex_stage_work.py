# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-step working spans of duplex chunks in a stage engine process (RFC #8325).

A duplex append whose envelope carries ``duplex.perf_chunk = [epoch, seq]`` is
measured: every engine step that includes the request records one host-side span,
from ``schedule()`` returning the batch to ``update_from_output()`` finishing it,
tagged with the chunk the request was working on. The spans ride back to the API
process on ``OmniEngineCoreOutput.stage_work`` as ``[epoch, seq, start_ns,
duration_ns]`` rows, the producing step's chunk last. Requests without a chunk id
are never touched. Until a stage first sees a chunk id, each step checks only
its newly scheduled requests, so collection costs one short loop when it is off.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.request import Request

PERF_CHUNK_KEY = "perf_chunk"
# Set on a stage Request when an update or a connector payload names its chunk.
_REQUEST_CHUNK_ATTR = "_omni_duplex_perf_chunk"
_REQUEST_SPANS_ATTR = "_omni_duplex_stage_work"
# Steps whose update never arrives (engine reset) must not grow the table forever.
_MAX_IN_FLIGHT_STEPS = 64
# Set once an update or connector payload in this process carries a chunk id.
_perf_chunk_seen = False


def parse_perf_chunk(value: object) -> tuple[int, int] | None:
    """Return ``(epoch, seq)`` from a ``perf_chunk`` value, or None."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    epoch, seq = value
    if isinstance(epoch, bool) or isinstance(seq, bool):
        return None
    try:
        return int(epoch), int(seq)
    except (TypeError, ValueError):
        return None


def perf_chunk_from_buffer(buffer: object) -> tuple[int, int] | None:
    """Read the chunk id from a ``model_intermediate_buffer`` duplex envelope."""
    duplex = buffer.get("duplex") if isinstance(buffer, Mapping) else None
    return parse_perf_chunk(duplex.get(PERF_CHUNK_KEY)) if isinstance(duplex, Mapping) else None


def set_request_perf_chunk(request: Request, chunk: tuple[int, int] | None) -> None:
    """Record the chunk an update or payload hands to ``request``; None keeps the previous one."""
    global _perf_chunk_seen
    if chunk is not None:
        setattr(request, _REQUEST_CHUNK_ATTR, chunk)
        _perf_chunk_seen = True


def request_perf_chunk(request: Request) -> tuple[int, int] | None:
    """The chunk ``request`` currently works on, or None when it is not measured."""
    chunk = getattr(request, _REQUEST_CHUNK_ATTR, None)
    if isinstance(chunk, tuple):
        return chunk
    return perf_chunk_from_buffer(getattr(request, "model_intermediate_buffer", None))


class DuplexStageWorkRecorder:
    """Times scheduler steps for measured duplex requests and attaches the spans to outputs."""

    def __init__(self, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock = clock
        self._active = False
        # id(scheduler_output) -> (begin_ns, {req_id: (request, chunk)})
        self._in_flight: dict[int, tuple[int, dict[str, tuple[Request, tuple[int, int]]]]] = {}

    def step_begin(self, scheduler_output: SchedulerOutput, requests: Mapping[str, Request]) -> None:
        if not self._active:
            # Fast path while stats are off: a measured request first shows up as
            # a new request whose envelope has a chunk id, or through an update.
            new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None) or ()
            if not _perf_chunk_seen and not any(
                (request := requests.get(data.req_id)) is not None and request_perf_chunk(request) is not None
                for data in new_reqs
            ):
                return
            self._active = True
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        entries = {}
        for req_id in scheduled:
            request = requests.get(req_id)
            chunk = request_perf_chunk(request) if request is not None else None
            if chunk is not None:
                entries[req_id] = (request, chunk)
        if not entries:
            return
        if len(self._in_flight) >= _MAX_IN_FLIGHT_STEPS:
            self._in_flight.pop(next(iter(self._in_flight)))
        self._in_flight[id(scheduler_output)] = (self._clock(), entries)

    def step_end(self, scheduler_output: SchedulerOutput, engine_core_outputs: Mapping[int, EngineCoreOutputs]) -> None:
        record = self._in_flight.pop(id(scheduler_output), None)
        if record is None:
            return
        begin_ns, entries = record
        end_ns = self._clock()
        by_id = {}
        for request, chunk in entries.values():
            _add_span(request, chunk, begin_ns, end_ns)
            by_id[request.request_id] = request
        for client_outputs in engine_core_outputs.values():
            for output in getattr(client_outputs, "outputs", ()):
                request = by_id.get(output.request_id)
                if request is not None and hasattr(output, "stage_work"):
                    spans = getattr(request, _REQUEST_SPANS_ATTR, None)
                    if spans:
                        output.stage_work = spans
                        setattr(request, _REQUEST_SPANS_ATTR, [])

    def step_chunk(self, scheduler_output: SchedulerOutput, req_id: str) -> tuple[int, int] | None:
        """The chunk ``req_id`` worked on in this step, fixed at ``schedule()``.

        ``update_from_output`` can apply a queued update before the step's output
        is saved for the next stage; this is the chunk that output belongs to.
        """
        record = self._in_flight.get(id(scheduler_output))
        entry = record[1].get(req_id) if record is not None else None
        return entry[1] if entry is not None else None


def _add_span(request: Request, chunk: tuple[int, int], begin_ns: int, end_ns: int) -> None:
    """Append one step span; touching or overlapping spans of the same chunk coalesce."""
    spans: list[list[int]] | None = getattr(request, _REQUEST_SPANS_ATTR, None)
    if spans is None:
        spans = []
        setattr(request, _REQUEST_SPANS_ATTR, spans)
    if spans:
        last = spans[-1]
        if (last[0], last[1]) == chunk and begin_ns <= last[2] + last[3]:
            last[3] = max(last[3], end_ns - last[2])
            return
    spans.append([chunk[0], chunk[1], begin_ns, max(0, end_ns - begin_ns)])


def stage_work_rows(rows: Iterable[object] | None) -> list[tuple[int, int, int, int]]:
    """Validate ``OmniEngineCoreOutput.stage_work`` rows on the receiving side."""
    result: list[tuple[int, int, int, int]] = []
    for row in rows or ():
        if isinstance(row, list | tuple) and len(row) == 4:
            try:
                epoch, seq, start_ns, duration_ns = (int(value) for value in row)
            except (TypeError, ValueError):
                continue
            result.append((epoch, seq, start_ns, duration_ns))
    return result
