# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict


class StreamEdgeSnapshot(TypedDict):
    from_stage: int
    to_stage: int
    segment_index: int
    first_chunk_seq: int | None
    clock_domains: dict[str, str]
    producer_first_nonempty_emit_ms: float | None
    edge_first_send_complete_ms: float | None
    consumer_first_accept_ms: float | None
    consumer_first_nonempty_output_ms: float | None


class StreamEdgeEvent(str, Enum):
    PRODUCER_EMIT = "producer_first_nonempty_emit"
    SEND_COMPLETE = "edge_first_send_complete"
    CONSUMER_ACCEPT = "consumer_first_accept"
    CONSUMER_OUTPUT = "consumer_first_nonempty_output"


@dataclass(frozen=True)
class StreamEdgeKey:
    from_stage: int
    to_stage: int
    segment_index: int = 0

    def __post_init__(self) -> None:
        if min(self.from_stage, self.to_stage, self.segment_index) < 0 or self.from_stage == self.to_stage:
            raise ValueError("A stream edge requires distinct non-negative stages and a non-negative segment")


@dataclass(frozen=True)
class _Observation:
    elapsed_ms: float
    clock_domain: str


@dataclass
class _FirstEvents:
    observations: dict[StreamEdgeEvent, _Observation] = field(default_factory=dict)
    first_chunk_seq: int | None = None


@dataclass(frozen=True)
class StreamEdgeRecorder:
    """A handle fenced by one request owner and one edge's segment generation."""

    _owner: "RequestStreamEdgeEvents"
    key: StreamEdgeKey

    def record(
        self,
        event: StreamEdgeEvent,
        *,
        elapsed_ms: float,
        clock_domain: str,
        meaningful: bool,
        chunk_seq: int | None = None,
    ) -> bool:
        """Record a qualified event once; return False for empty, duplicate or stale work.

        Callers classify model payloads and observe commit/admission outcomes.
        In particular, a failed send must never call this with SEND_COMPLETE.
        Values are elapsed milliseconds from the request/segment origin named
        by clock_domain, never wall-clock or absolute monotonic timestamps.
        """
        owner = self._owner
        if not meaningful or owner._closed or owner._segments.get((self.key.from_stage, self.key.to_stage)) != self.key:
            return False
        if not isinstance(event, StreamEdgeEvent):
            raise ValueError("Unknown stream-edge event")
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise ValueError("Event elapsed time must be finite and non-negative")
        if not clock_domain or not clock_domain.strip():
            raise ValueError("Event clock domain must be declared")
        if chunk_seq is not None and chunk_seq < 0:
            raise ValueError("Chunk sequence must be non-negative")
        state = owner._events.setdefault(self.key, _FirstEvents())
        if event in state.observations:
            return False
        state.observations[event] = _Observation(float(elapsed_ms), clock_domain)
        if event == StreamEdgeEvent.PRODUCER_EMIT:
            state.first_chunk_seq = chunk_seq
        return True


class RequestStreamEdgeEvents:
    """First-event state owned by one ClientRequestState, never a global ID map.

    Each recorder belongs to this object even when an external ID is reused.
    The entrypoint event loop serializes access. Workers must send observations
    to that owner rather than mutate it from background threads.
    """

    def __init__(self) -> None:
        self._events: dict[StreamEdgeKey, _FirstEvents] = {}
        self._segments: dict[tuple[int, int], StreamEdgeKey] = {}
        self._closed = False

    def start_segment(self, key: StreamEdgeKey) -> StreamEdgeRecorder:
        if self._closed:
            raise RuntimeError("Request stream-edge metrics have been released")
        edge = (key.from_stage, key.to_stage)
        current = self._segments.get(edge)
        if current is not None and key.segment_index < current.segment_index:
            raise ValueError("Cannot reopen a retired stream-edge segment")
        self._segments[edge] = key
        return StreamEdgeRecorder(self, key)

    def snapshot(self) -> dict[str, StreamEdgeSnapshot]:
        """Detach a JSON-compatible snapshot; missing events stay None."""
        snapshot: dict[str, StreamEdgeSnapshot] = {}
        for key, state in self._events.items():
            record: StreamEdgeSnapshot = {
                "from_stage": key.from_stage,
                "to_stage": key.to_stage,
                "segment_index": key.segment_index,
                "first_chunk_seq": state.first_chunk_seq,
                "clock_domains": {},
                "producer_first_nonempty_emit_ms": None,
                "edge_first_send_complete_ms": None,
                "consumer_first_accept_ms": None,
                "consumer_first_nonempty_output_ms": None,
            }
            producer = state.observations.get(StreamEdgeEvent.PRODUCER_EMIT)
            send = state.observations.get(StreamEdgeEvent.SEND_COMPLETE)
            accept = state.observations.get(StreamEdgeEvent.CONSUMER_ACCEPT)
            output = state.observations.get(StreamEdgeEvent.CONSUMER_OUTPUT)
            record["producer_first_nonempty_emit_ms"] = producer.elapsed_ms if producer is not None else None
            record["edge_first_send_complete_ms"] = send.elapsed_ms if send is not None else None
            record["consumer_first_accept_ms"] = accept.elapsed_ms if accept is not None else None
            record["consumer_first_nonempty_output_ms"] = output.elapsed_ms if output is not None else None
            record["clock_domains"] = {
                event.value: observation.clock_domain for event, observation in state.observations.items()
            }
            snapshot[f"{key.from_stage}->{key.to_stage}:{key.segment_index}"] = record
        return snapshot

    def elapsed_between(self, key: StreamEdgeKey, start: StreamEdgeEvent, end: StreamEdgeEvent) -> float | None:
        """Only derive a delta for events with a shared origin and clock domain."""
        state = self._events.get(key)
        if state is None:
            return None
        first, last = state.observations.get(start), state.observations.get(end)
        if first is None or last is None or first.clock_domain != last.clock_domain:
            return None
        delta = last.elapsed_ms - first.elapsed_ms
        return delta if delta >= 0 else None

    def close(self) -> None:
        """Idempotent canonical cleanup; retained handles cannot recreate state."""
        self._closed = True
        self._events.clear()
        self._segments.clear()
