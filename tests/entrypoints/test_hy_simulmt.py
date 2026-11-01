# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deterministic tests for the metadata contract, without a serving backend."""

import asyncio
import heapq
from dataclasses import replace

import pytest

from vllm_omni.entrypoints.hy_simulmt import (
    Admission,
    Hypothesis,
    InferenceBackend,
    InferenceRequest,
    Revision,
    SessionPhase,
    StreamingConfig,
    StreamingSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def session() -> StreamingSession:
    return StreamingSession(StreamingConfig(language="zh"))


def test_defaults_and_limits() -> None:
    config = StreamingConfig(language="en")
    assert (config.cadence_samples, config.reset_samples, config.max_pending_requests) == (25600, 480000, 2)
    assert StreamingConfig(language="zh", cadence_ms=1, reset_ms=1, max_pending_requests=1).reset_samples == 16
    assert StreamingConfig(language="zh", cadence_ms=30000, max_pending_requests=8).cadence_samples == 480000


@pytest.mark.parametrize(
    "field,value",
    [
        ("language", ""),
        ("language", "fr"),
        ("language", None),
        ("cadence_ms", 0),
        ("cadence_ms", -1),
        ("cadence_ms", 30001),
        ("cadence_ms", True),
        ("cadence_ms", 1.6),
        ("cadence_ms", float("nan")),
        ("cadence_ms", float("inf")),
        ("cadence_ms", "1600"),
        ("reset_ms", 1599),
        ("reset_ms", 30001),
        ("reset_ms", False),
        ("max_pending_requests", 0),
        ("max_pending_requests", 9),
        ("max_pending_requests", True),
        ("max_pending_requests", 2.0),
    ],
)
def test_invalid_config(field, value) -> None:
    with pytest.raises(ValueError):
        replace(StreamingConfig(language="zh"), **{field: value})


@pytest.mark.parametrize("count", [0, -1, True, 1.5, 480001])
def test_invalid_append_is_atomic(count) -> None:
    state = session()
    with pytest.raises(ValueError):
        state.append_samples(count)
    assert (state.segment_samples, state.context_samples) == (0, 0)


def test_cadence_and_backpressure() -> None:
    state = session()
    state.append_samples(25599)
    assert not state.partial_due
    with pytest.raises(ValueError, match="not due"):
        state.partial(Admission.ACCEPTED)
    state.append_samples(1)
    assert state.partial_due
    assert state.partial(Admission.BACKPRESSURE) is None
    assert state.partial_due
    first = state.partial(Admission.ACCEPTED)
    assert first == Revision(0, 0, 1, "partial")
    assert not state.partial_due
    state.append_samples(100000)
    second = state.partial(Admission.COALESCED)
    assert second == Revision(0, 0, 2, "partial")
    assert not state.partial_due  # No burst of catch-up requests.
    assert state.complete(second, "replacement") == Hypothesis(second, "replacement")
    assert state.complete(first, "stale") is None
    assert state.complete(second, "duplicate") is None


def test_backpressured_partial_does_not_invalidate_prior_result() -> None:
    state = session()
    state.append_samples(25600)
    first = state.partial(Admission.ACCEPTED)
    assert first is not None
    state.append_samples(25600)
    state.partial(Admission.BACKPRESSURE)
    assert state.complete(first, "still valid") is not None


def test_queued_revision_does_not_suppress_running_partial() -> None:
    state = session()
    state.append_samples(25600)
    first = state.partial(Admission.ACCEPTED)
    assert first is not None
    state.append_samples(25600)
    second = state.partial(Admission.ACCEPTED)
    assert second is not None
    assert state.is_current(first)
    assert state.complete(first, "useful progress") == Hypothesis(first, "useful progress")
    assert not state.is_current(first)
    assert state.is_current(second)
    assert state.complete(second, "newer progress") == Hypothesis(second, "newer progress")
    assert state.complete(first, "duplicate") is None
    assert state.complete(replace(second, revision_id=3), "never admitted") is None
    assert state.complete(replace(second, status="final"), "wrong status") is None


@pytest.mark.parametrize("inference_ms", [2000, 4000, 8000])
def test_sustained_slow_inference_publishes_partials(inference_ms: int) -> None:
    """Virtual time: one running request and one coalescing pending slot."""
    state = session()
    events = [(time, "append") for time in range(1600, 30000, 1600)] + [(30000, "commit")]
    heapq.heapify(events)
    running: Revision | None = None
    pending: Revision | None = None
    published: list[tuple[int, Hypothesis]] = []
    while events:
        time, action = heapq.heappop(events)
        if action == "append":
            state.append_samples(25600)
            admission = Admission.COALESCED if pending else Admission.ACCEPTED
            pending = state.partial(admission)
        elif action == "commit":
            state.append_samples(state.remaining_samples)
            pending = state.commit(Admission.ACCEPTED, end_input=True)
        else:
            assert running is not None
            result = state.complete(running, f"translation-{running.revision_id}")
            if result is not None:
                published.append((time, result))
            running = None
        if running is None and pending is not None:
            running, pending = pending, None
            heapq.heappush(events, (time + inference_ms, "complete"))
    partials = [(time, result) for time, result in published if result.revision.status == "partial"]
    assert len(partials) >= 2
    assert partials[0][0] == 1600 + inference_ms
    assert all(time < 30000 for time, _ in partials)
    ids = [result.revision.revision_id for _, result in published]
    assert ids == sorted(set(ids))
    assert published[-1][1].revision.status == "final"
    assert state.phase is SessionPhase.CLOSED


@pytest.mark.parametrize("end_input", [False, True])
def test_final_admission_and_barrier(end_input: bool) -> None:
    state = session()
    state.append_samples(25600)
    partial = state.partial(Admission.ACCEPTED)
    assert partial is not None
    assert state.commit(Admission.BACKPRESSURE, end_input=end_input) is None
    assert state.phase is SessionPhase.OPEN
    assert state.segment_samples == 25600
    with pytest.raises(ValueError, match="explicit admission"):
        state.commit(Admission.COALESCED, end_input=end_input)
    assert state.phase is SessionPhase.OPEN
    final = state.commit(Admission.ACCEPTED, end_input=end_input)
    assert final == Revision(0, 0, 2, "final")
    assert state.phase is (SessionPhase.DRAINING if end_input else SessionPhase.FINALIZING)
    assert not state.partial_due
    with pytest.raises(ValueError):
        state.append_samples(1)
    with pytest.raises(ValueError):
        state.commit(Admission.ACCEPTED)
    with pytest.raises(ValueError):
        state.reset_context()
    assert state.complete(partial, "too late") is None
    assert state.complete(final, "") == Hypothesis(final, "")
    assert state.complete(final, "duplicate") is None
    assert state.phase is (SessionPhase.CLOSED if end_input else SessionPhase.OPEN)
    assert state.segment_samples == 0
    if not end_input:
        state.append_samples(1)
        assert state.commit(Admission.ACCEPTED) == Revision(0, 1, 1, "final")


def test_empty_completion_and_empty_commit() -> None:
    state = session()
    with pytest.raises(ValueError, match="empty"):
        state.commit(Admission.ACCEPTED)
    assert state.phase is SessionPhase.OPEN
    assert state.commit(Admission.ACCEPTED, end_input=True) is None
    assert state.phase is SessionPhase.CLOSED
    state.cancel()
    assert state.phase is SessionPhase.CLOSED


def test_reset_boundary_across_segments_and_crossing_append() -> None:
    state = session()
    state.append_samples(100000)
    first = state.commit(Admission.ACCEPTED)
    assert first is not None
    state.complete(first, "first segment")
    assert state.remaining_samples == 380000
    state.append_samples(379999)
    with pytest.raises(ValueError, match="boundary"):
        state.append_samples(2)
    assert (state.segment_samples, state.remaining_samples) == (379999, 1)
    partial = state.partial(Admission.ACCEPTED)
    assert partial is not None
    state.append_samples(1)
    assert not state.partial_due  # Final takes precedence, including non-divisible cadence.
    assert state.commit(Admission.BACKPRESSURE) is None
    assert (state.epoch, state.remaining_samples) == (0, 0)
    final = state.commit(Admission.ACCEPTED)
    assert final is not None
    assert state.epoch == 0
    event = state.complete(final, "old context final")
    assert event is not None and event.revision.epoch == 0
    assert (state.epoch, state.segment_id, state.remaining_samples) == (1, 2, 480000)
    assert state.complete(partial, "old epoch") is None
    state.append_samples(25600)
    assert state.partial(Admission.ACCEPTED) == Revision(1, 2, 1, "partial")
    assert state.complete(first, "old segment") is None


def test_explicit_reset_requires_empty_open_boundary() -> None:
    state = session()
    state.reset_context()
    assert state.epoch == 1
    state.append_samples(1)
    with pytest.raises(ValueError, match="finalize"):
        state.reset_context()
    assert state.segment_samples == 1
    final = state.commit(Admission.ACCEPTED)
    assert final is not None
    state.complete(final, "done")
    state.reset_context()
    assert (state.epoch, state.segment_id, state.context_samples) == (2, 1, 0)
    assert state.complete(final, "late") is None


@pytest.mark.parametrize("phase", ["open", "finalizing", "draining"])
def test_cancel_invalidates_and_clears(phase: str) -> None:
    state = session()
    state.append_samples(25600)
    revision = state.partial(Admission.ACCEPTED)
    if phase != "open":
        revision = state.commit(Admission.ACCEPTED, end_input=phase == "draining")
    assert revision is not None
    state.cancel()
    state.cancel()
    assert state.phase is SessionPhase.CANCELLED
    assert (state.segment_samples, state.context_samples) == (0, 0)
    assert state.complete(revision, "late") is None
    for operation in (
        lambda: state.append_samples(1),
        lambda: state.partial(Admission.ACCEPTED),
        lambda: state.commit(Admission.ACCEPTED, end_input=True),
        state.reset_context,
    ):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize("end_input", [False, True])
def test_current_failure_cancels_without_completing(end_input: bool) -> None:
    state = session()
    state.append_samples(25600)
    partial = state.partial(Admission.ACCEPTED)
    assert partial is not None
    assert state.is_current(partial)
    final = state.commit(Admission.ACCEPTED, end_input=end_input)
    assert final is not None
    assert not state.is_current(partial)  # A stale failure must not cancel the final.
    assert state.is_current(final)
    state.cancel()
    assert state.phase is SessionPhase.CANCELLED
    assert not state.is_current(final)
    assert state.complete(final, "fabricated success") is None


class ControlledBackend:
    """Explicit completion/abort gates; cancellation may lose to completion."""

    def __init__(self) -> None:
        self.results: dict[str, asyncio.Future[str]] = {}
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.cancelled: set[str] = set()

    async def infer(self, request: InferenceRequest) -> str:
        result: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self.results[request.request_id] = result
        self.started.put_nowait(request.request_id)
        return await result

    async def cancel(self, request_id: str) -> None:
        self.cancelled.add(request_id)
        result = self.results.get(request_id)
        if result is not None and not result.done():
            result.cancel()


@pytest.mark.asyncio
async def test_backend_out_of_order_and_cancel_race() -> None:
    state = session()
    fake = ControlledBackend()
    backend: InferenceBackend = fake
    state.append_samples(25600)
    partial = state.partial(Admission.ACCEPTED)
    assert partial is not None
    partial_request = InferenceRequest("connection-a-1", partial, bytes(51200), "zh", "")
    old_task = asyncio.create_task(backend.infer(partial_request))
    assert await fake.started.get() == partial_request.request_id
    final = state.commit(Admission.ACCEPTED)
    assert final is not None
    final_request = replace(partial_request, request_id="connection-a-2", revision=final)
    final_task = asyncio.create_task(backend.infer(final_request))
    assert await fake.started.get() == final_request.request_id
    fake.results[final_request.request_id].set_result("authoritative")
    assert state.complete(final, await final_task) == Hypothesis(final, "authoritative")
    fake.results[partial_request.request_id].set_result("out of order")
    assert state.complete(partial, await old_task) is None
    await backend.cancel(partial_request.request_id)
    await backend.cancel(partial_request.request_id)
    await backend.cancel("unknown")

    state.append_samples(25600)
    revision = state.partial(Admission.ACCEPTED)
    assert revision is not None
    request = replace(partial_request, request_id="connection-a-3", revision=revision)
    task = asyncio.create_task(backend.infer(request))
    await fake.started.get()
    # Completion wins the engine race, but cancellation wins publication.
    fake.results[request.request_id].set_result("must not publish")
    state.cancel()
    await backend.cancel(request.request_id)
    assert state.complete(revision, await task) is None


@pytest.mark.asyncio
async def test_backend_abort_pending_work() -> None:
    fake = ControlledBackend()
    backend: InferenceBackend = fake
    request = InferenceRequest("connection-b-1", Revision(0, 0, 1, "partial"), bytes(32), "en", "")
    task = asyncio.create_task(backend.infer(request))
    await fake.started.get()
    await backend.cancel(request.request_id)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.cancelled == {request.request_id}
