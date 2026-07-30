# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vllm_omni.experimental.ar_diffusion.session import (
    ARDiffusionChunkMetadataError,
    ARDiffusionEventAcceptanceStatus,
    ARDiffusionEventQueueFullError,
    ARDiffusionPreparedControls,
    ARDiffusionSession,
    ARDiffusionSessionError,
    ARDiffusionSessionEvent,
    ARDiffusionSessionManager,
    ARDiffusionSessionStateError,
    ARDiffusionSessionStatus,
    ARDiffusionStaleEventError,
    ARDiffusionWorkerLifecycle,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionChunkMetadata,
    ARDiffusionControlInput,
    ARDiffusionTickRequest,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeTickConsumer:
    def __init__(self) -> None:
        self.ticks: list[ARDiffusionTickRequest] = []
        self.failures_remaining = 0
        self.started: asyncio.Event | None = None
        self.release: asyncio.Event | None = None
        self.metadata_override: ARDiffusionChunkMetadata | None = None

    async def execute_tick(self, tick: ARDiffusionTickRequest) -> ARDiffusionChunkMetadata:
        self.ticks.append(tick)
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("fake tick failed")
        return self.metadata_override or ARDiffusionChunkMetadata.from_tick(tick)

    def chunk_metadata(self, output: ARDiffusionChunkMetadata) -> ARDiffusionChunkMetadata:
        return output


class FakeLifecycle:
    def __init__(self) -> None:
        self.resets: list[str] = []
        self.closes: list[str] = []
        self.close_failures_remaining = 0

    async def reset_session(self, session_id: str) -> None:
        self.resets.append(session_id)

    async def close_session(self, session_id: str) -> None:
        self.closes.append(session_id)
        if self.close_failures_remaining:
            self.close_failures_remaining -= 1
            raise RuntimeError("fake close failed")


class FailingCommitReducer:
    def __init__(self) -> None:
        self.commit_calls = 0
        self.reset_calls = 0

    def prepare(
        self,
        *,
        current_controls: dict[str, ARDiffusionControlInput],
        events: tuple[ARDiffusionSessionEvent, ...],
        chunk_index: int,
    ) -> ARDiffusionPreparedControls:
        del current_controls, events, chunk_index
        return ARDiffusionPreparedControls(controls=())

    def commit(self, prepared: ARDiffusionPreparedControls) -> None:
        del prepared
        self.commit_calls += 1
        raise RuntimeError("fake reducer commit failed")

    def reset(self) -> None:
        self.reset_calls += 1


def make_session(
    *,
    max_pending_events: int = 8,
    consumer: FakeTickConsumer | None = None,
    lifecycle: FakeLifecycle | None = None,
) -> tuple[ARDiffusionSession, FakeTickConsumer, FakeLifecycle]:
    resolved_consumer = consumer or FakeTickConsumer()
    resolved_lifecycle = lifecycle or FakeLifecycle()
    session = ARDiffusionSession(
        "world-1",
        tick_consumer=resolved_consumer,
        lifecycle=resolved_lifecycle,
        max_pending_events=max_pending_events,
        request_id_factory=lambda session_id, chunk_index: f"{session_id}-request-{chunk_index}",
    )
    return session, resolved_consumer, resolved_lifecycle


@pytest.mark.asyncio
async def test_events_are_accepted_then_sorted_and_applied_at_chunk_boundary() -> None:
    session, consumer, _ = make_session()
    event_two = ARDiffusionSessionEvent(event_id=2, prompt="turn")
    event_one = ARDiffusionSessionEvent(
        event_id=1,
        controls=(
            ARDiffusionControlInput(
                track="camera",
                schema="example.camera.v1",
                data={"samples": [{"x": 1.0}]},
            ),
        ),
    )

    accepted_two = await session.accept_event(event_two)
    accepted_one = await session.accept_event(event_one)
    duplicate = await session.accept_event(event_two)

    assert accepted_two.status is ARDiffusionEventAcceptanceStatus.ACCEPTED
    assert accepted_one.status is ARDiffusionEventAcceptanceStatus.ACCEPTED
    assert duplicate.status is ARDiffusionEventAcceptanceStatus.DUPLICATE
    assert consumer.ticks == []

    metadata = await session.next_chunk()

    assert metadata.applied_event_ids == (1, 2)
    assert consumer.ticks[0] == ARDiffusionTickRequest(
        session_id="world-1",
        request_id="world-1-request-0",
        chunk_index=0,
        applied_event_ids=(1, 2),
        prompt="turn",
        controls=(
            ARDiffusionControlInput(
                track="camera",
                schema="example.camera.v1",
                data={"samples": [{"x": 1.0}]},
            ),
        ),
    )
    assert session.pending_event_count == 0

    applied_duplicate = await session.accept_event(event_two)
    assert applied_duplicate.status is ARDiffusionEventAcceptanceStatus.DUPLICATE

    next_metadata = await session.next_chunk()
    assert next_metadata.chunk_index == 1
    assert next_metadata.applied_event_ids == ()
    assert consumer.ticks[1].prompt == "turn"
    assert consumer.ticks[1].controls == consumer.ticks[0].controls


@pytest.mark.asyncio
async def test_composite_event_is_isolated_and_queue_admission_is_atomic() -> None:
    session, consumer, _ = make_session(max_pending_events=1)
    payload = {"samples": [{"x": 1.0}]}
    composite = ARDiffusionSessionEvent(
        event_id=5,
        prompt="new prompt",
        controls=(
            ARDiffusionControlInput(
                track="camera",
                schema="example.camera.v1",
                data=payload,
            ),
            ARDiffusionControlInput(
                track="action",
                schema="example.action.v1",
                data={"keys": ["left"]},
            ),
        ),
    )
    await session.accept_event(composite)
    payload["samples"][0]["x"] = 99.0

    duplicate = await session.accept_event(composite)
    assert duplicate.status is ARDiffusionEventAcceptanceStatus.DUPLICATE
    with pytest.raises(ARDiffusionEventQueueFullError, match="queue is full"):
        await session.accept_event(ARDiffusionSessionEvent(event_id=6, prompt="rejected"))

    await session.next_chunk()
    assert consumer.ticks[0].applied_event_ids == (5,)
    assert consumer.ticks[0].prompt == "new prompt"
    assert consumer.ticks[0].controls[0].data["keys"] == ("left",)
    assert consumer.ticks[0].controls[1].data["samples"] == ({"x": 1.0},)
    assert consumer.ticks[0].controls[0].to_dict()["data"] == {"keys": ["left"]}
    assert consumer.ticks[0].controls[1].to_dict()["data"] == {"samples": [{"x": 1.0}]}


def test_event_rejects_duplicate_tracks_and_non_transport_payloads() -> None:
    with pytest.raises(ValueError, match="at most once"):
        ARDiffusionSessionEvent(
            event_id=1,
            controls=(
                ARDiffusionControlInput(track="camera", schema="one", data={}),
                ARDiffusionControlInput(track="camera", schema="two", data={}),
            ),
        )

    with pytest.raises(ValueError, match="transport-safe"):
        ARDiffusionControlInput(
            track="camera",
            schema="example.camera.v1",
            data={"object": object()},
        )


@pytest.mark.asyncio
async def test_failed_tick_closes_worker_and_requires_reset() -> None:
    consumer = FakeTickConsumer()
    consumer.failures_remaining = 1
    lifecycle = FakeLifecycle()
    session, _, _ = make_session(consumer=consumer, lifecycle=lifecycle)
    await session.accept_event(ARDiffusionSessionEvent(event_id=3, prompt="retry me"))

    with pytest.raises(RuntimeError, match="fake tick failed"):
        await session.next_chunk()

    assert session.status is ARDiffusionSessionStatus.FAILED
    assert session.chunk_index == 0
    assert session.pending_event_count == 1
    assert lifecycle.closes == ["world-1"]
    with pytest.raises(ARDiffusionSessionStateError, match="failed"):
        await session.next_chunk()

    await session.reset()
    assert session.status is ARDiffusionSessionStatus.ACTIVE
    assert session.pending_event_count == 0
    assert lifecycle.resets == ["world-1"]
    await session.accept_event(ARDiffusionSessionEvent(event_id=4, prompt="after reset"))
    metadata = await session.next_chunk()
    assert metadata.chunk_index == 0
    assert metadata.applied_event_ids == (4,)
    assert [tick.applied_event_ids for tick in consumer.ticks] == [(3,), (4,)]


@pytest.mark.asyncio
async def test_inflight_snapshot_is_immutable_and_new_event_waits_for_next_chunk() -> None:
    consumer = FakeTickConsumer()
    consumer.started = asyncio.Event()
    consumer.release = asyncio.Event()
    session, _, _ = make_session(consumer=consumer)
    await session.accept_event(ARDiffusionSessionEvent(event_id=10, prompt="first"))

    first_chunk = asyncio.create_task(session.next_chunk())
    await consumer.started.wait()
    accepted = await session.accept_event(ARDiffusionSessionEvent(event_id=11, prompt="second"))
    assert accepted.status is ARDiffusionEventAcceptanceStatus.ACCEPTED
    assert consumer.ticks[0].prompt == "first"
    assert consumer.ticks[0].applied_event_ids == (10,)

    with pytest.raises(ARDiffusionStaleEventError, match="event floor is 10"):
        await session.accept_event(ARDiffusionSessionEvent(event_id=9, prompt="stale"))

    consumer.release.set()
    first_metadata = await first_chunk
    second_metadata = await session.next_chunk()
    assert first_metadata.applied_event_ids == (10,)
    assert second_metadata.applied_event_ids == (11,)
    assert consumer.ticks[1].prompt == "second"


@pytest.mark.asyncio
async def test_metadata_mismatch_does_not_commit_snapshot() -> None:
    consumer = FakeTickConsumer()
    consumer.metadata_override = ARDiffusionChunkMetadata(
        session_id="another-world",
        request_id="wrong",
        chunk_index=0,
    )
    lifecycle = FakeLifecycle()
    session, _, _ = make_session(consumer=consumer, lifecycle=lifecycle)
    await session.accept_event(ARDiffusionSessionEvent(event_id=1, prompt="pending"))

    with pytest.raises(ARDiffusionChunkMetadataError, match="does not match"):
        await session.next_chunk()

    assert session.chunk_index == 0
    assert session.pending_event_count == 1
    assert session.status is ARDiffusionSessionStatus.FAILED
    assert lifecycle.closes == ["world-1"]


@pytest.mark.asyncio
async def test_reducer_commit_failure_is_fail_closed_without_partial_session_commit() -> None:
    consumer = FakeTickConsumer()
    lifecycle = FakeLifecycle()
    reducer = FailingCommitReducer()
    session = ARDiffusionSession(
        "world-1",
        tick_consumer=consumer,
        lifecycle=lifecycle,
        max_pending_events=4,
        request_id_factory=lambda session_id, chunk_index: f"{session_id}-request-{chunk_index}",
        control_reducer=reducer,
    )
    await session.accept_event(ARDiffusionSessionEvent(event_id=1, prompt="pending"))

    with pytest.raises(RuntimeError, match="fake reducer commit failed"):
        await session.next_chunk()

    assert reducer.commit_calls == 1
    assert session.status is ARDiffusionSessionStatus.FAILED
    assert session.chunk_index == 0
    assert session.pending_event_count == 1
    assert lifecycle.closes == ["world-1"]
    with pytest.raises(ARDiffusionSessionStateError, match="failed"):
        await session.next_chunk()

    await session.reset()
    assert reducer.reset_calls == 1
    assert session.status is ARDiffusionSessionStatus.ACTIVE
    assert session.chunk_index == 0
    assert session.pending_event_count == 0


@pytest.mark.asyncio
async def test_reset_close_and_disconnect_reach_lifecycle_boundary() -> None:
    consumer = FakeTickConsumer()
    lifecycle = FakeLifecycle()
    manager = ARDiffusionSessionManager(
        tick_consumer=consumer,
        lifecycle=lifecycle,
        max_pending_events=4,
        request_id_factory=lambda session_id, chunk_index: f"{session_id}-{chunk_index}",
    )
    session = await manager.create_session("world-reset")
    await session.accept_event(ARDiffusionSessionEvent(event_id=4, prompt="before reset"))
    await session.next_chunk()
    await manager.reset_session("world-reset")

    assert lifecycle.resets == ["world-reset"]
    assert session.status is ARDiffusionSessionStatus.ACTIVE
    assert session.chunk_index == 0
    duplicate = await session.accept_event(ARDiffusionSessionEvent(event_id=4, prompt="duplicate"))
    assert duplicate.status is ARDiffusionEventAcceptanceStatus.DUPLICATE

    reset_tick = await session.next_chunk()
    assert reset_tick.chunk_index == 0
    assert consumer.ticks[-1].prompt is None
    assert consumer.ticks[-1].controls == ()

    await manager.close_session("world-reset")
    assert lifecycle.closes == ["world-reset"]
    assert session.status is ARDiffusionSessionStatus.CLOSED
    with pytest.raises(ARDiffusionSessionStateError, match="does not exist"):
        await manager.get_session("world-reset")

    disconnected = await manager.create_session("world-disconnect")
    await manager.disconnect("world-disconnect")
    assert disconnected.status is ARDiffusionSessionStatus.CLOSED
    assert lifecycle.closes[-1] == "world-disconnect"


@pytest.mark.asyncio
async def test_manager_keeps_interleaved_sessions_independent() -> None:
    consumer = FakeTickConsumer()
    manager = ARDiffusionSessionManager(
        tick_consumer=consumer,
        lifecycle=FakeLifecycle(),
        max_pending_events=4,
        request_id_factory=lambda session_id, chunk_index: f"{session_id}-request-{chunk_index}",
    )
    session_a = await manager.create_session("world-a")
    session_b = await manager.create_session("world-b")

    await session_a.accept_event(ARDiffusionSessionEvent(event_id=10, prompt="a0"))
    await session_a.next_chunk()
    await session_b.accept_event(ARDiffusionSessionEvent(event_id=1, prompt="b0"))
    await session_b.next_chunk()
    await session_a.accept_event(ARDiffusionSessionEvent(event_id=11, prompt="a1"))
    await session_a.next_chunk()

    assert [
        (
            tick.session_id,
            tick.request_id,
            tick.chunk_index,
            tick.applied_event_ids,
        )
        for tick in consumer.ticks
    ] == [
        ("world-a", "world-a-request-0", 0, (10,)),
        ("world-b", "world-b-request-0", 0, (1,)),
        ("world-a", "world-a-request-1", 1, (11,)),
    ]
    assert session_a.chunk_index == 2
    assert session_b.chunk_index == 1
    with pytest.raises(ARDiffusionStaleEventError, match="event floor is 11"):
        await session_a.accept_event(ARDiffusionSessionEvent(event_id=2, prompt="stale only for a"))
    assert (
        await session_b.accept_event(ARDiffusionSessionEvent(event_id=2, prompt="b1"))
    ).status is ARDiffusionEventAcceptanceStatus.ACCEPTED


@pytest.mark.asyncio
async def test_failed_close_retains_tombstone_until_cleanup_retry_succeeds() -> None:
    consumer = FakeTickConsumer()
    lifecycle = FakeLifecycle()
    lifecycle.close_failures_remaining = 1
    manager = ARDiffusionSessionManager(
        tick_consumer=consumer,
        lifecycle=lifecycle,
        max_pending_events=4,
    )
    session = await manager.create_session("world-close-retry")

    with pytest.raises(RuntimeError, match="fake close failed"):
        await manager.close_session("world-close-retry")

    assert session.status is ARDiffusionSessionStatus.CLEANUP_FAILED
    assert await manager.get_session("world-close-retry") is session
    with pytest.raises(ARDiffusionSessionStateError, match="already exists"):
        await manager.create_session("world-close-retry")
    with pytest.raises(ARDiffusionSessionStateError, match="cleanup_failed"):
        await manager.reset_session("world-close-retry")

    await manager.close_session("world-close-retry")

    assert lifecycle.closes == ["world-close-retry", "world-close-retry"]
    assert session.status is ARDiffusionSessionStatus.CLOSED
    with pytest.raises(ARDiffusionSessionStateError, match="does not exist"):
        await manager.get_session("world-close-retry")
    recreated = await manager.create_session("world-close-retry")
    assert recreated is not session


@pytest.mark.asyncio
async def test_failed_disconnect_retains_session_for_explicit_cleanup_retry() -> None:
    lifecycle = FakeLifecycle()
    lifecycle.close_failures_remaining = 1
    manager = ARDiffusionSessionManager(
        tick_consumer=FakeTickConsumer(),
        lifecycle=lifecycle,
        max_pending_events=4,
    )
    session = await manager.create_session("world-disconnect-retry")

    with pytest.raises(RuntimeError, match="fake close failed"):
        await manager.disconnect("world-disconnect-retry")

    assert session.status is ARDiffusionSessionStatus.CLEANUP_FAILED
    assert await manager.get_session("world-disconnect-retry") is session

    await manager.close_session("world-disconnect-retry")
    assert session.status is ARDiffusionSessionStatus.CLOSED


class FakeRPCClient:
    def __init__(self, result: list[Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def collective_rpc(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        return self.result


@pytest.mark.asyncio
async def test_worker_lifecycle_routes_to_selected_stage_rpc() -> None:
    client = FakeRPCClient([[True]])
    lifecycle = ARDiffusionWorkerLifecycle(client, stage_ids=[2], timeout=3.0)

    await lifecycle.reset_session("world-1")
    await lifecycle.close_session("world-1")

    assert client.calls == [
        {
            "method": "reset_ar_diffusion_session",
            "timeout": 3.0,
            "args": ("world-1",),
            "stage_ids": [2],
        },
        {
            "method": "close_ar_diffusion_session",
            "timeout": 3.0,
            "args": ("world-1",),
            "stage_ids": [2],
        },
    ]


@pytest.mark.asyncio
async def test_worker_lifecycle_rejects_unsupported_stage_result() -> None:
    lifecycle = ARDiffusionWorkerLifecycle(
        FakeRPCClient([{"supported": False, "error": "not an AR stage"}]),
        stage_ids=[1],
    )

    with pytest.raises(ARDiffusionSessionError, match="not an AR stage"):
        await lifecycle.close_session("world-1")
