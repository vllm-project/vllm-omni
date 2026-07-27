# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import logging
from dataclasses import FrozenInstanceError

import msgspec
import pytest

from vllm_omni.experimental.fullduplex.engine.duplex_control_plane import (
    DuplexControlPlane,
    DuplexOutputContext,
    DuplexRequestIdentity,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
)
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexRuntimeCapabilities,
)
from vllm_omni.experimental.fullduplex.engine.lease import DuplexLeaseActivity, DuplexLeaseConfig
from vllm_omni.experimental.fullduplex.engine.messages import (
    AppendDuplexInputMessage,
    CloseDuplexSessionMessage,
    DuplexFence,
    DuplexSessionLifecycleMessage,
    OpenDuplexSessionMessage,
    ResumeDuplexSessionMessage,
    SignalDuplexTurnMessage,
    TouchDuplexSessionMessage,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Extension:
    def configure_sampling_params(self, *, runtime_config, defaults):
        del runtime_config
        return tuple(f"configured-{stage_id}" for stage_id, _ in enumerate(defaults))

    def plan_append(
        self,
        *,
        request_id,
        fence,
        session_config,
        runtime_config,
        seq,
        turn_seq,
        mode,
        payload,
        final,
        sampling_params,
    ):
        del request_id, fence, session_config, runtime_config, seq, turn_seq, mode, payload, final
        assert sampling_params == "configured-0"
        return DuplexAppendPlan(prompt={"prompt_token_ids": [1, 2, 3]})

    def decide_output(self, **kwargs):
        del kwargs
        return None


class _TypedStagePort:
    stage_count = 2

    def __init__(self) -> None:
        self.ensure_calls: list[DuplexStageRequestContext] = []
        self.submit_calls: list[DuplexStageSubmission] = []
        self.cleanup_calls: list[tuple[list[str], bool]] = []

    def sampling_defaults(self) -> tuple[object, ...]:
        return ("default-0", "default-1")

    def ensure_request(self, context: DuplexStageRequestContext) -> None:
        self.ensure_calls.append(context)

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
        self.submit_calls.append(submission)
        return DuplexStageSubmissionResult(
            request_id=submission.context.request_id,
            stage_id=submission.context.stage_id,
            replica_id=3,
        )

    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
        self.cleanup_calls.append((request_ids, abort))


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@pytest.mark.asyncio
async def test_control_plane_uses_frozen_typed_stage_context_without_request_state() -> None:
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(),
        stage_port=stage_port,
        result_sink=result_sink,
    )
    fence = DuplexFence("typed-port")

    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open",
            fence=fence,
            session_id=fence.session_id,
            capabilities={
                "input_modes": [DuplexInputMode.APPEND_AUDIO_CHUNK.value],
            },
            session_config={"voice": "test"},
            runtime_config={"runtime": "test"},
        )
    )
    assert (await result_sink.get()).ok is True

    context = stage_port.ensure_calls[-1]
    assert context.session_id == fence.session_id
    assert context.fence == fence
    assert context.stage_id == 0
    assert context.final_stage_id == 1
    assert context.sampling_params == ("configured-0", "configured-1")
    assert context.session_config == {"voice": "test"}
    assert context.runtime_config == {"runtime": "test"}
    with pytest.raises(FrozenInstanceError):
        context.stage_id = 1  # type: ignore[misc]

    await plane.handle(
        AppendDuplexInputMessage(
            control_id="append",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
            final=True,
        )
    )
    append_result = await result_sink.get()
    assert append_result.ok is True
    assert append_result.stage_results[0]["replica_id"] == 3
    assert append_result.stage_results[0]["result"]["response_stage_id"] == 1

    submission = stage_port.submit_calls[-1]
    assert submission.context == stage_port.ensure_calls[-1]
    assert submission.prompt == {"prompt_token_ids": [1, 2, 3]}
    assert submission.already_submitted is False
    assert not hasattr(submission, "request_state")


def test_control_plane_accepts_only_typed_duplex_messages() -> None:
    plane = DuplexControlPlane(
        extension=None,
        stage_port=_TypedStagePort(),
        result_sink=asyncio.Queue(),
    )

    assert plane.accepts(
        OpenDuplexSessionMessage(
            control_id="open",
            fence=DuplexFence("typed-message"),
            session_id="typed-message",
            capabilities=DuplexRuntimeCapabilities().__dict__,
        )
    )
    assert plane.accepts(type("Lookalike", (), {"type": "open_duplex_session"})()) is False


@pytest.mark.parametrize(
    "message",
    [
        TouchDuplexSessionMessage(
            control_id="touch-1",
            fence=DuplexFence("sid-message"),
            session_id="sid-message",
            activity=DuplexLeaseActivity.HEARTBEAT.value,
        ),
        ResumeDuplexSessionMessage(
            control_id="resume-1",
            fence=DuplexFence("sid-message"),
            session_id="sid-message",
            expected_lease_generation=3,
        ),
        DuplexSessionLifecycleMessage(
            fence=DuplexFence("sid-message"),
            session_id="sid-message",
            event="expired",
            reason="idle_ttl_expired",
            lease_generation=4,
            submitted_request_ids=["req-submitted"],
            reserved_request_ids=["req-reserved"],
        ),
    ],
)
def test_duplex_lease_messages_round_trip(message) -> None:
    encoded = msgspec.json.encode(message)
    decoded = msgspec.json.decode(encoded, type=type(message))

    assert decoded == message


@pytest.mark.asyncio
async def test_control_plane_touches_append_signal_and_explicit_activity() -> None:
    clock = _Clock()
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    lifecycle_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(),
        stage_port=stage_port,
        result_sink=result_sink,
        lifecycle_sink=lifecycle_sink,
        lease_config=DuplexLeaseConfig(idle_ttl_s=30.0, disconnect_grace_s=5.0),
        clock=clock,
    )
    fence = DuplexFence("sid-touch")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open",
            fence=fence,
            session_id=fence.session_id,
            capabilities={"input_modes": [DuplexInputMode.APPEND_AUDIO_CHUNK.value]},
        )
    )
    await result_sink.get()
    session = plane.sessions.require(fence.session_id)

    clock.advance(1.0)
    await plane.handle(
        AppendDuplexInputMessage(
            control_id="append",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
        )
    )
    assert (await result_sink.get()).ok is True
    assert session.lease.last_activity == 1.0

    clock.advance(1.0)
    await plane.handle(
        SignalDuplexTurnMessage(
            control_id="signal",
            fence=fence,
            session_id=fence.session_id,
            event="session.update",
        )
    )
    assert (await result_sink.get()).ok is True
    assert session.lease.last_activity == 2.0

    submit_count = len(stage_port.submit_calls)
    clock.advance(1.0)
    await plane.handle(
        TouchDuplexSessionMessage(
            control_id="heartbeat",
            fence=fence,
            session_id=fence.session_id,
            activity=DuplexLeaseActivity.HEARTBEAT.value,
        )
    )
    touch_result = await result_sink.get()
    assert touch_result.ok is True
    assert session.lease.last_activity == 3.0
    assert len(stage_port.submit_calls) == submit_count
    assert lifecycle_sink.empty()


@pytest.mark.asyncio
async def test_control_plane_resume_requires_expected_lease_generation() -> None:
    clock = _Clock()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=None,
        stage_port=_TypedStagePort(),
        result_sink=result_sink,
        lifecycle_sink=asyncio.Queue(),
        lease_config=DuplexLeaseConfig(),
        clock=clock,
    )
    fence = DuplexFence("sid-resume-control")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open",
            fence=fence,
            session_id=fence.session_id,
            capabilities={},
        )
    )
    await result_sink.get()
    await plane.handle(
        TouchDuplexSessionMessage(
            control_id="detach",
            fence=fence,
            session_id=fence.session_id,
            activity=DuplexLeaseActivity.DETACH.value,
        )
    )
    await result_sink.get()

    await plane.handle(
        ResumeDuplexSessionMessage(
            control_id="resume",
            fence=fence,
            session_id=fence.session_id,
            expected_lease_generation=0,
        )
    )
    result = await result_sink.get()

    assert result.ok is True
    assert result.stage_results[0]["result"]["lease_generation"] == 1

    await plane.handle(
        ResumeDuplexSessionMessage(
            control_id="stale-resume",
            fence=fence,
            session_id=fence.session_id,
            expected_lease_generation=0,
        )
    )
    stale_result = await result_sink.get()
    assert stale_result.ok is False
    assert stale_result.accepted_fence == fence
    assert stale_result.lease_generation == 1


@pytest.mark.asyncio
async def test_control_plane_reaps_one_session_through_cleanup_and_lifecycle_sink() -> None:
    clock = _Clock()
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    lifecycle_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=None,
        stage_port=stage_port,
        result_sink=result_sink,
        lifecycle_sink=lifecycle_sink,
        lease_config=DuplexLeaseConfig(idle_ttl_s=2.0, disconnect_grace_s=1.0),
        clock=clock,
    )
    for session_id in ("sid-a", "sid-b"):
        fence = DuplexFence(session_id)
        await plane.handle(
            OpenDuplexSessionMessage(
                control_id=f"open-{session_id}",
                fence=fence,
                session_id=session_id,
                capabilities={},
            )
        )
        await result_sink.get()
    clock.advance(1.0)
    plane.sessions.require("sid-b").touch(DuplexFence("sid-b"), DuplexLeaseActivity.HEARTBEAT)
    clock.advance(1.1)

    expired_count = await plane.reap_expired()

    assert expired_count == 1
    assert plane.sessions.get("sid-a") is None
    assert plane.sessions.get("sid-b") is not None
    assert stage_port.cleanup_calls == []
    lifecycle = await lifecycle_sink.get()
    assert isinstance(lifecycle, DuplexSessionLifecycleMessage)
    assert lifecycle.session_id == "sid-a"
    assert lifecycle.reason == "idle_ttl_expired"
    assert result_sink.empty()


@pytest.mark.asyncio
async def test_control_plane_retries_expired_cleanup_before_publishing_lifecycle() -> None:
    class _FailOnceStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.failures_remaining = 1

        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise RuntimeError("transient cleanup failure")
            await super().cleanup(request_ids, abort=abort)

    clock = _Clock()
    stage_port = _FailOnceStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    lifecycle_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=None,
        stage_port=stage_port,
        result_sink=result_sink,
        lifecycle_sink=lifecycle_sink,
        lease_config=DuplexLeaseConfig(idle_ttl_s=1.0, disconnect_grace_s=1.0),
        clock=clock,
    )
    fence = DuplexFence("sid-retry-expiry")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-retry",
            fence=fence,
            session_id=fence.session_id,
            capabilities={},
        )
    )
    await result_sink.get()
    request_id = plane.stage_request_id(fence, stage_id=0)
    plane.sessions.require(fence.session_id).reserve_stage_request(0, request_id, fence=fence)
    clock.advance(2.0)

    assert await plane.reap_expired() == 0
    assert lifecycle_sink.empty()

    assert await plane.reap_expired() == 1
    assert (await lifecycle_sink.get()).session_id == fence.session_id


@pytest.mark.asyncio
async def test_expired_session_holds_admission_slot_while_cleanup_is_blocked() -> None:
    class _BlockedCleanupStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.cleanup_started = asyncio.Event()
            self.release_cleanup = asyncio.Event()

        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            self.cleanup_started.set()
            await self.release_cleanup.wait()
            await super().cleanup(request_ids, abort=abort)

    clock = _Clock()
    stage_port = _BlockedCleanupStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=None,
        stage_port=stage_port,
        result_sink=result_sink,
        lease_config=DuplexLeaseConfig(idle_ttl_s=1.0, disconnect_grace_s=1.0),
        clock=clock,
        max_sessions=1,
    )
    old_fence = DuplexFence("sid-expired-blocked")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-expired-blocked",
            fence=old_fence,
            session_id=old_fence.session_id,
            capabilities={},
        )
    )
    assert (await result_sink.get()).ok is True
    old_session = plane.sessions.require(old_fence.session_id)
    old_session.bind_stage_request(0, "req-expired-blocked", fence=old_fence)
    clock.advance(2.0)

    reap_task = asyncio.create_task(plane.reap_expired())
    await asyncio.wait_for(stage_port.cleanup_started.wait(), timeout=1.0)
    replacement_fence = DuplexFence("sid-replacement-blocked")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-replacement-blocked",
            fence=replacement_fence,
            session_id=replacement_fence.session_id,
            capabilities={},
        )
    )

    blocked_result = await result_sink.get()
    assert blocked_result.ok is False
    assert blocked_result.error is not None
    assert blocked_result.error.code == "resource_exhausted"
    assert plane.sessions.get(old_fence.session_id) is old_session

    stage_port.release_cleanup.set()
    assert await reap_task == 1
    assert plane.sessions.get(old_fence.session_id) is None

    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-replacement-after-cleanup",
            fence=replacement_fence,
            session_id=replacement_fence.session_id,
            capabilities={},
        )
    )
    assert (await result_sink.get()).ok is True


@pytest.mark.asyncio
async def test_failed_expiry_cleanup_retains_admission_slot_for_retry() -> None:
    class _FailingCleanupStagePort(_TypedStagePort):
        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            del request_ids, abort
            raise RuntimeError("cleanup failed")

    clock = _Clock()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=None,
        stage_port=_FailingCleanupStagePort(),
        result_sink=result_sink,
        lease_config=DuplexLeaseConfig(idle_ttl_s=1.0, disconnect_grace_s=1.0),
        clock=clock,
        max_sessions=1,
    )
    old_fence = DuplexFence("sid-expired-failed")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-expired-failed",
            fence=old_fence,
            session_id=old_fence.session_id,
            capabilities={},
        )
    )
    assert (await result_sink.get()).ok is True
    old_session = plane.sessions.require(old_fence.session_id)
    old_session.bind_stage_request(0, "req-expired-failed", fence=old_fence)
    clock.advance(2.0)

    assert await plane.reap_expired() == 0
    replacement_fence = DuplexFence("sid-replacement-failed")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-replacement-failed",
            fence=replacement_fence,
            session_id=replacement_fence.session_id,
            capabilities={},
        )
    )

    blocked_result = await result_sink.get()
    assert blocked_result.ok is False
    assert blocked_result.error is not None
    assert blocked_result.error.code == "resource_exhausted"
    assert plane.sessions.get(old_fence.session_id) is old_session


@pytest.mark.asyncio
async def test_capacity_rejection_is_an_info_event(caplog: pytest.LogCaptureFixture) -> None:
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=None,
        stage_port=_TypedStagePort(),
        result_sink=result_sink,
        max_sessions=1,
    )
    first_fence = DuplexFence("sid-capacity-first")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-capacity-first",
            fence=first_fence,
            session_id=first_fence.session_id,
            capabilities={},
        )
    )
    assert (await result_sink.get()).ok is True
    caplog.clear()

    rejected_fence = DuplexFence("sid-capacity-rejected")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-capacity-rejected",
            fence=rejected_fence,
            session_id=rejected_fence.session_id,
            capabilities={},
        )
    )

    result = await result_sink.get()
    assert result.error.code == "resource_exhausted"
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


@pytest.mark.asyncio
async def test_failed_request_cleanup_retries_before_releasing_admission() -> None:
    class _FailOnceCleanupStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            self.cleanup_calls.append((list(request_ids), abort))
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("transient request cleanup failure")

    stage_port = _FailOnceCleanupStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=None,
        stage_port=stage_port,
        result_sink=result_sink,
        max_sessions=1,
    )
    old_fence = DuplexFence("sid-request-cleanup")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-request-cleanup",
            fence=old_fence,
            session_id=old_fence.session_id,
            capabilities={},
        )
    )
    assert (await result_sink.get()).ok is True
    old_session = plane.sessions.require(old_fence.session_id)
    old_session.bind_stage_request(0, "req-request-cleanup", fence=old_fence)

    closed = plane.close_sessions_for_request_ids(
        ["req-request-cleanup"],
        abort=True,
    )
    assert closed == {old_fence.session_id: ["req-request-cleanup"]}
    assert plane.sessions.get(old_fence.session_id) is old_session
    with pytest.raises(RuntimeError, match="transient request cleanup failure"):
        await stage_port.cleanup(closed[old_fence.session_id], abort=True)

    replacement_fence = DuplexFence("sid-request-cleanup-replacement")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-request-cleanup-replacement-blocked",
            fence=replacement_fence,
            session_id=replacement_fence.session_id,
            capabilities={},
        )
    )
    assert (await result_sink.get()).error.code == "resource_exhausted"

    assert await plane.reap_expired() == 1
    assert plane.sessions.get(old_fence.session_id) is None
    assert stage_port.cleanup_calls == [
        (["req-request-cleanup"], True),
        (["req-request-cleanup"], True),
    ]

    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-request-cleanup-replacement",
            fence=replacement_fence,
            session_id=replacement_fence.session_id,
            capabilities={},
        )
    )
    assert (await result_sink.get()).ok is True


@pytest.mark.asyncio
async def test_stale_request_cleanup_does_not_finalize_reopened_incarnation() -> None:
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=None,
        stage_port=stage_port,
        result_sink=result_sink,
        max_sessions=1,
    )
    old_fence = DuplexFence("sid-request-cleanup-reopen")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-request-cleanup-old",
            fence=old_fence,
            session_id=old_fence.session_id,
            capabilities={},
        )
    )
    assert (await result_sink.get()).ok is True
    old_session = plane.sessions.require(old_fence.session_id)
    old_session.bind_stage_request(0, "req-request-cleanup-old", fence=old_fence)
    closed = plane.close_sessions_for_request_ids(
        ["req-request-cleanup-old"],
        abort=True,
        cleanup_in_progress=True,
    )
    assert closed == {old_fence.session_id: ["req-request-cleanup-old"]}
    plane.defer_request_cleanups([old_fence.session_id])

    await plane.handle(
        CloseDuplexSessionMessage(
            control_id="close-request-cleanup-old",
            fence=old_fence,
            session_id=old_fence.session_id,
            reason="client_close",
        )
    )
    assert (await result_sink.get()).ok is True
    assert plane.sessions.get(old_fence.session_id) is None

    new_fence = DuplexFence(old_fence.session_id, incarnation=1)
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-request-cleanup-new",
            fence=new_fence,
            session_id=new_fence.session_id,
            capabilities={},
        )
    )
    assert (await result_sink.get()).ok is True
    new_session = plane.sessions.require(new_fence.session_id)
    assert new_session.resume(new_fence, expected_lease_generation=0) == 1

    assert await plane.reap_expired() == 1
    assert plane.sessions.get(new_fence.session_id) is new_session


@pytest.mark.asyncio
async def test_failed_open_cleanup_is_retried_before_releasing_admission() -> None:
    class _FailOpenAndFirstCleanupStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.ensure_attempts = 0
            self.cleanup_attempts = 0

        def ensure_request(self, context: DuplexStageRequestContext) -> None:
            self.ensure_attempts += 1
            if self.ensure_attempts == 1:
                raise RuntimeError("initial stage setup failed")
            super().ensure_request(context)

        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            self.cleanup_calls.append((list(request_ids), abort))
            self.cleanup_attempts += 1
            if self.cleanup_attempts == 1:
                raise RuntimeError("transient open rollback failure")

    stage_port = _FailOpenAndFirstCleanupStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(),
        stage_port=stage_port,
        result_sink=result_sink,
        max_sessions=1,
    )
    failed_fence = DuplexFence("sid-open-rollback")

    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-fails-during-stage-setup",
            fence=failed_fence,
            session_id=failed_fence.session_id,
            capabilities={"input_modes": ["append_audio_chunk"]},
        )
    )

    failed_result = await result_sink.get()
    assert failed_result.ok is False
    assert plane.sessions.get(failed_fence.session_id) is not None

    replacement_fence = DuplexFence("sid-open-rollback-replacement")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-replacement-before-rollback",
            fence=replacement_fence,
            session_id=replacement_fence.session_id,
            capabilities={"input_modes": ["append_audio_chunk"]},
        )
    )
    assert (await result_sink.get()).error.code == "resource_exhausted"

    assert await plane.reap_expired() == 0
    assert plane.sessions.get(failed_fence.session_id) is None

    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-replacement-after-rollback",
            fence=replacement_fence,
            session_id=replacement_fence.session_id,
            capabilities={"input_modes": ["append_audio_chunk"]},
        )
    )
    assert (await result_sink.get()).ok is True


@pytest.mark.asyncio
async def test_expired_cleanup_failure_does_not_block_other_sessions() -> None:
    class _OneStuckExpiryStagePort(_TypedStagePort):
        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            if request_ids == [stuck_request_id]:
                raise RuntimeError("stuck expiry cleanup")
            await super().cleanup(request_ids, abort=abort)

    clock = _Clock()
    stage_port = _OneStuckExpiryStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    lifecycle_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=None,
        stage_port=stage_port,
        result_sink=result_sink,
        lifecycle_sink=lifecycle_sink,
        lease_config=DuplexLeaseConfig(idle_ttl_s=1.0, disconnect_grace_s=1.0),
        clock=clock,
    )
    stuck_fence = DuplexFence("sid-expiry-stuck")
    ready_fence = DuplexFence("sid-expiry-ready")
    stuck_request_id = plane.stage_request_id(stuck_fence, stage_id=0)
    ready_request_id = plane.stage_request_id(ready_fence, stage_id=0)
    for fence in (stuck_fence, ready_fence):
        await plane.handle(
            OpenDuplexSessionMessage(
                control_id=f"open-{fence.session_id}",
                fence=fence,
                session_id=fence.session_id,
                capabilities={},
            )
        )
        assert (await result_sink.get()).ok is True
        plane.sessions.require(fence.session_id).reserve_stage_request(
            0,
            plane.stage_request_id(fence, stage_id=0),
            fence=fence,
        )
    clock.advance(2.0)

    assert await plane.reap_expired() == 1

    assert (await lifecycle_sink.get()).session_id == ready_fence.session_id
    assert stage_port.cleanup_calls == [([ready_request_id], False)]


@pytest.mark.asyncio
async def test_cancel_retains_request_resources_until_abort_succeeds() -> None:
    class _FailOnceStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.failures_remaining = 1

        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise RuntimeError("transient abort failure")
            await super().cleanup(request_ids, abort=abort)

    stage_port = _FailOnceStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=None, stage_port=stage_port, result_sink=result_sink)
    cancelled_fence = DuplexFence("sid-cancel-retry")
    next_fence = DuplexFence("sid-cancel-retry", epoch=1)
    session = plane.sessions.open_session(cancelled_fence)
    session.bind_stage_request(0, "req-cancel-retry", fence=cancelled_fence)

    def signal(control_id: str) -> SignalDuplexTurnMessage:
        return SignalDuplexTurnMessage(
            control_id=control_id,
            fence=cancelled_fence,
            next_fence=next_fence,
            session_id=cancelled_fence.session_id,
            event="input.cancel",
        )

    await plane.handle(signal("cancel-first"))
    assert (await result_sink.get()).ok is False
    assert session.fence == next_fence
    assert session.resource_request_ids(cancelled_fence) == ["req-cancel-retry"]

    await plane.handle(signal("cancel-retry"))
    assert (await result_sink.get()).ok is True
    assert session.resource_request_ids(cancelled_fence) == []
    assert stage_port.cleanup_calls == [(["req-cancel-retry"], True)]


@pytest.mark.asyncio
async def test_close_retains_session_resources_until_cleanup_succeeds() -> None:
    class _FailOnceStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.failures_remaining = 1

        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise RuntimeError("transient close cleanup failure")
            await super().cleanup(request_ids, abort=abort)

    stage_port = _FailOnceStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=None, stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence("sid-close-retry")
    session = plane.sessions.open_session(fence)
    session.bind_stage_request(0, "req-close-retry", fence=fence)

    def close(control_id: str) -> CloseDuplexSessionMessage:
        return CloseDuplexSessionMessage(
            control_id=control_id,
            fence=fence,
            session_id=fence.session_id,
            reason="client_close",
        )

    await plane.handle(close("close-first"))
    assert (await result_sink.get()).ok is False
    assert plane.sessions.get(fence.session_id) is session
    assert session.resource_request_ids() == ["req-close-retry"]

    await plane.handle(close("close-retry"))
    assert (await result_sink.get()).ok is True
    assert plane.sessions.get(fence.session_id) is None
    assert stage_port.cleanup_calls == [(["req-close-retry"], True)]


@pytest.mark.asyncio
async def test_control_cleanup_failure_does_not_block_other_sessions() -> None:
    class _IndependentFailureStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.attempts: dict[str, int] = {}

        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            request_id = request_ids[0]
            self.attempts[request_id] = self.attempts.get(request_id, 0) + 1
            if request_id == "req-stuck" or self.attempts[request_id] == 1:
                raise RuntimeError(f"cleanup failed for {request_id}")
            await super().cleanup(request_ids, abort=abort)

    stage_port = _IndependentFailureStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=None, stage_port=stage_port, result_sink=result_sink)

    for session_id, request_id in (("sid-stuck", "req-stuck"), ("sid-ready", "req-ready")):
        fence = DuplexFence(session_id)
        session = plane.sessions.open_session(fence)
        session.bind_stage_request(0, request_id, fence=fence)
        await plane.handle(
            CloseDuplexSessionMessage(
                control_id=f"close-{session_id}",
                fence=fence,
                session_id=session_id,
                reason="client_close",
            )
        )
        assert (await result_sink.get()).ok is False

    await plane.reap_expired()

    assert plane.sessions.get("sid-stuck") is not None
    assert plane.sessions.get("sid-ready") is None
    assert stage_port.cleanup_calls == [(["req-ready"], True)]


@pytest.mark.asyncio
async def test_control_cleanup_is_single_flight_with_concurrent_reaper() -> None:
    class _SlowCleanupStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.cleanup_started = asyncio.Event()
            self.release_cleanup = asyncio.Event()
            self.cleanup_attempts = 0

        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            self.cleanup_attempts += 1
            self.cleanup_started.set()
            await self.release_cleanup.wait()
            await super().cleanup(request_ids, abort=abort)

    stage_port = _SlowCleanupStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=None, stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence("sid-concurrent-cleanup")
    session = plane.sessions.open_session(fence)
    session.bind_stage_request(0, "req-concurrent-cleanup", fence=fence)

    close_task = asyncio.create_task(
        plane.handle(
            CloseDuplexSessionMessage(
                control_id="close-concurrent-cleanup",
                fence=fence,
                session_id=fence.session_id,
                reason="client_close",
            )
        )
    )
    await asyncio.wait_for(stage_port.cleanup_started.wait(), timeout=1)

    reaper_task = asyncio.create_task(plane.reap_expired())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert stage_port.cleanup_attempts == 1

    stage_port.release_cleanup.set()
    await asyncio.gather(close_task, reaper_task)

    assert (await result_sink.get()).ok is True
    assert plane.sessions.get(fence.session_id) is None
    assert stage_port.cleanup_calls == [(["req-concurrent-cleanup"], True)]


@pytest.mark.asyncio
async def test_unknown_input_mode_rejects_open_without_registering_session() -> None:
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=None, stage_port=_TypedStagePort(), result_sink=result_sink)
    fence = DuplexFence("sid-unknown-mode")

    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-unknown-mode",
            fence=fence,
            session_id=fence.session_id,
            capabilities={"input_modes": ["future_mode"]},
        )
    )

    result = await result_sink.get()
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_capability"
    assert plane.sessions.get(fence.session_id) is None


@pytest.mark.asyncio
async def test_extension_free_turn_commit_does_not_allocate_or_submit_stage_request() -> None:
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=None, stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence("sid-turn-commit")

    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-turn-commit",
            fence=fence,
            session_id=fence.session_id,
            capabilities={"input_modes": [DuplexInputMode.TURN_COMMIT_ONLY.value]},
        )
    )
    assert (await result_sink.get()).ok is True

    await plane.handle(
        AppendDuplexInputMessage(
            control_id="append-turn-commit",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.TURN_COMMIT_ONLY.value,
            payload=None,
            final=True,
        )
    )
    result = await result_sink.get()

    assert result.ok is True
    assert result.stage_results[0]["result"]["data_plane_append"] is False
    assert stage_port.ensure_calls == []
    assert stage_port.submit_calls == []


@pytest.mark.asyncio
async def test_session_update_replaces_both_configs_with_one_generation() -> None:
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=_TypedStagePort(), result_sink=result_sink)
    fence = DuplexFence("sid-atomic-config")
    session = plane.sessions.open_session(fence)

    await plane.handle(
        SignalDuplexTurnMessage(
            control_id="update-config",
            fence=fence,
            session_id=fence.session_id,
            event="session.update",
            session_config={"voice": "new"},
            runtime_config={"temperature": 0.5},
        )
    )

    assert (await result_sink.get()).ok is True
    assert session.config_generation == 1
    assert session.session_config == {"voice": "new"}
    assert session.runtime_config == {"temperature": 0.5}


@pytest.mark.asyncio
async def test_stage_submission_is_compensated_when_local_commit_fails() -> None:
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence("sid-submit-compensation")
    session = plane.sessions.open_session(
        fence,
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK}),
    )
    request_id = plane.stage_request_id(fence, stage_id=0)

    def fail_commit(_reservation):
        raise RuntimeError("local commit failed")

    session.commit_append = fail_commit  # type: ignore[method-assign]
    await plane.handle(
        AppendDuplexInputMessage(
            control_id="append-compensation",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
        )
    )

    assert (await result_sink.get()).ok is False
    assert stage_port.cleanup_calls == [([request_id], True)]


@pytest.mark.asyncio
async def test_failed_submission_compensation_blocks_append_until_reaper_cleans_it() -> None:
    class _FailingCleanupStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.cleanup_failures = 2

        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            await super().cleanup(request_ids, abort=abort)
            if self.cleanup_failures > 0:
                self.cleanup_failures -= 1
                raise RuntimeError("stage abort failed")

    stage_port = _FailingCleanupStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence("sid-durable-submit-compensation")
    session = plane.sessions.open_session(
        fence,
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK}),
    )
    original_commit = session.commit_append

    def fail_commit(_reservation):
        raise RuntimeError("local commit failed")

    session.commit_append = fail_commit  # type: ignore[method-assign]
    first = AppendDuplexInputMessage(
        control_id="append-compensation-1",
        fence=fence,
        session_id=fence.session_id,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
        payload={"audio": b"pcm"},
    )
    await plane.handle(first)
    assert (await result_sink.get()).ok is False
    assert plane.pending_submission_cleanup_count == 1
    assert len(stage_port.submit_calls) == 1

    await plane.handle(
        AppendDuplexInputMessage(
            control_id="append-compensation-2",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
        )
    )
    assert (await result_sink.get()).ok is False
    assert plane.pending_submission_cleanup_count == 1
    assert len(stage_port.submit_calls) == 1

    session.commit_append = original_commit  # type: ignore[method-assign]
    stage_port.cleanup_failures = 0
    await plane.reap_expired()

    assert plane.pending_submission_cleanup_count == 0
    assert session.resource_request_ids() == []


def test_stale_output_is_rejected_before_extension_decision() -> None:
    class _CountingExtension(_Extension):
        def __init__(self) -> None:
            self.decision_calls = 0

        def decide_output(self, **kwargs):
            del kwargs
            self.decision_calls += 1
            return None

    extension = _CountingExtension()
    plane = DuplexControlPlane(extension=extension, stage_port=_TypedStagePort(), result_sink=asyncio.Queue())
    current = DuplexFence("sid-stale-output", epoch=1)
    plane.sessions.open_session(current)
    context = DuplexOutputContext(
        identity=DuplexRequestIdentity(
            session_id=current.session_id,
            fence=DuplexFence(current.session_id, epoch=0),
        ),
        final_stage_id=1,
        segment_finished=True,
    )

    assert plane.decide_output(0, object(), context) is None
    assert extension.decision_calls == 0


@pytest.mark.asyncio
async def test_control_dispatch_is_ordered_per_session_without_blocking_other_sessions() -> None:
    class _BlockingStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
            if submission.context.session_id == "sid-blocked":
                self.started.set()
                await self.release.wait()
            return await super().submit(submission)

    stage_port = _BlockingStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=stage_port, result_sink=result_sink)
    blocked = DuplexFence("sid-blocked")
    independent = DuplexFence("sid-independent")
    capabilities = DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK})
    plane.sessions.open_session(blocked, capabilities=capabilities)
    plane.sessions.open_session(independent, capabilities=capabilities)

    plane.dispatch(
        AppendDuplexInputMessage(
            control_id="blocked-append",
            fence=blocked,
            session_id=blocked.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
            operation_id="blocked-operation",
        )
    )
    await asyncio.wait_for(stage_port.started.wait(), timeout=1)
    plane.dispatch(
        AppendDuplexInputMessage(
            control_id="blocked-append-retry",
            fence=blocked,
            session_id=blocked.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
            operation_id="blocked-operation",
        )
    )
    plane.dispatch(
        TouchDuplexSessionMessage(
            control_id="blocked-touch",
            fence=blocked,
            session_id=blocked.session_id,
            activity=DuplexLeaseActivity.HEARTBEAT.value,
        )
    )
    plane.dispatch(
        TouchDuplexSessionMessage(
            control_id="independent-touch",
            fence=independent,
            session_id=independent.session_id,
            activity=DuplexLeaseActivity.HEARTBEAT.value,
        )
    )

    independent_result = await asyncio.wait_for(result_sink.get(), timeout=1)
    assert independent_result.control_id == "independent-touch"

    stage_port.release.set()
    await plane.drain()
    assert [result_sink.get_nowait().control_id for _ in range(3)] == [
        "blocked-append",
        "blocked-append-retry",
        "blocked-touch",
    ]
    assert [
        submission.context.session_id
        for submission in stage_port.submit_calls
        if submission.context.session_id == blocked.session_id
    ] == [blocked.session_id]
