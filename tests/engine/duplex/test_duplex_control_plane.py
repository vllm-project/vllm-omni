# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import logging
from dataclasses import FrozenInstanceError

import msgspec
import pytest

from vllm_omni.engine.duplex.control_plane import (
    DuplexControlPlane,
    DuplexOutputContext,
    DuplexRequestIdentity,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
)
from vllm_omni.engine.duplex.lease import DuplexLeaseActivity, DuplexLeaseConfig
from vllm_omni.engine.duplex.messages import (
    AppendDuplexInputMessage,
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexFence,
    DuplexSessionLifecycleMessage,
    OpenDuplexSessionMessage,
    ResumeDuplexSessionMessage,
    SignalDuplexTurnMessage,
    TouchDuplexSessionMessage,
)
from vllm_omni.engine.duplex.runtime import (
    DUPLEX_CONTRACT_VERSION,
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexRuntimeCapabilities,
)
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.capabilities import minicpmo45_native_capabilities

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

    def supports_scheduler_native_append(self, stage_id: int = 0) -> bool:
        del stage_id
        return False

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


class _DescriptorExtension(_Extension):
    adapter_id = "personaplex"
    runtime_extension_id = "personaplex"


class _NativeStagePort(_TypedStagePort):
    def __init__(self) -> None:
        super().__init__()
        self.wait_replay_safe_calls: list[str] = []

    def supports_scheduler_native_append(self, stage_id: int = 0) -> bool:
        return stage_id == 0

    async def wait_replay_safe(self, request_id: str, *, deadline_monotonic=None) -> None:
        del deadline_monotonic
        self.wait_replay_safe_calls.append(request_id)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@pytest.mark.asyncio
@pytest.mark.parametrize("first_control", ["open", "resume", "signal"])
@pytest.mark.parametrize("terminal", ["cancel", "close"])
@pytest.mark.parametrize("start_waiter", [False, True], ids=["same-tick", "waiting"])
async def test_preempted_queued_append_does_not_cancel_its_predecessor(
    monkeypatch, first_control, terminal, start_waiter
):
    results: asyncio.Queue[DuplexControlResultMessage] = asyncio.Queue()
    plane = DuplexControlPlane(extension=None, stage_port=_TypedStagePort(), result_sink=results)
    fence = DuplexFence("queued-control-preemption")
    entered, release = asyncio.Event(), asyncio.Event()
    cancelled: list[str] = []
    first_messages = {
        "open": OpenDuplexSessionMessage(control_id="first", session_id=fence.session_id, fence=fence, capabilities={}),
        "resume": ResumeDuplexSessionMessage(
            control_id="first", session_id=fence.session_id, fence=fence, expected_lease_generation=0
        ),
        "signal": SignalDuplexTurnMessage(
            control_id="first", session_id=fence.session_id, fence=fence, event="user_started"
        ),
    }

    async def handle(message):
        if message.control_id == "first":
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(first_control)
                raise
        await plane.put_result(
            message.control_id, fence=fence, operation=first_control, session_id=fence.session_id, stage_results=[]
        )

    monkeypatch.setattr(plane, "handle", handle)
    plane.dispatch(first_messages[first_control])
    await asyncio.wait_for(entered.wait(), 1)
    plane.dispatch(
        AppendDuplexInputMessage(
            control_id="append", session_id=fence.session_id, fence=fence, mode="append_tokens", payload={}
        )
    )
    if start_waiter:
        await asyncio.sleep(0)
    if terminal == "close":
        plane.dispatch(CloseDuplexSessionMessage(control_id="terminal", session_id=fence.session_id, fence=fence))
    else:
        plane.dispatch(
            SignalDuplexTurnMessage(
                control_id="terminal", session_id=fence.session_id, fence=fence, event="input.cancel"
            )
        )
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(plane.drain(), 1)
        assert cancelled == []
        replies = {result.control_id: result for result in (results.get_nowait() for _ in range(results.qsize()))}
        assert set(replies) == {"first", "append", "terminal"}
        assert replies["first"].ok and replies["terminal"].ok
        assert replies["append"].error is not None and replies["append"].error.code == "cancelled"
    finally:
        release.set()
        await plane.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cleanup_kind", "pause_at"),
    [
        ("close", "stage_cleanup"),
        ("replica_lost", "stage_cleanup"),
        ("replica_lost", "lifecycle_notification"),
        ("expiry", "stage_cleanup"),
        ("expiry", "lifecycle_notification"),
    ],
)
async def test_late_cleanup_does_not_finalize_reopened_incarnation(cleanup_kind, pause_at):
    entered, release = asyncio.Event(), asyncio.Event()
    clock = _Clock()

    class GatedPort(_TypedStagePort):
        async def cleanup(self, request_ids, *, abort=False):
            if pause_at == "stage_cleanup":
                entered.set()
                await release.wait()
            await super().cleanup(request_ids, abort=abort)

    class GatedSink:
        async def put(self, message):
            if pause_at == "lifecycle_notification":
                entered.set()
                await release.wait()

    plane = DuplexControlPlane(
        extension=None,
        stage_port=GatedPort(),
        result_sink=asyncio.Queue(),
        lifecycle_sink=GatedSink(),
        clock=clock,
        lease_config=DuplexLeaseConfig(idle_ttl_s=1.0),
    )
    old_fence = DuplexFence("cleanup-reopen", incarnation=1)
    old = plane.sessions.open_session(old_fence, lease_config=DuplexLeaseConfig(idle_ttl_s=1.0))
    old.bind_stage_request(0, "old-request", fence=old_fence)
    if cleanup_kind == "close":
        pending = asyncio.create_task(
            plane.handle_close(
                CloseDuplexSessionMessage(control_id="close-old", session_id=old_fence.session_id, fence=old_fence)
            )
        )
    elif cleanup_kind == "replica_lost":
        pending = asyncio.create_task(plane._terminate_replica_lost_session(old))
    else:
        clock.advance(2.0)
        pending = asyncio.create_task(plane.reap_expired())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        # Independent request cleanup completes before the original I/O await.
        plane.sessions.finalize_close_session(old)
        new_fence = DuplexFence(old_fence.session_id, incarnation=2)
        new = plane.sessions.open_session(new_fence)
        new.bind_stage_request(0, "new-request", fence=new_fence)
        if cleanup_kind == "expiry":
            # Both old expiry and a valid resume advance the lease counter.
            # Equal lease generations do not imply equal incarnations.
            new.resume(new_fence, expected_lease_generation=0)
            assert new.lease.generation == old.lease.generation
        release.set()
        await asyncio.wait_for(pending, 1)
        assert plane.sessions.get(new_fence.session_id) is new
        assert new.resource_request_ids() == ["new-request"]
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await plane.shutdown()


@pytest.mark.asyncio
async def test_model_native_open_rejects_missing_append_contract_before_admission() -> None:
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence("native-without-append")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open",
            fence=fence,
            session_id=fence.session_id,
            capabilities={"implementation_level": "model_native_duplex", "supports_scheduler_native_append": False},
        )
    )
    result = await result_sink.get()
    assert result.ok is False
    assert plane.sessions.session_count == 0
    assert stage_port.ensure_calls == []
    assert stage_port.submit_calls == []


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
    assert context.trace is not None
    assert context.trace.session_id == fence.session_id
    assert context.trace.request_id == context.request_id
    assert context.trace.event == "stage_request"
    with pytest.raises(FrozenInstanceError):
        context.stage_id = 1  # type: ignore[misc]

    await plane.handle(
        AppendDuplexInputMessage(
            control_id="append",
            operation_id="append-operation",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
            final=True,
            deadline_monotonic=123.0,
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
    assert submission.operation_id == "append-operation"
    assert submission.deadline_monotonic == 123.0
    assert not hasattr(submission, "request_state")


@pytest.mark.asyncio
async def test_completed_append_retry_returns_deduplicated_receipt_without_resubmit() -> None:
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence("sid-deduplicated-retry")
    plane.sessions.open_session(
        fence,
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK}),
    )
    append = AppendDuplexInputMessage(
        control_id="append-first",
        operation_id="stable-operation",
        fence=fence,
        session_id=fence.session_id,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
        payload={"audio": b"pcm"},
    )

    await plane.handle(append)
    first = await result_sink.get()
    await plane.handle(
        AppendDuplexInputMessage(
            control_id="append-retry",
            operation_id=append.operation_id,
            fence=fence,
            session_id=fence.session_id,
            mode=append.mode,
            payload=append.payload,
        )
    )
    retry = await result_sink.get()

    assert first.ok is True
    assert retry.ok is True
    assert retry.stage_results[0]["result"]["deduplicated"] is True
    assert len(stage_port.submit_calls) == 1


@pytest.mark.asyncio
async def test_open_rejects_scheduler_native_append_not_supported_by_engine() -> None:
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(),
        stage_port=stage_port,
        result_sink=result_sink,
    )
    fence = DuplexFence("native-unsupported")

    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-native",
            fence=fence,
            session_id=fence.session_id,
            capabilities={
                "input_modes": [DuplexInputMode.APPEND_AUDIO_CHUNK.value],
                "supports_scheduler_native_append": True,
            },
        )
    )

    result = await result_sink.get()
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "invalid_capability"
    assert "scheduler_native_append_not_supported_by_engine_stage_0" in result.error.message
    assert plane.sessions.get(fence.session_id) is None


@pytest.mark.asyncio
async def test_open_validates_plugin_descriptor_against_engine_stage_topology() -> None:
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_DescriptorExtension(),
        stage_port=stage_port,
        result_sink=result_sink,
    )
    fence = DuplexFence("descriptor-stage-mismatch")

    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-descriptor-stage-mismatch",
            fence=fence,
            session_id=fence.session_id,
            capabilities={
                "contract_version": DUPLEX_CONTRACT_VERSION,
                "adapter_id": "personaplex",
                "runtime_extension_id": "personaplex",
                "stage_count": 3,
                "input_modes": [DuplexInputMode.APPEND_AUDIO_CHUNK.value],
            },
        )
    )

    result = await result_sink.get()
    assert result.ok is False
    assert result.error is not None
    assert "duplex_plugin_stage_count_mismatch" in result.error.message
    assert plane.sessions.get(fence.session_id) is None


@pytest.mark.asyncio
async def test_open_rejects_explicit_unknown_contract_version_without_descriptor_fields() -> None:
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_DescriptorExtension(),
        stage_port=_TypedStagePort(),
        result_sink=result_sink,
    )
    fence = DuplexFence("descriptor-version-mismatch")

    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-descriptor-version-mismatch",
            fence=fence,
            session_id=fence.session_id,
            capabilities={
                "contract_version": "duplex.capabilities.v0",
                "input_modes": [DuplexInputMode.TURN_COMMIT_ONLY.value],
            },
        )
    )

    result = await result_sink.get()
    assert result.ok is False
    assert result.error is not None
    assert "unsupported duplex contract version" in result.error.message
    assert plane.sessions.get(fence.session_id) is None


@pytest.mark.asyncio
async def test_open_admission_queue_promotes_after_cleanup_with_structured_reason() -> None:
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(),
        stage_port=stage_port,
        result_sink=result_sink,
        max_sessions=1,
        max_pending_session_opens=1,
        session_open_queue_timeout_s=2.0,
    )
    first_fence = DuplexFence("admission-first")
    second_fence = DuplexFence("admission-second")

    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-first",
            fence=first_fence,
            session_id=first_fence.session_id,
            capabilities={"input_modes": [DuplexInputMode.APPEND_AUDIO_CHUNK.value]},
        )
    )
    first = await result_sink.get()
    assert first.ok is True
    assert first.admission["mode"] == "immediate"

    queued = asyncio.create_task(
        plane.handle(
            OpenDuplexSessionMessage(
                control_id="open-second",
                fence=second_fence,
                session_id=second_fence.session_id,
                capabilities={"input_modes": [DuplexInputMode.APPEND_AUDIO_CHUNK.value]},
            )
        )
    )
    await asyncio.sleep(0)
    assert not queued.done()

    await plane.handle(
        CloseDuplexSessionMessage(
            control_id="close-first",
            fence=first_fence,
            session_id=first_fence.session_id,
        )
    )
    await queued
    results = [await result_sink.get(), await result_sink.get()]
    close_result = next(item for item in results if item.session_id == first_fence.session_id)
    second = next(item for item in results if item.session_id == second_fence.session_id)
    assert close_result.ok is True
    assert second.ok is True
    assert second.session_id == second_fence.session_id
    assert second.admission["mode"] == "queued"
    assert second.admission["reason"] == "capacity_released"


@pytest.mark.asyncio
async def test_open_admission_queue_reports_queue_full_without_waiting() -> None:
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(),
        stage_port=_TypedStagePort(),
        result_sink=result_sink,
        max_sessions=1,
        max_pending_session_opens=1,
        session_open_queue_timeout_s=1.0,
    )
    first_fence = DuplexFence("queue-full-first")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-first",
            fence=first_fence,
            session_id=first_fence.session_id,
            capabilities={"input_modes": [DuplexInputMode.APPEND_AUDIO_CHUNK.value]},
        )
    )
    assert (await result_sink.get()).ok is True

    queued_fence = DuplexFence("queue-full-queued")
    blocked = asyncio.create_task(
        plane.handle(
            OpenDuplexSessionMessage(
                control_id="open-queued",
                fence=queued_fence,
                session_id=queued_fence.session_id,
                capabilities={"input_modes": [DuplexInputMode.APPEND_AUDIO_CHUNK.value]},
            )
        )
    )
    await asyncio.sleep(0)
    rejected_fence = DuplexFence("queue-full-rejected")
    await plane.handle(
        OpenDuplexSessionMessage(
            control_id="open-rejected",
            fence=rejected_fence,
            session_id=rejected_fence.session_id,
            capabilities={"input_modes": [DuplexInputMode.APPEND_AUDIO_CHUNK.value]},
        )
    )
    await blocked
    results = [await result_sink.get(), await result_sink.get()]
    rejected = next(item for item in results if item.session_id == rejected_fence.session_id)
    assert rejected.ok is False
    assert rejected.admission["reason"] == "queue_full"


@pytest.mark.asyncio
async def test_scheduler_native_initial_append_requires_operation_id_before_reserving_request() -> None:
    class _NativeStagePort(_TypedStagePort):
        def supports_scheduler_native_append(self, stage_id: int = 0) -> bool:
            return stage_id == 0

    stage_port = _NativeStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence("native-operation-required")
    session = plane.sessions.open_session(
        fence,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
            scheduler_native_append=True,
        ),
    )

    await plane.handle(
        AppendDuplexInputMessage(
            control_id="append-without-operation-id",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
        )
    )

    result = await result_sink.get()
    assert result.ok is False
    assert result.error.code == "invalid_argument"
    assert "non-empty operation_id" in result.error.message
    assert stage_port.ensure_calls == []
    assert stage_port.submit_calls == []
    assert session.resource_request_ids() == []


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


@pytest.mark.asyncio
@pytest.mark.parametrize("preempt_with", ["cancel", "close"])
async def test_cancel_or_close_preempts_native_append_waiting_for_core_reply(preempt_with: str) -> None:
    class _BlockedUpdateStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    stage_port = _BlockedUpdateStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence(f"sid-preempt-{preempt_with}")
    session = plane.sessions.open_session(
        fence,
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK}),
    )
    request_id = plane.stage_request_id(fence, stage_id=0)
    session.bind_stage_request(0, request_id, fence=fence)

    plane.dispatch(
        AppendDuplexInputMessage(
            control_id="append-blocked",
            operation_id="append-blocked-operation",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
        )
    )
    await asyncio.wait_for(stage_port.started.wait(), timeout=1)
    if preempt_with == "close":
        plane.dispatch(
            CloseDuplexSessionMessage(
                control_id="preempt-close",
                fence=fence,
                session_id=fence.session_id,
                reason="client_close",
            )
        )
    else:
        plane.dispatch(
            SignalDuplexTurnMessage(
                control_id="preempt-cancel",
                fence=fence,
                session_id=fence.session_id,
                event="response.cancel",
                next_fence=DuplexFence(fence.session_id, epoch=1),
            )
        )

    await asyncio.wait_for(plane.drain(), timeout=1)
    results = {result.control_id: result for result in (result_sink.get_nowait(), result_sink.get_nowait())}
    assert results["append-blocked"].error.code == "cancelled"
    assert results[f"preempt-{preempt_with}"].ok is True
    assert stage_port.cleanup_calls == [([request_id], True)]
    if preempt_with == "close":
        assert plane.sessions.get(fence.session_id) is None
    else:
        assert plane.sessions.require(fence.session_id).fence.epoch == 1


@pytest.mark.asyncio
async def test_close_does_not_recancel_append_while_cancel_reply_is_being_published() -> None:
    class _BlockedUpdateStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class _BlockingResultSink:
        def __init__(self) -> None:
            self.queue: asyncio.Queue = asyncio.Queue()
            self.append_reply_started = asyncio.Event()
            self.release_append_reply = asyncio.Event()

        async def put(self, message) -> None:
            if message.control_id == "append-blocked":
                self.append_reply_started.set()
                await self.release_append_reply.wait()
            await self.queue.put(message)

    stage_port = _BlockedUpdateStagePort()
    result_sink = _BlockingResultSink()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence("sid-cancel-then-close")
    next_fence = DuplexFence(fence.session_id, epoch=1)
    session = plane.sessions.open_session(
        fence,
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK}),
    )
    request_id = plane.stage_request_id(fence, stage_id=0)
    session.bind_stage_request(0, request_id, fence=fence)

    plane.dispatch(
        AppendDuplexInputMessage(
            control_id="append-blocked",
            operation_id="append-blocked-operation",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
        )
    )
    await asyncio.wait_for(stage_port.started.wait(), timeout=1)
    plane.dispatch(
        SignalDuplexTurnMessage(
            control_id="cancel-blocked-append",
            fence=fence,
            session_id=fence.session_id,
            event="input.cancel",
            next_fence=next_fence,
        )
    )
    await asyncio.wait_for(result_sink.append_reply_started.wait(), timeout=1)

    # The close arrives while the cancelled append is inside result_sink.put().
    # It must wait for the first cancellation reply instead of cancelling that
    # task a second time and losing the append's correlated result.
    plane.dispatch(
        CloseDuplexSessionMessage(
            control_id="close-after-cancel",
            fence=next_fence,
            session_id=fence.session_id,
        )
    )
    result_sink.release_append_reply.set()
    await asyncio.wait_for(plane.drain(), timeout=1)

    results = {result.control_id: result for result in (result_sink.queue.get_nowait() for _ in range(3))}
    assert results["append-blocked"].error.code == "cancelled"
    assert results["cancel-blocked-append"].ok is True
    assert results["close-after-cancel"].ok is True
    assert stage_port.cleanup_calls == [([request_id], True)]
    assert plane.sessions.get(fence.session_id) is None


@pytest.mark.asyncio
async def test_per_session_append_admission_is_bounded_and_close_still_preempts() -> None:
    class _BlockedStagePort(_TypedStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()

        async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    stage_port = _BlockedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(),
        stage_port=stage_port,
        result_sink=result_sink,
        max_pending_appends_per_session=1,
    )
    fence = DuplexFence("sid-append-admission")
    plane.sessions.open_session(
        fence,
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK}),
    )
    plane.dispatch(
        AppendDuplexInputMessage(
            control_id="append-admitted",
            operation_id="op-admitted",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"one"},
        )
    )
    await asyncio.wait_for(stage_port.started.wait(), timeout=1)
    plane.dispatch(
        AppendDuplexInputMessage(
            control_id="append-rejected",
            operation_id="op-rejected",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"two"},
        )
    )
    plane.dispatch(
        CloseDuplexSessionMessage(
            control_id="close-after-overload",
            fence=fence,
            session_id=fence.session_id,
        )
    )
    await asyncio.wait_for(plane.drain(), timeout=1)
    results = {result.control_id: result for result in (result_sink.get_nowait() for _ in range(3))}
    assert results["append-rejected"].error.code == "resource_exhausted"
    assert results["append-rejected"].error.retryable is True
    assert results["append-admitted"].error.code == "cancelled"
    assert results["close-after-overload"].ok is True
    assert plane.sessions.get(fence.session_id) is None


def _native_session(plane: DuplexControlPlane, fence: DuplexFence, *, prompt_replay: bool = True):
    return plane.sessions.open_session(
        fence,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
            scheduler_native_append=True,
            prompt_replay=prompt_replay,
        ),
    )


def _append_message(fence: DuplexFence, index: int, *, payload=None) -> AppendDuplexInputMessage:
    return AppendDuplexInputMessage(
        control_id=f"append-{index}",
        operation_id=f"operation-{index}",
        fence=fence,
        session_id=fence.session_id,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
        payload={"audio": f"pcm-{index}".encode()} if payload is None else payload,
    )


def test_prompt_replay_requires_native_append_in_public_and_engine_contracts() -> None:
    from vllm_omni.entrypoints.duplex.protocol import DuplexCapabilities

    with pytest.raises(ValueError, match="prompt replay requires scheduler-native append"):
        DuplexCapabilities(supports_prompt_replay=True)
    with pytest.raises(ValueError, match="prompt replay requires scheduler-native append"):
        DuplexRuntimeCapabilities(prompt_replay=True)
    with pytest.raises(ValueError, match="prompt replay requires scheduler-native append"):
        DuplexControlPlane.coerce_capabilities({"supports_prompt_replay": True})


@pytest.mark.parametrize("value", ["false", 1, None])
def test_prompt_replay_wire_capability_rejects_non_boolean(value) -> None:
    with pytest.raises(TypeError, match="supports_prompt_replay must be a boolean"):
        DuplexControlPlane.coerce_capabilities(
            {"supports_scheduler_native_append": True, "supports_prompt_replay": value}
        )


def test_prompt_replay_capability_defaults_off_and_round_trips_when_opted_in() -> None:
    from vllm_omni.entrypoints.duplex.protocol import DuplexCapabilities

    assert DuplexControlPlane.coerce_capabilities({"supports_scheduler_native_append": True}).prompt_replay is False
    public = DuplexCapabilities(supports_scheduler_native_append=True, supports_prompt_replay=True)
    assert DuplexControlPlane.coerce_capabilities(public.as_dict()).prompt_replay is True


@pytest.mark.parametrize("available", [False, True])
def test_minicpmo_replay_capability_requires_the_native_append_contract(monkeypatch, available) -> None:
    monkeypatch.setattr("vllm_omni.engine.kv_append.scheduler_native_append_available", lambda: available)
    public = minicpmo45_native_capabilities()
    assert public.supports_scheduler_native_append is available
    assert public.supports_prompt_replay is available


@pytest.mark.asyncio
async def test_native_append_without_replay_does_not_journal_rollover_or_recover() -> None:
    port = _NativeStagePort()
    sink: asyncio.Queue[DuplexControlResultMessage] = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(),
        stage_port=port,
        result_sink=sink,
        recovery_max_replay_tokens=4,
        rollover_retain_tokens=2,
    )
    fence = DuplexFence("sid-native-without-replay")
    session = _native_session(plane, fence, prompt_replay=False)
    session.update_scheduler_context(tokens=8, limit=10)

    for index in range(1, 4):
        await plane.handle(_append_message(fence, index))
        assert sink.get_nowait().ok is True

    assert session.input_seq == 3
    assert all(call.context.scheduler_native_append for call in port.submit_calls)
    assert not session.replay_appends
    assert session.replay_token_count == session.replay_byte_count == 0
    assert session.resource_generation == 0
    assert not port.wait_replay_safe_calls
    request_id = session.stage_bindings[0].request_id
    recoverable, terminal = plane.prepare_replica_recovery(0, [request_id])
    assert recoverable == set()
    assert terminal == {request_id}
    assert session.recovery_required is False

    with pytest.raises(RuntimeError, match="prompt replay is not supported"):
        await plane._rebuild_scheduler_native_session(session, (), deadline_monotonic=None)
    assert not port.cleanup_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_limit,append_count", [(1, 4), (256, 260)])
async def test_expired_operation_is_rejected_across_physical_kv_rollover(receipt_limit, append_count):
    class Extension(_Extension):
        def plan_append(self, **kwargs):
            return DuplexAppendPlan(prompt={"prompt_token_ids": [1, 2]})

    port = _NativeStagePort()
    sink: asyncio.Queue[DuplexControlResultMessage] = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=Extension(),
        stage_port=port,
        result_sink=sink,
        completed_append_limit=receipt_limit,
        recovery_max_replay_tokens=6,
        rollover_trigger_fraction=0,
        rollover_retain_tokens=3,
    )
    fence = DuplexFence("rollover-receipts")
    session = _native_session(plane, fence)
    for index in range(1, append_count + 1):
        await plane.handle(_append_message(fence, index))
        assert sink.get_nowait().ok
    assert session.resource_generation > 0
    before = len(port.submit_calls)
    await plane.handle(_append_message(fence, 2))
    result = sink.get_nowait()
    assert not result.ok
    assert result.error.code == "idempotency_window_expired"
    assert result.error.retryable is False
    assert len(port.submit_calls) == before
    assert session.input_seq == append_count


@pytest.mark.asyncio
async def test_session_tombstone_capacity_rejects_before_submit_and_resets_on_cancel(monkeypatch):
    from vllm_omni.engine.duplex import session as duplex_session

    monkeypatch.setattr(duplex_session, "_APPEND_TOMBSTONE_LIMIT", 1)
    port = _NativeStagePort()
    sink: asyncio.Queue[DuplexControlResultMessage] = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=port, result_sink=sink, completed_append_limit=1)
    fence = DuplexFence("receipt-capacity")
    session = _native_session(plane, fence)
    for index in (1, 2):
        await plane.handle(_append_message(fence, index))
        assert sink.get_nowait().ok
    before = len(port.submit_calls)
    await plane.handle(_append_message(fence, 3))
    result = sink.get_nowait()
    assert not result.ok
    assert result.error.code == "resource_exhausted"
    assert not result.error.retryable
    assert len(port.submit_calls) == before
    assert len(session.retired_append_ids) == 1
    next_fence = DuplexFence(fence.session_id, epoch=1)
    await plane.handle(
        SignalDuplexTurnMessage(
            control_id="cancel",
            session_id=fence.session_id,
            fence=fence,
            event="input.cancel",
            next_fence=next_fence,
        )
    )
    assert sink.get_nowait().ok
    assert not session.retired_append_ids
    await plane.handle(_append_message(next_fence, 1))
    assert sink.get_nowait().ok


@pytest.mark.asyncio
async def test_committed_native_journal_recovers_on_new_physical_request_generation() -> None:
    stage_port = _NativeStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence("sid-recover-committed")
    session = _native_session(plane, fence)

    await plane.handle(_append_message(fence, 1))
    assert (await result_sink.get()).ok is True
    old_request_id = session.stage_bindings[0].request_id

    recoverable, terminal = plane.prepare_replica_recovery(0, [old_request_id])
    assert recoverable == {old_request_id}
    assert terminal == set()
    assert session.recovery_required is True
    assert session.resource_request_ids() == []

    await plane.handle(_append_message(fence, 2))
    result = await result_sink.get()

    assert result.ok is True
    assert session.resource_generation == 1
    assert session.recovery_required is False
    assert [append.operation_id for append in session.replay_appends] == ["operation-1", "operation-2"]
    new_request_id = session.stage_bindings[0].request_id
    assert new_request_id != old_request_id
    assert "stage0g1" in new_request_id
    assert [submission.recovery_replay for submission in stage_port.submit_calls] == [False, True, False]
    assert stage_port.submit_calls[1].already_submitted is False
    assert stage_port.submit_calls[2].already_submitted is True


@pytest.mark.asyncio
async def test_uncertain_native_append_fails_closed_instead_of_cross_replica_replay() -> None:
    stage_port = _NativeStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence("sid-recover-uncertain")
    session = _native_session(plane, fence)

    await plane.handle(_append_message(fence, 1))
    assert (await result_sink.get()).ok is True
    request_id = session.stage_bindings[0].request_id

    recoverable, terminal = plane.prepare_replica_recovery(
        0,
        [request_id],
        uncertain_request_ids=[request_id],
    )

    assert recoverable == set()
    assert terminal == {request_id}
    assert session.recovery_required is False
    assert session.stage_bindings[0].request_id == request_id


@pytest.mark.asyncio
async def test_replay_failure_aborts_new_generation_and_returns_typed_error() -> None:
    class _FailReplayStagePort(_NativeStagePort):
        async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
            self.submit_calls.append(submission)
            if submission.recovery_replay:
                raise RuntimeError("deterministic replay rejected")
            return DuplexStageSubmissionResult(
                request_id=submission.context.request_id,
                stage_id=submission.context.stage_id,
                replica_id=3,
            )

    stage_port = _FailReplayStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence("sid-replay-failure")
    session = _native_session(plane, fence)

    await plane.handle(_append_message(fence, 1))
    assert (await result_sink.get()).ok is True
    old_request_id = session.stage_bindings[0].request_id
    assert plane.prepare_replica_recovery(0, [old_request_id]) == ({old_request_id}, set())

    await plane.handle(_append_message(fence, 2))
    result = await result_sink.get()

    assert result.ok is False
    assert result.error.code == "kv_recovery_failed"
    assert result.error.retryable is False
    assert session.input_seq == 1
    assert session.recovery_required is True
    assert session.resource_request_ids() == []
    assert stage_port.cleanup_calls == [([plane.stage_request_id(fence, stage_id=0, resource_generation=1)], True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("preempt_with", ["cancel", "close"])
async def test_cancel_or_close_preempts_replica_recovery_replay(preempt_with: str) -> None:
    class _BlockedReplayStagePort(_NativeStagePort):
        def __init__(self) -> None:
            super().__init__()
            self.replay_started = asyncio.Event()

        async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
            self.submit_calls.append(submission)
            if submission.recovery_replay:
                self.replay_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            return DuplexStageSubmissionResult(
                request_id=submission.context.request_id,
                stage_id=submission.context.stage_id,
                replica_id=3,
            )

    stage_port = _BlockedReplayStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence(f"sid-recovery-preempt-{preempt_with}")
    session = _native_session(plane, fence)

    await plane.handle(_append_message(fence, 1))
    assert (await result_sink.get()).ok is True
    old_request_id = session.stage_bindings[0].request_id
    assert plane.prepare_replica_recovery(0, [old_request_id]) == ({old_request_id}, set())

    plane.dispatch(_append_message(fence, 2))
    await asyncio.wait_for(stage_port.replay_started.wait(), timeout=1)
    if preempt_with == "close":
        plane.dispatch(
            CloseDuplexSessionMessage(
                control_id="preempt-recovery-close",
                fence=fence,
                session_id=fence.session_id,
            )
        )
    else:
        plane.dispatch(
            SignalDuplexTurnMessage(
                control_id="preempt-recovery-cancel",
                fence=fence,
                session_id=fence.session_id,
                event="input.cancel",
                next_fence=DuplexFence(fence.session_id, epoch=1),
            )
        )

    await asyncio.wait_for(plane.drain(), timeout=1)
    results = {result.control_id: result for result in (result_sink.get_nowait(), result_sink.get_nowait())}
    assert results["append-2"].error.code == "cancelled"
    assert results[f"preempt-recovery-{preempt_with}"].ok is True
    new_request_id = plane.stage_request_id(fence, stage_id=0, resource_generation=1)
    assert stage_port.cleanup_calls == [([new_request_id], True)]
    if preempt_with == "close":
        assert plane.sessions.get(fence.session_id) is None
    else:
        surviving = plane.sessions.require(fence.session_id)
        assert surviving.fence.epoch == 1
        assert surviving.resource_request_ids() == []


@pytest.mark.asyncio
async def test_context_rollover_keeps_recent_complete_units_and_rebases_first_replay() -> None:
    class _RolloverExtension(_Extension):
        def __init__(self) -> None:
            self.recovery_initial: list[bool] = []

        def plan_append(self, *, request_id, payload, **kwargs):
            del kwargs
            return DuplexAppendPlan(
                prompt={
                    "prompt_token_ids": [int(payload["token"])] * 2,
                    "model_intermediate_buffer": {"request_id": request_id, "duplex": {}},
                }
            )

        def prepare_recovery_prompt(self, *, prompt, request_id, initial):
            self.recovery_initial.append(initial)
            copied: OmniTokensPrompt = {
                "prompt_token_ids": list(prompt["prompt_token_ids"]),
                "model_intermediate_buffer": {
                    "request_id": request_id,
                    "duplex": dict(prompt["model_intermediate_buffer"]["duplex"]),
                },
            }
            copied["model_intermediate_buffer"]["duplex"]["rebased_initial"] = initial
            return copied

    stage_port = _NativeStagePort()
    extension = _RolloverExtension()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=extension,
        stage_port=stage_port,
        result_sink=result_sink,
        recovery_max_replay_tokens=6,
        recovery_max_replay_bytes=4096,
        rollover_trigger_fraction=0,
        rollover_retain_tokens=3,
    )
    fence = DuplexFence("sid-context-rollover")
    session = _native_session(plane, fence)

    for index in range(1, 5):
        await plane.handle(_append_message(fence, index, payload={"token": index}))
        assert (await result_sink.get()).ok is True

    assert session.resource_generation == 1
    assert [append.operation_id for append in session.replay_appends] == ["operation-3", "operation-4"]
    assert session.replay_token_count == 4
    assert extension.recovery_initial == [True]
    replay_submission = [submission for submission in stage_port.submit_calls if submission.recovery_replay]
    assert len(replay_submission) == 1
    assert replay_submission[0].prompt["model_intermediate_buffer"]["duplex"]["rebased_initial"] is True
    assert replay_submission[0].prompt["model_intermediate_buffer"]["duplex"]["recovery_replay"] is True
    assert stage_port.wait_replay_safe_calls == [plane.stage_request_id(fence, stage_id=0)]
    assert stage_port.cleanup_calls == [([plane.stage_request_id(fence, stage_id=0)], True)]


@pytest.mark.asyncio
async def test_context_rollover_accounts_for_rebased_prefix_and_can_make_live_append_initial() -> None:
    class _ContextBudgetExtension(_Extension):
        def plan_append(self, *, request_id, payload, **kwargs):
            del kwargs
            return DuplexAppendPlan(
                prompt={
                    "prompt_token_ids": [int(payload["token"])] * 2,
                    "model_intermediate_buffer": {"request_id": request_id, "duplex": {}},
                }
            )

        def prepare_recovery_prompt(self, *, prompt, request_id, initial):
            copied: OmniTokensPrompt = {
                "prompt_token_ids": list(prompt["prompt_token_ids"]),
                "model_intermediate_buffer": {
                    "request_id": request_id,
                    "duplex": dict(prompt["model_intermediate_buffer"]["duplex"]),
                },
            }
            if initial:
                copied["prompt_token_ids"] += [99, 99, 99]
            copied["model_intermediate_buffer"]["duplex"]["rebased_initial"] = initial
            return copied

    stage_port = _NativeStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_ContextBudgetExtension(),
        stage_port=stage_port,
        result_sink=result_sink,
        recovery_max_replay_tokens=8,
        recovery_max_replay_bytes=4096,
        rollover_trigger_fraction=0.8,
        rollover_retain_tokens=4,
    )
    fence = DuplexFence("sid-context-prefix-budget")
    session = _native_session(plane, fence)

    await plane.handle(_append_message(fence, 1, payload={"token": 1}))
    assert (await result_sink.get()).ok is True
    session.update_scheduler_context(tokens=5, limit=6)

    await plane.handle(_append_message(fence, 2, payload={"token": 2}))
    assert (await result_sink.get()).ok is True

    assert session.resource_generation == 1
    assert [append.operation_id for append in session.replay_appends] == ["operation-2"]
    assert [submission.recovery_replay for submission in stage_port.submit_calls] == [False, False]
    rebuilt_submission = stage_port.submit_calls[-1]
    assert rebuilt_submission.already_submitted is False
    assert len(rebuilt_submission.prompt["prompt_token_ids"]) == 5
    assert rebuilt_submission.prompt["model_intermediate_buffer"]["duplex"]["rebased_initial"] is True
    # Model metadata, not just the control-plane flag, drives output suppression.
    assert rebuilt_submission.prompt["model_intermediate_buffer"]["duplex"].get("recovery_replay", False) is False
    assert stage_port.cleanup_calls == [([plane.stage_request_id(fence, stage_id=0)], True)]


@pytest.mark.asyncio
async def test_context_rollover_rejects_oversized_rebased_candidate_before_destroying_old_kv() -> None:
    class _OversizedInitialExtension(_Extension):
        def plan_append(self, *, request_id, **kwargs):
            del kwargs
            return DuplexAppendPlan(
                prompt={
                    "prompt_token_ids": [1, 1],
                    "model_intermediate_buffer": {"request_id": request_id, "duplex": {}},
                }
            )

        def prepare_recovery_prompt(self, *, prompt, request_id, initial):
            copied: OmniTokensPrompt = {
                "prompt_token_ids": list(prompt["prompt_token_ids"]),
                "model_intermediate_buffer": {
                    "request_id": request_id,
                    "duplex": dict(prompt["model_intermediate_buffer"]["duplex"]),
                },
            }
            if initial:
                copied["prompt_token_ids"] += [99] * 5
            return copied

    stage_port = _NativeStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_OversizedInitialExtension(),
        stage_port=stage_port,
        result_sink=result_sink,
        recovery_max_replay_tokens=8,
        recovery_max_replay_bytes=4096,
        rollover_trigger_fraction=0.8,
        rollover_retain_tokens=4,
    )
    fence = DuplexFence("sid-context-prefix-too-large")
    session = _native_session(plane, fence)

    await plane.handle(_append_message(fence, 1))
    assert (await result_sink.get()).ok is True
    old_request_id = session.stage_bindings[0].request_id
    session.update_scheduler_context(tokens=5, limit=6)

    await plane.handle(_append_message(fence, 2))
    result = await result_sink.get()

    assert result.ok is False
    assert result.error.code == "resource_exhausted"
    assert result.error.retryable is False
    assert session.input_seq == 1
    assert session.resource_generation == 0
    assert session.stage_bindings[0].request_id == old_request_id
    assert stage_port.cleanup_calls == []


@pytest.mark.asyncio
async def test_context_exhaustion_is_nonretryable_and_cancel_reclaims_resident_kv() -> None:
    class _ContextLimitedStagePort(_TypedStagePort):
        async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
            raise RuntimeError("streaming_prompt_context_limit_exceeded: projected=65, limit=64")

    stage_port = _ContextLimitedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(extension=_Extension(), stage_port=stage_port, result_sink=result_sink)
    fence = DuplexFence("sid-context-limit")
    session = plane.sessions.open_session(
        fence,
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK}),
    )
    request_id = plane.stage_request_id(fence, stage_id=0)
    session.bind_stage_request(0, request_id, fence=fence)

    await plane.handle(
        AppendDuplexInputMessage(
            control_id="append-over-context",
            operation_id="op-over-context",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
        )
    )
    result = await result_sink.get()

    assert result.error.code == "resource_exhausted"
    assert result.error.retryable is False
    assert session.input_seq == 0
    assert session.stage_bindings[0].request_id == request_id

    next_fence = DuplexFence(fence.session_id, epoch=1)
    await plane.handle(
        SignalDuplexTurnMessage(
            control_id="cancel-exhausted-context",
            fence=fence,
            session_id=fence.session_id,
            event="input.cancel",
            next_fence=next_fence,
        )
    )

    assert (await result_sink.get()).ok is True
    assert stage_port.cleanup_calls == [([request_id], True)]
    assert session.fence == next_fence
    assert session.resource_request_ids() == []


@pytest.mark.asyncio
async def test_replica_affinity_loss_terminally_closes_native_session() -> None:
    class _LostReplicaStagePort(_TypedStagePort):
        async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
            raise RuntimeError("resident KV cannot be reassigned")

    stage_port = _LostReplicaStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    lifecycle_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(),
        stage_port=stage_port,
        result_sink=result_sink,
        lifecycle_sink=lifecycle_sink,
    )
    fence = DuplexFence("sid-replica-lost")
    session = plane.sessions.open_session(
        fence,
        capabilities=DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK}),
    )
    request_id = plane.stage_request_id(fence, stage_id=0)
    session.bind_stage_request(0, request_id, fence=fence)

    await plane.handle(
        AppendDuplexInputMessage(
            control_id="append-after-replica-loss",
            operation_id="op-after-replica-loss",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": b"pcm"},
        )
    )
    result = await result_sink.get()
    lifecycle = await lifecycle_sink.get()

    assert result.error.code == "replica_lost"
    assert result.error.retryable is False
    assert lifecycle.event == "terminated"
    assert lifecycle.reason == "native_kv_replica_lost"
    assert stage_port.cleanup_calls == [([request_id], True)]
    assert plane.sessions.get(fence.session_id) is None


@pytest.mark.asyncio
async def test_many_sessions_make_progress_without_cross_session_state_leakage() -> None:
    stage_port = _TypedStagePort()
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(),
        stage_port=stage_port,
        result_sink=result_sink,
        max_sessions=32,
    )
    capabilities = DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK})
    fences = [DuplexFence(f"sid-load-{index}") for index in range(32)]
    for fence in fences:
        plane.sessions.open_session(fence, capabilities=capabilities)
        plane.dispatch(
            AppendDuplexInputMessage(
                control_id=f"append-load-{fence.session_id}",
                operation_id=f"operation-{fence.session_id}",
                fence=fence,
                session_id=fence.session_id,
                mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
                payload={"audio": fence.session_id.encode()},
            )
        )

    await asyncio.wait_for(plane.drain(), timeout=2)
    results = [result_sink.get_nowait() for _ in fences]

    assert all(result.ok for result in results)
    assert {result.session_id for result in results} == {fence.session_id for fence in fences}
    assert len({submission.context.request_id for submission in stage_port.submit_calls}) == len(fences)
    assert all(plane.sessions.require(fence.session_id).input_seq == 1 for fence in fences)


@pytest.mark.asyncio
async def test_hot_session_backlog_cannot_starve_other_sessions_and_close_reclaims_it() -> None:
    """A saturated session must not consume the control plane's global progress.

    Keep one Stage0 submission blocked, fill that session's bounded append
    backlog, and prove 31 independent sessions still complete before the hot
    session is released. Closing the hot session must then preempt every
    admitted append and leave no queued work behind.
    """

    class _HotSessionBlockingStagePort(_TypedStagePort):
        def __init__(self, hot_session_id: str) -> None:
            super().__init__()
            self.hot_session_id = hot_session_id
            self.hot_append_started = asyncio.Event()

        async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
            if submission.context.session_id != self.hot_session_id:
                return await super().submit(submission)
            self.submit_calls.append(submission)
            self.hot_append_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    hot_fence = DuplexFence("sid-load-hot")
    cold_fences = [DuplexFence(f"sid-load-cold-{index}") for index in range(31)]
    stage_port = _HotSessionBlockingStagePort(hot_fence.session_id)
    result_sink: asyncio.Queue = asyncio.Queue()
    plane = DuplexControlPlane(
        extension=_Extension(),
        stage_port=stage_port,
        result_sink=result_sink,
        max_sessions=32,
        max_pending_appends_per_session=4,
    )
    capabilities = DuplexRuntimeCapabilities(input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK})
    for fence in [hot_fence, *cold_fences]:
        plane.sessions.open_session(fence, capabilities=capabilities)

    def append_message(fence: DuplexFence, suffix: str) -> AppendDuplexInputMessage:
        return AppendDuplexInputMessage(
            control_id=f"append-{suffix}",
            operation_id=f"operation-{suffix}",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"audio": suffix.encode()},
        )

    plane.dispatch(append_message(hot_fence, "hot-0"))
    await asyncio.wait_for(stage_port.hot_append_started.wait(), timeout=1)
    for index in range(1, 4):
        plane.dispatch(append_message(hot_fence, f"hot-{index}"))
    plane.dispatch(append_message(hot_fence, "hot-overflow"))
    for index, fence in enumerate(cold_fences):
        plane.dispatch(append_message(fence, f"cold-{index}"))

    early_results = [await asyncio.wait_for(result_sink.get(), timeout=1) for _ in range(32)]
    early_by_control = {result.control_id: result for result in early_results}
    assert set(early_by_control) == {
        "append-hot-overflow",
        *(f"append-cold-{index}" for index in range(31)),
    }
    assert early_by_control["append-hot-overflow"].error.code == "resource_exhausted"
    assert early_by_control["append-hot-overflow"].error.retryable is True
    assert all(early_by_control[f"append-cold-{index}"].ok for index in range(31))
    assert plane.sessions.require(hot_fence.session_id).input_seq == 0

    plane.dispatch(
        CloseDuplexSessionMessage(
            control_id="close-hot",
            fence=hot_fence,
            session_id=hot_fence.session_id,
        )
    )
    await asyncio.wait_for(plane.drain(), timeout=1)
    terminal_results = [result_sink.get_nowait() for _ in range(5)]
    terminal_by_control = {result.control_id: result for result in terminal_results}
    assert terminal_by_control["close-hot"].ok is True
    assert all(terminal_by_control[f"append-hot-{index}"].error.code == "cancelled" for index in range(4))
    assert plane.sessions.get(hot_fence.session_id) is None
    assert hot_fence.session_id not in plane._pending_append_counts
    assert hot_fence.session_id not in plane._session_control_tails
