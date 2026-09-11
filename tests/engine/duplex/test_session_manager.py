# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``DuplexSessionManager``: admission, control RPC, command backpressure, reaper and request cleanup."""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import FrozenInstanceError, dataclass, field
from typing import Any

import pytest

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import session_manager as session_manager_module
from vllm_omni.engine.duplex.commands import AppendAudio, CloseSession, Commit, DuplexCommand, Heartbeat
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig, DuplexSessionState
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexStagePort,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
    duplex_resource_request_belongs_to_session,
)
from vllm_omni.engine.duplex.events import (
    DuplexEvent,
    ErrorEvent,
    SessionClosed,
    SessionCreated,
    SessionExpired,
    SessionHeartbeatAck,
)
from vllm_omni.engine.duplex.lease import DuplexLeaseActivity
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    DuplexSessionEventMessage,
    OpenDuplexSessionMessage,
    ResumeDuplexSessionMessage,
    TouchDuplexSessionMessage,
)
from vllm_omni.engine.duplex.plugin import (
    DuplexDataPlane,
    DuplexModelPlugin,
    DuplexModelSessionState,
    DuplexRuntimeConfigError,
    PcmAppendBuffer,
    PcmAppendReservation,
)
from vllm_omni.engine.duplex.session import DuplexEngineSession
from vllm_omni.engine.duplex.session_manager import DuplexSessionManager
from vllm_omni.engine.duplex.session_runner import _Internal

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# --------------------------------------------------------------------------- #
# Fakes                                                                       #
# --------------------------------------------------------------------------- #


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakePcmAppendBuffer(PcmAppendBuffer):
    @property
    def pending_byte_count(self) -> int:
        return 0

    def clear(self) -> None:
        pass

    def clear_force_listen(self) -> None:
        pass

    def has_pending(self) -> bool:
        return False

    def has_reserved(self) -> bool:
        return False

    def prepare_append(self, payload, *, operation_id, chunk_period_ms, allow_emit) -> PcmAppendReservation | None:
        del payload, operation_id, chunk_period_ms, allow_emit
        return None

    def prepare_commit(self, *, operation_id, chunk_period_ms) -> PcmAppendReservation:
        del operation_id, chunk_period_ms
        raise NotImplementedError("fake buffer never commits")

    def flush(self, *, chunk_period_ms) -> dict[str, object] | None:
        del chunk_period_ms
        return None


class FakeModelSessionState(DuplexModelSessionState):
    def __init__(self) -> None:
        self.audio_buffer = FakePcmAppendBuffer()
        self.input_since_commit = False
        self.speech_since_commit = False
        self.context_locked = False
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        self.continuation_owner_id = None
        self.continuation_units = 0
        self.pending_silence_task = None
        self.pending_silence_owner_id = None

    def retain_committed_audio(self, payload, *, operation_id, reserved_bytes=0) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes = reserved_bytes

    def clear_committed_audio(self) -> int:
        reserved = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        return reserved

    def clear_continuation(self) -> None:
        self.continuation_owner_id = None
        self.continuation_units = 0


class FakeDataPlane(DuplexDataPlane):
    def __init__(self) -> None:
        self.closed_sessions: list[str] = []

    def begin_request(self, request_id: str) -> None:
        pass

    def is_terminal(self, request_id: str | None) -> bool:
        return False

    def mark_terminal(self, request_id: str) -> None:
        pass

    def close_stream(self, request_id: str) -> None:
        pass

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        self.closed_sessions.append(session_id)

    def project(self, result, *, context=None):
        del result, context
        return []


class FakePlugin(DuplexModelPlugin):
    plugin_id = "fake-duplex"

    def __init__(self, encode_audio=None) -> None:
        super().__init__(encode_audio or (lambda *args: None))
        self.data_plane = FakeDataPlane()
        self.runtime_config: dict[str, object] = {"runtime": "test"}
        self.runtime_config_error: Exception | None = None
        #: Session ids whose ``prepare_runtime_config`` waits for ``runtime_config_gate``.
        self.blocked_session_ids: set[str] = set()
        self.runtime_config_gate = asyncio.Event()
        self.runtime_config_started = asyncio.Event()

    def configure_sampling_params(self, *, runtime_config, defaults):
        del runtime_config
        return tuple(f"configured-{stage_id}" for stage_id, _ in enumerate(defaults))

    def plan_append(self, **kwargs) -> DuplexAppendPlan:
        del kwargs
        return DuplexAppendPlan(prompt={"prompt_token_ids": [1, 2, 3]})

    def decide_output(self, **kwargs):
        del kwargs
        return None

    def create_session_state(self) -> DuplexModelSessionState:
        return FakeModelSessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        del max_sessions
        return DuplexCapabilities(supports_input_append=True)

    def validate_client_extra_body(self, extra_body: object) -> None:
        pass

    async def prepare_runtime_config(self, config: DuplexSessionConfig, *, model_config: Any) -> dict[str, object]:
        del model_config
        if self.runtime_config_error is not None:
            raise self.runtime_config_error
        if config.instructions in self.blocked_session_ids:
            self.runtime_config_started.set()
            await self.runtime_config_gate.wait()
        return dict(self.runtime_config)

    def runtime_config_for_update(self, config, current):
        del config
        return dict(current)

    def data_plane_context(self, **kwargs) -> object:
        return dict(kwargs)


class FakeStagePort(DuplexStagePort):
    def __init__(self, stage_count: int = 2) -> None:
        self._stage_count = stage_count
        self.ensure_calls: list[DuplexStageRequestContext] = []
        self.submit_calls: list[DuplexStageSubmission] = []
        #: Every cleanup attempt, including the ones that were made to fail.
        self.cleanup_calls: list[tuple[list[str], bool]] = []
        self.abort_calls: list[list[str]] = []
        self.cleanup_failures_remaining = 0
        self.cleanup_gate: asyncio.Event | None = None
        self.cleanup_started = asyncio.Event()

    @property
    def stage_count(self) -> int:
        return self._stage_count

    def sampling_defaults(self) -> tuple[object, ...]:
        return tuple(f"default-{stage_id}" for stage_id in range(self._stage_count))

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
        self.cleanup_calls.append((list(request_ids), abort))
        self.cleanup_started.set()
        if self.cleanup_gate is not None:
            await self.cleanup_gate.wait()
        if self.cleanup_failures_remaining:
            self.cleanup_failures_remaining -= 1
            raise RuntimeError("transient cleanup failure")

    async def abort_requests(self, request_ids: list[str]) -> None:
        self.abort_calls.append(list(request_ids))


# --------------------------------------------------------------------------- #
# Harness                                                                     #
# --------------------------------------------------------------------------- #


def _runtime_config(**overrides: Any) -> DuplexSessionRuntimeConfig:
    values: dict[str, Any] = {
        "idle_ttl_s": 300.0,
        "disconnect_grace_s": 30.0,
        "max_sessions": 4,
        "max_pending_input_bytes_per_session": 1024,
        "max_pending_turns_per_session": 2,
    }
    values.update(overrides)
    return DuplexSessionRuntimeConfig(**values)


def stage0_request_id(session_id: str, *, epoch: int = 0) -> str:
    encoded = base64.urlsafe_b64encode(session_id.encode("utf-8")).decode("ascii").rstrip("=")
    return f"duplex-s.{encoded}.e.{epoch}.r.stage0"


@dataclass
class Harness:
    manager: DuplexSessionManager
    plugin: FakePlugin
    stage_port: FakeStagePort
    clock: FakeClock
    output_sink: asyncio.Queue = field(default_factory=asyncio.Queue)
    result_sink: asyncio.Queue = field(default_factory=asyncio.Queue)

    @classmethod
    def create(
        cls,
        *,
        plugin: FakePlugin | None = None,
        stage_port: FakeStagePort | None = None,
        clock: FakeClock | None = None,
        **runtime_overrides: Any,
    ) -> Harness:
        plugin = plugin or FakePlugin()
        stage_port = stage_port or FakeStagePort()
        clock = clock or FakeClock()
        output_sink: asyncio.Queue = asyncio.Queue()
        result_sink: asyncio.Queue = asyncio.Queue()
        manager = DuplexSessionManager(
            plugin=plugin,
            stage_port=stage_port,
            output_sink=output_sink,
            result_sink=result_sink,
            runtime_config=_runtime_config(**runtime_overrides),
            model_config=None,
            clock=clock,
        )
        return cls(manager, plugin, stage_port, clock, output_sink, result_sink)

    async def __aenter__(self) -> Harness:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.manager.shutdown()

    async def open(self, session_id: str, config: DuplexSessionConfig | None = None) -> DuplexControlResultMessage:
        await self.manager.handle(
            OpenDuplexSessionMessage(
                control_id=f"open-{session_id}",
                session_id=session_id,
                session_config=config or DuplexSessionConfig(model="fake-model"),
            )
        )
        return await self.result()

    async def close(self, session_id: str, *, reason: str = "client_close") -> DuplexControlResultMessage:
        await self.manager.handle(
            CloseDuplexSessionMessage(control_id=f"close-{session_id}", session_id=session_id, reason=reason)
        )
        return await self.result()

    async def resume(self, session_id: str, *, expected_lease_generation: int) -> DuplexControlResultMessage:
        await self.manager.handle(
            ResumeDuplexSessionMessage(
                control_id=f"resume-{session_id}-{expected_lease_generation}",
                session_id=session_id,
                expected_lease_generation=expected_lease_generation,
            )
        )
        return await self.result()

    async def touch(self, session_id: str, activity: str) -> DuplexControlResultMessage:
        await self.manager.handle(
            TouchDuplexSessionMessage(
                control_id=f"touch-{session_id}-{activity}", session_id=session_id, activity=activity
            )
        )
        return await self.result()

    def command(self, session_id: str, command: DuplexCommand) -> None:
        self.manager.dispatch(DuplexSessionCommandMessage(session_id=session_id, command=command))

    async def result(self) -> DuplexControlResultMessage:
        result = await asyncio.wait_for(self.result_sink.get(), timeout=2.0)
        assert isinstance(result, DuplexControlResultMessage)
        return result

    def events(self, session_id: str | None = None) -> list[DuplexEvent]:
        events: list[DuplexEvent] = []
        while not self.output_sink.empty():
            message = self.output_sink.get_nowait()
            assert isinstance(message, DuplexSessionEventMessage)
            assert message.event.session_id == message.session_id
            if session_id is None or message.session_id == session_id:
                events.append(message.event)
        return events

    def session(self, session_id: str) -> DuplexEngineSession:
        session = self.manager.get(session_id)
        assert session is not None
        return session

    def capture_submissions(self, session_id: str) -> list[DuplexCommand]:
        """Replace the runner mailbox with a list so admitted commands can be inspected."""
        runner = self.manager.runners[session_id]
        submitted: list[DuplexCommand] = []
        runner.submit = submitted.append  # type: ignore[method-assign]
        return submitted


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


# --------------------------------------------------------------------------- #
# Open                                                                        #
# --------------------------------------------------------------------------- #


async def test_open_answers_with_capabilities_and_emits_session_created() -> None:
    async with Harness.create() as harness:
        config = DuplexSessionConfig(model="fake-model", voice="test")
        result = await harness.open("sid-open", config)

        assert result.ok is True
        assert result.operation == "open"
        assert result.control_id == "open-sid-open"
        assert result.session_id == "sid-open"
        assert result.lease_generation == 0
        assert result.capabilities == DuplexCapabilities(supports_input_append=True)
        assert result.public_session is not None
        assert result.public_session["id"] == "sid-open"
        assert result.public_session["voice"] == "test"
        assert result.error_code is None

        events = harness.events("sid-open")
        assert isinstance(events[0], SessionCreated)
        assert events[0].epoch == 0
        assert events[0].session["id"] == "sid-open"
        wire = events[0].to_realtime()
        assert wire["type"] == "session.created"
        assert "incarnation" not in wire and "incarnation" not in wire["session"]

        session = harness.session("sid-open")
        assert session.state == DuplexSessionState.OPEN
        assert session.runtime_config == {"runtime": "test"}
        assert isinstance(session.model_state, FakeModelSessionState)
        assert harness.manager.active_count() == 1
        assert harness.manager.sessions() == {"sid-open": session}


async def test_open_reserves_the_fenced_stage0_request_with_a_frozen_context() -> None:
    async with Harness.create() as harness:
        await harness.open("sid-context", DuplexSessionConfig(model="fake-model", voice="test"))

        request_id = stage0_request_id("sid-context")
        context = harness.stage_port.ensure_calls[-1]
        assert isinstance(context, DuplexStageRequestContext)
        assert context.request_id == request_id
        assert context.session_id == "sid-context"
        assert context.fence == DuplexFence("sid-context")
        assert context.stage_id == 0
        assert context.final_stage_id == 1
        assert context.config_generation == 0
        assert context.sampling_params == ("configured-0", "configured-1")
        assert context.stage_sampling_params == "configured-0"
        assert context.session_config["voice"] == "test"
        assert context.runtime_config == {"runtime": "test"}
        with pytest.raises(FrozenInstanceError):
            context.stage_id = 1  # type: ignore[misc]

        session = harness.session("sid-context")
        assert session.resource_request_ids() == [request_id]
        assert session.stage_request_submitted(0, request_id) is False
        assert harness.manager.runner_for_request_id(request_id) is harness.manager.runners["sid-context"]
        assert harness.stage_port.submit_calls == []


def test_stage_request_id_is_derived_from_the_fence() -> None:
    fence = DuplexFence("sid/with+special", epoch=2, turn_id=5)

    request_id = DuplexSessionManager.stage_request_id(fence, stage_id=0)

    assert request_id == stage0_request_id("sid/with+special", epoch=2)
    assert duplex_resource_request_belongs_to_session(request_id, "sid/with+special")
    assert not duplex_resource_request_belongs_to_session(request_id, "sid/other")


async def test_ensure_stage_request_ignores_stages_beyond_the_pipeline() -> None:
    async with Harness.create() as harness:
        await harness.open("sid-stages")
        session = harness.session("sid-stages")

        assert harness.manager.ensure_stage_request(session, stage_id=2) is None
        context = harness.manager.ensure_stage_request(session, stage_id=1)

        assert context is not None
        assert context.request_id.endswith(".r.stage1")
        assert harness.manager.runner_for_request_id(context.request_id) is harness.manager.runners["sid-stages"]
        assert session.resource_request_ids() == [stage0_request_id("sid-stages"), context.request_id]


async def test_sampling_params_for_validates_the_plugin_policy() -> None:
    class ShortSamplingPlugin(FakePlugin):
        def configure_sampling_params(self, *, runtime_config, defaults):
            if runtime_config:
                return defaults[:1]
            return tuple(defaults)

    class ListSamplingPlugin(FakePlugin):
        def configure_sampling_params(self, *, runtime_config, defaults):
            if runtime_config:
                return list(defaults)
            return tuple(defaults)

    async with Harness.create(plugin=ShortSamplingPlugin()) as harness:
        result = await harness.open("sid-short")
        assert result.ok is False
        assert result.error_code == "invalid_argument"
        assert "one sampling parameter per stage" in str(result.error_message)
        assert harness.manager.get("sid-short") is None
        assert harness.manager.active_count() == 0

    async with Harness.create(plugin=ListSamplingPlugin()) as harness:
        result = await harness.open("sid-list")
        assert result.ok is False
        assert result.error_code == "invalid_argument"
        assert "as a tuple" in str(result.error_message)


async def test_duplicate_open_is_rejected_with_session_exists() -> None:
    async with Harness.create() as harness:
        assert (await harness.open("sid-dup")).ok is True
        first_session = harness.session("sid-dup")

        result = await harness.open("sid-dup")

        assert result.ok is False
        assert result.error_code == "session_exists"
        assert result.error_retryable is False
        assert harness.session("sid-dup") is first_session
        assert harness.manager.active_count() == 1
        assert len(harness.stage_port.ensure_calls) == 1


async def test_capacity_rejection_is_retryable_and_logged_at_info(caplog: pytest.LogCaptureFixture) -> None:
    manager_logger = session_manager_module.logger
    manager_logger.addHandler(caplog.handler)
    try:
        async with Harness.create(max_sessions=1) as harness:
            assert (await harness.open("sid-first")).ok is True
            caplog.clear()

            with caplog.at_level(logging.INFO, logger=manager_logger.name):
                result = await harness.open("sid-rejected")

            assert result.ok is False
            assert result.error_code == "resource_exhausted"
            assert result.error_retryable is True
            assert harness.manager.get("sid-rejected") is None
            assert harness.manager.active_count() == 1
            rejections = [record for record in caplog.records if "rejected" in record.getMessage()]
            assert rejections and all(record.levelno == logging.INFO for record in rejections)
            assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    finally:
        manager_logger.removeHandler(caplog.handler)


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ValueError("reference audio not found"), "invalid_duplex_runtime_config"),
        (DuplexRuntimeConfigError("bad ref", code="unsupported_ref_audio_path"), "unsupported_ref_audio_path"),
        (DuplexRuntimeConfigError("bad voice", code="invalid_voice"), "invalid_voice"),
    ],
)
async def test_plugin_runtime_config_rejection_rolls_back_the_open(error: Exception, expected_code: str) -> None:
    plugin = FakePlugin()
    plugin.runtime_config_error = error
    async with Harness.create(plugin=plugin) as harness:
        result = await harness.open("sid-runtime")

        assert result.ok is False
        assert result.error_code == expected_code
        assert result.error_message == str(error)
        assert result.capabilities is None
        assert harness.manager.get("sid-runtime") is None
        assert harness.manager.active_count() == 0
        assert harness.stage_port.ensure_calls == []
        assert harness.events() == []

        # Nothing was registered under the rejected id and the slot is free again.
        plugin.runtime_config_error = None
        assert (await harness.open("sid-runtime")).ok is True


# --------------------------------------------------------------------------- #
# Command dispatch                                                            #
# --------------------------------------------------------------------------- #


async def test_command_for_unknown_session_emits_unknown_session_error() -> None:
    async with Harness.create() as harness:
        harness.command("sid-missing", Heartbeat(event_id="evt-hb"))

        events = harness.events("sid-missing")
        assert len(events) == 1
        error = events[0]
        assert isinstance(error, ErrorEvent)
        assert error.code == "unknown_session"
        assert error.related_event_id == "evt-hb"
        assert error.session_id == "sid-missing"
        assert error.epoch is None
        assert error.to_realtime()["error"]["code"] == "unknown_session"


async def test_command_for_closed_session_emits_unknown_session_error() -> None:
    async with Harness.create() as harness:
        await harness.open("sid-closed")
        assert (await harness.close("sid-closed")).ok is True
        harness.events()

        harness.command("sid-closed", Heartbeat(event_id="evt-late"))

        events = harness.events("sid-closed")
        assert [type(event) for event in events] == [ErrorEvent]
        assert events[0].code == "unknown_session"
        assert events[0].related_event_id == "evt-late"


async def test_admitted_command_reaches_the_session_runner_in_order() -> None:
    async with Harness.create() as harness:
        await harness.open("sid-runner")
        harness.events()
        harness.clock.advance(5.0)

        harness.command("sid-runner", Heartbeat(event_id="evt-1"))
        harness.command("sid-runner", Heartbeat(event_id="evt-2"))
        await _settle()

        events = harness.events("sid-runner")
        assert [type(event) for event in events] == [SessionHeartbeatAck, SessionHeartbeatAck]
        assert all(event.epoch == 0 for event in events)
        assert harness.session("sid-runner").lease.last_activity == 5.0


async def test_append_audio_is_backpressured_by_pending_input_bytes() -> None:
    async with Harness.create(max_pending_input_bytes_per_session=8) as harness:
        await harness.open("sid-bytes")
        harness.events()
        session = harness.session("sid-bytes")
        submitted = harness.capture_submissions("sid-bytes")
        assert session.reserve_input_bytes(6, limit=8)

        harness.command("sid-bytes", AppendAudio(audio=b"pcm", event_id="evt-over"))
        harness.command("sid-bytes", AppendAudio(audio=b"pc", event_id="evt-fits"))

        events = harness.events("sid-bytes")
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)
        assert events[0].code == "input_backpressure"
        assert events[0].related_event_id == "evt-over"
        assert events[0].epoch == 0
        assert [command.event_id for command in submitted] == ["evt-fits"]
        # Admission reserved the wire size of the admitted append; the runner
        # released it on dequeue and re-reserved the decoded size (2 bytes).
        assert session.pending_input_bytes == 8


async def test_commit_holds_a_pending_turn_reservation_until_the_runner_dequeues_it() -> None:
    async with Harness.create(max_pending_turns_per_session=1) as harness:
        await harness.open("sid-turns")
        harness.events()
        session = harness.session("sid-turns")
        submitted = harness.capture_submissions("sid-turns")

        harness.command("sid-turns", Commit(event_id="evt-commit-1"))
        harness.command("sid-turns", Commit(event_id="evt-commit-2"))

        assert [command.event_id for command in submitted] == ["evt-commit-1"]
        assert session.pending_input_turns == 1
        events = harness.events("sid-turns")
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)
        assert events[0].code == "input_backpressure"
        assert events[0].related_event_id == "evt-commit-2"

        # The runner releases the reservation when it dequeues the commit.
        session.release_pending_turn()
        harness.command("sid-turns", Commit(event_id="evt-commit-3"))
        assert [command.event_id for command in submitted] == ["evt-commit-1", "evt-commit-3"]
        assert harness.events("sid-turns") == []


def test_accepts_only_duplex_message_types() -> None:
    manager = DuplexSessionManager(
        plugin=FakePlugin(),
        stage_port=FakeStagePort(),
        output_sink=asyncio.Queue(),
        result_sink=asyncio.Queue(),
        runtime_config=_runtime_config(),
        model_config=None,
        clock=FakeClock(),
    )
    try:
        config = DuplexSessionConfig()
        assert manager.accepts(OpenDuplexSessionMessage(control_id="c", session_id="s", session_config=config))
        assert manager.accepts(CloseDuplexSessionMessage(control_id="c", session_id="s"))
        assert manager.accepts(ResumeDuplexSessionMessage(control_id="c", session_id="s", expected_lease_generation=0))
        assert manager.accepts(TouchDuplexSessionMessage(control_id="c", session_id="s", activity="heartbeat"))
        assert manager.accepts(DuplexSessionCommandMessage(session_id="s", command=Heartbeat()))
        assert manager.accepts(type("Lookalike", (), {"type": "open_duplex_session", "session_id": "s"})()) is False
        assert (
            manager.accepts(DuplexControlResultMessage(control_id="c", operation="open", session_id="s", ok=True))
            is False
        )
        assert manager.accepts(Heartbeat()) is False
        with pytest.raises(TypeError, match="Unsupported duplex control message"):
            manager.dispatch(object())
    finally:
        manager.executor.shutdown(wait=False)


# --------------------------------------------------------------------------- #
# Close / resume / touch                                                      #
# --------------------------------------------------------------------------- #


async def test_close_releases_stage_resources_and_is_idempotent() -> None:
    async with Harness.create() as harness:
        await harness.open("sid-close")
        harness.events()
        session = harness.session("sid-close")
        request_id = stage0_request_id("sid-close")
        session.bind_stage_request(0, request_id, fence=session.fence)
        reserved_context = harness.manager.ensure_stage_request(session, stage_id=1)
        assert reserved_context is not None

        result = await harness.close("sid-close", reason="client_close")

        assert result.ok is True
        assert result.operation == "close"
        assert result.lease_generation == 1
        assert result.public_session is not None and result.public_session["state"] == "closed"
        assert session.state == DuplexSessionState.CLOSED
        assert session.lease.terminal_reason == "client_close"
        assert session.resource_request_ids() == []
        assert harness.manager.get("sid-close") is None
        assert harness.manager.active_count() == 0
        assert harness.manager.runner_for_request_id(request_id) is None
        assert harness.stage_port.cleanup_calls == [([request_id], True), ([reserved_context.request_id], False)]
        assert harness.plugin.data_plane.closed_sessions == ["sid-close"]
        closed = [event for event in harness.events("sid-close") if isinstance(event, SessionClosed)]
        assert len(closed) == 1
        assert closed[0].reason == "client_close"
        assert closed[0].is_terminal is True

        again = await harness.close("sid-close")
        assert again.ok is True
        assert again.lease_generation is None
        assert again.public_session is None
        assert harness.stage_port.cleanup_calls[2:] == []
        assert harness.events() == []


async def test_close_retains_the_admission_slot_until_stage_cleanup_succeeds() -> None:
    async with Harness.create(max_sessions=1) as harness:
        await harness.open("sid-close-retry")
        session = harness.session("sid-close-retry")
        session.bind_stage_request(0, stage0_request_id("sid-close-retry"), fence=session.fence)
        harness.stage_port.cleanup_failures_remaining = 1

        result = await harness.close("sid-close-retry")

        # The session is closed from the client's point of view; only the
        # stage cleanup is outstanding and the reaper retries it.
        assert result.ok is True
        assert harness.manager.get("sid-close-retry") is None
        assert session.state == DuplexSessionState.CLOSED
        assert (await harness.open("sid-replacement")).error_code == "resource_exhausted"

        assert await harness.manager.reap_expired() == 1
        assert harness.stage_port.cleanup_calls == [([stage0_request_id("sid-close-retry")], True)] * 2
        assert (await harness.open("sid-replacement")).ok is True


async def test_resume_requires_the_expected_lease_generation() -> None:
    async with Harness.create() as harness:
        await harness.open("sid-resume")
        session = harness.session("sid-resume")
        harness.clock.advance(3.0)
        assert (await harness.touch("sid-resume", DuplexLeaseActivity.DETACH.value)).ok is True
        assert session.lease.detached_at == 3.0

        harness.clock.advance(1.0)
        result = await harness.resume("sid-resume", expected_lease_generation=0)

        assert result.ok is True
        assert result.operation == "resume"
        assert result.lease_generation == 1
        assert result.public_session is not None and result.public_session["id"] == "sid-resume"
        assert session.lease_generation == 1
        assert session.lease.detached_at is None
        assert session.lease.last_activity == 4.0

        stale = await harness.resume("sid-resume", expected_lease_generation=0)
        assert stale.ok is False
        assert stale.error_code == "session_resume_conflict"
        assert stale.lease_generation == 1
        assert session.lease_generation == 1

        unknown = await harness.resume("sid-unknown", expected_lease_generation=0)
        assert unknown.ok is False
        assert unknown.error_code == "unknown_session"
        assert unknown.lease_generation is None


async def test_touch_records_activity_and_detach_marks_the_lease_detached() -> None:
    async with Harness.create() as harness:
        await harness.open("sid-touch")
        session = harness.session("sid-touch")
        assert session.lease.last_activity == 0.0

        harness.clock.advance(2.0)
        result = await harness.touch("sid-touch", DuplexLeaseActivity.HEARTBEAT.value)
        assert result.ok is True
        assert result.operation == "touch"
        assert session.lease.last_activity == 2.0
        assert session.lease.detached_at is None

        harness.clock.advance(1.0)
        assert (await harness.touch("sid-touch", DuplexLeaseActivity.DETACH.value)).ok is True
        assert session.lease.last_activity == 3.0
        assert session.lease.detached_at == 3.0

        harness.clock.advance(1.0)
        assert (await harness.touch("sid-touch", DuplexLeaseActivity.ATTACH.value)).ok is True
        assert session.lease.last_activity == 4.0
        # Only resume clears a detach.
        assert session.lease.detached_at == 3.0

        invalid = await harness.touch("sid-touch", "bogus")
        assert invalid.ok is False
        assert invalid.error_code == "invalid_argument"

        unknown = await harness.touch("sid-unknown", DuplexLeaseActivity.HEARTBEAT.value)
        assert unknown.ok is False
        assert unknown.error_code == "unknown_session"
        assert harness.stage_port.submit_calls == []


# --------------------------------------------------------------------------- #
# Reaper                                                                      #
# --------------------------------------------------------------------------- #


async def test_reaper_expires_idle_sessions_and_emits_session_expired() -> None:
    async with Harness.create(idle_ttl_s=2.0, disconnect_grace_s=1.0) as harness:
        await harness.open("sid-idle")
        await harness.open("sid-active")
        harness.events()
        harness.clock.advance(1.0)
        assert (await harness.touch("sid-active", DuplexLeaseActivity.HEARTBEAT.value)).ok is True
        harness.clock.advance(1.1)

        assert await harness.manager.reap_expired() == 1

        assert harness.manager.get("sid-idle") is None
        assert harness.manager.get("sid-active") is not None
        assert harness.manager.active_count() == 1
        expired = harness.events("sid-idle")
        assert len(expired) == 1
        assert isinstance(expired[0], SessionExpired)
        assert expired[0].reason == "idle_ttl_expired"
        assert expired[0].to_realtime() == {
            "type": "session.expired",
            "event_id": expired[0].event_id,
            "session_id": "sid-idle",
            "reason": "idle_ttl_expired",
        }
        assert harness.events("sid-active") == []
        assert harness.stage_port.cleanup_calls == [([stage0_request_id("sid-idle")], False)]
        assert harness.result_sink.empty()

        assert await harness.manager.reap_expired() == 0


async def test_reaper_expires_detached_sessions_after_the_disconnect_grace() -> None:
    async with Harness.create(idle_ttl_s=300.0, disconnect_grace_s=1.0) as harness:
        await harness.open("sid-detached")
        harness.events()
        assert (await harness.touch("sid-detached", DuplexLeaseActivity.DETACH.value)).ok is True
        harness.clock.advance(0.5)
        assert await harness.manager.reap_expired() == 0
        harness.clock.advance(0.5)

        assert await harness.manager.reap_expired() == 1

        assert harness.manager.get("sid-detached") is None
        expired = harness.events("sid-detached")
        assert [type(event) for event in expired] == [SessionExpired]
        assert expired[0].reason == "disconnect_grace_expired"
        assert (await harness.resume("sid-detached", expected_lease_generation=0)).error_code == "unknown_session"


async def test_session_closed_is_emitted_only_after_stage_cleanup_so_a_reopen_is_admitted() -> None:
    async with Harness.create(max_sessions=1) as harness:
        await harness.open("sid-closing")
        harness.events()
        session = harness.session("sid-closing")
        session.bind_stage_request(0, stage0_request_id("sid-closing"), fence=session.fence)
        harness.stage_port.cleanup_gate = asyncio.Event()

        close_task = asyncio.create_task(harness.close("sid-closing"))
        await asyncio.wait_for(harness.stage_port.cleanup_started.wait(), timeout=1.0)
        # The runner is gone but the stage request is still being aborted:
        # no session.closed yet, so a client cannot race the admission slot.
        assert harness.events("sid-closing") == []

        harness.stage_port.cleanup_gate.set()
        assert (await close_task).ok is True
        events = harness.events("sid-closing")
        assert [type(event) for event in events] == [SessionClosed]
        assert events[0].reason == "client_close"
        assert (await harness.open("sid-replacement")).ok is True


async def test_wire_close_command_frees_the_slot_before_session_closed() -> None:
    async with Harness.create(max_sessions=1) as harness:
        await harness.open("sid-wire-close")
        harness.events()
        session = harness.session("sid-wire-close")
        request_id = stage0_request_id("sid-wire-close")
        session.bind_stage_request(0, request_id, fence=session.fence)
        harness.stage_port.cleanup_gate = asyncio.Event()

        harness.command("sid-wire-close", CloseSession(reason="client_close"))
        await asyncio.wait_for(harness.stage_port.cleanup_started.wait(), timeout=1.0)
        assert harness.manager.get("sid-wire-close") is None
        assert harness.events("sid-wire-close") == []

        harness.stage_port.cleanup_gate.set()
        for _ in range(50):
            if harness.manager._admission_count() == 0:
                break
            await asyncio.sleep(0.01)
        assert harness.stage_port.cleanup_calls == [([request_id], True)]
        events = harness.events("sid-wire-close")
        assert [type(event) for event in events] == [SessionClosed]
        assert events[0].reason == "client_close"
        assert (await harness.open("sid-replacement")).ok is True


async def test_wire_close_stops_the_worker_and_refuses_later_control_ops() -> None:
    async with Harness.create(max_sessions=1) as harness:
        await harness.open("sid-wire")
        harness.events()
        runner = harness.manager.runners["sid-wire"]
        worker = runner._worker
        assert worker is not None
        harness.stage_port.cleanup_gate = asyncio.Event()

        harness.command("sid-wire", CloseSession(reason="client_close"))
        await asyncio.wait_for(harness.stage_port.cleanup_started.wait(), timeout=1.0)
        # Control operations issued while the close is in flight are refused,
        # not silently accepted on a half-torn session.
        resumed = await harness.resume("sid-wire", expected_lease_generation=0)
        assert resumed.ok is False
        assert resumed.error_code in {"session_closed", "unknown_session"}
        harness.command("sid-wire", Heartbeat())

        harness.stage_port.cleanup_gate.set()
        for _ in range(50):
            if worker.done() and harness.manager._admission_count() == 0:
                break
            await asyncio.sleep(0.01)
        assert worker.done(), "the mailbox worker must exit after a wire close"
        events = harness.events("sid-wire")
        assert [type(event) for event in events] == [ErrorEvent, SessionClosed] or [
            type(event) for event in events
        ] == [SessionClosed, ErrorEvent]
        assert (await harness.open("sid-after")).ok is True


async def test_concurrent_opens_cannot_exceed_max_sessions() -> None:
    async with Harness.create(max_sessions=1) as harness:
        harness.plugin.blocked_session_ids.add("first")
        first = asyncio.create_task(
            harness.open("first", DuplexSessionConfig(model="fake-model", instructions="first"))
        )
        await asyncio.wait_for(harness.plugin.runtime_config_started.wait(), timeout=1.0)

        # The first open holds the only slot while it awaits the plugin.
        second = await harness.open("second", DuplexSessionConfig(model="fake-model", instructions="second"))
        assert second.ok is False
        assert second.error_code == "resource_exhausted"

        harness.plugin.runtime_config_gate.set()
        assert (await first).ok is True
        assert harness.manager.active_count() == 1


async def test_runtime_close_without_stage_requests_frees_the_slot() -> None:
    async with Harness.create(max_sessions=1) as harness:
        await harness.open("sid-runtime")
        harness.events()
        runner = harness.manager.runners["sid-runtime"]
        runner.session.release_all_requests()

        await runner._close_from_runtime("model_failed")
        for _ in range(50):
            if harness.manager._admission_count() == 0:
                break
            await asyncio.sleep(0.01)

        assert harness.manager.get("sid-runtime") is None
        assert [type(event) for event in harness.events("sid-runtime")] == [SessionClosed]
        assert (await harness.open("sid-next")).ok is True


async def test_close_of_a_forgotten_session_is_idempotent() -> None:
    async with Harness.create(max_sessions=1) as harness:
        result = await harness.close("never-opened")

        assert result.ok is True


async def test_append_bytes_are_reserved_at_admission_until_the_runner_dequeues() -> None:
    async with Harness.create(max_sessions=1, max_pending_input_bytes_per_session=8) as harness:
        await harness.open("sid-bytes")
        harness.events()
        session = harness.session("sid-bytes")
        runner = harness.manager.runners["sid-bytes"]
        # Park the worker so admitted commands stay in the mailbox.
        gate = asyncio.Event()
        runner._mailbox.put_nowait(_Internal("wait", {"gate": gate}))
        runner._on_internal = lambda item: gate.wait()  # type: ignore[method-assign]
        await asyncio.sleep(0)

        harness.command("sid-bytes", AppendAudio(audio=b"12345"))
        assert session.pending_input_bytes == 5
        harness.command("sid-bytes", AppendAudio(audio=b"12345"))
        errors = [event for event in harness.events("sid-bytes") if isinstance(event, ErrorEvent)]
        assert [error.code for error in errors] == ["input_backpressure"]
        gate.set()


async def test_expired_session_retains_the_admission_slot_until_cleanup_succeeds() -> None:
    async with Harness.create(max_sessions=1, idle_ttl_s=1.0) as harness:
        await harness.open("sid-expired")
        harness.events()
        session = harness.session("sid-expired")
        request_id = stage0_request_id("sid-expired")
        session.bind_stage_request(0, request_id, fence=session.fence)
        harness.stage_port.cleanup_failures_remaining = 1
        harness.clock.advance(2.0)

        assert await harness.manager.reap_expired() == 0

        assert harness.manager.get("sid-expired") is None
        assert session.state == DuplexSessionState.CLOSED
        assert [type(event) for event in harness.events("sid-expired")] == [SessionExpired]
        assert (await harness.open("sid-replacement")).error_code == "resource_exhausted"

        assert await harness.manager.reap_expired() == 1

        assert harness.stage_port.cleanup_calls == [([request_id], True)] * 2
        assert session.resource_request_ids() == []
        assert harness.events("sid-expired") == []
        assert (await harness.open("sid-replacement")).ok is True


async def test_expired_session_holds_the_admission_slot_while_cleanup_is_blocked() -> None:
    async with Harness.create(max_sessions=1, idle_ttl_s=1.0) as harness:
        await harness.open("sid-blocked")
        session = harness.session("sid-blocked")
        session.bind_stage_request(0, stage0_request_id("sid-blocked"), fence=session.fence)
        harness.stage_port.cleanup_gate = asyncio.Event()
        harness.clock.advance(2.0)

        reap_task = asyncio.create_task(harness.manager.reap_expired())
        await asyncio.wait_for(harness.stage_port.cleanup_started.wait(), timeout=1.0)
        assert (await harness.open("sid-replacement")).error_code == "resource_exhausted"

        harness.stage_port.cleanup_gate.set()
        assert await reap_task == 1
        assert (await harness.open("sid-replacement")).ok is True


async def test_expired_cleanup_failure_does_not_block_other_sessions() -> None:
    class OneStuckStagePort(FakeStagePort):
        async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
            await super().cleanup(request_ids, abort=abort)
            if request_ids == [stage0_request_id("sid-stuck")]:
                raise RuntimeError("stuck expiry cleanup")

    async with Harness.create(stage_port=OneStuckStagePort(), idle_ttl_s=1.0) as harness:
        await harness.open("sid-stuck")
        await harness.open("sid-fine")
        harness.events()
        harness.clock.advance(2.0)

        assert await harness.manager.reap_expired() == 1

        assert harness.manager.active_count() == 0
        assert {type(event) for event in harness.events()} == {SessionExpired}
        assert (await harness.open("sid-fine")).ok is True
        assert (await harness.open("sid-stuck")).error_code == "session_exists"


# --------------------------------------------------------------------------- #
# Request-triggered cleanup (orchestrator error paths)                        #
# --------------------------------------------------------------------------- #


async def test_close_sessions_for_request_ids_expires_the_owner_and_retries_cleanup() -> None:
    async with Harness.create(max_sessions=1) as harness:
        await harness.open("sid-request")
        harness.events()
        session = harness.session("sid-request")
        request_id = stage0_request_id("sid-request")
        session.bind_stage_request(0, request_id, fence=session.fence)
        harness.stage_port.cleanup_failures_remaining = 1

        closed = harness.manager.close_sessions_for_request_ids([request_id, "req-unrelated"], abort=True)

        assert closed == {"sid-request": [request_id]}
        assert harness.manager.get("sid-request") is None
        assert session.lease.terminal_reason == "request_cleanup"
        await _settle()
        expired = harness.events("sid-request")
        assert [type(event) for event in expired] == [SessionExpired]
        assert expired[0].reason == "request_cleanup"
        assert session.state == DuplexSessionState.CLOSED
        # The stage resources are still owned by the pending cleanup.
        assert session.resource_request_ids() == [request_id]
        assert (await harness.open("sid-replacement")).error_code == "resource_exhausted"

        assert await harness.manager.reap_expired() == 0
        assert harness.stage_port.cleanup_calls == [([request_id], True)]
        assert (await harness.open("sid-replacement")).error_code == "resource_exhausted"

        assert await harness.manager.reap_expired() == 1
        assert harness.stage_port.cleanup_calls == [([request_id], True)] * 2
        assert session.resource_request_ids() == []
        assert harness.manager.runner_for_request_id(request_id) is None
        assert (await harness.open("sid-replacement")).ok is True
        assert harness.manager.close_sessions_for_request_ids([request_id], abort=True) == {}


async def test_in_progress_request_cleanup_is_finalized_by_the_orchestrator_not_the_reaper() -> None:
    async with Harness.create(max_sessions=1) as harness:
        await harness.open("sid-in-progress")
        session = harness.session("sid-in-progress")
        request_id = stage0_request_id("sid-in-progress")
        session.bind_stage_request(0, request_id, fence=session.fence)

        closed = harness.manager.close_sessions_for_request_ids([request_id], abort=True, cleanup_in_progress=True)
        assert closed == {"sid-in-progress": [request_id]}

        # The orchestrator owns this cleanup: the reaper neither retries nor releases it.
        assert await harness.manager.reap_expired() == 0
        assert harness.stage_port.cleanup_calls == []
        assert (await harness.open("sid-replacement")).error_code == "resource_exhausted"

        harness.manager.finalize_closed_sessions(["sid-in-progress"])

        assert session.resource_request_ids() == []
        assert session.state == DuplexSessionState.CLOSED
        assert harness.manager.runner_for_request_id(request_id) is None
        assert await harness.manager.reap_expired() == 0
        assert harness.stage_port.cleanup_calls == []
        assert (await harness.open("sid-replacement")).ok is True


async def test_deferred_request_cleanup_is_retried_by_the_reaper() -> None:
    async with Harness.create(max_sessions=1) as harness:
        await harness.open("sid-deferred")
        session = harness.session("sid-deferred")
        request_id = stage0_request_id("sid-deferred")
        session.bind_stage_request(0, request_id, fence=session.fence)
        harness.manager.close_sessions_for_request_ids([request_id], abort=True, cleanup_in_progress=True)
        assert await harness.manager.reap_expired() == 0
        assert harness.stage_port.cleanup_calls == []

        # The orchestrator's own cleanup failed: hand the retry to the reaper.
        harness.manager.defer_request_cleanups(["sid-deferred"])

        assert await harness.manager.reap_expired() == 1
        assert harness.stage_port.cleanup_calls == [([request_id], True)]
        assert session.resource_request_ids() == []
        assert (await harness.open("sid-replacement")).ok is True


async def test_request_cleanup_closes_the_owner_of_any_of_its_stage_requests() -> None:
    async with Harness.create() as harness:
        await harness.open("sid-merge")
        session = harness.session("sid-merge")
        stage0 = stage0_request_id("sid-merge")
        session.bind_stage_request(0, stage0, fence=session.fence)
        stage1 = harness.manager.ensure_stage_request(session, stage_id=1)
        assert stage1 is not None

        closed = harness.manager.close_sessions_for_request_ids([stage1.request_id])

        assert closed == {"sid-merge": [stage0, stage1.request_id]}
        assert harness.manager.close_sessions_for_request_ids([stage0]) == {}
        assert await harness.manager.reap_expired() == 1
        assert harness.stage_port.cleanup_calls == [([stage0, stage1.request_id], False)]


# --------------------------------------------------------------------------- #
# Dispatch ordering / shutdown                                                #
# --------------------------------------------------------------------------- #


async def test_control_dispatch_is_ordered_per_session_without_blocking_other_sessions() -> None:
    plugin = FakePlugin()
    plugin.blocked_session_ids.add("sid-blocked")
    async with Harness.create(plugin=plugin) as harness:
        manager = harness.manager
        blocked_config = DuplexSessionConfig(model="fake-model", instructions="sid-blocked")

        manager.dispatch(
            OpenDuplexSessionMessage(control_id="blocked-open", session_id="sid-blocked", session_config=blocked_config)
        )
        await asyncio.wait_for(plugin.runtime_config_started.wait(), timeout=1.0)
        manager.dispatch(
            TouchDuplexSessionMessage(
                control_id="blocked-touch",
                session_id="sid-blocked",
                activity=DuplexLeaseActivity.HEARTBEAT.value,
            )
        )
        manager.dispatch(
            OpenDuplexSessionMessage(
                control_id="independent-open",
                session_id="sid-independent",
                session_config=DuplexSessionConfig(model="fake-model"),
            )
        )
        manager.dispatch(
            TouchDuplexSessionMessage(
                control_id="independent-touch",
                session_id="sid-independent",
                activity=DuplexLeaseActivity.HEARTBEAT.value,
            )
        )

        assert [(await harness.result()).control_id for _ in range(2)] == ["independent-open", "independent-touch"]
        assert harness.result_sink.empty()
        assert manager.get("sid-blocked") is None

        plugin.runtime_config_gate.set()
        results = [await harness.result() for _ in range(2)]
        assert [result.control_id for result in results] == ["blocked-open", "blocked-touch"]
        assert all(result.ok for result in results)
        assert manager.active_count() == 2


async def test_shutdown_closes_every_runner_and_stops_dispatch_tasks() -> None:
    harness = Harness.create()
    await harness.open("sid-a")
    await harness.open("sid-b")
    sessions = [harness.session("sid-a"), harness.session("sid-b")]
    harness.events()

    await harness.manager.shutdown()

    assert harness.manager.active_count() == 0
    assert harness.manager.runners == {}
    assert all(session.state == DuplexSessionState.CLOSED for session in sessions)
    assert harness.manager.runner_for_request_id(stage0_request_id("sid-b")) is None
    # Shutdown is silent: no session.closed / session.expired is emitted.
    assert harness.events() == []
    assert harness.stage_port.cleanup_calls == []


def test_manager_rejects_a_plugin_whose_sampling_policy_mismatches_the_stages() -> None:
    class WrongStageCountPlugin(FakePlugin):
        def configure_sampling_params(self, *, runtime_config, defaults):
            del runtime_config
            return defaults[:1]

    with pytest.raises(ValueError, match="one sampling parameter per stage"):
        DuplexSessionManager(
            plugin=WrongStageCountPlugin(),
            stage_port=FakeStagePort(),
            output_sink=asyncio.Queue(),
            result_sink=asyncio.Queue(),
            runtime_config=_runtime_config(),
            model_config=None,
        )


# --------------------------------------------------------------------------- #
# Admission bookkeeping                                                       #
# --------------------------------------------------------------------------- #


async def test_duplicate_open_in_flight_does_not_release_the_first_opens_slot() -> None:
    """A duplicate id arriving mid-flight must not free the slot the first open holds.

    The rejection runs the same ``finally`` as a successful open, so without a
    guard it discards an ``_admitting`` entry it never added, and the next open
    of a *different* id sails past ``max_sessions``. Server-allocated ids make
    this a guard rather than a live path, which is exactly why a sequential
    duplicate cannot exercise it.
    """
    async with Harness.create(max_sessions=1) as harness:
        harness.plugin.blocked_session_ids.add("dup")
        first = asyncio.create_task(harness.open("dup", DuplexSessionConfig(model="fake-model", instructions="dup")))
        await asyncio.wait_for(harness.plugin.runtime_config_started.wait(), timeout=1.0)

        duplicate = await harness.open("dup", DuplexSessionConfig(model="fake-model", instructions="dup"))
        assert duplicate.ok is False
        assert duplicate.error_code == "session_exists"

        # The in-flight open still holds the only slot.
        overflow = await harness.open("other", DuplexSessionConfig(model="fake-model", instructions="other"))
        assert overflow.ok is False
        assert overflow.error_code == "resource_exhausted"

        harness.plugin.runtime_config_gate.set()
        assert (await first).ok is True
        assert harness.manager.active_count() == 1


async def test_expiry_emits_a_terminal_event_even_when_a_close_deferred_it() -> None:
    """Every session ends with exactly one terminal event, whoever tears it down.

    A wire ``session.close`` defers its ``session.closed`` to the manager. If a
    stage failure then takes the runner away from the manager (the orchestrator
    cleanup path), nobody would emit the deferred event unless ``expire``
    upgrades it.
    """
    async with Harness.create() as harness:
        await harness.open("sid-deferred")
        runner = harness.manager.runners["sid-deferred"]
        # What the runner does for a wire close: tear down, defer the terminal.
        await runner.close("client_close", emit_closed=False)
        harness.events("sid-deferred")

        await runner.expire("request_cleanup", emit_expired=True)

        terminal = [
            event for event in harness.events("sid-deferred") if isinstance(event, SessionClosed | SessionExpired)
        ]
        assert len(terminal) == 1, terminal
        assert isinstance(terminal[0], SessionExpired)
        assert terminal[0].reason == "request_cleanup"


async def test_a_terminal_event_is_emitted_once_even_if_expiry_runs_twice() -> None:
    async with Harness.create() as harness:
        await harness.open("sid-once")
        runner = harness.manager.runners["sid-once"]

        await runner.expire("idle_ttl_expired", emit_expired=True)
        await runner.expire("idle_ttl_expired", emit_expired=True)

        terminal = [event for event in harness.events("sid-once") if isinstance(event, SessionClosed | SessionExpired)]
        assert len(terminal) == 1, terminal


# --------------------------------------------------------------------------- #
# Resume and takeover lifecycle                                               #
# --------------------------------------------------------------------------- #


async def test_resume_inside_the_grace_window_stops_the_reaper_expiring_the_session() -> None:
    """Resuming clears the disconnect deadline, not just the lease generation.

    The reaper decides expiry from ``lease.detached_at``. A resume that bumped
    the generation without clearing it would let the reaper kill a session a
    client had already come back to.
    """
    async with Harness.create(disconnect_grace_s=30.0) as harness:
        await harness.open("sid-grace")
        session = harness.session("sid-grace")
        assert (await harness.touch("sid-grace", DuplexLeaseActivity.DETACH.value)).ok is True
        harness.clock.advance(10.0)

        assert (await harness.resume("sid-grace", expected_lease_generation=0)).ok is True
        assert session.lease.detached_at is None

        # Past the original deadline: the session survives because it is attached.
        harness.clock.advance(25.0)
        assert await harness.manager.reap_expired() == 0
        assert harness.manager.get("sid-grace") is not None


async def test_resume_after_the_grace_window_reports_unknown_session() -> None:
    """Once the reaper has expired a detached session, resume has nothing to attach to."""
    async with Harness.create(disconnect_grace_s=5.0) as harness:
        await harness.open("sid-late")
        assert (await harness.touch("sid-late", DuplexLeaseActivity.DETACH.value)).ok is True
        harness.clock.advance(6.0)
        assert await harness.manager.reap_expired() == 1

        late = await harness.resume("sid-late", expected_lease_generation=0)

        assert late.ok is False
        assert late.error_code == "unknown_session"


async def test_takeover_invalidates_the_generation_the_replaced_client_held() -> None:
    """Two clients cannot both hold the session: the first one's generation goes stale.

    This is the engine half of a websocket takeover — the transport picks the
    winning socket, and the lease generation is what stops the loser from
    resuming behind it.
    """
    async with Harness.create() as harness:
        await harness.open("sid-takeover")
        session = harness.session("sid-takeover")
        assert (await harness.touch("sid-takeover", DuplexLeaseActivity.DETACH.value)).ok is True

        winner = await harness.resume("sid-takeover", expected_lease_generation=0)
        assert winner.ok is True
        assert winner.lease_generation == 1

        # The replaced client still believes it holds generation 0.
        loser = await harness.resume("sid-takeover", expected_lease_generation=0)
        assert loser.ok is False
        assert loser.error_code == "session_resume_conflict"
        assert session.lease_generation == 1

        # The winner can still act on the session it took over.
        assert (await harness.touch("sid-takeover", DuplexLeaseActivity.HEARTBEAT.value)).ok is True


async def test_resume_is_refused_once_a_close_has_begun() -> None:
    async with Harness.create() as harness:
        await harness.open("sid-closing")
        await harness.close("sid-closing")

        resumed = await harness.resume("sid-closing", expected_lease_generation=0)

        assert resumed.ok is False
        assert resumed.error_code == "unknown_session"


async def test_wire_close_of_an_idle_session_still_emits_session_closed() -> None:
    """The protocol smoke shape: open, send nothing, close.

    A session that never bound a stage request has nothing for the reaper to
    clean up, so the terminal event is the only thing the client ever gets
    back for its ``session.close``. Regression guard for a close path that
    only emitted after a cleanup it never performed.
    """
    async with Harness.create(max_sessions=1) as harness:
        await harness.open("sid-idle-close")
        harness.events()

        harness.command("sid-idle-close", CloseSession(reason="client_close"))
        events: list[DuplexEvent] = []
        for _ in range(200):
            events.extend(harness.events("sid-idle-close"))
            if events:
                break
            await asyncio.sleep(0.01)

        assert [type(event) for event in events] == [SessionClosed]
        assert events[0].reason == "client_close"
        assert harness.manager.get("sid-idle-close") is None
        assert (await harness.open("sid-idle-replacement")).ok is True


# --------------------------------------------------------------------------- #
# Reaper loop                                                                 #
# --------------------------------------------------------------------------- #


async def test_reaper_loop_waits_between_ticks() -> None:
    """The loop paces itself; it does not spin on the expiry check.

    Ported from the pre-framework ``Orchestrator._duplex_reaper_loop`` tests:
    the loop moved to the session manager with the sessions it reaps.
    """
    manager = object.__new__(DuplexSessionManager)
    calls = 0

    async def reap_expired(now: float | None = None) -> int:
        nonlocal calls
        calls += 1
        return 0

    manager.reap_expired = reap_expired  # type: ignore[method-assign]
    manager.runtime_config = DuplexSessionRuntimeConfig(reaper_interval_s=0.01)
    shutdown = asyncio.Event()

    task = asyncio.create_task(manager.reaper_loop(shutdown))
    await asyncio.sleep(0.035)
    shutdown.set()
    await asyncio.wait_for(task, timeout=5.0)

    assert 2 <= calls <= 5


@pytest.mark.parametrize("first_cleanup_delay", [0.0, 0.05], ids=["immediate", "delayed"])
async def test_reaper_loop_survives_one_cleanup_failure(first_cleanup_delay: float) -> None:
    """A failed expiry sweep is retried on the next tick, not fatal to the loop."""
    recovered = asyncio.Event()
    calls = 0

    manager = object.__new__(DuplexSessionManager)

    async def reap_expired(now: float | None = None) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(first_cleanup_delay)
            raise RuntimeError("transient cleanup failure")
        recovered.set()
        return 0

    manager.reap_expired = reap_expired  # type: ignore[method-assign]
    manager.runtime_config = DuplexSessionRuntimeConfig(reaper_interval_s=0.01)
    shutdown = asyncio.Event()

    task = asyncio.create_task(manager.reaper_loop(shutdown))
    try:
        await asyncio.wait_for(recovered.wait(), timeout=5.0)
        assert calls >= 2
    finally:
        shutdown.set()
        await asyncio.wait_for(task, timeout=5.0)
