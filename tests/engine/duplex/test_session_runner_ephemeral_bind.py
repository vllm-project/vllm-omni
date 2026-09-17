# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ephemeral (non-resumable) Stage0 binding for turn-commit models.

A model whose Stage0 cannot resume appends on a finished request gets one
ordinary, turn-scoped request per committed turn (``...r.stage0_t{N}``)
instead of the single resident ``...r.stage0`` id. These tests pin the id
shape, the capability branching, the two-turn submit flow through the real
session runner with a recording fake stage port, and the observe hook that
lets an intermediate stage reach the client without stopping the pipeline.
Silence continuation is refused on the non-resumable path, a finished
observed intermediate stage does not close the stream, a listen-only
append keeps the turn-scoped Stage0 id bound, and a later commit re-binds
a stale submitted id without aborting leftover downstream bindings.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.commands import DuplexCommand
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexOutputContext,
    DuplexOutputDecision,
    DuplexRequestIdentity,
    DuplexStagePort,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
    duplex_ephemeral_stage_request_id,
    duplex_resource_request_id,
    is_stable_stage0_placeholder,
)
from vllm_omni.engine.duplex.events import DuplexEvent
from vllm_omni.engine.duplex.messages import (
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    DuplexSessionEventMessage,
    OpenDuplexSessionMessage,
)
from vllm_omni.engine.duplex.plugin import (
    DuplexDataPlane,
    DuplexModelPlugin,
    DuplexModelSessionState,
    PcmAppendBuffer,
    PcmAppendReservation,
)
from vllm_omni.engine.duplex.session.helpers import stage0_request_id
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION_ID = "duplex-ephemeral-test"


# --------------------------------------------------------------------------- #
# L1: id shape and capability branching                                       #
# --------------------------------------------------------------------------- #


def test_stage_request_id_ephemeral_when_not_resumable() -> None:
    fence = DuplexFence(SESSION_ID, epoch=0, turn_id=3)
    request_id = DuplexSessionManager.stage_request_id(fence, stage_id=0, resumable=False)
    assert request_id == duplex_ephemeral_stage_request_id(fence, stage_id=0)
    assert request_id.endswith(".r.stage0_t3")
    assert not is_stable_stage0_placeholder(request_id, session_id=SESSION_ID, epoch=0)


def test_stage_request_id_resumable_keeps_stable_role() -> None:
    fence = DuplexFence(SESSION_ID, epoch=1, turn_id=3)
    request_id = DuplexSessionManager.stage_request_id(fence, stage_id=0, resumable=True)
    assert request_id == duplex_resource_request_id(fence, "stage0")
    assert is_stable_stage0_placeholder(request_id, session_id=SESSION_ID, epoch=1)


def test_is_stable_stage0_placeholder_rejects_ephemeral_ids() -> None:
    fence = DuplexFence(SESSION_ID, epoch=0, turn_id=3)
    assert is_stable_stage0_placeholder(duplex_resource_request_id(fence, "stage0"), session_id=SESSION_ID, epoch=0)
    assert not is_stable_stage0_placeholder(
        duplex_ephemeral_stage_request_id(fence, stage_id=0),
        session_id=SESSION_ID,
        epoch=0,
    )


def test_stage_submission_resumable_defaults_to_true() -> None:
    context = DuplexStageRequestContext(
        request_id="req",
        session_id=SESSION_ID,
        fence=DuplexFence(SESSION_ID),
        stage_id=0,
        final_stage_id=1,
        config_generation=0,
        sampling_params=(SamplingParams(max_tokens=8),),
    )
    submission = DuplexStageSubmission(context=context, prompt={"prompt_token_ids": [1]}, already_submitted=False)
    assert submission.resumable is True


def test_helpers_stage0_request_id_branches_on_capability() -> None:
    ephemeral_session = SimpleNamespace(
        session_id=SESSION_ID,
        turn_id=2,
        capabilities=DuplexCapabilities(supports_core_resumable_request=False),
    )
    ephemeral_id = stage0_request_id(ephemeral_session, 0)
    assert ephemeral_id.endswith(".r.stage0_t2")
    assert not is_stable_stage0_placeholder(ephemeral_id, session_id=SESSION_ID, epoch=0)
    resident_session = SimpleNamespace(
        session_id=SESSION_ID,
        turn_id=2,
        capabilities=DuplexCapabilities(supports_core_resumable_request=True),
    )
    resident_id = stage0_request_id(resident_session, 0)
    assert is_stable_stage0_placeholder(resident_id, session_id=SESSION_ID, epoch=0)


def test_observe_stage_output_defaults_off() -> None:
    plugin = EphemeralFakePlugin()
    assert plugin.observe_stage_output(stage_id=0, output=object(), context=object()) is False


# --------------------------------------------------------------------------- #
# Fakes: an ephemeral turn-commit plugin                                      #
# --------------------------------------------------------------------------- #


class _Reservation(PcmAppendReservation):
    def __init__(self, *, operation_id: str, payload: dict[str, object] | None, byte_count: int) -> None:
        self.operation_id = operation_id
        self.payload = payload
        self._byte_count = byte_count
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def byte_count(self) -> int:
        return self._byte_count

    def commit(self) -> None:
        self._active = False

    def rollback(self) -> None:
        self._active = False


class _CommitOnlyPcmBuffer(PcmAppendBuffer):
    """Accumulates PCM and emits the whole utterance only on commit."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    @property
    def pending_byte_count(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def clear_force_listen(self) -> None:
        return

    def has_pending(self) -> bool:
        return bool(self._buffer)

    def has_reserved(self) -> bool:
        return False

    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        allow_emit: bool,
    ) -> _Reservation | None:
        del operation_id, chunk_period_ms, allow_emit
        audio = payload.get("audio")
        if isinstance(audio, str):
            try:
                self._buffer.extend(base64.b64decode(audio, validate=True))
            except (binascii.Error, ValueError):
                pass
        return None  # commit-only: never emit per chunk

    def prepare_commit(self, *, operation_id: str, chunk_period_ms: int) -> _Reservation:
        del chunk_period_ms
        raw = bytes(self._buffer)
        self._buffer.clear()
        payload: dict[str, object] | None = None
        if raw:
            payload = {
                "type": "audio",
                "audio": base64.b64encode(raw).decode("ascii"),
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
                "final": True,
                "is_speech": True,
            }
        return _Reservation(operation_id=operation_id, payload=payload, byte_count=len(raw))

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        reservation = self.prepare_commit(operation_id="flush", chunk_period_ms=chunk_period_ms)
        reservation.commit()
        return reservation.payload


@dataclass(slots=True)
class _FakeSessionState(DuplexModelSessionState):
    audio_buffer: _CommitOnlyPcmBuffer = field(default_factory=_CommitOnlyPcmBuffer)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    context_locked: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes += max(0, int(reserved_bytes))

    def clear_committed_audio(self) -> int:
        reserved_bytes = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        return reserved_bytes

    def clear_continuation(self) -> None:
        self.continuation_owner_id = None
        self.continuation_units = 0
        self.pending_silence_task = None
        self.pending_silence_owner_id = None


class _FakeDataPlane(DuplexDataPlane):
    """Projects every delivered output as one audio event for its request."""

    def __init__(self) -> None:
        self._terminal: set[str] = set()
        self.closed_streams: list[str] = []

    def begin_request(self, request_id: str) -> None:
        self._terminal.discard(request_id)

    def is_terminal(self, request_id: str | None) -> bool:
        return request_id in self._terminal if request_id is not None else False

    def mark_terminal(self, request_id: str) -> None:
        self._terminal.add(request_id)

    def close_stream(self, request_id: str) -> None:
        self.closed_streams.append(request_id)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        self._terminal.clear()

    def project(self, result: object, *, context: object | None = None) -> list[dict[str, object]]:
        del context
        if not isinstance(result, dict):
            return []
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list):
            return []
        events: list[dict[str, object]] = []
        for output in outputs:
            request_id = getattr(output, "request_id", None)
            if not isinstance(request_id, str) or not request_id:
                continue
            mm = getattr(output, "multimodal_output", None)
            model_turn_id = mm.get("model_turn_id") if isinstance(mm, Mapping) else None
            if model_turn_id is None:
                model_turn_id = getattr(output, "duplex_turn_id", None)
            # Stage ``finished`` is not the duplex turn ending; only an explicit
            # ``end_of_turn`` (final-stage / model turn_eos) closes the response.
            end_of_turn = bool(mm.get("end_of_turn")) if isinstance(mm, Mapping) else False
            events.append(
                {
                    "stage_role": "tts",
                    "is_listen": False,
                    "data_plane_request_id": request_id,
                    "audio": "wav-fake",
                    "sample_rate_hz": 24000,
                    "model_turn_id": model_turn_id,
                    "end_of_turn": end_of_turn,
                }
            )
        return events


class EphemeralFakePlugin(DuplexModelPlugin):
    """Turn-commit-only fake: one ordinary Stage0 request per committed turn."""

    plugin_id = "fake-ephemeral"

    def __init__(self, *, observe_stage0: bool = False) -> None:
        super().__init__(lambda audio, sample_rate_hz, fmt, speed: None)
        self.data_plane = _FakeDataPlane()
        self._observe_stage0 = observe_stage0
        self.planned_payloads: list[dict[str, object]] = []

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        return tuple(defaults)

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, object],
        runtime_config: dict[str, object],
        seq: int,
        turn_seq: int,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del session_config, runtime_config, seq, turn_seq, sampling_params
        if not isinstance(payload, Mapping) or not final:
            raise AssertionError("ephemeral fake only plans final commit payloads")
        self.planned_payloads.append({"request_id": request_id, "turn_id": fence.turn_id, "payload": payload})
        return DuplexAppendPlan(
            prompt={
                "prompt_token_ids": [1],
                "additional_information": {"session_id": fence.session_id, "turn_id": fence.turn_id},
            }
        )

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, object],
        output: object,
    ) -> DuplexOutputDecision | None:
        return None

    def observe_stage_output(self, *, stage_id: int, output: object, context: object) -> bool:
        del output, context
        return self._observe_stage0 and stage_id == 0

    def create_session_state(self) -> _FakeSessionState:
        return _FakeSessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return DuplexCapabilities(
            supports_model_native_turn_policy=False,
            supports_barge_in=False,
            supports_input_append=True,
            supports_turn_commit_only=True,
            supports_core_resumable_request=False,
            supports_realtime_endpoint=True,
            supports_chat_completions=True,
            adapter_patterns=["turn_commit"],
            chunk_period_ms=1000,
        )

    def validate_client_extra_body(self, extra_body: object) -> None:
        return

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: object | None
    ) -> dict[str, object]:
        del config, model_config
        return {}

    def runtime_config_for_update(
        self, config: DuplexSessionConfig, current: Mapping[str, object]
    ) -> dict[str, object]:
        del config
        return dict(current)

    def data_plane_context(
        self,
        *,
        epoch: int,
        turn_id: int,
        active_response_turn_id: int | None,
        active_response_id: str | None,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> object:
        return SimpleNamespace(
            epoch=epoch,
            turn_id=turn_id,
            active_response_turn_id=active_response_turn_id,
            active_response_id=active_response_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )


# --------------------------------------------------------------------------- #
# L2 harness: real session runner + recording fake stage port                 #
# --------------------------------------------------------------------------- #


class RecordingStagePort(DuplexStagePort):
    """Records what the runner asks of the orchestrator; never talks to a stage."""

    def __init__(self, *, stage_count: int = 2) -> None:
        self._stage_count = stage_count
        self.ensured: list[DuplexStageRequestContext] = []
        self.submissions: list[DuplexStageSubmission] = []
        self.cleanups: list[tuple[list[str], bool]] = []
        self.aborts: list[list[str]] = []

    @property
    def stage_count(self) -> int:
        return self._stage_count

    def sampling_defaults(self) -> tuple[object, ...]:
        return tuple(SamplingParams(max_tokens=8) for _ in range(self._stage_count))

    def ensure_request(self, context: DuplexStageRequestContext) -> None:
        self.ensured.append(context)

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
        self.submissions.append(submission)
        return DuplexStageSubmissionResult(
            request_id=submission.context.request_id,
            stage_id=submission.context.stage_id,
            replica_id=0,
        )

    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
        self.cleanups.append((list(request_ids), abort))

    async def abort_requests(self, request_ids: list[str]) -> None:
        self.aborts.append(list(request_ids))


@dataclass
class Harness:
    manager: DuplexSessionManager
    port: RecordingStagePort
    plugin: EphemeralFakePlugin
    output: asyncio.Queue[Any]
    results: asyncio.Queue[Any]
    runner: DuplexSessionRunner
    events: list[DuplexEvent] = field(default_factory=list)

    @property
    def session(self):  # noqa: ANN202
        return self.runner.session

    def submit(self, command: DuplexCommand) -> None:
        self.manager.dispatch(DuplexSessionCommandMessage(session_id=SESSION_ID, command=command))

    async def settle(self, *, idle_s: float = 0.05, timeout_s: float = 3.0) -> list[DuplexEvent]:
        """Run the loop until the runner mailbox and append tasks are quiet; return new events."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        quiet_since: float | None = None
        collected: list[DuplexEvent] = []
        while True:
            drained = False
            while not self.output.empty():
                message = self.output.get_nowait()
                if isinstance(message, DuplexSessionEventMessage):
                    collected.append(message.event)
                    drained = True
            busy = (
                drained
                or not self.runner._mailbox.empty()
                or any(not task.done() for task in self.runner.tasks.append_tasks)
                or any(not task.done() for task in self.runner._background_tasks)
            )
            now = loop.time()
            if busy:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = now
            elif now - quiet_since >= idle_s:
                break
            if now >= deadline:
                break
            await asyncio.sleep(0.005)
        self.events.extend(collected)
        return collected

    async def run(self, command: DuplexCommand) -> list[DuplexEvent]:
        self.submit(command)
        return await self.settle()

    def deliver(
        self,
        output: object,
        *,
        stage_id: int,
        epoch: int | None = None,
    ) -> bool:
        session = self.session
        fence = session.fence if epoch is None else DuplexFence(SESSION_ID, epoch=epoch)
        context = DuplexOutputContext(
            identity=DuplexRequestIdentity(session_id=SESSION_ID, fence=fence),
            final_stage_id=self.port.stage_count - 1,
            segment_finished=bool(getattr(output, "finished", False)),
        )
        return self.runner.on_stage_output(
            stage_id, output, None, request_id=getattr(output, "request_id"), context=context
        )

    async def deliver_and_settle(self, output: object, *, stage_id: int) -> list[DuplexEvent]:
        self.deliver(output, stage_id=stage_id)
        return await self.settle()


async def open_harness(*, observe_stage0: bool = False, stage_count: int = 2) -> Harness:
    plugin = EphemeralFakePlugin(observe_stage0=observe_stage0)
    port = RecordingStagePort(stage_count=stage_count)
    output: asyncio.Queue[Any] = asyncio.Queue()
    results: asyncio.Queue[Any] = asyncio.Queue()
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=DuplexSessionRuntimeConfig(),
        model_config=None,
    )
    config = DuplexSessionConfig(
        model="fake/ephemeral-turn-commit",
        modalities=["text", "audio"],
        instructions="You are a concise assistant.",
        extra_body={},
    )
    await manager.handle(OpenDuplexSessionMessage(control_id="c-open", session_id=SESSION_ID, session_config=config))
    result = await asyncio.wait_for(results.get(), timeout=2.0)
    assert isinstance(result, DuplexControlResultMessage) and result.ok, result
    harness = Harness(
        manager=manager, port=port, plugin=plugin, output=output, results=results, runner=manager.runners[SESSION_ID]
    )
    await harness.settle()
    return harness


async def close_harness(harness: Harness) -> None:
    await harness.manager.shutdown()


def pcm_f32(samples: int, *, value: float = 0.05) -> bytes:
    return struct.pack(f"<{samples}f", *([value] * samples))


def append_audio(samples: int = 16000) -> commands.AppendAudio:
    return commands.AppendAudio(
        audio=pcm_f32(samples),
        format="pcm_f32le",
        sample_rate_hz=16000,
        is_speech=True,
    )


def fake_output(
    request_id: str,
    *,
    finished: bool,
    turn_id: int | None = 0,
    end_of_turn: bool = False,
) -> SimpleNamespace:
    """A stage output the way the orchestrator hands it to the runner."""
    # ``model_turn_id`` must sit in multimodal_output to survive
    # OmniRequestOutput.from_stage_output; a raw duplex_turn_id attribute does not.
    multimodal_output: dict[str, object] = {"end_of_turn": end_of_turn}
    if turn_id is not None:
        multimodal_output["model_turn_id"] = turn_id
    return SimpleNamespace(
        request_id=request_id,
        finished=finished,
        duplex_turn_id=turn_id,
        outputs=[SimpleNamespace(text="", token_ids=[], multimodal_output=multimodal_output)],
        multimodal_output=multimodal_output,
    )


def types(events: Sequence[DuplexEvent]) -> list[str]:
    return [event.type for event in events]


# --------------------------------------------------------------------------- #
# L2: two committed turns through the real runner                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ephemeral_two_committed_turns_get_fresh_stage0_ids() -> None:
    h = await open_harness()
    try:
        # Admission reserves the turn-scoped Stage0 id, not the resident one.
        assert [context.request_id for context in h.port.ensured] == [
            duplex_ephemeral_stage_request_id(DuplexFence(SESSION_ID, epoch=0, turn_id=0), stage_id=0)
        ]

        # Turn 0: append + commit => one ephemeral ordinary request.
        await h.run(append_audio())
        assert not h.port.submissions  # commit-only buffer: nothing per chunk
        events = await h.run(commands.Commit(create_response=True))
        assert "input_audio_buffer.committed" in types(events)
        assert "response.created" in types(events)
        assert len(h.port.submissions) == 1
        first = h.port.submissions[0]
        assert first.context.request_id.endswith(".r.stage0_t0")
        assert first.already_submitted is False
        assert first.resumable is False

        # The model answers; the data plane reports model_turn_id so the turn completes.
        events = await h.deliver_and_settle(
            fake_output(first.context.request_id, finished=True, turn_id=0, end_of_turn=True),
            stage_id=1,
        )
        assert "response.done" in types(events)
        assert h.session.turn_id == 1

        # Turn 1: a fresh ephemeral id, again a first-time submit.
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        assert len(h.port.submissions) == 2
        second = h.port.submissions[1]
        assert second.context.request_id.endswith(".r.stage0_t1")
        assert second.context.request_id != first.context.request_id
        assert second.already_submitted is False
        assert second.resumable is False
        # Happy path: no stale ephemeral had to be aborted.
        assert h.port.cleanups == []
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_ephemeral_rebind_after_turn_without_model_turn_id() -> None:
    """A turn that never completed (no model_turn_id) re-binds on the next commit."""
    h = await open_harness()
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        assert len(h.port.submissions) == 1
        first_id = h.port.submissions[0].context.request_id
        assert first_id.endswith(".r.stage0_t0")

        # The response ends but the model turn is never completed (no id).
        events = await h.deliver_and_settle(
            fake_output(first_id, finished=True, turn_id=None, end_of_turn=True),
            stage_id=1,
        )
        assert "response.done" in types(events)
        assert h.session.turn_id == 0

        leftover_stage1 = duplex_resource_request_id(h.session.fence, "stage1")
        h.session.bind_stage_request(1, leftover_stage1, fence=h.session.fence)

        # The next commit cannot reuse the finished t0 id: the turn is
        # completed mechanically, a fresh t1 id is minted, and only the stale
        # Stage0 request is aborted. A leftover downstream binding stays.
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        assert len(h.port.submissions) == 2
        second = h.port.submissions[1]
        assert second.context.request_id.endswith(".r.stage0_t1")
        assert second.already_submitted is False
        assert h.session.turn_id == 1
        assert h.port.cleanups == [([first_id], True)]
        assert leftover_stage1 not in {rid for ids, _abort in h.port.cleanups for rid in ids}
        assert (1, leftover_stage1) in h.session.request_resources
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_listen_only_submit_rebinds_on_the_next_commit() -> None:
    """A real listen-only Stage0 submit stays bound; the next commit re-binds it."""
    h = await open_harness()
    try:
        payload = {
            "type": "audio",
            "audio": base64.b64encode(pcm_f32(16000)).decode("ascii"),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
            "final": True,
            "is_speech": True,
        }
        task = await h.runner._start_append(payload, final=True, precreate_response=False)
        assert await task is True
        await h.settle()

        assert len(h.port.submissions) == 1
        first = h.port.submissions[0]
        first_id = first.context.request_id
        assert first_id.endswith(".r.stage0_t0")
        assert first.already_submitted is False
        assert first.resumable is False
        assert h.session.active_request_id == first_id
        assert not is_stable_stage0_placeholder(first_id, session_id=SESSION_ID, epoch=0)
        assert h.session.turn_id == 0
        assert h.session.active_response_id is None
        assert "response.created" not in types(h.events)

        leftover_stage1 = duplex_resource_request_id(h.session.fence, "stage1")
        h.session.bind_stage_request(1, leftover_stage1, fence=h.session.fence)

        await h.run(append_audio())
        events = await h.run(commands.Commit(create_response=True))
        assert "input_audio_buffer.committed" in types(events)
        assert "response.created" in types(events)
        assert len(h.port.submissions) == 2
        second = h.port.submissions[1]
        assert second.context.request_id.endswith(".r.stage0_t1")
        assert second.already_submitted is False
        assert h.session.turn_id == 1
        assert h.port.cleanups == [([first_id], True)]
        assert leftover_stage1 not in {rid for ids, _abort in h.port.cleanups for rid in ids}
        assert (1, leftover_stage1) in h.session.request_resources
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_observe_stage0_projects_and_still_forwards() -> None:
    h = await open_harness(observe_stage0=True)
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        request_id = h.port.submissions[0].context.request_id

        # An intermediate-stage output is observed: projected to the client
        # (an event is emitted) yet NOT consumed (on_stage_output returns
        # False so the orchestrator still forwards it to the next stage).
        forwarded = h.deliver(fake_output(request_id, finished=False, turn_id=0), stage_id=0)
        assert forwarded is False
        events = await h.settle()
        assert "response.output_audio.delta" in types(events)

        # With the plugin's observe off, the same delivery emits nothing.
        h.plugin._observe_stage0 = False
        forwarded = h.deliver(fake_output(request_id, finished=False, turn_id=0), stage_id=0)
        assert forwarded is False
        events = await h.settle()
        assert "response.output_audio.delta" not in types(events)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_observe_finished_intermediate_does_not_close_the_stream() -> None:
    """A finished observed stage is that stage ending, not the duplex turn."""
    h = await open_harness(observe_stage0=True)
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        request_id = h.port.submissions[0].context.request_id

        forwarded = h.deliver(fake_output(request_id, finished=True, turn_id=0), stage_id=0)
        events = await h.settle()
        assert forwarded is False
        assert "response.output_audio.delta" in types(events)
        assert "response.done" not in types(events)
        assert h.plugin.data_plane.closed_streams == []
        assert len(h.port.submissions) == 1
        assert h.runner.model_state.continuation_units == 0
        assert h.session.turn_id == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_finished_final_stage_does_not_schedule_silence_continuation() -> None:
    """Non-resumable stage0 cannot submit_update a silence unit after TTS ends."""
    h = await open_harness()
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        request_id = h.port.submissions[0].context.request_id

        events = await h.deliver_and_settle(
            fake_output(request_id, finished=True, turn_id=0, end_of_turn=True),
            stage_id=1,
        )
        assert "response.done" in types(events)
        assert request_id in h.plugin.data_plane.closed_streams
        assert len(h.port.submissions) == 1
        assert h.runner.model_state.continuation_units == 0
        assert h.runner.model_state.continuation_owner_id is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_listen_only_append_keeps_ephemeral_stage0_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn-scoped ids must survive a listen-only append; only ``...r.stage0`` is cleared."""
    h = await open_harness()
    try:

        async def listen_only(*args: object, **kwargs: object) -> tuple[bool, bool]:
            del args, kwargs
            return True, False

        monkeypatch.setattr(h.runner.model, "append_runtime_input", listen_only)
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        bound = h.session.active_request_id
        assert isinstance(bound, str)
        assert bound.endswith(".r.stage0_t0")
        assert not is_stable_stage0_placeholder(bound, session_id=SESSION_ID, epoch=0)
        assert h.port.submissions == []
    finally:
        await close_harness(h)
