# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Engine-resident session runner scenarios (the old serving handler tests, re-homed).

The runner is driven exactly the way ``DuplexOrchestrator`` drives it: typed
commands go through ``DuplexSessionManager.dispatch``, stage outputs are pushed
with ``runner.on_stage_output`` and everything the session says is read back
from the manager's output sink as typed events. The stage port is a recording
fake; the model plugin is the real MiniCPM-o 4.5 one so append planning and
output projection are exercised end to end.
"""

from __future__ import annotations

import asyncio
import base64
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.commands import DuplexCommand
from vllm_omni.engine.duplex.config import DuplexPlaybackCommitPolicy, DuplexSessionConfig, DuplexSessionState
from vllm_omni.engine.duplex.contracts import (
    DuplexOutputContext,
    DuplexRequestIdentity,
    DuplexStagePort,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex.events import DuplexEvent
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    DuplexSessionEventMessage,
    OpenDuplexSessionMessage,
)
from vllm_omni.engine.duplex.session.engine_session import RESPONSE_REQUEST_MEASUREMENT_ORIGIN
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner
from vllm_omni.metrics.stats import OrchestratorAggregator, StageRequestStats, StageStats
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION_ID = "duplex-test"
LISTEN_TOKEN_ID = 7


# --------------------------------------------------------------------------- #
# Fakes                                                                       #
# --------------------------------------------------------------------------- #


class RecordingStagePort(DuplexStagePort):
    """Records what the runner asks of the orchestrator; never talks to a stage."""

    def __init__(self, *, stage_count: int = 2) -> None:
        self._stage_count = stage_count
        self.ensured: list[DuplexStageRequestContext] = []
        self.submissions: list[DuplexStageSubmission] = []
        self.cleanups: list[tuple[list[str], bool]] = []
        self.aborts: list[list[str]] = []
        self.fail_submit: Exception | None = None
        #: When set, ``submit`` parks on it after signalling ``submit_started``.
        self.submit_gate: asyncio.Event | None = None
        self.submit_started = asyncio.Event()

    @property
    def stage_count(self) -> int:
        return self._stage_count

    def sampling_defaults(self) -> tuple[object, ...]:
        return tuple(SamplingParams(max_tokens=8) for _ in range(self._stage_count))

    def ensure_request(self, context: DuplexStageRequestContext) -> None:
        self.ensured.append(context)

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
        if self.fail_submit is not None:
            raise self.fail_submit
        if self.submit_gate is not None:
            self.submit_started.set()
            await self.submit_gate.wait()
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


def _fake_encode_audio(audio: object, sample_rate_hz: int, response_format: str, speed: float | None) -> str | None:
    del sample_rate_hz, response_format, speed
    if audio is None:
        return None
    samples = int(np.asarray(audio, dtype=np.float32).size)
    return f"wav-{samples}" if samples > 0 else None


@dataclass
class Harness:
    manager: DuplexSessionManager
    port: RecordingStagePort
    output: asyncio.Queue[Any]
    results: asyncio.Queue[Any]
    runner: DuplexSessionRunner
    events: list[DuplexEvent] = field(default_factory=list)

    @property
    def session(self):
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
        output: SimpleNamespace,
        *,
        stage_id: int = 1,
        segment_finished: bool = False,
        segment_output_metadata: dict[str, Any] | None = None,
        segment_token_ids: Sequence[int] = (),
        epoch: int | None = None,
        metrics: Any = None,
    ) -> bool:
        session = self.session
        fence = session.fence if epoch is None else session.fence.__class__(SESSION_ID, epoch=epoch)
        context = DuplexOutputContext(
            identity=DuplexRequestIdentity(session_id=SESSION_ID, fence=fence),
            final_stage_id=self.port.stage_count - 1,
            segment_finished=segment_finished,
            segment_token_ids=tuple(segment_token_ids),
            segment_output_metadata=dict(segment_output_metadata or {}),
        )
        request_id = getattr(output, "request_id", None)
        if not isinstance(request_id, str):
            raise AssertionError("stage output is missing request_id")
        return self.runner.on_stage_output(stage_id, output, metrics, request_id=request_id, context=context)

    async def deliver_and_settle(self, output: SimpleNamespace, **kwargs: Any) -> list[DuplexEvent]:
        self.deliver(output, **kwargs)
        return await self.settle()

    def stage0_request_id(self, epoch: int | None = None) -> str:
        fence = self.session.fence if epoch is None else self.session.fence.__class__(SESSION_ID, epoch=epoch)
        return duplex_resource_request_id(fence, "stage0")


async def open_harness(
    *,
    auto_response: bool = True,
    modalities: Sequence[str] = ("text",),
    extra_body: dict[str, object] | None = None,
    runtime_config: DuplexSessionRuntimeConfig | None = None,
    stage_count: int = 2,
    clock: Any = None,
    log_stats: bool = False,
) -> Harness:
    plugin = MiniCPMO45DuplexPlugin(_fake_encode_audio)
    port = RecordingStagePort(stage_count=stage_count)
    output: asyncio.Queue[Any] = asyncio.Queue()
    results: asyncio.Queue[Any] = asyncio.Queue()
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=runtime_config or DuplexSessionRuntimeConfig(),
        model_config=None,
        log_stats=log_stats,
        clock=clock,
    )
    body: dict[str, object] = {"auto_response": auto_response, **(extra_body or {})}
    config = DuplexSessionConfig(
        model="openbmb/MiniCPM-o-4_5",
        modalities=list(modalities),
        instructions="You are a concise assistant.",
        extra_body=body,
    )
    await manager.handle(OpenDuplexSessionMessage(control_id="c-open", session_id=SESSION_ID, session_config=config))
    result = await asyncio.wait_for(results.get(), timeout=2.0)
    assert isinstance(result, DuplexControlResultMessage) and result.ok, result
    harness = Harness(manager=manager, port=port, output=output, results=results, runner=manager.runners[SESSION_ID])
    await harness.settle()
    return harness


async def close_harness(harness: Harness) -> None:
    await harness.manager.shutdown()


# --------------------------------------------------------------------------- #
# Payload / output builders                                                   #
# --------------------------------------------------------------------------- #


def pcm_f32(samples: int, *, value: float = 0.05) -> bytes:
    return struct.pack(f"<{samples}f", *([value] * samples))


def append_audio(
    samples: int = 16000,
    *,
    value: float = 0.05,
    is_speech: bool | None = True,
    event_id: str | None = None,
) -> commands.AppendAudio:
    return commands.AppendAudio(
        audio=pcm_f32(samples, value=value),
        format="pcm_f32le",
        sample_rate_hz=16000,
        is_speech=is_speech,
        event_id=event_id,
    )


def stage_stats(
    *,
    stage_id: int,
    request_id: str,
    num_tokens_out: int,
    itls_ms: list[float] | None = None,
) -> StageRequestStats:
    """The per-request stats the orchestrator hands to the runner with an output."""
    return StageRequestStats(
        batch_id=0,
        batch_size=1,
        num_tokens_in=7,
        num_tokens_out=num_tokens_out,
        stage_gen_time_ms=120.0,
        rx_transfer_bytes=0,
        rx_decode_time_ms=0.0,
        rx_in_flight_time_ms=0.0,
        stage_stats=StageStats(),
        stage_id=stage_id,
        request_id=request_id,
        final_output_type="text",
        vllm_itls_ms=list(itls_ms or []),
    )


def tts_output(
    request_id: str,
    *,
    samples: int = 24000,
    finished: bool = False,
    text: str = "hello",
    tts_is_last_chunk: bool = False,
    turn_end: bool = False,
    turn_id: int = 0,
    epoch: int = 0,
) -> SimpleNamespace:
    """A Stage1 (TTS) output the way the orchestrator hands it to the runner."""
    return SimpleNamespace(
        request_id=request_id,
        finished=finished,
        outputs=[SimpleNamespace(text=text, token_ids=[], multimodal_output={})],
        multimodal_output={
            "audio": np.zeros(samples, dtype=np.float32),
            "sr": 24000,
            "meta.tts_is_last_chunk": np.array([int(tts_is_last_chunk)], dtype=np.int32),
            "meta.turn_end": np.array([int(turn_end)], dtype=np.int32),
            "meta.duplex_turn_id": np.array([turn_id], dtype=np.int32),
            "meta.duplex_epoch": np.array([epoch], dtype=np.int32),
        },
    )


def listen_output(request_id: str) -> SimpleNamespace:
    """A finished Stage0 segment that ends with the listen token."""
    return SimpleNamespace(
        request_id=request_id,
        finished=True,
        outputs=[
            SimpleNamespace(
                text="",
                token_ids=[11, 12, LISTEN_TOKEN_ID],
                stop_reason=LISTEN_TOKEN_ID,
                multimodal_output={},
            )
        ],
        multimodal_output={"meta.listen_token_id": LISTEN_TOKEN_ID},
    )


def types(events: Sequence[DuplexEvent]) -> list[str]:
    return [event.type for event in events]


def find(events: Sequence[DuplexEvent], wire_type: str) -> DuplexEvent:
    for event in events:
        if event.type == wire_type:
            return event
    raise AssertionError(f"no {wire_type!r} in {types(events)}")


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# --------------------------------------------------------------------------- #
# Session lifecycle                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_open_announces_the_session_and_reserves_the_stage0_request() -> None:
    h = await open_harness()
    try:
        assert types(h.events) == ["session.created", "session.updated"]
        created = h.events[0]
        assert created.session_id == SESSION_ID
        assert created.session["id"] == SESSION_ID
        assert created.session["capabilities"]["supports_input_append"] is True
        assert created.to_realtime()["type"] == "session.created"
        assert "incarnation" not in created.to_realtime()
        assert h.session.state == DuplexSessionState.OPEN
        assert h.session.config.playback_commit_policy == DuplexPlaybackCommitPolicy.COMMIT_ALL_ON_DONE.value
        assert created.session["playback_commit_policy"] == DuplexPlaybackCommitPolicy.COMMIT_ALL_ON_DONE.value
        # Admission reserves the resumable Stage0 request atomically with the open.
        assert [context.request_id for context in h.port.ensured] == [h.stage0_request_id(epoch=0)]
        assert h.stage0_request_id(epoch=0) == "duplex-s.ZHVwbGV4LXRlc3Q.e.0.r.stage0"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_close_emits_session_closed_and_releases_stage_requests() -> None:
    h = await open_harness()
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    await h.manager.handle(
        CloseDuplexSessionMessage(control_id="c-close", session_id=SESSION_ID, reason="client_close")
    )
    result = await asyncio.wait_for(h.results.get(), timeout=2.0)
    events = await h.settle()

    assert result.ok and result.operation == "close"
    assert types(events) == ["session.closed"]
    assert events[0].reason == "client_close"
    assert events[0].is_terminal
    assert h.session.state == DuplexSessionState.CLOSED
    assert h.port.cleanups == [([request_id], True)]
    assert SESSION_ID not in h.manager.runners
    await close_harness(h)


@pytest.mark.asyncio
async def test_idle_expiry_emits_session_expired() -> None:
    clock = {"now": 100.0}
    h = await open_harness(runtime_config=DuplexSessionRuntimeConfig(idle_ttl_s=1.0), clock=lambda: clock["now"])
    clock["now"] += 2.0
    assert await h.manager.reap_expired() == 1
    events = await h.settle()

    assert types(events) == ["session.expired"]
    assert events[0].reason == "idle_ttl_expired"
    assert events[0].is_terminal
    assert SESSION_ID not in h.manager.runners
    assert not h.manager._closing
    await close_harness(h)


@pytest.mark.asyncio
async def test_commands_for_a_closed_session_are_answered_with_unknown_session() -> None:
    h = await open_harness()
    await h.manager.handle(CloseDuplexSessionMessage(control_id="c-close", session_id=SESSION_ID, reason="bye"))
    await h.results.get()
    await h.settle()

    h.submit(commands.Heartbeat(event_id="evt-late"))
    events = await h.settle()

    assert types(events) == ["error"]
    assert events[0].code == "unknown_session"
    assert events[0].related_event_id == "evt-late"
    await close_harness(h)


# --------------------------------------------------------------------------- #
# Input path                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_append_plans_and_submits_one_model_unit_in_wire_order() -> None:
    h = await open_harness()
    try:
        events = await h.run(append_audio())
        assert types(events) == ["input_audio_buffer.speech_started"]
        assert len(h.port.submissions) == 1
        first = h.port.submissions[0]
        assert first.context.request_id == h.stage0_request_id(epoch=0)
        assert first.context.stage_id == 0
        assert first.already_submitted is False
        duplex = first.prompt["model_intermediate_buffer"]["duplex"]
        assert duplex["session_id"] == SESSION_ID
        assert (duplex["epoch"], duplex["seq"], duplex["turn_id"], duplex["turn_seq"]) == (0, 1, 0, 1)
        assert duplex["final"] is False
        assert "incarnation" not in duplex
        assert len(base64.b64decode(duplex["payload"]["audio"])) == 16000 * 4
        assert duplex["payload"]["is_speech"] is True
        assert h.session.input_seq == 1
        assert h.session.stage_request_submitted(0, first.context.request_id)

        await h.run(append_audio())
        second = h.port.submissions[1]
        assert second.already_submitted is True
        assert second.prompt["model_intermediate_buffer"]["duplex"]["seq"] == 2
        assert h.session.input_seq == 2
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_partial_units_are_buffered_until_a_whole_chunk_is_available() -> None:
    h = await open_harness()
    try:
        await h.run(append_audio(samples=8000))
        assert h.port.submissions == []
        assert h.runner.model_state.audio_buffer.has_pending()

        await h.run(append_audio(samples=8000))
        assert len(h.port.submissions) == 1
        assert not h.runner.model_state.audio_buffer.has_pending()
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_input_backpressure_is_rejected_before_the_buffer_is_touched() -> None:
    h = await open_harness(runtime_config=DuplexSessionRuntimeConfig(max_pending_input_bytes_per_session=1000))
    try:
        events = await h.run(append_audio(event_id="evt-big"))
        assert types(events) == ["error"]
        assert events[0].code == "input_backpressure"
        assert events[0].related_event_id == "evt-big"
        assert h.port.submissions == []
        assert not h.runner.model_state.audio_buffer.has_pending()
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_text_append_is_not_supported_by_the_native_runtime() -> None:
    h = await open_harness()
    try:
        events = await h.run(commands.AppendText(text="hello", event_id="evt-text"))
        assert types(events) == ["error"]
        assert events[0].code == "native_text_append_unsupported"
        assert events[0].related_event_id == "evt-text"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_clear_input_drops_buffered_audio() -> None:
    h = await open_harness()
    try:
        await h.run(append_audio(samples=8000))
        assert h.runner.model_state.audio_buffer.has_pending()
        events = await h.run(commands.ClearInput())
        assert types(events) == ["input_audio_buffer.cleared"]
        assert not h.runner.model_state.audio_buffer.has_pending()
        assert h.session.pending_input_bytes == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_commit_without_audio_is_rejected() -> None:
    h = await open_harness()
    try:
        events = await h.run(commands.Commit(event_id="evt-commit"))
        assert types(events) == ["error"]
        assert events[0].code == "input_audio_buffer_empty"
        assert events[0].related_event_id == "evt-commit"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_empty_silent_commit_acks_without_opening_response() -> None:
    """#3: empty is_speech=False commit keeps the silent-ack pair, even auto-respond."""
    h = await open_harness(auto_response=True)
    try:
        # resolve_commit only admits empty silent commits after non-speech was seen.
        projector = h.runner._require_projector()
        projector.input_audio_buffer_had_non_speech = True
        events = await h.run(commands.Commit())
        assert "error" not in types(events), types(events)
        assert "input_audio_buffer.committed" in types(events), types(events)
        assert "response.listen" in types(events), types(events)
        listen = next(event for event in events if event.type == "response.listen")
        assert listen.details["reason"] == "silence_or_noise"
        committed = next(event for event in events if event.type == "input_audio_buffer.committed")
        assert committed.details.get("empty") is True
        assert committed.details.get("no_response") is True
        assert h.port.submissions == []
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_commit_projects_the_user_item_and_does_not_resubmit_consumed_audio() -> None:
    h = await open_harness()
    try:
        await h.run(append_audio())
        events = await h.run(commands.Commit())
        assert types(events) == [
            "input_audio_buffer.speech_stopped",
            "conversation.item.added",
            "conversation.item.created",
            "input_audio_buffer.committed",
            "conversation.item.done",
        ]
        committed = find(events, "input_audio_buffer.committed")
        assert committed.item_id == find(events, "conversation.item.added").item_id
        # The unit already reached Stage0; a commit is only a turn marker here.
        assert len(h.port.submissions) == 1
        assert h.session.input_commit_seq == 1
        assert h.session.history[-1]["role"] == "user"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_heartbeat_touches_the_lease_and_acks() -> None:
    clock = {"now": 100.0}
    h = await open_harness(clock=lambda: clock["now"])
    try:
        clock["now"] = 150.0
        events = await h.run(commands.Heartbeat())
        assert types(events) == ["session.heartbeat_ack"]
        assert h.session.lease.last_activity == 150.0
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Model output path                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stage1_audio_opens_a_response_and_streams_deltas() -> None:
    h = await open_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()

        events = await h.deliver_and_settle(tts_output(request_id, samples=24000, text="he"))
        assert types(events) == [
            "response.created",
            "conversation.item.added",
            "conversation.item.created",
            "response.output_item.added",
            "response.speak",
            "response.content_part.added",
            "response.output_audio.delta",
            "response.output_audio_transcript.delta",
        ]
        response_id = find(events, "response.created").response_id
        assert response_id == h.session.active_response_id
        assert h.session.active_request_id == request_id
        delta = find(events, "response.output_audio.delta")
        assert (delta.delta, delta.format, delta.sample_rate_hz) == ("wav-24000", "wav", 24000)
        assert delta.response_id == response_id
        assert find(events, "response.output_audio_transcript.delta").delta == "he"
        assert h.session.playback.sent_ms == 1000

        # Cumulative Stage1 audio is sliced to the new samples only.
        events = await h.deliver_and_settle(tts_output(request_id, samples=48000, text="hello"))
        assert types(events) == ["response.output_audio.delta", "response.output_audio_transcript.delta"]
        assert find(events, "response.output_audio.delta").delta == "wav-24000"
        assert find(events, "response.output_audio_transcript.delta").delta == "llo"
        assert h.session.playback.sent_ms == 2000
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_turn_end_completes_the_response_and_advances_the_model_turn() -> None:
    h = await open_harness()
    try:
        await h.run(commands.UpdateSession(patch={"playback_commit_policy": "ack_only"}))
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))

        events = await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello", turn_end=True))
        assert types(events) == [
            "response.output_audio.done",
            "response.output_audio_transcript.done",
            "response.content_part.done",
            "response.output_item.done",
            "conversation.item.done",
            "response.done",
            "rate_limits.updated",
        ]
        done = find(events, "response.done")
        assert done.status == "completed"
        assert find(events, "response.output_audio_transcript.done").transcript == "hello"
        assert h.session.active_response_id is None
        assert h.session.turn_id == 1
        # ACK-only playback was explicitly selected: the assistant text enters
        # history on playback ack.
        assert h.session.history == ()

        ack = await h.run(commands.AckPlayback(played_ms=1000, response_id=done.response_id))
        assert types(ack) == ["playback.acknowledged"]
        assert ack[0].details["history_committed"] is True
        assert ack[0].details["playback"]["committed_ms"] == 1000
        assert h.session.history == ({"role": "assistant", "content": "hello"},)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_playback_ack_for_an_unknown_response_is_rejected() -> None:
    h = await open_harness()
    try:
        events = await h.run(commands.AckPlayback(played_ms=10, response_id="resp-unknown", event_id="evt-ack"))
        assert types(events) == ["error"]
        assert events[0].code == "playback_item_not_found"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_stale_epoch_output_is_dropped_after_barge_in() -> None:
    h = await open_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="he"))

        events = await h.run(commands.BargeIn())
        assert h.session.epoch == 1
        assert h.session.active_response_id is None
        assert h.port.aborts == [[request_id]]
        assert h.port.cleanups == [([request_id], True)]
        done = find(events, "response.done")
        assert done.status == "cancelled"
        assert types(events)[: types(events).index("response.done")] == [
            "response.output_audio.done",
            "response.output_audio_transcript.done",
            "response.content_part.done",
            "response.output_item.done",
            "conversation.item.done",
        ]

        late = tts_output(request_id, samples=48000, text="hello", epoch=0)
        assert h.deliver(late, epoch=0) is True  # consumed, never forwarded
        assert await h.settle() == []
    finally:
        await close_harness(h)


async def test_barge_in_aborts_draining_tts_as_well_as_the_active_request() -> None:
    h = await open_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="he"))
        h.session.bind_draining_request("duplex-drain-tts", "resp-old")
        # Draining TTS sits on an older turn fence. cancel_fence only drops
        # the fence being cancelled, so these bindings must be popped by id.
        from vllm_omni.engine.duplex.contracts import DuplexFence
        from vllm_omni.engine.duplex.session.engine_session import DuplexRequestResource

        older = DuplexFence(h.session.session_id, epoch=h.session.epoch, turn_id=h.session.turn_id + 5)
        h.session.request_resources[(2, "duplex-drain-tts")] = DuplexRequestResource(
            stage_id=2, request_id="duplex-drain-tts", fence=older, submitted=True
        )
        h.session.request_resources[(3, "duplex-drain-tts")] = DuplexRequestResource(
            stage_id=3, request_id="duplex-drain-tts", fence=older, submitted=True
        )
        h.session.request_resources[(1, "still-live")] = DuplexRequestResource(
            stage_id=1, request_id="still-live", fence=older, submitted=True
        )
        events = await h.run(commands.BargeIn())
        assert h.port.aborts == [[request_id, "duplex-drain-tts"]]
        assert not h.session.is_draining_request("duplex-drain-tts")
        done_ids = [
            event.response["id"]
            for event in events
            if getattr(event, "wire_type", "") == "response.done" and isinstance(getattr(event, "response", None), dict)
        ]
        assert "resp-old" in done_ids
        assert (2, "duplex-drain-tts") not in h.session.request_resources
        assert (3, "duplex-drain-tts") not in h.session.request_resources
        assert (0, request_id) not in h.session.request_resources
        assert (1, "still-live") in h.session.request_resources
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_cancel_response_without_active_response_is_rejected() -> None:
    h = await open_harness()
    try:
        events = await h.run(commands.CancelResponse(event_id="evt-cancel"))
        assert types(events) == ["error"]
        assert events[0].code == "response_not_active"
        assert events[0].related_event_id == "evt-cancel"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_cancel_response_aborts_the_stage_request_and_reports_playback() -> None:
    h = await open_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
        await h.run(commands.AckPlayback(played_ms=400, response_id=h.session.active_response_id))

        events = await h.run(commands.CancelResponse())
        done = find(events, "response.done")
        assert done.status == "cancelled"
        assert done.response["status_details"]["reason"] == "client_cancelled"
        assert h.port.aborts == [[request_id]]
        assert h.session.epoch == 1
        # Only the played prefix of the cancelled answer is kept in history.
        assert h.session.history == ({"role": "assistant", "content": "he"},)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_cancel_response_absorbs_a_response_task_that_outlives_the_wait() -> None:
    """The 0.25 s wait after cancelling the response task may time out.

    On Python 3.10 that timeout is ``asyncio.TimeoutError``, not the builtin
    ``TimeoutError``, and it must not stop the cancel from completing.
    """

    async def outlives_the_first_cancel() -> None:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            await asyncio.sleep(1)
            raise

    h = await open_harness()
    task = asyncio.create_task(outlives_the_first_cancel())
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
        h.runner.tasks.active_response_task = task

        events = await h.run(commands.CancelResponse())
        # The runner is still inside that wait when the mailbox goes quiet.
        events += await h.settle(idle_s=0.6, timeout_s=5.0)
        assert find(events, "response.done").status == "cancelled"
        assert "error" not in types(events)
        assert h.session.epoch == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


@pytest.mark.asyncio
async def test_listen_decision_is_consumed_and_never_forwarded_to_tts() -> None:
    h = await open_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        # An unfinished Stage0 segment is plain text for the TTS stage: forwarded.
        text_output = SimpleNamespace(
            request_id=request_id,
            finished=False,
            outputs=[SimpleNamespace(text="hi", token_ids=[11], multimodal_output={})],
            multimodal_output={},
        )
        assert h.deliver(text_output, stage_id=0) is False

        consumed = h.deliver(
            listen_output(request_id),
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[11, 12, LISTEN_TOKEN_ID],
            segment_output_metadata={"meta.listen_token_id": LISTEN_TOKEN_ID},
        )
        events = await h.settle()
        assert consumed is True
        assert types(events) == ["response.listen"]
        assert events[0].details["reason"] == "model_listen"
        assert events[0].details["model_listen"] is True
        assert events[0].to_realtime()["response"]["status"] == "listening"
        assert h.session.active_response_id is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_listen_decision_on_a_resumable_request_closes_the_bounded_response() -> None:
    """A live stage-0 request is resumable: its output never says ``finished``,
    only its segment does. The listen that answers the last continuation unit
    must still close the response, or the session sits forever with no
    terminal event (the live-client E2E hang)."""
    h = await open_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
        response_id = h.session.active_response_id
        assert response_id is not None
        # The continuation budget is spent: the unit being answered was the forced listen.
        model_state = h.runner.model_state
        model_state.continuation_owner_id = f"response:{response_id}"
        model_state.continuation_units = h.runner.model._AUTO_RESPONSE_MAX_CONTINUATION_UNITS

        listen = listen_output(request_id)
        listen.finished = False
        events = await h.deliver_and_settle(
            listen,
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[11, 12, LISTEN_TOKEN_ID],
            segment_output_metadata={"meta.listen_token_id": LISTEN_TOKEN_ID},
        )

        assert types(events)[0] == "response.listen"
        assert events[0].details["model_listen"] is True
        assert find(events, "response.done").response_id == response_id
        assert h.session.active_response_id is None
        assert model_state.continuation_units == 0
        assert len(h.port.submissions) == 1  # no further silence unit was scheduled
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_direct_response_listen_still_emits_response_done_after_continuation_clear() -> None:
    """AURA-like silent DIRECT_RESPONSE: after continuation budget is spent, emit response.done.

    Non-resumable turn-commit still closes the active response on a listen/direct
    decision so the client is not left without a terminal event.
    """
    from dataclasses import replace

    h = await open_harness()
    try:
        h.session.capabilities = replace(h.session.capabilities, supports_core_resumable_request=False)
        await h.run(append_audio())
        assert h.port.submissions, "ephemeral Stage0 must submit"
        request_id = h.port.submissions[-1].context.request_id
        assert "-turn" in request_id
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="ok"))
        response_id = h.session.active_response_id
        assert response_id is not None
        model_state = h.runner.model_state
        model_state.continuation_owner_id = f"response:{response_id}"
        model_state.continuation_units = h.runner.model._AUTO_RESPONSE_MAX_CONTINUATION_UNITS

        listen = listen_output(request_id)
        listen.finished = True
        events = await h.deliver_and_settle(
            listen,
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[11, 12, LISTEN_TOKEN_ID],
            segment_output_metadata={"meta.listen_token_id": LISTEN_TOKEN_ID},
        )
        assert "response.listen" in types(events)
        done = find(events, "response.done")
        assert done.response_id == response_id
        assert h.session.active_response_id is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_listen_decision_on_a_resumable_request_keeps_the_turn_going() -> None:
    """Same unfinished listen, budget left: it must schedule the next unit.

    A non-terminal auto-response listen is answered with the next silence
    unit, not with a ``response.listen`` event, so the submission is the
    observable. Before the fix the projector saw ``finished=False`` and
    yielded nothing: no silence unit, the response left open with nothing in
    flight -- the same hang, one unit earlier.
    """
    h = await open_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
        response_id = h.session.active_response_id
        assert response_id is not None
        submissions_before = len(h.port.submissions)

        listen = listen_output(request_id)
        listen.finished = False
        events = await h.deliver_and_settle(
            listen,
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[11, 12, LISTEN_TOKEN_ID],
            segment_output_metadata={"meta.listen_token_id": LISTEN_TOKEN_ID},
        )

        assert not [event for event in events if event.type == "error"], types(events)
        assert len(h.port.submissions) == submissions_before + 1, "the next silence unit was not scheduled"
        assert h.runner.model_state.continuation_units == 1
        assert h.session.active_response_id == response_id
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_tts_segment_end_schedules_a_silence_continuation_unit() -> None:
    h = await open_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
        await h.deliver_and_settle(
            tts_output(request_id, samples=48000, text="hello", tts_is_last_chunk=True, finished=True)
        )

        assert len(h.port.submissions) == 2
        silence = h.port.submissions[1]
        assert silence.already_submitted is True
        duplex = silence.prompt["model_intermediate_buffer"]["duplex"]
        assert duplex["seq"] == 2
        assert duplex["payload"]["duplex_turn_id"] == 0
        assert set(base64.b64decode(duplex["payload"]["audio"])) == {0}
        assert h.runner.model_state.continuation_units == 1
        # The response stays open across the segment boundary.
        assert h.session.active_response_id is not None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_stage_failure_fails_the_active_response() -> None:
    h = await open_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))

        h.runner.on_stage_failure(1, RuntimeError("boom"))
        events = await h.settle()
        assert types(events)[0] == "error"
        assert events[0].code == "runtime_data_plane_stream_failed"
        done = find(events, "response.done")
        assert done.status == "failed"
        assert done.response["status_details"]["reason"] == "runtime_data_plane_stream_failed"
        assert h.session.active_response_id is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_stage_failure_on_draining_request_fails_that_response_only() -> None:
    """Overlapped: a draining older request failure must not wipe the active response."""
    from dataclasses import replace

    h = await open_harness()
    try:
        h.session.capabilities = replace(h.session.capabilities, supports_concurrent_turn_requests=True)
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
        r1 = h.session.active_response_id
        assert r1 is not None
        draining_req = "req-r1-talker-drain"
        h.session.bind_draining_request(draining_req, r1)
        # Open a newer active response while R1 TTS is still draining.
        r2 = h.session.begin_response(turn_id=(h.session.turn_id or 0) + 1)
        assert h.session.active_response_id == r2
        assert r2 != r1

        h.runner.on_stage_failure(2, RuntimeError("drain-boom"), request_id=draining_req)
        events = await h.settle()
        assert types(events)[0] == "error"
        done = find(events, "response.done")
        assert done.response_id == r1
        assert done.status == "failed"
        assert h.session.active_response_id == r2
        assert not h.session.is_draining_request(draining_req)
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Turn mode (no auto response)                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_turn_mode_skips_silent_chunks() -> None:
    h = await open_harness(auto_response=False)
    try:
        events = await h.run(append_audio(is_speech=False, value=0.0))
        assert types(events) == ["response.listen"]
        assert events[0].details["reason"] == "silence_or_noise"
        assert h.port.submissions == []
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_turn_mode_keeps_silent_chunks_with_video_frames() -> None:
    """Engine vision-follow buffers silent+frames; MiniCPM commit stays speech-gated."""
    from dataclasses import replace

    h = await open_harness(auto_response=False)
    try:
        h.session.capabilities = replace(
            h.session.capabilities,
            required_input_modalities=frozenset({"video"}),
            optional_input_modalities=frozenset({"audio"}),
        )
        # Minimal 1x1 JPEG (base64) — wire validation only checks non-empty str.
        frame = (
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
            "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
            "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAA"
            "AAAAAAAAAAj/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAA"
            "AAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAGfAP/EABQQ"
            "AQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAQUCf//EABQRAQAAAAAAAAAAAAAAAAAA"
            "AAD/2gAIAQMBAT8Bf//EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQIBAT8Bf//E"
            "ABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEABj8Cf//EABQQAQAAAAAAAAAAAAAA"
            "AAAAAAD/2gAIAQEAAT8hf//Z"
        )
        cmd = commands.AppendAudio(
            audio=pcm_f32(160, value=0.0),
            format="pcm_f32le",
            sample_rate_hz=16000,
            is_speech=False,
            video_frames=(frame,),
        )
        events = await h.run(cmd)
        assert "response.listen" not in types(events)
        assert h.runner.model_state.audio_buffer.has_pending()
        events = await h.run(commands.Commit(create_response=True))
        # MiniCPM prepare_commit is speech-gated (no frames-only Stage0).
        assert "input_audio_buffer.committed" in types(events), types(events)
        assert h.port.submissions == []
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_turn_mode_keeps_silent_vision_while_response_in_progress() -> None:
    """TTS still playing: vision-follow must buffer, not drop as silence_or_noise."""
    from dataclasses import replace

    h = await open_harness(auto_response=False)
    try:
        h.session.capabilities = replace(
            h.session.capabilities,
            required_input_modalities=frozenset({"video"}),
            optional_input_modalities=frozenset({"audio"}),
        )
        h.session._response.active_response_id = "resp-tts"
        frame = (
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
            "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
            "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAA"
            "AAAAAAAAAAj/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAA"
            "AAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAGfAP/EABQQ"
            "AQAAAAAAAAAAAAAAAAAAAAD/2gAIAQMBAT8Bf//EABQRAQAAAAAAAAAAAAAAAAAA"
            "AAD/2gAIAQIBAT8Bf//EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEABj8Cf//E"
            "ABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAT8hf//Z"
        )
        events = await h.run(
            commands.AppendAudio(
                audio=pcm_f32(160, value=0.0),
                format="pcm_f32le",
                sample_rate_hz=16000,
                is_speech=False,
                video_frames=(frame,),
            )
        )
        listen_reasons = [
            getattr(event, "details", {}) or {} for event in events if getattr(event, "type", None) == "response.listen"
        ]
        assert all(details.get("reason") != "silence_or_noise" for details in listen_reasons)
        assert h.runner.model_state.audio_buffer.has_pending()
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_turn_mode_commit_with_response_create_starts_one_response() -> None:
    h = await open_harness(auto_response=False)
    try:
        events = await h.run(append_audio())
        assert types(events) == ["input_audio_buffer.speech_started"]
        assert h.port.submissions == []  # buffered until the commit

        events = await h.run(commands.Commit(create_response=True))
        assert "input_audio_buffer.committed" in types(events)
        assert "response.created" in types(events)
        assert len(h.port.submissions) == 1
        final = h.port.submissions[0]
        assert final.prompt["model_intermediate_buffer"]["duplex"]["final"] is True
        request_id = final.context.request_id
        response_id = h.session.active_response_id
        assert response_id is not None

        events = await h.deliver_and_settle(tts_output(request_id, samples=24000, text="sure"))
        assert types(events)[-2:] == ["response.output_audio.delta", "response.output_audio_transcript.delta"]
        events = await h.deliver_and_settle(tts_output(request_id, samples=24000, text="sure", finished=True))
        done = find(events, "response.done")
        assert done.response_id == response_id
        assert done.status == "completed"
        assert h.session.active_response_id is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_response_create_with_no_input_at_all_is_rejected() -> None:
    """Nothing committed and no items: refuse rather than emit an empty turn."""
    h = await open_harness(auto_response=False)
    try:
        events = await h.run(commands.CreateResponse(event_id="evt-resp"))
        assert types(events) == ["error"]
        assert events[0].code == "response_create_without_input"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_an_item_only_response_create_is_refused_not_left_hanging() -> None:
    """Items are context, not a turn, and the refusal must be immediate.

    A model-native model decides to speak from the audio it hears, so a turn
    with no audio in it is one it never answers. Opening a response anyway
    submits work that never completes, and the caller only finds out when the
    session idles out -- observed as a 300-second HTTP 500 against a real
    server. Refusing costs the same information and none of the wait.
    """
    h = await open_harness(auto_response=False)
    try:
        await h.run(
            commands.CreateItem(
                item={
                    "id": "item_prompt",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is 2+2?"}],
                }
            )
        )
        events = await h.run(commands.CreateResponse(event_id="evt-resp"))
        assert types(events) == ["error"], events
        assert events[0].code == "text_only_turn_unsupported"
        assert not h.port.submissions, "nothing may be submitted for a turn the model cannot answer"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_conversation_items_are_consumed_by_the_turn_they_start() -> None:
    """The items belong to the turn that answered them, not to the next one.

    Without this, a session that once received an item would let every later
    ``response.create`` start an empty turn off the same stale input.
    """
    h = await open_harness(auto_response=False)
    try:
        await h.run(
            commands.CreateItem(
                item={
                    "id": "item_hi",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            )
        )
        assert h.session.unanswered_user_items() == 1
        # Consumed by the response.create that refused them: a later one must
        # not see the same stale input and refuse for a second time.
        await h.run(commands.CreateResponse(event_id="evt-first"))
        assert h.session.unanswered_user_items() == 0
        events = await h.run(commands.CreateResponse(event_id="evt-second"))
        assert events[0].code == "response_create_without_input"
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Session update and conversation items                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_session_update_rejects_instruction_changes_after_the_context_is_locked() -> None:
    h = await open_harness()
    try:
        await h.run(append_audio())
        assert h.runner.model_state.context_locked
        events = await h.run(commands.UpdateSession(patch={"instructions": "changed"}, event_id="evt-upd"))
        assert types(events) == ["error"]
        assert events[0].code == "instructions_update_unsupported"
        assert events[0].related_event_id == "evt-upd"
        assert h.session.config.instructions == "You are a concise assistant."
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_session_update_replaces_config_and_runtime_config_together() -> None:
    h = await open_harness()
    try:
        generation = h.session.config_generation
        events = await h.run(commands.UpdateSession(patch={"temperature": 0.3}))
        assert types(events) == ["session.updated"]
        assert events[0].session["temperature"] == 0.3
        assert h.session.config.temperature == 0.3
        assert h.session.runtime_config["duplex_stage_sampling_params"]["0"]["temperature"] == 0.3
        assert h.session.config_generation == generation + 2
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_conversation_items_can_be_injected_and_deleted() -> None:
    h = await open_harness()
    try:
        item = {"id": "item_hist", "type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        events = await h.run(commands.CreateItem(item=item))
        assert types(events) == ["conversation.item.added", "conversation.item.created", "conversation.item.done"]
        assert h.session.history == ({"role": "user", "content": "hi"},)

        events = await h.run(commands.DeleteItem(item_id="item_hist"))
        assert types(events) == ["conversation.item.deleted"]
        assert h.session.history == ()

        events = await h.run(commands.DeleteItem(item_id="item_missing", event_id="evt-del"))
        assert types(events) == ["error"]
        assert events[0].code == "item_not_found"
    finally:
        await close_harness(h)


def _stage_metrics_of(event: DuplexEvent) -> dict[str, dict[str, object]]:
    """Per-stage engine metrics as the client reads them off one wire event."""
    payload = event.to_realtime()
    metadata = payload.get("metadata")
    assert isinstance(metadata, dict), payload
    vllm_omni = metadata.get("vllm_omni")
    assert isinstance(vllm_omni, dict), metadata
    stage_metrics = vllm_omni.get("stage_metrics")
    assert isinstance(stage_metrics, dict), vllm_omni
    return stage_metrics


@pytest.mark.asyncio
async def test_stage0_metrics_reach_the_response_even_though_its_output_feeds_tts() -> None:
    """A pass-through Stage0 output still has to report its tokens.

    Stage0 text with no decision is forwarded to the TTS stage rather than
    consumed, and before sessions moved into the engine the orchestrator
    published its metrics separately. Returning ``False`` here must not also
    throw the metrics away, or every response reports no engine token count and
    the benchmark cannot compute TPOT.
    """
    h = await open_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        text_output = SimpleNamespace(
            request_id=request_id,
            finished=False,
            outputs=[SimpleNamespace(text="hi", token_ids=[11], multimodal_output={})],
            multimodal_output={},
        )
        forwarded = h.deliver(
            text_output,
            stage_id=0,
            metrics=stage_stats(stage_id=0, request_id=request_id, num_tokens_out=3, itls_ms=[9.0, 11.0]),
        )
        assert forwarded is False
        assert await h.settle() == []

        events = await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hi"))
        stage_metrics = _stage_metrics_of(find(events, "response.output_audio.delta"))
        assert stage_metrics["0"]["num_tokens_out"] == 3
        assert stage_metrics["0"]["vllm_itls_ms"] == [9.0, 11.0]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_projected_stage1_metrics_reach_the_spoken_audio_event() -> None:
    """Thinker text is projected, but its TTFT/TPOT still have to ride the audio event."""
    h = await open_harness(stage_count=3)
    try:
        plugin = h.runner.model._ctx.plugin

        def project_stage1(*, stage_id: int, output: object, context: object) -> bool:
            del output, context
            return stage_id == 1

        plugin.project_intermediate_output = project_stage1
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        thinker = SimpleNamespace(
            request_id=request_id,
            finished=True,
            outputs=[SimpleNamespace(text="hello", token_ids=[7, 8], multimodal_output={})],
            multimodal_output={},
        )
        forwarded = h.deliver(
            thinker,
            stage_id=1,
            segment_finished=False,
            metrics=stage_stats(stage_id=1, request_id=request_id, num_tokens_out=2, itls_ms=[6.0]),
        )
        assert forwarded is False
        events = await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"), stage_id=2)
        stage_metrics = _stage_metrics_of(find(events, "response.output_audio.delta"))
        assert stage_metrics["1"]["num_tokens_out"] == 2
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_stage0_metrics_from_several_units_are_summed_into_one_response() -> None:
    """Two pass-through segments before the first audio are one response's tokens."""
    h = await open_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        for tokens in (3, 4):
            h.deliver(
                SimpleNamespace(
                    request_id=request_id,
                    finished=False,
                    outputs=[SimpleNamespace(text="hi", token_ids=[11], multimodal_output={})],
                    multimodal_output={},
                ),
                stage_id=0,
                metrics=stage_stats(stage_id=0, request_id=request_id, num_tokens_out=tokens),
            )
        assert await h.settle() == []

        events = await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hi"))
        stage_metrics = _stage_metrics_of(find(events, "response.output_audio.delta"))
        assert stage_metrics["0"]["num_tokens_out"] == 7
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_response_done_logs_orchestrator_stage_table(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[OrchestratorAggregator] = []

    def _capture(self: OrchestratorAggregator) -> dict[str, object]:
        logged.append(self)
        return {}

    monkeypatch.setattr(OrchestratorAggregator, "build_and_log_summary", _capture)
    h = await open_harness(log_stats=True)
    try:
        assert h.session.log_stats is True
        assert h.session.num_stages == 2
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        h.deliver(
            SimpleNamespace(
                request_id=request_id,
                finished=False,
                outputs=[SimpleNamespace(text="hi", token_ids=[11], multimodal_output={})],
                multimodal_output={},
            ),
            stage_id=0,
            metrics=stage_stats(stage_id=0, request_id=request_id, num_tokens_out=3),
        )
        assert await h.settle() == []

        events = await h.deliver_and_settle(
            tts_output(request_id, samples=24000, text="hi", turn_end=True),
            metrics=stage_stats(stage_id=1, request_id=request_id, num_tokens_out=4),
        )
        done = find(events, "response.done")
        assert done.status == "completed"
        assert len(logged) == 1
        aggregator = logged[0]
        assert aggregator.num_stages == 2
        stage_ids = [event.stage_id for event in aggregator.stage_events[done.response_id]]
        assert stage_ids == [0, 1]
        assert done.response_id in aggregator.e2e_done
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_duplex_stage_request_stamps_wall_clock_request_timestamp() -> None:
    """serving_time_to_first_output_ms is (first_output_ts - request_timestamp)*1000.

    Duplex used to leave request_timestamp at 0, so the table printed unix_ts*1000.
    """
    from tests.engine.test_duplex_orchestrator import (
        SESSION_ID as ORCH_SESSION_ID,
    )
    from tests.engine.test_duplex_orchestrator import (
        _build,
        _close,
        _open,
        _stage0_request_id,
    )
    from vllm_omni.engine.duplex_orchestrator import DuplexOrchestratorRequestState

    orchestrator, _, rpc_q, _ = _build()
    try:
        result = await _open(orchestrator, rpc_q)
        assert result.ok
        state = orchestrator.request_states[_stage0_request_id()]
        assert isinstance(state, DuplexOrchestratorRequestState)
        assert state.request_timestamp > 1_000_000_000.0
    finally:
        if ORCH_SESSION_ID in orchestrator.session_manager.runners:
            await _close(orchestrator, rpc_q)
        await orchestrator.session_manager.shutdown()


def _response_request_metrics_of(event: object) -> dict[str, object]:
    """Server request-start clocks as the client reads them off one wire event."""
    payload = event.to_realtime()
    metadata = payload.get("metadata")
    assert isinstance(metadata, dict), payload
    vllm_omni = metadata.get("vllm_omni")
    assert isinstance(vllm_omni, dict), metadata
    metrics = vllm_omni.get("response_request_metrics")
    assert isinstance(metrics, dict), vllm_omni
    return metrics


@pytest.mark.asyncio
async def test_first_audio_delta_carries_server_request_start_metrics() -> None:
    clock = {"now": 1000.0}
    h = await open_harness(clock=lambda: clock["now"])
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        clock["now"] = 1001.3
        events = await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hi"))
        metrics = _response_request_metrics_of(find(events, "response.output_audio.delta"))
        assert metrics["source"] == "server_monotonic_request_start"
        assert metrics["measurement_origin"] == dict(RESPONSE_REQUEST_MEASUREMENT_ORIGIN)
        assert metrics["ttft_ms"] == pytest.approx(1300.0)
        assert metrics["ttfp_ms"] == pytest.approx(1300.0)
        speak_metrics = _response_request_metrics_of(find(events, "response.speak"))
        assert speak_metrics["ttft_ms"] == pytest.approx(1300.0)
        assert speak_metrics["ttfp_ms"] == pytest.approx(1300.0)

        clock["now"] = 1002.0
        later = await h.deliver_and_settle(tts_output(request_id, samples=48000, text="hello"))
        later_payload = find(later, "response.output_audio.delta").to_realtime()
        later_metadata = later_payload.get("metadata")
        later_vllm_omni = later_metadata.get("vllm_omni") if isinstance(later_metadata, dict) else None
        later_metrics = later_vllm_omni.get("response_request_metrics") if isinstance(later_vllm_omni, dict) else None
        assert later_metrics is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_stale_epoch_append_does_not_own_request_start() -> None:
    clock = {"now": 1000.0}
    h = await open_harness(clock=lambda: clock["now"])
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        clock["now"] = 1000.5
        append_ok, emitted = await h.runner.model.append_runtime_input(
            {"duplex_turn_id": 0},
            final=False,
            expected_epoch=h.session.epoch + 1,
        )
        assert append_ok is True
        assert emitted is False
        clock["now"] = 1001.3
        events = await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hi"))
        metrics = _response_request_metrics_of(find(events, "response.output_audio.delta"))
        assert metrics["ttft_ms"] == pytest.approx(1300.0)
        assert metrics["ttfp_ms"] == pytest.approx(1300.0)
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Server VAD                                                                  #
# --------------------------------------------------------------------------- #


class _ScriptedDetector:
    """A turn detector that answers each appended chunk with a pre-decided result."""

    def __init__(self, results: list[object]) -> None:
        self._results = list(results)
        self.resets = 0

    def process(self, base64_audio: str, *, fmt: str, sample_rate_hz: int | None, audio_end_ms: int | None = None):
        del base64_audio, fmt, sample_rate_hz, audio_end_ms
        return self._results.pop(0)

    def reset(self) -> None:
        self.resets += 1


def _speech_then_stop() -> _ScriptedDetector:
    from vllm_omni.engine.duplex.turn_detection import TurnDetectionResult

    return _ScriptedDetector(
        [
            TurnDetectionResult(is_speech=True, speech_active=True, speech_started=True, speech_probability=0.9),
            TurnDetectionResult(
                is_speech=False,
                speech_active=False,
                speech_stopped=True,
                speech_probability=0.05,
                should_commit=True,
                create_response=True,
            ),
        ]
    )


def _final_submissions(h: Harness) -> list[object]:
    return [s for s in h.port.submissions if s.prompt["model_intermediate_buffer"]["duplex"].get("final") is True]


@pytest.mark.asyncio
async def test_server_vad_speech_stopped_does_not_end_the_turn_of_an_auto_response_session() -> None:
    """A model-native session keeps its turn open past the detector's stop.

    Committing at ``speech_stopped`` cut the utterance short: the model
    listened on that early final unit, the trailing silence became a second,
    near-empty turn, and the client's own commit answered nothing (the
    server-VAD hard-interrupt E2E timed out waiting for the follow-up
    response). The old translator synthesized that commit for turn-based
    server VAD only.
    """
    h = await open_harness()
    try:
        h.runner.control._detector = _speech_then_stop()
        first = await h.run(append_audio())
        assert "input_audio_buffer.speech_started" in types(first)
        # A partial unit, like the trailing silence of a real utterance: it is
        # what a commit here would submit as the turn's final append.
        second = await h.run(append_audio(samples=8000, is_speech=False, value=0.0))
        assert "input_audio_buffer.speech_stopped" in types(second)
        assert "input_audio_buffer.committed" not in types(first) + types(second)
        assert _final_submissions(h) == [], "the detector must not close a model-native turn"

        events = await h.run(commands.Commit())
        assert types(events).count("input_audio_buffer.committed") == 1
        assert len(_final_submissions(h)) == 1, "the client's commit closes the turn exactly once"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_server_vad_speech_stopped_still_commits_a_turn_mode_session() -> None:
    """Turn-based server VAD keeps the contract the old translator implemented."""
    h = await open_harness(auto_response=False)
    try:
        h.runner.control._detector = _speech_then_stop()
        await h.run(append_audio())
        events = await h.run(append_audio(is_speech=False, value=0.0))
        assert "input_audio_buffer.speech_stopped" in types(events)
        assert "input_audio_buffer.committed" in types(events)
        assert len(_final_submissions(h)) == 1, "the detector's stop commits the turn and starts the response"
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Cancelled append and its precreated response (#7636 Issue 2)                #
# --------------------------------------------------------------------------- #


async def _commit_with_pending_response(h: Harness) -> str:
    """Commit a turn whose final append parks in the stage port; return the precreated response id."""
    await h.run(append_audio())
    h.port.submit_gate = asyncio.Event()
    h.submit(commands.Commit(create_response=True))
    await asyncio.wait_for(h.port.submit_started.wait(), timeout=2.0)
    # The parked append keeps the runner "busy": settle only until the mailbox is quiet.
    events = await h.settle(timeout_s=0.3)
    assert "response.created" in types(events)
    response_id = h.session.active_response_id
    assert response_id is not None
    assert h.runner.tasks.append_tail is not None and not h.runner.tasks.append_tail.done()
    return response_id


@pytest.mark.asyncio
async def test_an_append_cancelled_from_outside_fails_the_response_it_precreated() -> None:
    """A cancelled append gives back the response it precreated, as its docstring promises.

    The ``CancelledError`` branch used to roll back only the PCM reservation,
    so ``active_response_id`` stayed pinned to a response nobody would ever
    finish: the client had seen ``response.created`` and waited forever.
    """
    h = await open_harness(auto_response=False)
    try:
        response_id = await _commit_with_pending_response(h)

        pending = h.runner.tasks.append_tail
        assert pending is not None
        pending.cancel()
        events = await h.settle()

        done = find(events, "response.done")
        assert done.response_id == response_id
        assert done.status == "failed"
        assert done.response["status_details"]["reason"] == "append_cancelled"
        assert h.session.active_response_id is None
        assert h.session.state == DuplexSessionState.OPEN

        # The cancelled append counts as a failed one: the chain behind it is
        # refused with the documented error, not left hanging.
        h.port.submit_gate = None
        await h.run(append_audio())
        events = await h.run(commands.Commit(create_response=True))
        assert "response.created" not in types(events)
        assert find(events, "error").code == "commit_aborted"
        assert h.session.active_response_id is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_response_cancel_of_a_pending_append_ends_the_response_once_as_cancelled() -> None:
    """When the runner cancels the append itself, it owns the response: one terminal, status cancelled.

    The append must not also fail the response on its way out, or the client
    would get a failed ``response.done`` for a response it cancelled.
    """
    h = await open_harness(auto_response=False)
    try:
        response_id = await _commit_with_pending_response(h)

        events = await h.run(commands.CancelResponse())

        terminals = [event for event in events if event.type == "response.done"]
        assert [event.response_id for event in terminals] == [response_id]
        assert terminals[0].status == "cancelled"
        assert h.session.active_response_id is None
        assert h.session.state == DuplexSessionState.OPEN
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_closing_the_session_with_a_pending_append_emits_no_failed_response() -> None:
    """A close ends the active response itself; the append it cancels stays quiet."""
    h = await open_harness(auto_response=False)
    try:
        await _commit_with_pending_response(h)

        events = await h.run(commands.CloseSession(reason="client_close"))
        events += await h.settle()

        assert [event.type for event in events if event.type == "response.done"] == []
        assert "session.closed" in types(events)
        assert h.session.active_response_id is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_input_clear_racing_a_queued_append_drops_it_as_the_clients_doing() -> None:
    """An ``input_audio_buffer.clear`` that lands while an append waits its turn is not a runtime failure.

    The clear deactivates the queued append's reservation; when its turn
    comes it gives everything back and stops, without an error and without
    failing the session, and its reason is the client's clear rather than
    ``runtime_append_failed``.
    """
    h = await open_harness()
    try:
        h.port.submit_gate = asyncio.Event()
        h.submit(append_audio())
        await asyncio.wait_for(h.port.submit_started.wait(), timeout=2.0)
        h.submit(append_audio())
        await h.settle(timeout_s=0.3)
        assert len(h.runner.tasks.append_tasks) == 2, "the second append is queued behind the parked one"
        queued = h.runner.tasks.append_tail
        assert queued is not None and not queued.done()

        h.submit(commands.ClearInput())
        events = await h.settle(timeout_s=0.3)
        assert types(events) == ["input_audio_buffer.cleared"]

        h.port.submit_gate.set()
        events = await h.settle()
        assert queued.done() and queued.result() is False, "the cleared append never ran"
        assert len(h.port.submissions) == 1, "only the append that was already in flight reached the stage"
        assert [event.type for event in events if event.type in {"error", "response.done", "session.closed"}] == []
        assert h.session.pending_input_bytes == 0
        assert h.session.state == DuplexSessionState.OPEN
    finally:
        await close_harness(h)
