# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Engine-resident session runner scenarios (the old serving handler tests, re-homed).

The runner is driven exactly the way ``DuplexOrchestrator`` drives it: typed
commands go through ``DuplexSessionManager.dispatch``, stage outputs are pushed
with ``runner.on_stage_output`` and everything the session says is read back
from the session's output buffer as typed events. The stage port is a recording
fake; the model plugin is the real MiniCPM-o 4.5 one so append planning and
output projection are exercised end to end.
"""

from __future__ import annotations

import asyncio
import base64
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.commands import DuplexCommand
from vllm_omni.engine.duplex.config import DuplexSessionConfig, DuplexSessionState
from vllm_omni.engine.duplex.contracts import (
    DuplexOutputContext,
    DuplexRequestIdentity,
    DuplexStagePort,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.events import AudioDelta, DuplexEvent
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    DuplexSessionEventMessage,
    OpenDuplexSessionMessage,
)
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin
from vllm_omni.engine.duplex.session.emitter import SessionEmitter
from vllm_omni.engine.duplex.session.engine_session import RESPONSE_REQUEST_MEASUREMENT_ORIGIN
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner
from vllm_omni.entrypoints.duplex.realtime_input import RealtimeEnvelope
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
        self.abort_started = asyncio.Event()
        self.abort_gate: asyncio.Event | None = None
        self.fail_submit: Exception | None = None
        #: When set, ``submit`` parks on it after signalling ``submit_started``.
        self.submit_gate: asyncio.Event | None = None
        self.submit_started = asyncio.Event()
        self.cleanup_failures_remaining = 0

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
        if self.cleanup_failures_remaining:
            self.cleanup_failures_remaining -= 1
            raise RuntimeError("transient cleanup failure")

    async def abort_requests(self, request_ids: list[str]) -> None:
        self.aborts.append(list(request_ids))
        self.abort_started.set()
        if self.abort_gate is not None:
            await self.abort_gate.wait()


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
    output_buffer: DuplexOutputBuffer
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
            while self.output_buffer.pending_events:
                event = await self.output_buffer.get()
                assert event is not None
                if self.output_buffer.is_valid(event):
                    collected.append(event)
                drained = True
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
    plugin: DuplexModelPlugin | None = None,
) -> Harness:
    if plugin is None:
        plugin = MiniCPMO45DuplexPlugin(_fake_encode_audio)
    port = RecordingStagePort(stage_count=stage_count)
    output: asyncio.Queue[Any] = asyncio.Queue()
    results: asyncio.Queue[Any] = asyncio.Queue()
    limits = runtime_config or DuplexSessionRuntimeConfig()
    output_buffer = DuplexOutputBuffer(
        max_bytes=limits.max_pending_output_bytes_per_session,
        max_events=limits.max_pending_output_events_per_session,
    )
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=limits,
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
    await manager.handle(
        OpenDuplexSessionMessage(
            control_id="c-open", session_id=SESSION_ID, session_config=config, output_buffer=output_buffer
        )
    )
    result = await asyncio.wait_for(results.get(), timeout=2.0)
    assert isinstance(result, DuplexControlResultMessage) and result.ok, result
    harness = Harness(
        manager=manager,
        port=port,
        output=output,
        output_buffer=output_buffer,
        results=results,
        runner=manager.runners[SESSION_ID],
    )
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


def _is_silence_unit(submission: DuplexStageSubmission) -> bool:
    """A continuation unit: the only append whose audio is all zeros."""
    payload = submission.prompt["model_intermediate_buffer"]["duplex"]["payload"]
    return set(base64.b64decode(payload["audio"])) == {0}


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
@pytest.mark.parametrize("camera_update", [False, True])
async def test_aura_asr_transcript_does_not_project_as_assistant_output(camera_update: bool) -> None:
    from io import BytesIO

    from PIL import Image

    from vllm_omni.model_executor.models.aura_omni.duplex.plugin import AuraDuplexPlugin

    plugin = AuraDuplexPlugin(_fake_encode_audio)
    h = await open_harness(plugin=plugin, stage_count=4, auto_response=not camera_update)
    try:
        image = BytesIO()
        Image.new("RGB", (8, 8), "red").save(image, format="PNG")
        await h.run(
            commands.AppendAudio(
                audio=pcm_f32(1600),
                format="pcm_f32le",
                sample_rate_hz=16000,
                is_speech=True,
                video_frames=(b64(image.getvalue()),),
            )
        )
        if camera_update:
            events = await h.run(commands.Commit(final=True, create_response=False))
            speech_item_id = find(events, "input_audio_buffer.committed").to_realtime()["item_id"]
            assert not h.port.submissions
            image = BytesIO()
            Image.new("RGB", (8, 8), "blue").save(image, format="PNG")
            await h.run(commands.AppendAudio(video_frames=(b64(image.getvalue()),)))
        await h.run(commands.Commit(final=True, create_response=True))
        submission = h.port.submissions[-1]
        if camera_update:
            info = submission.prompt["additional_information"]
            assert info["realtime_item_id"] == speech_item_id
            audio, rate = submission.prompt["multi_modal_data"]["audio"]
            assert rate == 16000 and audio.tobytes() == pcm_f32(1600)
            video = info["deferred_multi_modal_data"]["video"][0][0]
            assert [tuple(frame[0, 0]) for frame in video] == [(255, 0, 0), (0, 0, 255)]
        request_id = submission.context.request_id
        output = SimpleNamespace(
            request_id=request_id,
            finished=True,
            outputs=[SimpleNamespace(text="Watch the red light.", token_ids=[], multimodal_output={})],
            multimodal_output={},
        )
        transcript = plugin.user_transcript(stage_id=0, output=output, prompt=submission.prompt, finished=True)
        context = DuplexOutputContext(
            identity=DuplexRequestIdentity(session_id=SESSION_ID, fence=submission.context.fence),
            final_stage_id=3,
            segment_finished=True,
        )
        assert not h.runner.on_stage_output(
            0,
            output,
            stage_stats(stage_id=0, request_id=request_id, num_tokens_out=5),
            request_id=request_id,
            context=context,
            input_transcript=transcript,
            input_item_id=submission.prompt["additional_information"]["realtime_item_id"],
        )
        events = await h.settle()
        transcription = find(events, "conversation.item.input_audio_transcription.completed")
        assert transcription.transcript == transcript
        if camera_update:
            assert transcription.to_realtime()["item_id"] == speech_item_id
            assert h.session.pending_input_bytes == 0
        assert not any(event.type.startswith("response.output_") for event in events)
        assert (h.session.active_response_id is not None) == camera_update

        output.outputs[0].text = "I will watch it."
        assert not h.deliver(output, stage_id=1, segment_finished=True)
        events = await h.settle()
        delta = find(events, "response.output_audio_transcript.delta")
        assert delta.delta == "I will watch it."
        # AURA starts the response on thinker output; earlier ASR metrics are
        # stashed until then and must appear on the public transcript event.
        assert _stage_metrics_of(delta)["0"]["num_tokens_out"] == 5
    finally:
        await h.manager.shutdown()


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
async def test_close_cancels_an_inflight_response_and_aborts_its_request() -> None:
    saw_cancel = asyncio.Event()

    async def inflight() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            saw_cancel.set()
            raise

    h = await open_harness()
    task = asyncio.create_task(inflight())
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
        h.runner.tasks.active_response_task = task
        await h.manager.handle(
            CloseDuplexSessionMessage(control_id="c-close-inflight", session_id=SESSION_ID, reason="client_close")
        )
        await asyncio.wait_for(h.results.get(), timeout=2.0)
        await h.settle()
        assert saw_cancel.is_set()
        assert any(request_id in batch for batch in h.port.aborts)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


@pytest.mark.asyncio
async def test_close_skips_retirement_after_the_ledger_is_closed() -> None:
    h = await open_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
        asked: list[str] = []

        def explode(config, current):
            del config, current
            asked.append("retire")
            raise RuntimeError("the plugin cannot prove its queue is clean")

        h.runner.plugin.retire_pending_function_output = explode
        epoch_before = h.runner.session.epoch
        await h.manager.handle(
            CloseDuplexSessionMessage(control_id="c-close-retire", session_id=SESSION_ID, reason="client_close")
        )
        result = await asyncio.wait_for(h.results.get(), timeout=2.0)
        events = await h.settle()

        assert result.ok and result.operation == "close"
        assert h.runner.session.epoch > epoch_before
        assert asked == []
        assert "tool_result_retirement_failed" not in [getattr(event, "code", None) for event in events]
        assert "session.closed" in types(events)
        assert any(request_id in batch for batch in h.port.aborts)
    finally:
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
async def test_manual_vision_frames_honor_the_pending_byte_limit() -> None:
    """Frame-only units have no PCM, but the base64 is still pending input."""
    frame_a = "a" * 80
    frame_b = "b" * 80
    h = await open_harness(
        auto_response=False,
        runtime_config=DuplexSessionRuntimeConfig(max_pending_input_bytes_per_session=100),
    )
    try:
        await h.run(commands.AppendAudio(format="pcm_f32le", sample_rate_hz=16000, video_frames=(frame_a,)))
        assert h.session.pending_input_bytes == len(frame_a)
        assert h.runner.model_state.audio_buffer.pending_byte_count == len(frame_a)

        events = await h.run(commands.AppendAudio(format="pcm_f32le", sample_rate_hz=16000, video_frames=(frame_b,)))
        assert types(events) == ["error"]
        assert events[0].code == "input_backpressure"
        assert h.session.pending_input_bytes == len(frame_a)
        assert [item["video_frames"] for item in h.runner.model_state.audio_buffer._vision_queue] == [[frame_a]]

        await h.run(commands.ClearInput())
        assert h.session.pending_input_bytes == 0
        assert not h.runner.model_state.audio_buffer.has_pending()

        await h.run(commands.AppendAudio(format="pcm_f32le", sample_rate_hz=16000, video_frames=(frame_a,)))
        await h.run(commands.Commit(create_response=True))
        assert h.session.pending_input_bytes == 0
        assert not h.runner.model_state.audio_buffer.has_pending()
        assert len(h.port.submissions) == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("discard", ["clear", "cancel", "promote"])
async def test_deferred_vision_prefixes_stay_charged_until_discard(discard: str) -> None:
    """Held VISION prefixes still count toward the pending-byte limit."""
    frame = "a" * 40
    h = await open_harness(
        auto_response=False,
        runtime_config=DuplexSessionRuntimeConfig(max_pending_input_bytes_per_session=100),
    )
    try:
        await h.run(append_audio(samples=8))
        await h.run(commands.Commit(create_response=True))
        request_id = h.port.submissions[0].context.request_id
        response_id = h.session.active_response_id
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="sure"))
        await h.run(commands.AckPlayback(response_id=response_id, played_ms=1000))

        for _ in range(2):
            await h.run(commands.AppendAudio(format="pcm_f32le", sample_rate_hz=16000, video_frames=(frame,)))
        assert h.session.pending_input_bytes == 80
        await h.run(commands.Commit(create_response=False))
        assert h.session.pending_input_bytes == 80
        assert h.runner.model_state.commit_prefix_reserved_bytes == [40]

        rejected = await h.run(commands.AppendAudio(format="pcm_f32le", sample_rate_hz=16000, video_frames=(frame,)))
        assert types(rejected) == ["error"]
        assert rejected[0].code == "input_backpressure"
        assert h.session.pending_input_bytes == 80

        if discard == "clear":
            await h.run(commands.ClearInput())
        elif discard == "cancel":
            await h.run(commands.CancelResponse())
        else:
            await h.deliver_and_settle(tts_output(request_id, samples=0, text="", finished=True))
            await h.run(commands.CreateResponse())
        assert h.session.pending_input_bytes == 0
        assert not h.runner.model_state.commit_prefix_payloads
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
@pytest.mark.parametrize("phase", ["decode", "vad"])
@pytest.mark.parametrize("fails", [False, True])
async def test_audio_admission_is_held_until_preprocessing_finishes(monkeypatch, phase, fails) -> None:
    h = await open_harness(
        auto_response=False,
        runtime_config=DuplexSessionRuntimeConfig(max_pending_input_bytes_per_session=64_000),
    )
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    owner = h.runner if phase == "decode" else h.runner.control
    method = "offload" if phase == "decode" else "run_turn_detection"
    original = getattr(owner, method)
    on_command = h.runner._on_command

    async def track_command(command):
        try:
            await on_command(command)
        finally:
            finished.set()

    async def blocked(*args, **kwargs):
        started.set()
        await release.wait()
        if fails:
            raise ValueError("preprocessing failed")
        return await original(*args, **kwargs)

    monkeypatch.setattr(owner, method, blocked)
    monkeypatch.setattr(h.runner, "_on_command", track_command)
    try:
        h.submit(append_audio(samples=8000))
        await asyncio.wait_for(started.wait(), timeout=2.0)
        assert h.session.pending_input_bytes == 32_000
        h.submit(append_audio(samples=10_000, event_id="over-limit"))
        events = await h.settle()
        assert any(
            event.type == "error" and event.code == "input_backpressure" and event.related_event_id == "over-limit"
            for event in events
        )

        release.set()
        await asyncio.wait_for(finished.wait(), timeout=2.0)
        events = await h.settle()
        assert h.session.pending_input_bytes == (0 if fails else 32_000)
        assert h.session._input.queued_input_bytes == 0
        if fails:
            assert any(event.type == "error" and "preprocessing failed" in event.message for event in events)
        else:
            assert h.runner.model_state.audio_buffer.pending_byte_count == 32_000
        await h.run(commands.ClearInput())
        assert h.session.pending_input_bytes == 0
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
async def test_clear_input_preserves_queued_append_byte_reservations(monkeypatch, committed) -> None:
    h = await open_harness(
        auto_response=False,
        runtime_config=DuplexSessionRuntimeConfig(max_pending_input_bytes_per_session=64_000),
    )
    blocked = asyncio.Event()
    resume = asyncio.Event()
    on_command = h.runner._on_command

    async def pause_queued_append(command):
        if command.event_id == "queued":
            blocked.set()
            await resume.wait()
        await on_command(command)

    try:
        await h.run(append_audio(samples=8000))
        if committed:
            await h.run(commands.Commit(create_response=False))
            assert h.runner.model_state.committed_audio_reserved_bytes == 32_000
        assert h.session.pending_input_bytes == 32_000
        monkeypatch.setattr(h.runner, "_on_command", pause_queued_append)
        h.submit(commands.ClearInput())
        h.submit(append_audio(samples=8000, event_id="queued"))
        assert h.session.pending_input_bytes == 64_000
        await asyncio.wait_for(blocked.wait(), timeout=2.0)

        # Only the old buffer was cleared; the later append still owns its budget.
        assert h.session.pending_input_bytes == 32_000
        assert not h.runner.model_state.audio_buffer.has_pending()
        assert h.runner.model_state.committed_audio_reserved_bytes == 0
        h.submit(append_audio(samples=10_000, event_id="over-limit"))
        events = await h.settle()
        assert any(
            event.type == "error" and event.code == "input_backpressure" and event.related_event_id == "over-limit"
            for event in events
        )

        resume.set()
        await h.settle()
        assert h.session.pending_input_bytes == 32_000
        assert h.runner.model_state.audio_buffer.pending_byte_count == 32_000
        await h.run(commands.ClearInput())
        assert h.session.pending_input_bytes == 0
    finally:
        resume.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_cleared_in_flight_append_cannot_release_later_audio(monkeypatch) -> None:
    h = await open_harness()
    started = asyncio.Event()
    release = asyncio.Event()
    processed = asyncio.Event()
    submit = h.port.submit
    on_command = h.runner._on_command

    async def blocked_submit(submission):
        started.set()
        await release.wait()
        return await submit(submission)

    async def observe_command(command):
        await on_command(command)
        if command.event_id == "later":
            processed.set()

    monkeypatch.setattr(h.port, "submit", blocked_submit)
    monkeypatch.setattr(h.runner, "_on_command", observe_command)
    try:
        h.submit(append_audio())
        await asyncio.wait_for(started.wait(), timeout=2.0)
        assert h.session.pending_input_bytes == 64_000
        h.submit(commands.ClearInput())
        h.submit(append_audio(samples=8000, event_id="later"))
        await asyncio.wait_for(processed.wait(), timeout=2.0)
        assert h.session.pending_input_bytes == 32_000
        release.set()
        await h.settle()
        assert h.session.pending_input_bytes == 32_000
        assert h.runner.model_state.audio_buffer.pending_byte_count == 32_000
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_append_chained_behind_a_cleared_append_still_submits() -> None:
    h = await open_harness()
    h.port.submit_gate = asyncio.Event()
    try:
        before = len(h.port.submissions)
        h.submit(append_audio(event_id="in-flight"))
        await asyncio.wait_for(h.port.submit_started.wait(), timeout=2.0)
        h.submit(append_audio(event_id="cleared"))
        h.submit(commands.ClearInput())
        h.submit(append_audio(event_id="after-clear"))
        for _ in range(200):
            if h.runner._mailbox.empty() and len(h.runner.tasks.append_tasks) == 3:
                break
            await asyncio.sleep(0.005)
        assert len(h.runner.tasks.append_tasks) == 3

        h.port.submit_gate.set()
        await h.settle()
        # The cleared append is called off; the one behind it is not rolled back.
        assert len(h.port.submissions) == before + 2
        assert not h.runner.model_state.audio_buffer.has_pending()
        assert h.session.pending_input_bytes == 0
    finally:
        h.port.submit_gate.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_admitted_pcm16_append_that_decodes_past_the_limit_is_rejected_cleanly() -> None:
    h = await open_harness(
        auto_response=False,
        runtime_config=DuplexSessionRuntimeConfig(max_pending_input_bytes_per_session=40_000),
    )
    try:
        pcm16 = np.full(16000, 1600, dtype="<i2").tobytes()  # 32_000 wire bytes, 64_000 as f32
        events = await h.run(commands.AppendAudio(audio=pcm16, format="pcm16", sample_rate_hz=16000, is_speech=True))
        # Admission passes on wire bytes; the PCM handoff enforces the limit on the decoded size.
        assert any(event.type == "error" and event.code == "input_backpressure" for event in events)
        assert not h.runner.model_state.audio_buffer.has_pending()
        assert not h.runner.model_state.input_since_commit
        assert h.session.pending_input_bytes == 0
        # Nothing stays charged, so audio that fits is still accepted.
        await h.run(append_audio(samples=8000))
        assert h.session.pending_input_bytes == h.runner.model_state.audio_buffer.pending_byte_count == 32_000
    finally:
        await close_harness(h)


class _SplitDetector:
    """Splits every append into one 1 s segment and its remainder."""

    def process_segments(self, base64_audio: str, *, fmt: str, sample_rate_hz: int | None):
        from vllm_omni.engine.duplex.turn_detection import TurnDetectionResult, TurnDetectionSegment

        del fmt, sample_rate_hz
        audio = base64.b64decode(base64_audio)
        result = TurnDetectionResult(is_speech=True, speech_active=True)
        return [
            TurnDetectionSegment(base64.b64encode(part).decode(), result) for part in (audio[:64_000], audio[64_000:])
        ]

    def reset(self) -> None:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_vad_split_append_keeps_unprocessed_segments_charged(monkeypatch, fails) -> None:
    h = await open_harness(runtime_config=DuplexSessionRuntimeConfig(max_pending_input_bytes_per_session=120_000))
    h.runner.control._detector = _SplitDetector()
    h.port.submit_gate = asyncio.Event()
    handed_off = asyncio.Event()
    resume = asyncio.Event()
    start_append = h.runner._start_append

    async def pause_after_first_handoff(*args, **kwargs):
        task = await start_append(*args, **kwargs)
        if not handed_off.is_set():
            handed_off.set()
            await resume.wait()
            if fails:
                raise RuntimeError("append failed after handoff")
        return task

    monkeypatch.setattr(h.runner, "_start_append", pause_after_first_handoff)
    try:
        # 96_000 bytes: the first 1 s segment is handed off and its stage
        # submission stays pending; the 32_000-byte tail is not processed yet.
        h.submit(append_audio(samples=24_000))
        await asyncio.wait_for(handed_off.wait(), timeout=2.0)
        assert h.session.pending_input_bytes == 96_000
        # Fits only if the unprocessed tail is forgotten.
        h.submit(append_audio(samples=8000, event_id="over-limit"))
        events = await h.settle(timeout_s=0.2)
        assert any(
            event.type == "error" and event.code == "input_backpressure" and event.related_event_id == "over-limit"
            for event in events
        )

        resume.set()
        h.port.submit_gate.set()
        await h.settle()
        # Every charge is released once: the tail is buffered, or released on failure.
        assert h.session._input.queued_input_bytes == 0
        assert h.session.pending_input_bytes == (0 if fails else 32_000)
        await h.run(commands.ClearInput())
        assert h.session.pending_input_bytes == 0
    finally:
        resume.set()
        h.port.submit_gate.set()
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize(("limit", "kept"), [(80_000, 64_000), (48_000, 32_000)])
async def test_vad_split_append_past_the_limit_matches_main(limit, kept) -> None:
    h = await open_harness(
        auto_response=False,
        runtime_config=DuplexSessionRuntimeConfig(max_pending_input_bytes_per_session=limit),
    )
    h.runner.control._detector = _SplitDetector()
    try:
        pcm16 = np.full(24_000, 1600, dtype="<i2").tobytes()  # 48_000 wire bytes, 64_000 + 32_000 as f32
        events = await h.run(commands.AppendAudio(audio=pcm16, format="pcm16", sample_rate_hz=16000, is_speech=True))
        # Each segment is checked as on main: the still-queued tail never blocks an
        # earlier segment, so only the segment that does not fit is rejected.
        assert [event.code for event in events if event.type == "error"] == ["input_backpressure"]
        assert h.session.pending_input_bytes == h.runner.model_state.audio_buffer.pending_byte_count == kept
        assert h.session._input.queued_input_bytes == 0
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
async def test_auto_response_commit_releases_residual_audio_before_the_next_turn() -> None:
    h = await open_harness(extra_body={"force_listen_count": 0})
    try:
        for turn in range(2):
            # Match streaming input with a partial final model unit. Give the
            # turns different PCM so replaying the first tail cannot go unnoticed.
            value = 0.05 + turn * 0.05
            for _ in range(7):
                await h.run(append_audio(3200, value=value))
            await h.run(commands.Commit())

            assert h.runner.model_state.committed_audio_payload is None
            assert h.session.pending_input_bytes == 0
            assert len(h.port.submissions) == (turn + 1) * 2
            streaming_append = h.port.submissions[-2].prompt["model_intermediate_buffer"]["duplex"]
            assert streaming_append["final"] is False
            assert base64.b64decode(streaming_append["payload"]["audio"]) == pcm_f32(16000, value=value)
            final_append = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]
            assert final_append["final"] is True
            assert base64.b64decode(final_append["payload"]["audio"]) == (
                pcm_f32(6400, value=value) + pcm_f32(9600, value=0.0)
            )

            samples = (turn + 1) * 24000
            await h.deliver_and_settle(tts_output(h.stage0_request_id(), samples=samples, turn_id=turn))
            events = await h.deliver_and_settle(
                tts_output(h.stage0_request_id(), samples=samples, turn_end=True, turn_id=turn)
            )
            done = find(events, "response.done")
            assert done.status == "completed"
            ack = await h.run(commands.AckPlayback(played_ms=1000, response_id=done.response_id))
            assert types(ack) == ["playback.acknowledged"]
            assert ack[0].details["history_committed"] is True
            assert ack[0].details["playback"]["committed_ms"] == 1000
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
        # Ack-only playback: the assistant text enters history on playback ack.
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
    release_abort = asyncio.Event()
    h.port.abort_gate = release_abort
    try:
        # Admit the older response just as the model channel does before its
        # first output; it remains live while the newer response is generated.
        h.runner.emit({"type": "response.created", "response_id": "resp-old"})
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
        h.manager.emit(
            h.session,
            [
                AudioDelta(response_id="resp-old", delta="AAAA"),
                AudioDelta(response_id="resp-old", delta="BBBB"),
                AudioDelta(response_id=h.session.active_response_id, delta="CCCC"),
            ],
        )
        held = await h.output_buffer.get()
        assert held is not None and h.output_buffer.is_valid(held)
        h.submit(commands.BargeIn())
        await asyncio.wait_for(h.port.abort_started.wait(), timeout=2.0)
        # Both queued audio and an event held by the consumer become stale
        # before the stage abort can yield, including the older draining turn.
        assert not h.output_buffer.is_valid(held)
        assert h.output_buffer.pending_events == 0
        release_abort.set()
        events = await h.settle()
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
        release_abort.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_close_invalidates_draining_audio_before_stage_abort_finishes() -> None:
    h = await open_harness()
    release_abort = asyncio.Event()
    h.port.abort_gate = release_abort
    try:
        h.session.bind_draining_request("duplex-drain-tts", "resp-old")
        h.manager.emit(
            h.session,
            [AudioDelta(response_id="resp-old", delta="AAAA"), AudioDelta(response_id="resp-old", delta="BBBB")],
        )
        held = await h.output_buffer.get()
        assert held is not None and h.output_buffer.is_valid(held)
        h.submit(commands.CloseSession())
        await asyncio.wait_for(h.port.abort_started.wait(), timeout=2.0)
        # Session close suppresses audio.cancelled notifications; it must still
        # invalidate the pending output without relying on those later events.
        assert not h.output_buffer.is_valid(held)
        assert h.output_buffer.pending_events == 0
        assert h.port.aborts == [["duplex-drain-tts"]]
        release_abort.set()
        events = await h.settle()
        assert "session.closed" in types(events)
        assert h.manager.active_count() == 0
    finally:
        release_abort.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_cancel_cleanup_failure_retains_requests_for_close_retry() -> None:
    h = await open_harness()
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="he"))
        h.port.cleanup_failures_remaining = 1

        events = await h.run(commands.BargeIn())

        assert h.session.epoch == 1
        assert find(events, "error").code == "runtime_signal_failed"
        assert h.port.cleanups == [([request_id], True)]
        assert h.session.resource_request_ids() == [request_id]

        await h.manager.handle(
            CloseDuplexSessionMessage(control_id="c-close", session_id=SESSION_ID, reason="client_close")
        )
        result = await asyncio.wait_for(h.results.get(), timeout=2.0)

        assert result.ok and result.operation == "close"
        assert h.port.cleanups == [([request_id], True), ([request_id], True)]
        assert h.session.resource_request_ids() == []
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("ended", [False, True])
@pytest.mark.parametrize("native_signal", [False, True])
async def test_audio_truncation_preserves_retrieved_and_terminal_transcript(ended, native_signal) -> None:
    h = await open_harness()
    try:
        session = h.session
        response = session.begin_response()
        item_id = f"item_{response}"
        h.runner.emit(h.runner.model.response_created_payload(response, epoch=session.epoch))
        session.append_assistant_text("abcdefghij")
        session.mark_audio_sent(10_000, text_chars=10, audio_complete=True)
        h.runner.emit(
            {
                "type": "response.output_audio.delta",
                "response_id": response,
                "audio": "AAA=",
                "text": "abcdefghij",
                "audio_duration_ms": 10_000,
                "audio_text_marks": [{"text_chars": 10, "audio_end_ms": 10_000}],
            }
        )
        if ended:
            session.register_history_item(item_id, session.end_response())
            h.runner.emit({"type": "response.done", "response_id": response})
        await h.settle()

        if native_signal:
            command = commands.SignalTurn(
                event="conversation.item.truncate", signal_payload={"item_id": item_id, "audio_end_ms": 4_000}
            )
        else:
            command = commands.TruncateItem(item_id=item_id, audio_end_ms=4_000)
        events = await h.run(command)
        assert "error" not in types(events)
        assert find(events, "conversation.item.truncated").audio_end_ms == 4_000
        for _ in range(2):
            events = await h.run(
                commands.SignalTurn(event="conversation.item.retrieve", signal_payload={"item_id": item_id})
            )
            assert find(events, "conversation.item.retrieved").item["content"][0]["transcript"] == "abcd"

        if not ended:
            session.register_history_item(item_id, session.end_response())
            h.runner.emit({"type": "response.done", "response_id": response})
            events = await h.settle()
            done = find(events, "response.done")
            assert done.response["output"][0]["content"][0]["transcript"] == "abcd"
            events = await h.run(
                commands.SignalTurn(event="conversation.item.retrieve", signal_payload={"item_id": item_id})
            )
            assert find(events, "conversation.item.retrieved").item["content"][0]["transcript"] == "abcd"
        assert session.history_item_ids[item_id]["content"] == "abcd"
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
@pytest.mark.parametrize("aligned", [True, False])
@pytest.mark.parametrize("hard_cutoff_ms", [None, 500])
async def test_full_ack_releases_cancelled_replies_without_history_calibration(aligned, hard_cutoff_ms) -> None:
    h = await open_harness()
    try:
        assert h.runner.ctx.history_calibration is None
        for _ in range(8):
            await h.run(append_audio())
            await h.deliver_and_settle(
                tts_output(h.stage0_request_id(), samples=24000, text="hello", epoch=h.session.epoch)
            )
            response = h.session.active_response_id
            assert response is not None
            if not aligned:
                # Exercise the proportional fallback for outputs without alignment.
                h.session._response.assistant_audio_text_marks.clear()
            await h.run(commands.CancelResponse())
            assert not h.session.playback_for_response(response).audio_complete
            if hard_cutoff_ms is not None:
                await h.run(commands.TruncateItem(item_id=f"item_{response}", audio_end_ms=hard_cutoff_ms))
            events = await h.run(commands.AckPlayback(response_id=response, played_ms=1000))
            # TruncateItem already applied its own ACK and finalized the stop;
            # this later progress ACK must not report another history commit.
            assert find(events, "playback.acknowledged").details["history_committed"] is (hard_cutoff_ms is None)
            assert not h.session._conversation.assistant_response_snapshots
            assert not h.session._playback.by_response
            assert not h.session._conversation.hard_truncations_ms
        expected_text = "hello" if hard_cutoff_ms is None else "he"
        assert [message["content"] for message in h.session.history] == [expected_text] * 8
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_full_ack_preserves_cancelled_reply_while_history_calibration_is_pending() -> None:
    from vllm_omni.engine.duplex.session.history_calibration import HistoryCalibration

    entered, release = asyncio.Event(), asyncio.Event()

    async def calibrate(snapshot):
        entered.set()
        await release.wait()
        return 2

    h = await open_harness()
    controller = HistoryCalibration(h.session, calibrate, requires_audio=False)
    h.runner.ctx.history_calibration = controller
    try:
        await h.run(append_audio())
        await h.deliver_and_settle(tts_output(h.stage0_request_id(), samples=24000, text="hello"))
        response = h.session.active_response_id
        assert response is not None
        await h.run(commands.AckPlayback(response_id=response, played_ms=1000))
        await h.run(commands.CancelResponse())
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert not h.session._conversation.heard_prefixes
        await h.run(commands.AckPlayback(response_id=response, played_ms=1000))
        assert response in h.session._conversation.assistant_response_snapshots
        assert response in h.session._playback.by_response
        assert response in controller.tasks
        release.set()
        await controller.before_prompt()
        assert h.session.history[-1]["content"] == "he"
    finally:
        release.set()
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


async def _open_active_response(h: Harness) -> str:
    """Drive the harness until a stage1 response is streaming, as the docs assume."""
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
    return request_id


@pytest.mark.asyncio
async def test_input_cancel_with_an_active_response_acknowledges_cleared_input_once() -> None:
    """An input.cancel that also tears down a response still acks the clear.

    Regression: ``input_audio_buffer.cleared`` only existed on the queued-input
    path, so cancelling an active response emitted no acknowledgement even
    though ``input.cancel`` promises one. The epoch still advances exactly once.
    """
    h = await open_harness()
    try:
        request_id = await _open_active_response(h)
        assert h.session.epoch == 0
        assert h.session.active_response_id is not None

        events = await h.run(commands.CancelInput())

        assert types(events).count("input_audio_buffer.cleared") == 1
        assert find(events, "response.done").status == "cancelled"
        assert h.port.aborts == [[request_id]]
        assert h.session.epoch == 1, "one cancellation advances the epoch exactly once"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_input_cancel_with_a_data_plane_stream_acknowledges_cleared_input_once() -> None:
    """The stream-only fallback path acks the clear too, without a second epoch.

    With no active response but a resumable data-plane stream, the cancel is
    owned by the stream fallback (a different branch from the queued-input
    path); it must still emit exactly one ``input_audio_buffer.cleared``.
    """
    h = await open_harness()
    try:
        h.runner.run.stream_request_id = h.stage0_request_id()
        assert h.session.active_response_id is None

        events = await h.run(commands.CancelInput())

        assert types(events).count("input_audio_buffer.cleared") == 1
        assert h.runner.run.stream_request_id is None
        assert h.session.epoch == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_input_cancel_with_only_queued_input_acknowledges_cleared_input_once() -> None:
    """The queued-input path emits one clear, one epoch bump, and the client reason.

    The internal acknowledgement carries the client's cancel reason, not the
    acoustic ``barge_in`` reason it used to inherit by falling through the
    barge-in branch.
    """
    h = await open_harness()
    try:
        await h.run(append_audio(samples=8000))
        assert h.runner.model_state.audio_buffer.has_pending()

        emitted: list[dict[str, object]] = []
        original_emit = h.runner.emit

        def capture(payload: dict[str, object]) -> None:
            emitted.append(dict(payload))
            original_emit(payload)

        h.runner.emit = capture  # type: ignore[method-assign]
        events = await h.run(commands.CancelInput())

        assert types(events).count("input_audio_buffer.cleared") == 1
        assert h.session.epoch == 1
        assert not h.runner.model_state.audio_buffer.has_pending()
        acked = [payload for payload in emitted if payload.get("type") == "input.cancelled"]
        assert len(acked) == 1
        assert acked[0]["reason"] == "client_cancelled"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_a_barge_in_with_only_queued_input_keeps_its_input_cleared_event() -> None:
    """Preserve the established pending-only barge-in acknowledgement.

    This fix corrects the explicit ``input.cancel`` acknowledgement; it must not
    remove the input-clear a barge-in has always emitted for queued input. That
    acknowledgement carries the acoustic ``barge_in`` reason and still advances
    the epoch exactly once.
    """
    h = await open_harness()
    try:
        await h.run(append_audio(samples=8000))
        assert h.runner.model_state.audio_buffer.has_pending()
        assert h.session.active_response_id is None

        emitted: list[dict[str, object]] = []
        original_emit = h.runner.emit

        def capture(payload: dict[str, object]) -> None:
            emitted.append(dict(payload))
            original_emit(payload)

        h.runner.emit = capture  # type: ignore[method-assign]
        events = await h.run(commands.BargeIn())

        assert types(events).count("input_audio_buffer.cleared") == 1
        assert h.session.epoch == 1
        acked = [payload for payload in emitted if payload.get("type") == "input.cancelled"]
        assert len(acked) == 1
        assert acked[0]["reason"] == "barge_in"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_a_barge_in_that_cancels_a_response_gains_no_input_clear() -> None:
    """Input-clear must not grow onto the active barge-in path.

    Only explicit ``input.cancel`` gains a new acknowledgement; a barge-in that
    tears down an active response keeps emitting output-cancel events alone.
    """
    h = await open_harness()
    try:
        await _open_active_response(h)

        events = await h.run(commands.BargeIn())

        assert "input_audio_buffer.cleared" not in types(events)
        assert find(events, "response.done").status == "cancelled"
        assert h.session.epoch == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_a_response_cancel_never_acknowledges_cleared_input() -> None:
    """``response.cancel`` tears down output only; it must not clear input."""
    h = await open_harness()
    try:
        await _open_active_response(h)

        events = await h.run(commands.CancelResponse())

        assert "input_audio_buffer.cleared" not in types(events)
        assert find(events, "response.done").status == "cancelled"
        assert h.session.epoch == 1
    finally:
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
async def test_speech_after_silence_continuation_is_not_the_first_audio_window() -> None:
    """VISION then generated silence consumes the 1035 ms window.

    Two buffered seconds of later speech are two normal chunks (25 slots),
    not the single first-window unit (13).
    """
    h = await open_harness()
    try:
        await h.run(commands.AppendAudio(format="pcm_f32le", sample_rate_hz=16000, video_frames=("frame",)))
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
        await h.deliver_and_settle(
            tts_output(request_id, samples=48000, text="hello", tts_is_last_chunk=True, finished=True)
        )
        assert h.runner.model_state.audio_buffer._audio_window_started is True
        assert h.session.active_response_id is not None
        h.session.config.overlap_policy = "barge_in_on_speech"

        await h.run(
            commands.AppendAudio(
                audio=pcm_f32(32000),
                format="pcm_f32le",
                sample_rate_hz=16000,
                is_speech=True,
                hints={"overlap_action": "listen"},
            )
        )
        assert h.runner.model_state.audio_buffer.pending_byte_count == 32000 * 4
        before = len(h.port.submissions)
        await h.run(commands.Commit(create_response=False))
        if len(h.port.submissions) == before:
            await h.deliver_and_settle(tts_output(request_id, samples=0, text="", finished=True, turn_end=True))
        speech = h.port.submissions[before:]
        assert len(speech) == 1
        payload = speech[0].prompt["model_intermediate_buffer"]["duplex"]["payload"]
        assert "first_audio_window" not in payload
        assert len(speech[0].prompt["prompt_token_ids"]) == 25
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_answer_continues_when_the_segment_end_arrives_before_the_answer() -> None:
    """A continuation planned for the model turn is re-aimed at the answer that opened (#7729)."""
    h = await open_harness()
    try:
        for _ in range(4):
            await h.run(append_audio())
        request_id = h.stage0_request_id()

        # A TTS segment ends with no text or audio before any response exists.
        # This schedules a continuation, which waits for up to one chunk period.
        h.deliver(tts_output(request_id, samples=0, text="", tts_is_last_chunk=True))
        await asyncio.sleep(0.1)

        # During that wait the last real units arrive and the answer starts.
        h.submit(append_audio())
        await asyncio.sleep(0.1)
        h.deliver(tts_output(request_id, samples=24000, text="no"))
        await asyncio.sleep(0.1)
        h.submit(append_audio())
        h.submit(commands.Commit())
        await h.settle(timeout_s=5.0)

        # More of the answer. Its segment only ends after another unit is processed.
        for samples, text in ((48000, "no fi"), (72000, "no fight, bu"), (96000, "no fight, but yes,")):
            await h.deliver_and_settle(tts_output(request_id, samples=samples, text=text))
        await h.settle(idle_s=2.5, timeout_s=8.0)

        assert h.session.active_response_id is not None
        # 6 real units, then a silence unit that keeps the answer going.
        assert len(h.port.submissions) == 7
        assert _is_silence_unit(h.port.submissions[6])
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_a_dropped_continuation_is_not_re_aimed_across_a_barge_in() -> None:
    """The re-aim in #7729 stays inside the epoch that planned the unit.

    The next answer must open while the old continuation is still waiting out
    its chunk period, so this drives the runner with ``submit``/``deliver`` and
    short sleeps. ``run``/``settle`` would wait for that continuation to drop
    first, and then a fix that re-arms with the current epoch passes too.
    """
    h = await open_harness()
    try:
        for _ in range(2):
            await h.run(append_audio())
        request_id = h.stage0_request_id()

        # Plan a continuation for the model turn of epoch 0. It waits for up to
        # one chunk period (1 s) before it submits.
        h.deliver(tts_output(request_id, samples=0, text="", tts_is_last_chunk=True))
        await asyncio.sleep(0.1)

        # The user barges in while it waits, and the next turn opens its own answer.
        h.submit(commands.BargeIn())
        await asyncio.sleep(0.1)
        assert h.session.epoch == 1
        h.submit(append_audio())
        await asyncio.sleep(0.1)
        assert any(task.get_name() == "duplex-continue" and not task.done() for task in h.runner._background_tasks), (
            "the old continuation should still be waiting when the new answer opens"
        )
        h.deliver(tts_output(h.stage0_request_id(), samples=24000, text="hi", epoch=1))
        await h.settle(idle_s=2.5, timeout_s=8.0)

        assert h.session.epoch == 1
        assert h.session.active_response_id is not None
        assert [s for s in h.port.submissions if _is_silence_unit(s)] == []
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


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["barge_in", "stage_failure"])
@pytest.mark.parametrize("deleted", [False, True])
async def test_interrupted_drain_releases_only_deleted_item_playback(termination: str, deleted: bool) -> None:
    from dataclasses import replace

    h = await open_harness()
    try:
        h.session.capabilities = replace(h.session.capabilities, supports_concurrent_turn_requests=True)
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
        old_response = h.session.active_response_id
        assert old_response is not None
        item_id = f"item_{old_response}"
        h.session.snapshot_active_response_for_drain()
        # Multiple stage requests can drain for the same older response.
        for draining_request in ("old-talker", "old-vocoder"):
            h.session.bind_draining_request(draining_request, old_response)
        active_response = h.session.begin_response(turn_id=h.session.turn_id + 1)
        h.runner.emit({"type": "response.created", "response_id": active_response})
        h.session.append_assistant_text("newer reply")
        h.session.mark_audio_sent(duration_ms=250)
        await h.settle()
        if deleted:
            events = await h.run(commands.DeleteItem(item_id=item_id))
            assert types(events) == ["conversation.item.deleted"]
        assert h.session.playback_for_response(old_response).sent_ms == 1000

        if termination == "barge_in":
            events = await h.run(commands.BargeIn())
            assert h.port.aborts == [[request_id, "old-talker", "old-vocoder"]]
            assert h.session.active_response_id is None
        else:
            h.runner.on_stage_failure(1, RuntimeError("drain failed"), request_id="old-talker")
            events = await h.settle()
            assert h.session.active_response_id == active_response
        done = next(event for event in events if event.type == "response.done" and event.response_id == old_response)
        assert done.status == ("cancelled" if termination == "barge_in" else "failed")
        assert done.response["metadata"]["playback"]["sent_ms"] == 1000
        assert not h.session.response_has_draining_request(old_response)
        assert h.session.playback_for_response(active_response).sent_ms == 250
        assert (old_response in h.session._playback.by_response) is not deleted

        ack = await h.run(commands.AckPlayback(response_id=old_response, played_ms=500))
        if deleted:
            assert types(ack) == ["error"]
            assert ack[0].code == "playback_item_not_found"
            assert old_response not in h.session._playback.by_response
        else:
            assert types(ack) == ["playback.acknowledged"]
            assert h.session.playback_for_response(old_response).committed_ms == 500
            events = await h.run(commands.TruncateItem(item_id=item_id, audio_end_ms=250))
            assert types(events) == ["conversation.item.truncated", "playback.acknowledged"]
            assert find(events, "playback.acknowledged").details["playback"]["committed_ms"] == 250
            assert h.session.history_item_ids[item_id]["content"] == "h"
            assert old_response not in h.session._playback.by_response
            assert old_response not in h.session._conversation.assistant_response_snapshots
            assert item_id not in h.session._conversation.hard_truncations_ms
            assert h.session.playback_for_response(active_response).sent_ms == 250
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["barge_in", "stage_failure"])
async def test_interrupted_silent_drain_releases_snapshot_cursor_and_placeholder(termination: str) -> None:
    from dataclasses import replace

    h = await open_harness()
    try:
        h.session.capabilities = replace(h.session.capabilities, supports_concurrent_turn_requests=True)
        await h.run(append_audio())
        old_response = h.session.begin_response(turn_id=h.session.turn_id)
        h.runner.emit({"type": "response.created", "response_id": old_response})
        h.session.append_assistant_text("not yet spoken")
        h.session.snapshot_active_response_for_drain()
        h.session.bind_draining_request("silent-talker", old_response)
        active_response = h.session.begin_response(turn_id=h.session.turn_id + 1)
        h.runner.emit({"type": "response.created", "response_id": active_response})
        h.session.append_assistant_text("newer reply")
        h.session.mark_audio_sent(duration_ms=250)
        await h.settle()

        if termination == "barge_in":
            events = await h.run(commands.BargeIn())
        else:
            h.runner.on_stage_failure(1, RuntimeError("silent drain failed"), request_id="silent-talker")
            events = await h.settle()
            assert h.session.active_response_id == active_response
        done = next(event for event in events if event.type == "response.done" and event.response_id == old_response)
        assert done.response["metadata"]["playback"]["sent_ms"] == 0
        assert not h.session.response_has_draining_request(old_response)
        assert old_response not in h.session._conversation.assistant_response_snapshots
        assert old_response not in h.session._playback.by_response
        assert f"item_{old_response}" not in h.session._conversation.history_item_placeholders
        assert h.session.playback_for_response(active_response).sent_ms == 250
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
async def test_uncaught_session_update_error_names_that_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure outside the handled update branches still answers that command.

    The serving reader matches ``related_event_id``. Without it, the next
    append waits forever.     The failure is before the session adopts the candidate, so the live
    config stays and the staged detector is not installed.
    """

    def explode(self, patch, *, session_id=None, audio_started=False):
        del self, patch, session_id, audio_started
        raise RuntimeError("update exploded")

    class _StagedDetector:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    staged = _StagedDetector()

    def fake_prepare(cls, payload, *, backend_provider=None, semantic_backend_provider=None):
        del cls, payload, backend_provider, semantic_backend_provider
        from vllm_omni.engine.duplex.turn_detection import PendingTurnDetectionUpdate

        return PendingTurnDetectionUpdate(config=None, detector=staged)

    monkeypatch.setattr("vllm_omni.engine.duplex.config.DuplexSessionConfig.apply_realtime_update", explode)
    monkeypatch.setattr(
        "vllm_omni.engine.duplex.session.control.PendingTurnDetectionUpdate.prepare",
        classmethod(fake_prepare),
    )
    h = await open_harness()
    try:
        temperature = h.session.config.temperature
        events = await h.run(commands.UpdateSession(patch={"temperature": 0.2}, event_id="evt-boom"))
        assert types(events) == ["error"]
        assert events[0].code == "internal_error"
        assert events[0].related_event_id == "evt-boom"
        assert h.session.config.temperature == temperature
        assert h.runner.control._detector is not staged
        later = await h.run(commands.UpdateSession(patch={"temperature": 0.4}, event_id="evt-later"))
        assert later[0].code == "internal_error"
        assert later[0].related_event_id == "evt-later"
        assert h.runner.control._detector is not staged
        assert h.session.config.temperature == temperature
    finally:
        await close_harness(h)


def _accepted_format_patch() -> dict[str, object]:
    return {
        "temperature": 0.3,
        "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}},
    }


def _assert_serving_applies_adopted_session(session: object) -> None:
    """The object serving applies must move the live wire defaults, not only the event type."""
    assert isinstance(session, dict)
    assert session["id"] == SESSION_ID
    assert session["temperature"] == 0.3
    envelope = RealtimeEnvelope()
    assert envelope.defaults.input_sample_rate_hz == 16000
    envelope.apply_accepted_session(session)
    assert envelope.defaults.input_audio_format == "pcm16"
    assert envelope.defaults.input_sample_rate_hz == 24000


@pytest.mark.asyncio
async def test_adopted_session_update_publishes_when_the_snapshot_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the live config is replaced, a snapshot failure is still ``session.updated``.

    A stamped ``internal_error`` would tell serving the update was rejected
    while the engine had already adopted it. The fallback session is the
    accepted client payload, so serving's apply path picks up the new rate.
    """

    def explode(self):
        del self
        raise RuntimeError("snapshot exploded")

    h = await open_harness()
    monkeypatch.setattr(
        "vllm_omni.engine.duplex.session.engine_session.DuplexEngineSession.as_public_dict",
        explode,
    )
    try:
        events = await h.run(commands.UpdateSession(patch=_accepted_format_patch(), event_id="evt-pub"))
        assert types(events) == ["session.updated"]
        assert h.session.config.temperature == 0.3
        assert all(getattr(event, "code", None) != "internal_error" for event in events)
        _assert_serving_applies_adopted_session(events[0].session)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_adopted_session_update_publishes_when_replace_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed replace still publishes ``session.updated`` so serving takes the new rate.

    The live config did not change, so the event must not echo the patch's
    temperature. Projection fills the format and rate from the projector.
    """

    def explode(self, config: object) -> None:
        del self, config
        raise RuntimeError("replace exploded")

    h = await open_harness()
    monkeypatch.setattr(
        "vllm_omni.engine.duplex.session.engine_session.DuplexEngineSession.replace_config",
        explode,
    )
    try:
        temperature = h.session.config.temperature
        events = await h.run(commands.UpdateSession(patch=_accepted_format_patch(), event_id="evt-replace"))
        assert types(events) == ["session.updated"]
        assert h.session.config.temperature == temperature
        assert events[0].session.get("temperature") != 0.3
        assert all(getattr(event, "code", None) != "internal_error" for event in events)
        envelope = RealtimeEnvelope()
        envelope.apply_accepted_session(events[0].session)
        assert envelope.defaults.input_audio_format == "pcm16"
        assert envelope.defaults.input_sample_rate_hz == 24000
    finally:
        await close_harness(h)


def _stage_turn_detection(monkeypatch: pytest.MonkeyPatch):
    """Stage a detector without building Silero. ``prepare`` still leaves the patch in place."""

    class _StagedDetector:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    staged = _StagedDetector()

    def fake_prepare(cls, payload, *, backend_provider=None, semantic_backend_provider=None):
        del cls, payload, backend_provider, semantic_backend_provider
        from vllm_omni.engine.duplex.turn_detection import PendingTurnDetectionUpdate

        return PendingTurnDetectionUpdate(config=None, detector=staged)

    monkeypatch.setattr(
        "vllm_omni.engine.duplex.session.control.PendingTurnDetectionUpdate.prepare",
        classmethod(fake_prepare),
    )
    return staged


def _patch_with_turn_detection() -> dict[str, object]:
    patch = _accepted_format_patch()
    patch["turn_detection"] = {"type": "server_vad", "threshold": 0.5}
    return patch


@pytest.mark.asyncio
async def test_runtime_replace_failure_finishes_adoption_before_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runtime-config failure after ``replace_config`` is finished before the echo.

    The first replace raises. The except path applies the runtime config and
    commits the staged detector, so the echoed ``turn_detection`` is live and
    the engine is not left on the new config with the old runtime config.
    """
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

    calls = {"n": 0}
    original = DuplexEngineSession.replace_runtime_config

    def flaky(self, runtime_config: object) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("runtime replace exploded")
        original(self, runtime_config)

    staged = _stage_turn_detection(monkeypatch)
    h = await open_harness()
    monkeypatch.setattr(DuplexEngineSession, "replace_runtime_config", flaky)
    try:
        events = await h.run(commands.UpdateSession(patch=_patch_with_turn_detection(), event_id="evt-runtime"))
        assert calls["n"] == 2
        assert types(events) == ["session.updated"]
        assert h.session.config.temperature == 0.3
        assert h.session.runtime_config["duplex_stage_sampling_params"]["0"]["temperature"] == 0.3
        assert h.runner.control._detector is staged
        assert staged.reset_calls == 0
        assert events[0].session["turn_detection"]["type"] == "server_vad"
        assert all(getattr(event, "code", None) != "internal_error" for event in events)
        _assert_serving_applies_adopted_session(events[0].session)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_runtime_replace_that_cannot_finish_drops_the_uncommitted_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the runtime config never lands, restore the previous config and reject the detector.

    The echo must not advertise the patch's ``turn_detection``. Projection still
    carries the format and rate the projector already adopted.
    """

    def explode(self, runtime_config: object) -> None:
        del self, runtime_config
        raise RuntimeError("runtime replace exploded")

    staged = _stage_turn_detection(monkeypatch)
    h = await open_harness()
    temperature = h.session.config.temperature
    monkeypatch.setattr(
        "vllm_omni.engine.duplex.session.engine_session.DuplexEngineSession.replace_runtime_config",
        explode,
    )
    try:
        events = await h.run(commands.UpdateSession(patch=_patch_with_turn_detection(), event_id="evt-runtime-stuck"))
        assert types(events) == ["session.updated"]
        assert h.session.config.temperature == temperature
        assert h.runner.control._detector is not staged
        # Projection fills a missing turn_detection with None. That is not the
        # new server_vad config prepare wrote into the patch.
        assert events[0].session.get("turn_detection") is None
        assert all(getattr(event, "code", None) != "internal_error" for event in events)
        envelope = RealtimeEnvelope()
        envelope.apply_accepted_session(events[0].session)
        assert envelope.defaults.input_audio_format == "pcm16"
        assert envelope.defaults.input_sample_rate_hz == 24000
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_rejected_runtime_replace_does_not_count_the_candidate_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restoring the previous config must not go through ``replace_config``.

    The forward replace already counted the candidate seed and bumped
    ``config_generation``. A second ``replace_config`` would count the old
    seed as well, so a later ``response.create`` sees input that never landed.
    """
    from vllm_omni.engine.duplex.config import DuplexSessionConfig

    original = DuplexSessionConfig.apply_realtime_update

    def apply_seed(self, patch, *, session_id=None, audio_started=False):
        original(self, patch, session_id=session_id, audio_started=audio_started)
        seed = patch.get("initial_user_text")
        if isinstance(seed, str):
            self.initial_user_text = seed

    def explode(self, runtime_config: object) -> None:
        del self, runtime_config
        raise RuntimeError("runtime replace exploded")

    h = await open_harness()
    h.session.config.initial_user_text = "ask about A"
    unanswered = h.session.unanswered_user_items()
    generation = h.session.config_generation
    temperature = h.session.config.temperature
    monkeypatch.setattr(DuplexSessionConfig, "apply_realtime_update", apply_seed)
    monkeypatch.setattr(
        "vllm_omni.engine.duplex.session.engine_session.DuplexEngineSession.replace_runtime_config",
        explode,
    )
    try:
        patch = _accepted_format_patch()
        patch["initial_user_text"] = "ask about B"
        events = await h.run(commands.UpdateSession(patch=patch, event_id="evt-seed"))
        assert types(events) == ["session.updated"]
        assert h.session.config.initial_user_text == "ask about A"
        assert h.session.config.temperature == temperature
        assert h.session.unanswered_user_items() == unanswered
        assert h.session.config_generation == generation
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_detector_reset_failure_still_adopts_the_staged_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reset failure while swapping detectors still leaves one consistent update.

    ``commit`` discards the old detector either way. Keeping it after the new
    config and overlap policy landed would advertise a VAD the runner is not
    running.
    """
    from vllm_omni.engine.duplex.turn_detection import (
        PendingTurnDetectionUpdate,
        TurnDetectionConfig,
        normalize_turn_detection_session_payload,
    )

    class _OldDetector:
        def __init__(self) -> None:
            self.config = TurnDetectionConfig(threshold=0.2)
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1
            raise RuntimeError("reset exploded")

    class _StagedDetector:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    staged = _StagedDetector()
    old = _OldDetector()

    def fake_prepare(cls, payload, *, backend_provider=None, semantic_backend_provider=None):
        del cls, backend_provider, semantic_backend_provider
        _configured, config = normalize_turn_detection_session_payload(payload)
        return PendingTurnDetectionUpdate(config=config, detector=staged)

    monkeypatch.setattr(
        "vllm_omni.engine.duplex.session.control.PendingTurnDetectionUpdate.prepare",
        classmethod(fake_prepare),
    )
    h = await open_harness()
    h.runner.control._detector = old
    h.runner.control._config = old.config
    try:
        events = await h.run(commands.UpdateSession(patch=_patch_with_turn_detection(), event_id="evt-reset"))
        assert types(events) == ["session.updated"]
        assert old.reset_calls == 1
        assert h.runner.control._detector is staged
        assert h.session.config.overlap_policy == "barge_in_on_speech"
        assert events[0].session["overlap_policy"] == h.session.config.overlap_policy
        assert events[0].session["turn_detection"]["type"] == "server_vad"
        assert h.session.config.extra_body["realtime_turn_detection"]["type"] == "server_vad"
        assert h.runner.control._config is not old.config
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_publish_failure_after_replace_carries_the_accepted_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the live config is replaced, a failed publish still echoes the accepted patch."""
    calls = {"n": 0}
    original = SessionEmitter.emit

    def flaky(self: SessionEmitter, payload: dict[str, object]) -> None:
        if payload.get("type") == "session.updated" and calls["n"] == 0:
            calls["n"] += 1
            raise RuntimeError("emit exploded")
        original(self, payload)

    h = await open_harness()
    monkeypatch.setattr(SessionEmitter, "emit", flaky)
    try:
        events = await h.run(commands.UpdateSession(patch=_accepted_format_patch(), event_id="evt-emit"))
        assert calls["n"] == 1
        assert types(events) == ["session.updated"]
        assert h.session.config.temperature == 0.3
        _assert_serving_applies_adopted_session(events[0].session)
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


def _response_request_metrics_of(event: DuplexEvent) -> dict[str, object]:
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

    def process_segments(self, base64_audio: str, *, fmt: str, sample_rate_hz: int | None):
        from vllm_omni.engine.duplex.turn_detection import TurnDetectionSegment

        return [TurnDetectionSegment(base64_audio, self.process(base64_audio, fmt=fmt, sample_rate_hz=sample_rate_hz))]

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


@pytest.mark.asyncio
async def test_input_clear_resets_the_turn_detector() -> None:
    """A clear starts a new turn, so the detector's speech/pause state goes with the audio."""
    h = await open_harness(auto_response=False)
    try:
        detector = _speech_then_stop()
        h.runner.control._detector = detector
        await h.run(append_audio())
        await h.run(commands.ClearInput())
        assert detector.resets == 1
    finally:
        await close_harness(h)


def _semantic_detector(*turn_probabilities: float | Exception):
    from vllm_omni.engine.duplex.turn_detection import SemanticTurnDetector, TurnDetectionConfig

    answers = list(turn_probabilities)

    def turn_scorer(_audio: np.ndarray) -> float:
        if not answers:
            pytest.fail("Smart Turn was called more often than the test expects")
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return SemanticTurnDetector(
        TurnDetectionConfig.from_realtime({"type": "semantic_vad", "eagerness": "high"}),
        turn_scorer=turn_scorer,
        frame_scorer=lambda frame: 0.95 if float(np.abs(frame).max()) >= 0.01 else 0.02,
    )


@pytest.mark.parametrize(
    ("answer", "decision"),
    [(0.1, "smart_turn_incomplete"), (RuntimeError("inference failed"), "smart_turn_error")],
)
@pytest.mark.asyncio
async def test_semantic_timeout_logs_session_and_last_decision(answer, decision, monkeypatch) -> None:
    from vllm_omni.engine.duplex.session import control

    messages: list[str] = []
    monkeypatch.setattr(control.logger, "info", lambda fmt, *args: messages.append(fmt % args))
    h = await open_harness(auto_response=False)
    try:
        h.runner.control._detector = _semantic_detector(answer)
        await h.run(append_audio(samples=8000, is_speech=None))
        pause = await h.run(append_audio(samples=9600, value=0.0, is_speech=None))
        assert "input_audio_buffer.committed" not in types(pause)
        decisions = [message for message in messages if f"semantic_decision={decision}" in message]
        assert len(decisions) == 1
        assert f"session={SESSION_ID}" in decisions[0] and "commit=False" in decisions[0]
        events = await h.run(append_audio(samples=32000, value=0.0, is_speech=None))
        assert types(events).count("input_audio_buffer.committed") == 1
        commits = [message for message in messages if "commit_reason=silence_timeout" in message]
        assert len(commits) == 1
        assert f"session={SESSION_ID}" in commits[0] and f"semantic_decision={decision}" in commits[0]
        assert "commit=True" in commits[0]
    finally:
        await close_harness(h)


@pytest.mark.parametrize("command", [commands.ClearInput, commands.CancelInput, commands.BargeIn])
@pytest.mark.asyncio
async def test_discard_input_starts_a_fresh_semantic_turn(command) -> None:
    """Discarding an incomplete turn must also discard its pending semantic pause."""
    h = await open_harness(auto_response=False)
    try:
        h.runner.control._detector = _semantic_detector(0.1, 0.9)
        await h.run(append_audio(samples=8000, is_speech=None))
        pause = await h.run(append_audio(samples=9600, value=0.0, is_speech=None))
        assert "input_audio_buffer.committed" not in types(pause)

        await h.run(command())
        events = await h.run(append_audio(samples=8000, is_speech=None))
        events += await h.run(append_audio(samples=9600, value=0.0, is_speech=None))
        assert types(events).count("input_audio_buffer.speech_started") == 1
        assert types(events).count("input_audio_buffer.speech_stopped") == 1
        assert types(events).count("input_audio_buffer.committed") == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_semantic_vad_keeps_a_thinking_pause_inside_one_committed_turn() -> None:
    h = await open_harness(auto_response=False)
    try:
        h.runner.control._detector = _semantic_detector(0.1, 0.9)
        events = await h.run(append_audio(samples=8000, is_speech=None))
        # Smart Turn calls this pause incomplete: no stop, no commit, and the
        # silence is buffered rather than skipped as silence_or_noise.
        pause = await h.run(append_audio(samples=9600, value=0.0, is_speech=None))
        assert "input_audio_buffer.speech_stopped" not in types(pause)
        assert "input_audio_buffer.committed" not in types(pause)
        assert "response.listen" not in types(pause)
        events += pause
        events += await h.run(append_audio(samples=8000, is_speech=None))
        events += await h.run(append_audio(samples=9600, value=0.0, is_speech=None))

        assert types(events).count("input_audio_buffer.speech_started") == 1
        assert types(events).count("input_audio_buffer.speech_stopped") == 1
        assert types(events).count("input_audio_buffer.committed") == 1
        assert len(_final_submissions(h)) == 1
    finally:
        await close_harness(h)


@pytest.mark.parametrize("mode", ["server_vad", "semantic_vad"])
@pytest.mark.asyncio
async def test_one_append_preserves_multiple_vad_turns(mode: str) -> None:
    from vllm_omni.engine.duplex.turn_detection import SemanticTurnDetector, TurnDetectionConfig

    h = await open_harness(auto_response=False)
    try:
        config = TurnDetectionConfig(type=mode, create_response=False)
        if mode == "semantic_vad":
            detector = SemanticTurnDetector(config, turn_scorer=lambda _: 0.1)
        else:
            detector = config.build_detector()
        detector.vad._frame_scorer = lambda frame: 0.95 if np.abs(frame).max() > 0.1 else 0.02
        h.runner.control._detector = detector
        audio = np.concatenate([np.full(16000, 0.2), np.zeros(80000), np.full(16000, 0.4), np.zeros(80000)]).astype(
            "<f4"
        )
        events = await h.run(commands.AppendAudio(audio=audio.tobytes(), format="pcm_f32le", sample_rate_hz=16000))
        assert types(events).count("input_audio_buffer.speech_started") == 2
        assert types(events).count("input_audio_buffer.speech_stopped") == 2
        assert types(events).count("input_audio_buffer.committed") == 2
        assert "error" not in types(events)
        assert "response.created" not in types(events)
    finally:
        await close_harness(h)


@pytest.mark.parametrize("command", [commands.CancelResponse, commands.ClearOutputAudio])
@pytest.mark.asyncio
async def test_cancelling_output_keeps_the_pending_semantic_turn(command) -> None:
    h = await open_harness(auto_response=False)
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        assert h.session.active_response_id is not None
        h.runner.control._detector = _semantic_detector(0.1, 0.9)
        await h.run(append_audio(samples=8000, is_speech=None))
        await h.run(append_audio(samples=9600, value=0.0, is_speech=None))
        await h.run(command())
        events = await h.run(append_audio(samples=8000, is_speech=None))
        events += await h.run(append_audio(samples=9600, value=0.0, is_speech=None))
        assert "input_audio_buffer.speech_started" not in types(events)
        assert types(events).count("input_audio_buffer.committed") == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_auto_commit_does_not_reset_later_speech_in_the_same_append() -> None:
    h = await open_harness(auto_response=False)
    try:
        h.runner.control._detector = _semantic_detector(0.9, 0.9)
        audio = np.concatenate([np.full(16000, 0.2), np.zeros(16000), np.full(8000, 0.2)])
        first = await h.run(
            commands.AppendAudio(audio=audio.astype("<f4").tobytes(), format="pcm_f32le", sample_rate_hz=16000)
        )
        assert types(first).count("input_audio_buffer.speech_started") == 2
        assert types(first).count("input_audio_buffer.committed") == 1
        second = await h.run(append_audio(samples=9600, value=0.0, is_speech=None))
        assert "input_audio_buffer.speech_started" not in types(second)
        assert types(second).count("input_audio_buffer.committed") == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_a_missing_smart_turn_model_disables_turn_detection_with_its_own_code(tmp_path) -> None:
    from vllm_omni.engine.duplex.smart_turn import SmartTurnBackendProvider
    from vllm_omni.engine.duplex.turn_detection import SemanticTurnDetector, TurnDetectionConfig

    h = await open_harness(auto_response=False)
    try:
        h.runner.control._detector = SemanticTurnDetector(
            TurnDetectionConfig.from_realtime({"type": "semantic_vad"}),
            smart_turn_provider=SmartTurnBackendProvider(model_path=str(tmp_path / "missing.onnx")),
            frame_scorer=lambda _frame: 0.0,
        )
        events = await h.run(append_audio(is_speech=None))
        assert find(events, "error").code == "semantic_vad_unavailable"
        assert h.runner.control._detector is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_semantic_vad_is_not_started_for_an_auto_responding_session() -> None:
    h = await open_harness(auto_response=True, extra_body={"realtime_turn_detection": {"type": "semantic_vad"}})
    try:
        assert h.runner.control._detector is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_auto_response_cannot_be_turned_on_under_semantic_vad() -> None:
    h = await open_harness(auto_response=False)
    try:
        await h.run(commands.UpdateSession(patch={"turn_detection": {"type": "semantic_vad"}}))
        assert h.runner.control._detector is not None
        events = await h.run(commands.UpdateSession(patch={"extra_body": {"auto_response": True}}))
        assert find(events, "error").code == "unsupported_turn_detection"
        assert h.session.config.extra_body.get("auto_response") is False
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_semantic_vad_is_refused_for_an_auto_responding_session() -> None:
    h = await open_harness(auto_response=True)
    try:
        events = await h.run(
            commands.UpdateSession(patch={"turn_detection": {"type": "semantic_vad"}}, event_id="evt-sem")
        )
        error = find(events, "error")
        assert error.code == "unsupported_turn_detection"
        assert "semantic_vad" in error.message
        assert h.runner.control._detector is None
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
async def test_commit_aborted_does_not_claim_the_next_commit_item_id() -> None:
    """An aborted commit drops its own id, so a later FIFO ack keeps its own.

    ``resolve_commit`` pushes the id before ``_on_commit``. The abort discards
    that id, not whatever happens to be at the head. The next acknowledgement
    carries no explicit id, so it pops the FIFO.
    """
    h = await open_harness(auto_response=False)
    try:
        await _commit_with_pending_response(h)
        pending = h.runner.tasks.append_tail
        assert pending is not None
        pending.cancel()
        await h.settle()
        h.port.submit_gate = None

        await h.run(append_audio())
        aborted = await h.run(commands.Commit(create_response=False, realtime_item_id="item_abort"))
        assert find(aborted, "error").code == "commit_aborted"
        assert "input_audio_buffer.committed" not in types(aborted)
        projector = h.runner.out.require_projector()
        assert projector.pending_commit_item_ids == []

        # No explicit id: the ack has to pop the FIFO. A leftover item_abort
        # would come out here instead of item_fifo.
        projector.pending_commit_item_ids.append("item_fifo")
        h.runner.emit({"type": "input.committed", "message": {"role": "user", "content": "later"}})
        fifo = await h.settle()
        assert find(fifo, "input_audio_buffer.committed").to_realtime()["item_id"] == "item_fifo"
        assert projector.pending_commit_item_ids == []

        # The failed append stays on the tail. A non-resumable retry is what
        # starts a new chain, so the next commit can be acknowledged.
        h.session.capabilities = replace(h.session.capabilities, supports_core_resumable_request=False)
        await h.run(append_audio())
        accepted = await h.run(commands.Commit(create_response=False))
        stopped = find(accepted, "input_audio_buffer.speech_stopped")
        committed = find(accepted, "input_audio_buffer.committed")
        assert committed.to_realtime()["item_id"] == stopped.item_id
        assert stopped.item_id != "item_abort"
        assert projector.pending_commit_item_ids == []
        assert h.session.state == DuplexSessionState.OPEN
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
        assert queued.done() and queued.result() is True, "called off, not failed: later appends still run"
        assert len(h.port.submissions) == 1, "only the append that was already in flight reached the stage"
        assert [event.type for event in events if event.type in {"error", "response.done", "session.closed"}] == []
        assert h.session.pending_input_bytes == 0
        assert h.session.state == DuplexSessionState.OPEN
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_minicpmo_video_only_append_submits_without_commit() -> None:
    h = await open_harness()
    try:
        assert h.session.capabilities.allows_video_without_audio()
        events = await h.run(
            commands.AppendAudio(
                format="pcm_f32le",
                sample_rate_hz=16000,
                video_frames=("frame",),
            )
        )
        assert "error" not in types(events)
        assert len(h.port.submissions) == 1
        payload = h.port.submissions[0].prompt["model_intermediate_buffer"]["duplex"]["payload"]
        assert payload["video_frames"] == ["frame"]
        assert not payload.get("audio")
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_speech", [False, True])
@pytest.mark.parametrize("create_response", [False, True])
async def test_manual_commit_drains_vision_units_before_response(with_speech: bool, create_response: bool) -> None:
    h = await open_harness(auto_response=False)
    try:
        for frame in ("frame-1", "frame-2"):
            await h.run(commands.AppendAudio(format="pcm_f32le", sample_rate_hz=16000, video_frames=(frame,)))
        if with_speech:
            await h.run(
                commands.AppendAudio(audio=pcm_f32(1600), format="pcm_f32le", sample_rate_hz=16000, is_speech=True)
            )
        assert not h.port.submissions
        events = await h.run(commands.Commit(create_response=create_response))
        assert "input_audio_buffer.committed" in types(events)
        assert not h.runner.model_state.audio_buffer.has_pending()
        if not create_response:
            await h.run(commands.CreateResponse())
        payloads = [s.prompt["model_intermediate_buffer"]["duplex"]["payload"] for s in h.port.submissions]
        assert [p["video_frames"] for p in payloads if p.get("video_frames")] == [["frame-1"], ["frame-2"]]
        assert len(payloads) == (3 if with_speech else 2)
        assert all(p.get("force_listen") is True for p in payloads[:-1])
        assert bool(payloads[-1].get("audio")) is with_speech
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("create_response", [False, True])
@pytest.mark.parametrize("with_speech", [False, True])
async def test_deferred_commit_keeps_vision_units_ahead_of_the_speech_tail(
    create_response: bool, with_speech: bool
) -> None:
    """A commit during an active response must not collapse queued VISION units.

    Overlap deferral retains the turn until the response ends. Taking only the
    first VISION unit there leaves the other frames and the speech tail in the
    buffer, and prepare_commit clears the speech flag so that tail is stranded.
    """
    h = await open_harness(auto_response=False)
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        assert h.session.active_response_id is not None
        assert len(h.port.submissions) == 1

        request_id = h.port.submissions[0].context.request_id
        response_id = h.session.active_response_id
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="sure"))
        await h.run(commands.AckPlayback(response_id=response_id, played_ms=1000))
        for frame in ("frame-1", "frame-2"):
            await h.run(commands.AppendAudio(format="pcm_f32le", sample_rate_hz=16000, video_frames=(frame,)))
        if with_speech:
            await h.run(
                commands.AppendAudio(audio=pcm_f32(16000), format="pcm_f32le", sample_rate_hz=16000, is_speech=True)
            )
        events = await h.run(commands.Commit(create_response=create_response))
        assert "input_audio_buffer.committed" in types(events)
        assert len(h.port.submissions) == 1
        assert not h.runner.model_state.audio_buffer.has_pending()
        assert [p["video_frames"] for p in h.runner.model_state.commit_prefix_payloads] == (
            [["frame-1"], ["frame-2"]] if with_speech else [["frame-1"]]
        )

        await h.deliver_and_settle(tts_output(request_id, samples=0, text="", finished=True))
        if not create_response:
            create_events = await h.run(commands.CreateResponse())
            assert "error" not in types(create_events), create_events
        payloads = [s.prompt["model_intermediate_buffer"]["duplex"]["payload"] for s in h.port.submissions]
        assert [p.get("video_frames") for p in payloads[1:]] == (
            [["frame-1"], ["frame-2"], None] if with_speech else [["frame-1"], ["frame-2"]]
        )
        assert all(p.get("force_listen") is True for p in payloads[1:-1])
        assert bool(payloads[-1].get("audio")) is with_speech
        assert not h.runner.model_state.commit_prefix_payloads
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["deferred_twice", "retained_twice", "deferred_then_flushed"])
async def test_later_commit_keeps_earlier_retained_vision_units_in_wire_order(scenario: str) -> None:
    """A retained turn must neither merge into nor trail a later commit's VISION units."""

    async def commit_frames(frames: tuple[str, ...], *, create_response: bool) -> None:
        for frame in frames:
            await h.run(commands.AppendAudio(format="pcm_f32le", sample_rate_hz=16000, video_frames=(frame,)))
        await h.run(commands.Commit(create_response=create_response))

    h = await open_harness(auto_response=False)
    try:
        request_id = None
        if scenario != "retained_twice":
            await h.run(append_audio())
            await h.run(commands.Commit(create_response=True))
            request_id = h.port.submissions[0].context.request_id
            response_id = h.session.active_response_id
            await h.deliver_and_settle(tts_output(request_id, samples=24000, text="sure"))
            await h.run(commands.AckPlayback(response_id=response_id, played_ms=1000))

        await commit_frames(("a1", "a2"), create_response=scenario == "deferred_twice")
        if scenario == "deferred_then_flushed":
            assert isinstance(request_id, str)
            await h.deliver_and_settle(tts_output(request_id, samples=0, text="", finished=True))
        await commit_frames(("b1", "b2"), create_response=scenario != "retained_twice")
        if scenario == "deferred_twice":
            assert isinstance(request_id, str)
            await h.deliver_and_settle(tts_output(request_id, samples=0, text="", finished=True))
        if scenario == "retained_twice":
            await h.run(commands.CreateResponse())

        payloads = [s.prompt["model_intermediate_buffer"]["duplex"]["payload"] for s in h.port.submissions]
        turn = payloads if scenario == "retained_twice" else payloads[1:]
        assert [p.get("video_frames") for p in turn] == [["a1"], ["a2"], ["b1"], ["b2"]]
        assert all(p.get("force_listen") is True for p in turn[:-1])
        assert not h.runner.model_state.commit_prefix_payloads
    finally:
        await close_harness(h)
