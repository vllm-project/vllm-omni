# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Engine-resident Qwen3 chat-fallback lifecycle scenarios."""

from __future__ import annotations

import asyncio
import base64
import io
import struct
import wave
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.commands import DuplexCommand
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexStagePort,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
)
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    DuplexSessionEventMessage,
    DuplexSessionFallbackCancelMessage,
    DuplexSessionFallbackFailedMessage,
    DuplexSessionFallbackOutputMessage,
    DuplexSessionFallbackRequestMessage,
    DuplexSessionFallbackStartedMessage,
    OpenDuplexSessionMessage,
)
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner
from vllm_omni.engine.duplex.turn_detection import TurnDetectionResult
from vllm_omni.entrypoints.duplex.chat_fallback import build_chat_request
from vllm_omni.model_executor.models.qwen3_omni.duplex.chat_fallback_plugin import Qwen3OmniChatFallbackPlugin
from vllm_omni.model_executor.models.qwen3_omni.duplex.policy import INTERRUPTION_NOTE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION_ID = "qwen3-duplex-test"
_UNSUPPORTED_TOOL_OPTIONS = [
    {"tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}]},
    {"tool_choice": "auto"},
    {"tool_choice": {"type": "function", "name": "lookup"}},
    {"extra_body": {"tools": [{"type": "function", "function": {"name": "lookup"}}]}},
    {"extra_body": {"realtime_tools": [{"type": "function", "name": "lookup"}]}},
    {"extra_body": {"realtime_response_tool_choice": "required"}},
]


class RecordingStagePort(DuplexStagePort):
    """Stage fake that proves Qwen fallback turns never reach native append."""

    def __init__(self) -> None:
        self.ensured: list[DuplexStageRequestContext] = []
        self.submissions: list[DuplexStageSubmission] = []
        self.cleanups: list[tuple[list[str], bool]] = []
        self.aborts: list[list[str]] = []

    @property
    def stage_count(self) -> int:
        return 1

    def sampling_defaults(self) -> tuple[object, ...]:
        return (None,)

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


def _encode_audio(audio: object, sample_rate_hz: int, response_format: str, speed: float | None) -> str | None:
    del audio, sample_rate_hz, response_format, speed
    return None


def _pcm_f32(value: float = 0.05, samples: int = 16_000) -> bytes:
    return struct.pack(f"<{samples}f", *([value] * samples))


class _ScriptedDetector:
    """Return deterministic VAD results without loading the Silero model."""

    def __init__(self, results: list[TurnDetectionResult]) -> None:
        self._results = list(results)

    def process(
        self,
        base64_audio: str,
        *,
        fmt: str,
        sample_rate_hz: int | None,
        audio_end_ms: int | None = None,
    ) -> TurnDetectionResult:
        del base64_audio, fmt, sample_rate_hz, audio_end_ms
        return self._results.pop(0)

    def reset(self) -> None:
        pass


def _speech_then_stop_detector() -> _ScriptedDetector:
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


@dataclass
class Harness:
    manager: DuplexSessionManager
    port: RecordingStagePort
    output: asyncio.Queue[Any]
    output_buffer: DuplexOutputBuffer
    results: asyncio.Queue[Any]
    runner: DuplexSessionRunner
    events: list[Any] = field(default_factory=list)
    fallback_requests: list[DuplexSessionFallbackRequestMessage] = field(default_factory=list)
    fallback_cancels: list[DuplexSessionFallbackCancelMessage] = field(default_factory=list)

    @property
    def session(self):
        return self.runner.session

    def submit(self, command: DuplexCommand) -> None:
        self.manager.dispatch(DuplexSessionCommandMessage(session_id=SESSION_ID, command=command))

    async def settle(self, *, timeout_s: float = 3.0) -> list[Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        collected: list[Any] = []
        quiet_since: float | None = None
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
                drained = True
                if isinstance(message, DuplexSessionEventMessage):
                    collected.append(message.event)
                elif isinstance(message, DuplexSessionFallbackRequestMessage):
                    self.fallback_requests.append(message)
                elif isinstance(message, DuplexSessionFallbackCancelMessage):
                    self.fallback_cancels.append(message)
                else:
                    raise AssertionError(f"unexpected engine output: {message!r}")
            busy = (
                drained
                or not self.runner._mailbox.empty()
                or any(not task.done() for task in self.runner._background_tasks)
            )
            now = loop.time()
            if busy:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = now
            elif now - quiet_since >= 0.05:
                break
            if now >= deadline:
                raise AssertionError("duplex harness did not settle")
            await asyncio.sleep(0.005)
        self.events.extend(collected)
        return collected

    async def run(self, command: DuplexCommand) -> list[Any]:
        self.submit(command)
        return await self.settle()

    async def start_fallback_turn(self) -> DuplexSessionFallbackRequestMessage:
        await self.run(
            commands.AppendAudio(
                audio=_pcm_f32(),
                format="pcm_f32le",
                sample_rate_hz=16_000,
                is_speech=True,
            )
        )
        await self.run(commands.Commit())
        assert self.fallback_requests, "commit did not emit an internal fallback request"
        return self.fallback_requests.pop(0)

    async def send_fallback_started(self, request: DuplexSessionFallbackRequestMessage) -> list[Any]:
        self.manager.dispatch(
            DuplexSessionFallbackStartedMessage(
                session_id=SESSION_ID,
                request_id=request.request.request_id,
                response_id=request.request.response_id,
                epoch=request.request.epoch,
            )
        )
        return await self.settle()

    async def send_fallback_output(
        self,
        request: DuplexSessionFallbackRequestMessage,
        output: dict[str, object],
        *,
        epoch: int | None = None,
    ) -> list[Any]:
        self.manager.dispatch(
            DuplexSessionFallbackOutputMessage(
                session_id=SESSION_ID,
                request_id=request.request.request_id,
                response_id=request.request.response_id,
                epoch=request.request.epoch if epoch is None else epoch,
                output=output,
            )
        )
        return await self.settle()

    async def send_fallback_failed(
        self,
        request: DuplexSessionFallbackRequestMessage,
        *,
        error: str = "provider failed",
        error_code: str = "provider_error",
    ) -> list[Any]:
        self.manager.dispatch(
            DuplexSessionFallbackFailedMessage(
                session_id=SESSION_ID,
                request_id=request.request.request_id,
                response_id=request.request.response_id,
                epoch=request.request.epoch,
                error=error,
                error_code=error_code,
            )
        )
        return await self.settle()


async def open_harness(
    *,
    initial_user_text: str | None = None,
    modalities: Sequence[str] = ("text",),
    limits: DuplexSessionRuntimeConfig | None = None,
) -> Harness:
    port = RecordingStagePort()
    output: asyncio.Queue[Any] = asyncio.Queue()
    results: asyncio.Queue[Any] = asyncio.Queue()
    limits = limits or DuplexSessionRuntimeConfig()
    output_buffer = DuplexOutputBuffer(
        max_bytes=limits.max_pending_output_bytes_per_session,
        max_events=limits.max_pending_output_events_per_session,
    )
    manager = DuplexSessionManager(
        plugin=Qwen3OmniChatFallbackPlugin(_encode_audio),
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=limits,
        model_config=None,
    )
    await manager.handle(
        OpenDuplexSessionMessage(
            control_id="open-qwen3",
            session_id=SESSION_ID,
            output_buffer=output_buffer,
            session_config=DuplexSessionConfig(
                model="qwen3-omni",
                modalities=list(modalities),
                initial_user_text=initial_user_text,
            ),
        )
    )
    result = await asyncio.wait_for(results.get(), timeout=2.0)
    assert isinstance(result, DuplexControlResultMessage) and result.ok, result
    runner = manager.runners[SESSION_ID]
    harness = Harness(manager, port, output, output_buffer, results, runner)
    await harness.settle()
    return harness


async def close_harness(harness: Harness) -> None:
    await harness.manager.handle(
        CloseDuplexSessionMessage(control_id="close-qwen3", session_id=SESSION_ID, reason="test_cleanup")
    )
    await harness.results.get()
    await harness.settle()
    await harness.manager.shutdown()


def _event(events: Sequence[Any], event_type: str) -> Any:
    for event in events:
        if event.type == event_type:
            return event
    raise AssertionError(f"missing {event_type}: {[event.type for event in events]}")


@pytest.mark.asyncio
async def test_qwen_runner_resamples_wire_float_capture_from_24k_to_16k() -> None:
    harness = await open_harness()
    try:
        append = commands.command_from_realtime(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(_pcm_f32(samples=24_000)).decode(),
                "format": "pcm_f32le",
                "sample_rate_hz": 24_000,
                "is_speech": True,
            }
        )
        assert append.sample_rate_hz == 24_000
        await harness.run(append)
        await harness.run(commands.Commit())

        [request] = harness.fallback_requests
        assert request.request.input_payload["sample_rate_hz"] == 16_000
        assert len(base64.b64decode(request.request.input_payload["audio"])) == 16_000 * 4
        assert not [event for event in harness.events if event.type == "error"]
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_qwen_commit_emits_internal_fallback_request_and_no_native_stage_submission() -> None:
    harness = await open_harness()
    try:
        request = await harness.start_fallback_turn()
        created = _event(harness.events, "response.created")

        assert request.request.session_id == SESSION_ID
        assert request.request.request_id == f"duplex-fallback-{SESSION_ID}-0-1"
        assert request.request.response_id == created.response_id
        assert request.request.history[-1]["role"] == "user"
        assert harness.port.submissions == []
        assert not any(event.type == "duplex_session_fallback_request" for event in harness.events)
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_fallback_history_keeps_audio_per_turn_without_public_metadata() -> None:
    harness = await open_harness()
    try:
        first = await harness.start_fallback_turn()
        first_audio = first.request.history[-1]["_duplex_fallback_audio"]["audio"]
        assert "_duplex_fallback_audio" not in harness.session.history[-1]

        await harness.send_fallback_started(first)
        await harness.send_fallback_output(
            first,
            {"text": "first", "data_plane_request_id": first.request.request_id, "end_of_turn": True},
        )
        await harness.run(
            commands.AppendAudio(
                audio=_pcm_f32(0.1),
                format="pcm_f32le",
                sample_rate_hz=16_000,
                is_speech=True,
            )
        )
        await harness.run(commands.Commit())

        assert len(harness.fallback_requests) == 1
        second = harness.fallback_requests.pop()
        audio_messages = [
            message for message in second.request.history if message.get("_duplex_fallback_audio") is not None
        ]
        assert len(audio_messages) == 2
        assert [message["_duplex_fallback_audio"]["audio"] for message in audio_messages] == [
            first_audio,
            base64.b64encode(_pcm_f32(0.1)).decode(),
        ]
        assert (
            audio_messages[0]["_duplex_fallback_audio"]["audio"] != audio_messages[1]["_duplex_fallback_audio"]["audio"]
        )
        assert all("_duplex_fallback_audio" not in message for message in harness.session.history)
        assert all("_duplex_fallback_audio" not in str(event.to_realtime()) for event in harness.events)
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_five_plus_completed_audio_turns_bound_only_the_model_input_history() -> None:
    harness = await open_harness()
    try:
        for index in range(6):
            request = await harness.start_fallback_turn()
            chat = build_chat_request(request.request, model="qwen")
            audio_parts = [
                part
                for message in chat.messages
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if part.get("type") == "audio_url"
            ]
            assert len(audio_parts) == min(index + 1, 4)
            replies = [message["content"] for message in chat.messages if message["role"] == "assistant"]
            assert replies == [f"answer-{turn}" for turn in range(max(0, index - 3), index)]
            assert all("_duplex_fallback_audio" not in message for message in chat.messages)

            await harness.send_fallback_started(request)
            await harness.send_fallback_output(request, {"text": f"answer-{index}", "end_of_turn": True})
            assert len(harness.session.history) == 2 * (index + 1)
            assert all("_duplex_fallback_audio" not in message for message in harness.session.history)
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_audio_requested_fallback_emits_text_immediately_without_audio_playback_or_history() -> None:
    harness = await open_harness(modalities=("text", "audio"))
    try:
        request = await harness.start_fallback_turn()
        assert harness.session.playback.text_requires_complete_audio is True
        assert harness.session.playback.audio_complete is False
        await harness.send_fallback_started(request)
        events = await harness.send_fallback_output(request, {"text": "hello", "audio_duration_ms": 9999})

        assert _event(events, "response.text.delta").delta == "hello"
        assert not any(event.type == "response.output_audio_transcript.delta" for event in events)
        assert harness.session.playback.generated_ms == harness.session.playback.sent_ms == 0
        assert harness.session.assistant_audio_text_marks == ()
        assert not any(
            message.get("role") == "assistant" and message.get("content") for message in harness.session.history
        )
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [commands.CancelResponse(), commands.BargeIn()])
async def test_cancel_before_first_fallback_audio_does_not_commit_visible_text(command) -> None:
    harness = await open_harness(modalities=("text", "audio"))
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_output(request, {"text": "visible but not heard"})
        await harness.run(command)

        assert harness.session.active_response_id is None
        assert harness.session.playback.generated_ms == harness.session.playback.sent_ms == 0
        assert not any(
            message.get("role") == "assistant" and message.get("content") for message in harness.session.history
        )
        assert harness.fallback_cancels[0].response_id == request.request.response_id
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_ack_of_current_fallback_audio_before_done_cannot_commit_unaligned_text() -> None:
    harness = await open_harness(modalities=("text", "audio"))
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_output(request, {"text": "hello"})
        await harness.send_fallback_output(
            request,
            {
                "audio": base64.b64encode(b"\0\0" * 8_000).decode(),
                "audio_format": "pcm",
                "sample_rate_hz": 16_000,
                "audio_duration_ms": 500,
                "audio_complete": True,
            },
        )
        events = await harness.run(commands.AckPlayback(response_id=request.request.response_id, played_ms=500))

        assert harness.session.playback.audio_complete is False
        assert harness.session.assistant_audio_text_marks == ()
        assert _event(events, "playback.acknowledged").details["history_committed"] is False
        assert not any(
            message.get("role") == "assistant" and message.get("content") for message in harness.session.history
        )
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("played_ms", [500, 1000])
async def test_done_requires_full_audio_ack_before_committing_the_entire_unaligned_text(played_ms: int) -> None:
    harness = await open_harness(modalities=("text", "audio"))
    try:
        request = await harness.start_fallback_turn()
        for text in ("hello", " world"):
            await harness.send_fallback_output(request, {"text": text})
            assert harness.session.playback.audio_complete is False
            await harness.send_fallback_output(
                request,
                {
                    "audio": base64.b64encode(b"\0\0" * 8_000).decode(),
                    "audio_format": "pcm",
                    "sample_rate_hz": 16_000,
                    "audio_duration_ms": 500,
                },
            )
        assert harness.session.assistant_audio_text_marks == ()
        events = await harness.send_fallback_output(request, {"end_of_turn": True, "audio_duration_ms": 99_999})

        assert _event(events, "response.done").status == "completed"
        assert harness.session.playback.generated_ms == harness.session.playback.sent_ms == 1000
        assert harness.session.playback.audio_complete is True
        assert not any(
            message.get("role") == "assistant" and message.get("content") for message in harness.session.history
        )
        await harness.run(commands.AckPlayback(response_id=request.request.response_id, played_ms=played_ms))

        assistant_texts = [
            message.get("content") for message in harness.session.history if message.get("role") == "assistant"
        ]
        assert assistant_texts == (["hello world"] if played_ms == 1000 else [""])
        if played_ms == 500:
            await harness.run(commands.AckPlayback(response_id=request.request.response_id, played_ms=1000))
            assert harness.session.history[-1] == {"role": "assistant", "content": "hello world"}
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_audio_requested_fallback_done_without_audio_does_not_commit_text_as_heard() -> None:
    harness = await open_harness(modalities=("text", "audio"))
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_output(request, {"text": "visible but not heard"})
        await harness.send_fallback_output(request, {"end_of_turn": True})
        await harness.run(commands.AckPlayback(response_id=request.request.response_id, played_ms=0))

        assert harness.session.playback.generated_ms == harness.session.playback.sent_ms == 0
        assert not any(
            message.get("role") == "assistant" and message.get("content") for message in harness.session.history
        )
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_fallback_response_audio_option_uses_the_response_scoped_playback_policy() -> None:
    harness = await open_harness(initial_user_text="hello", modalities=("text",))
    try:
        await harness.run(commands.CreateResponse(options={"modalities": ["text", "audio"]}))
        [request] = harness.fallback_requests
        assert harness.session.config.modalities == ["text"]
        assert harness.session.playback.text_requires_complete_audio is True
        await harness.send_fallback_output(request, {"text": "visible but not heard"})
        await harness.run(commands.CancelResponse())
        assert not any(
            message.get("role") == "assistant" and message.get("content") for message in harness.session.history
        )
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_fallback_output_projects_text_done_and_history() -> None:
    harness = await open_harness()
    try:
        request = await harness.start_fallback_turn()
        assert harness.session.playback.text_requires_complete_audio is False
        await harness.send_fallback_started(request)
        events = await harness.send_fallback_output(
            request,
            {
                "text": "hello",
                "data_plane_request_id": request.request.request_id,
                "end_of_turn": True,
            },
        )

        assert _event(events, "response.text.delta").delta == "hello"
        assert not any(event.type == "response.output_audio_transcript.delta" for event in events)
        assert _event(events, "response.done").status == "completed"
        assert harness.session.active_response_id is None
        assert harness.session.active_request_id is None
        assert harness.runner.run.fallback_request_id is None
        assert harness.runner.plugin.data_plane.is_terminal(request.request.request_id)
        assert harness.session.history[-1] == {"role": "assistant", "content": "hello"}

        late_events = await harness.send_fallback_output(
            request,
            {"text": "late", "data_plane_request_id": request.request.request_id},
        )
        assert late_events == []
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_fallback_audio_output_projects_its_transcript() -> None:
    harness = await open_harness(modalities=("text", "audio"))
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_started(request)
        audio = base64.b64encode(b"\x00\x00" * 240).decode()
        events = await harness.send_fallback_output(
            request,
            {
                "audio": audio,
                "audio_format": "pcm",
                "sample_rate_hz": 24_000,
                "audio_duration_ms": 10,
                "text": "hello",
                "data_plane_request_id": request.request.request_id,
            },
        )

        assert _event(events, "response.output_audio_transcript.delta").delta == "hello"

        terminal_events = await harness.send_fallback_output(
            request,
            {"data_plane_request_id": request.request.request_id, "end_of_turn": True},
        )
        assert _event(terminal_events, "response.output_audio_transcript.done").transcript == "hello"
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_fallback_done_marker_does_not_emit_an_empty_text_delta() -> None:
    harness = await open_harness()
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_started(request)
        text_events = await harness.send_fallback_output(
            request,
            {"text": "hello", "data_plane_request_id": request.request.request_id},
        )
        assert _event(text_events, "response.text.delta").delta == "hello"

        terminal_events = await harness.send_fallback_output(
            request,
            {"data_plane_request_id": request.request.request_id, "end_of_turn": True},
        )

        assert not any(event.type == "response.text.delta" for event in terminal_events)
        assert _event(terminal_events, "response.done").status == "completed"
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_two_500ms_pcm_chunks_accumulate_one_second_of_playback() -> None:
    harness = await open_harness(modalities=("text", "audio"))
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_started(request)
        for expected_ms in (500, 1000):
            events = await harness.send_fallback_output(
                request,
                {
                    "audio": base64.b64encode(b"\0\0" * 8_000).decode(),
                    "audio_format": "pcm",
                    "sample_rate_hz": 16_000,
                    "audio_duration_ms": 500,
                },
            )
            delta = _event(events, "response.output_audio.delta")
            assert delta.sample_rate_hz == 16_000
            assert harness.session.playback.generated_ms == expected_ms
            assert harness.session.playback.sent_ms == expected_ms
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_explicit_false_commit_does_not_use_qwen_default_auto_response() -> None:
    harness = await open_harness()
    try:
        await harness.run(
            commands.AppendAudio(
                audio=_pcm_f32(),
                format="pcm_f32le",
                sample_rate_hz=16_000,
                is_speech=True,
            )
        )
        events = await harness.run(commands.Commit(create_response=False))

        assert not harness.fallback_requests
        assert not any(event.type == "response.created" for event in events)
        assert harness.session.active_response_id is None
        assert harness.session.active_request_id is None
        assert harness.runner.model_state.committed_audio_payload is not None
    finally:
        await close_harness(harness)


def _prompt_audio_samples(request: DuplexSessionFallbackRequestMessage) -> list[np.ndarray]:
    chat = build_chat_request(request.request, model="qwen")
    audios: list[np.ndarray] = []
    for message in chat.messages:
        content = message.get("content")
        for part in content if isinstance(content, list) else ():
            if part.get("type") == "audio_url":
                raw = base64.b64decode(part["audio_url"]["url"].split(",", 1)[1])
                with wave.open(io.BytesIO(raw), "rb") as wav:
                    assert wav.getframerate() == 16_000
                    audios.append(np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2"))
    return audios


@pytest.mark.asyncio
@pytest.mark.parametrize("response_mode", ["manual", "auto", "explicit"])
async def test_multiple_commits_submit_each_items_audio_exactly_once(response_mode: str) -> None:
    harness = await open_harness()
    try:
        if response_mode == "explicit":
            await harness.run(commands.UpdateSession(patch={"extra_body": {"auto_response": False}}))
        for value, create_response in ((0.125, False), (0.25, False if response_mode == "manual" else None)):
            await harness.run(
                commands.AppendAudio(audio=_pcm_f32(value), format="pcm_f32le", sample_rate_hz=16_000, is_speech=True)
            )
            if value == 0.25 and response_mode == "explicit":
                create_response = True
            await harness.run(commands.Commit(create_response=create_response))
        if response_mode == "manual":
            await harness.run(commands.CreateResponse())

        [request] = harness.fallback_requests
        audios = _prompt_audio_samples(request)
        assert len(audios) == 2
        np.testing.assert_array_equal(audios[0], np.full(16_000, 4096, dtype="<i2"))
        np.testing.assert_array_equal(audios[1], np.full(16_000, 8192, dtype="<i2"))
        assert len(harness.session.history) == 2
        assert harness.session.pending_input_bytes == 0
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_multiple_deferred_commits_do_not_duplicate_the_aggregate_scheduling_audio() -> None:
    harness = await open_harness()
    try:
        first = await harness.start_fallback_turn()
        await harness.send_fallback_started(first)
        for value, create_response in ((0.125, False), (0.25, True)):
            await harness.run(
                commands.AppendAudio(audio=_pcm_f32(value), format="pcm_f32le", sample_rate_hz=16_000, is_speech=True)
            )
            await harness.run(commands.Commit(create_response=create_response))
        await harness.send_fallback_output(first, {"text": "first answer", "end_of_turn": True})

        [second] = harness.fallback_requests
        audios = _prompt_audio_samples(second)
        assert len(audios) == 3
        np.testing.assert_array_equal(audios[1], np.full(16_000, 4096, dtype="<i2"))
        np.testing.assert_array_equal(audios[2], np.full(16_000, 8192, dtype="<i2"))
        assert harness.session.pending_input_bytes == 0
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_manual_overlap_commit_waits_for_response_create_after_the_active_response_finishes() -> None:
    harness = await open_harness()
    try:
        first = await harness.start_fallback_turn()
        await harness.run(
            commands.AppendAudio(audio=_pcm_f32(0.125), format="pcm_f32le", sample_rate_hz=16_000, is_speech=True)
        )
        await harness.run(commands.Commit(create_response=False))
        await harness.send_fallback_output(first, {"text": "first answer", "end_of_turn": True})

        assert harness.fallback_requests == []
        assert harness.runner.model_state.committed_audio_payload is not None
        assert harness.session.pending_input_bytes == 16_000 * 4
        await harness.run(commands.CreateResponse())
        assert len(harness.fallback_requests) == 1
        assert harness.session.pending_input_bytes == 0
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_response", [None, True])
async def test_listen_only_overlap_retains_sustained_speech_with_fallback_auto_response(auto_response) -> None:
    harness = await open_harness()
    try:
        if auto_response is not None:
            await harness.run(commands.UpdateSession(patch={"extra_body": {"auto_response": auto_response}}))
        first = await harness.start_fallback_turn()
        await harness.send_fallback_started(first)
        await harness.run(
            commands.AppendAudio(audio=_pcm_f32(0.1), format="pcm_f32le", sample_rate_hz=16_000, is_speech=True)
        )

        assert harness.session.overlap_speech_ms == 1000
        events = await harness.run(commands.Commit())
        committed = _event(events, "input_audio_buffer.committed").to_realtime()
        assert committed["event"]["response_create_deferred"] is True
        assert len(harness.session.history) == 2
        assert harness.runner.model_state.committed_audio_payload["audio"] == base64.b64encode(_pcm_f32(0.1)).decode()
        assert harness.session.active_response_id == first.request.response_id
        assert harness.fallback_cancels == []
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("response_state", ["active", "playing"])
@pytest.mark.parametrize("server_vad", [False, True])
async def test_listen_only_quiet_tail_preserves_the_buffered_fallback_speech_turn(
    response_state: str, server_vad: bool
) -> None:
    harness = await open_harness(modalities=("text", "audio"))
    try:
        first = await harness.start_fallback_turn()
        await harness.send_fallback_started(first)
        await harness.send_fallback_output(first, {"text": "first answer"})
        await harness.send_fallback_output(
            first,
            {
                "audio": base64.b64encode(b"\0\0" * 16_000).decode(),
                "audio_format": "pcm",
                "sample_rate_hz": 16_000,
                "audio_duration_ms": 1000,
            },
        )
        if response_state == "playing":
            await harness.send_fallback_output(first, {"end_of_turn": True})
            assert harness.session.active_response_id is None
        assert harness.session.playback.sent_ms == 1000
        assert harness.session.playback.committed_ms == 0

        if server_vad:
            harness.runner.control._detector = _ScriptedDetector(
                [
                    TurnDetectionResult(is_speech=True, speech_active=True, speech_started=True),
                    TurnDetectionResult(is_speech=False, speech_active=False, speech_stopped=True, audio_end_ms=4500),
                    TurnDetectionResult(is_speech=False, speech_active=False),
                ]
            )
        speech = _pcm_f32(0.125, samples=4 * 16_000)
        events = await harness.run(
            commands.AppendAudio(audio=speech, format="pcm_f32le", sample_rate_hz=16_000, is_speech=True)
        )
        assert _event(events, "overlap.decision").reason == "policy_listen_only"
        projector = harness.runner._require_projector()
        item_id = projector.active_input_item_id
        for audio_end_ms in (4500, 5000):
            events.extend(
                await harness.run(
                    commands.AppendAudio(
                        audio=_pcm_f32(0.0, samples=8_000),
                        format="pcm_f32le",
                        sample_rate_hz=16_000,
                        is_speech=False,
                        audio_end_ms=audio_end_ms,
                    )
                )
            )
            assert projector.input_audio_buffer_has_audio is True
            assert projector.active_input_item_id == item_id
            assert harness.session.pending_input_bytes == len(speech)
            assert harness.runner.model_state.audio_buffer.pending_byte_count == len(speech)
            assert harness.runner.model_state.speech_since_commit is True
        assert [event.reason for event in events if event.type == "overlap.decision"] == [
            "policy_listen_only",
            "silence_or_noise",
            "silence_or_noise",
        ]
        events.extend(await harness.run(commands.Commit(create_response=True)))

        assert not [event for event in events if event.type == "error"]
        assert sum(event.type == "input_audio_buffer.speech_started" for event in events) == 1
        assert sum(event.type == "input_audio_buffer.speech_stopped" for event in events) == 1
        assert sum(event.type == "input_audio_buffer.committed" for event in events) == 1
        assert _event(events, "input_audio_buffer.committed").item_id == item_id
        assert projector.input_audio_buffer_has_audio is False
        assert projector.input_speech_started is False
        assert harness.session.pending_input_turns == 0
        if response_state == "active":
            assert harness.fallback_requests == []
            assert harness.session.pending_input_bytes == len(speech)
        await harness.run(commands.AckPlayback(response_id=first.request.response_id, played_ms=1000))
        if response_state == "active":
            await harness.send_fallback_output(first, {"end_of_turn": True})

        [second] = harness.fallback_requests
        assert second.request.response_id != first.request.response_id
        assert base64.b64decode(second.request.input_payload["audio"]) == speech
        np.testing.assert_array_equal(_prompt_audio_samples(second)[-1], np.full(4 * 16_000, 4096, dtype="<i2"))
        assert harness.session.pending_input_bytes == 0
        assert harness.runner.model_state.audio_buffer.pending_byte_count == 0
        assert harness.fallback_cancels == []
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("response_state", ["active", "playing"])
async def test_quiet_only_fallback_overlap_stays_discarded_without_a_new_response(response_state: str) -> None:
    harness = await open_harness(modalities=("text", "audio"))
    try:
        first = await harness.start_fallback_turn()
        await harness.send_fallback_output(
            first,
            {
                "text": "first answer",
                "audio": base64.b64encode(b"\0\0" * 16_000).decode(),
                "audio_format": "pcm",
                "sample_rate_hz": 16_000,
                "audio_duration_ms": 1000,
            },
        )
        if response_state == "playing":
            await harness.send_fallback_output(first, {"end_of_turn": True})
        events = []
        for _ in range(2):
            events.extend(
                await harness.run(
                    commands.AppendAudio(
                        audio=_pcm_f32(0.0, samples=8_000),
                        format="pcm_f32le",
                        sample_rate_hz=16_000,
                        is_speech=False,
                    )
                )
            )
        assert [event.reason for event in events if event.type == "overlap.decision"] == [
            "silence_or_noise",
            "silence_or_noise",
        ]
        assert not any(event.type == "input_audio_buffer.speech_stopped" for event in events)
        assert harness.runner._require_projector().input_audio_buffer_has_audio is False
        assert harness.session.pending_input_bytes == 0
        assert harness.session.pending_input_turns == 0
        assert not harness.runner.model_state.audio_buffer.has_pending()
        assert harness.runner.model_state.speech_since_commit is False
        events = await harness.run(commands.Commit(create_response=True))
        assert _event(events, "error").code == "input_audio_buffer_empty"
        await harness.run(commands.AckPlayback(response_id=first.request.response_id, played_ms=1000))
        if response_state == "active":
            await harness.send_fallback_output(first, {"end_of_turn": True})
        assert harness.fallback_requests == []
        assert harness.session.pending_input_turns == 0
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_promoted_overlap_starts_another_fallback_request() -> None:
    harness = await open_harness()
    try:
        first = await harness.start_fallback_turn()
        await harness.send_fallback_started(first)
        await harness.run(commands.UpdateSession(patch={"extra_body": {"auto_response": False}}))
        await harness.run(
            commands.AppendAudio(audio=_pcm_f32(0.1), format="pcm_f32le", sample_rate_hz=16_000, is_speech=True)
        )
        await harness.run(commands.Commit(create_response=True))
        assert harness.session.pending_input_bytes == 16_000 * 4
        assert harness.fallback_requests == []

        events = await harness.send_fallback_output(first, {"text": "first answer", "end_of_turn": True})

        [second] = harness.fallback_requests
        assert second.request.response_id != first.request.response_id
        assert second.request.request_id != first.request.request_id
        assert second.request.epoch == first.request.epoch
        assert second.request.history[-1]["role"] == "user"
        assert second.request.input_payload["audio"] == base64.b64encode(_pcm_f32(0.1)).decode()
        assert harness.session.active_response_id == second.request.response_id
        assert _event(events, "response.created").response_id == second.request.response_id
        assert harness.session.pending_input_bytes == 0
        assert harness.session.pending_input_turns == 0
        assert harness.session.unanswered_user_items() == 0
        assert harness.runner.model_state.committed_audio_payload is None
        assert harness.port.submissions == []
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("output_modality", ["text", "audio"])
@pytest.mark.parametrize("end_of_turn", [False, True])
async def test_output_overflow_cancels_the_exact_fallback_identity_and_keeps_delivery_bounded(
    output_modality: str, end_of_turn: bool
) -> None:
    limits = DuplexSessionRuntimeConfig(max_pending_output_bytes_per_session=16_384)
    harness = await open_harness(limits=limits, modalities=("text", "audio"))
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_started(request)
        output: dict[str, object] = (
            {"text": "x" * 32_768}
            if output_modality == "text"
            else {
                "audio": base64.b64encode(b"\0\0" * 16_000).decode(),
                "audio_format": "pcm",
                "sample_rate_hz": 16_000,
                "audio_duration_ms": 1000,
            }
        )
        events = await harness.send_fallback_output(request, {**output, "end_of_turn": end_of_turn})

        [cancel] = harness.fallback_cancels
        assert (cancel.request_id, cancel.response_id, cancel.epoch) == (
            request.request.request_id,
            request.request.response_id,
            request.request.epoch,
        )
        assert cancel.reason == "output_backpressure"
        assert _event(events, "error").code == "output_backpressure"
        assert _event(events, "response.done").status == "failed"
        assert _event(events, "session.closed").reason == "output_backpressure"
        assert not any(event.type in {"response.text.delta", "response.output_audio.delta"} for event in events)
        assert harness.output_buffer.pending_events == 0
        assert harness.session.active_response_id is None
        assert harness.runner.run.fallback_request_id is None
        assert harness.port.aborts == []
        assert await harness.send_fallback_output(request, {"text": "late"}) == []
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [commands.CreateResponse(), commands.ClearInput()])
async def test_queued_client_commands_cannot_double_start_or_revive_a_promoted_turn(command) -> None:
    harness = await open_harness()
    try:
        first = await harness.start_fallback_turn()
        await harness.run(
            commands.AppendAudio(audio=_pcm_f32(0.125), format="pcm_f32le", sample_rate_hz=16_000, is_speech=True)
        )
        await harness.run(commands.Commit(create_response=True))
        harness.manager.dispatch(
            DuplexSessionFallbackOutputMessage(
                session_id=SESSION_ID,
                request_id=first.request.request_id,
                response_id=first.request.response_id,
                epoch=first.request.epoch,
                output={"text": "first answer", "end_of_turn": True},
            )
        )
        harness.submit(command)
        events = await harness.settle()

        expected_requests = 1 if isinstance(command, commands.CreateResponse) else 0
        assert len(harness.fallback_requests) == expected_requests
        assert sum(event.type == "response.created" for event in events) == expected_requests
        assert harness.session.pending_input_bytes == 0
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_session_auto_response_false_defers_fallback_until_response_create() -> None:
    harness = await open_harness()
    try:
        await harness.run(commands.UpdateSession(patch={"extra_body": {"auto_response": False}}))
        await harness.run(
            commands.AppendAudio(audio=_pcm_f32(), format="pcm_f32le", sample_rate_hz=16_000, is_speech=True)
        )
        await harness.run(commands.Commit())

        assert harness.fallback_requests == []
        assert harness.session.active_response_id is None
        await harness.run(commands.CreateResponse())
        assert len(harness.fallback_requests) == 1
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_qwen_server_vad_stop_commits_the_buffered_fallback_turn() -> None:
    harness = await open_harness()
    try:
        harness.runner.control._detector = _speech_then_stop_detector()
        await harness.run(
            commands.AppendAudio(
                audio=_pcm_f32(),
                format="pcm_f32le",
                sample_rate_hz=16_000,
                is_speech=True,
            )
        )
        events = await harness.run(
            commands.AppendAudio(
                audio=_pcm_f32(value=0.0, samples=8_000),
                format="pcm_f32le",
                sample_rate_hz=16_000,
                is_speech=False,
            )
        )

        assert "input_audio_buffer.speech_stopped" in [event.type for event in events]
        assert "input_audio_buffer.committed" in [event.type for event in events]
        assert "response.created" in [event.type for event in events]
        assert len(harness.fallback_requests) == 1
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("options", _UNSUPPORTED_TOOL_OPTIONS)
async def test_qwen_fallback_session_admission_rejects_unsupported_tools(options) -> None:
    harness = await open_harness(limits=DuplexSessionRuntimeConfig(max_sessions=2))
    try:
        await harness.manager.handle(
            OpenDuplexSessionMessage(
                control_id="open-tools",
                session_id="qwen-tools",
                session_config=DuplexSessionConfig.from_realtime({"model": "qwen", **options}),
                output_buffer=DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512),
            )
        )
        result = await harness.results.get()

        assert result.ok is False
        assert result.error_code == "unsupported_tools"
        assert "not implemented" in result.error_message
        assert "qwen-tools" not in harness.manager.runners
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("options", _UNSUPPORTED_TOOL_OPTIONS)
async def test_qwen_fallback_session_updates_reject_unsupported_tools_atomically(options) -> None:
    harness = await open_harness()
    try:
        original = harness.session.config.as_dict()
        events = await harness.run(commands.UpdateSession(event_id="tools-update", patch=options))

        error = _event(events, "error")
        assert error.code == "unsupported_tools"
        assert error.related_event_id == "tools-update"
        assert harness.session.config.as_dict() == original
        assert not any(event.type == "session.updated" for event in events)
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("options", _UNSUPPORTED_TOOL_OPTIONS)
async def test_qwen_fallback_response_options_reject_unsupported_tools_without_consuming_input(options) -> None:
    harness = await open_harness()
    try:
        await harness.run(
            commands.AppendAudio(audio=_pcm_f32(), format="pcm_f32le", sample_rate_hz=16_000, is_speech=True)
        )
        await harness.run(commands.Commit(create_response=False))
        events = await harness.run(commands.CreateResponse(event_id="tools-response", options=options))

        error = _event(events, "error")
        assert error.code == "unsupported_tools"
        assert "tool" in error.message.lower()
        assert error.related_event_id == "tools-response"
        assert harness.fallback_requests == []
        assert harness.session.active_response_id is None
        assert harness.runner.model_state.committed_audio_payload is not None
        await harness.run(commands.CreateResponse())
        assert len(harness.fallback_requests) == 1
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_qwen_fallback_rejects_function_outputs_without_reporting_them_created() -> None:
    harness = await open_harness()
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_output(
            request,
            {"function_call": True, "call_id": "call-lookup", "name": "lookup", "arguments": "{}"},
        )
        original_history = harness.session.history
        events = await harness.run(
            commands.CreateItem(
                event_id="tool-output",
                item={"id": "tool-result", "type": "function_call_output", "call_id": "call-lookup", "output": "42"},
            )
        )

        error = _event(events, "error")
        assert error.code == "unsupported_tools"
        assert not any(event.type in {"conversation.item.created", "conversation.item.added"} for event in events)
        assert harness.session.history == original_history
        assert harness.session.active_response_id == request.request.response_id
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_empty_tool_definitions_are_accepted_for_a_tool_free_fallback_turn() -> None:
    harness = await open_harness(initial_user_text="hello")
    try:
        events = await harness.run(commands.UpdateSession(patch={"tools": [], "tool_choice": None}))
        assert _event(events, "session.updated")
        await harness.run(commands.CreateResponse(options={"tools": [], "tool_choice": None}))
        assert len(harness.fallback_requests) == 1
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_qwen_fallback_accepts_response_create_options() -> None:
    harness = await open_harness()
    try:
        await harness.run(
            commands.AppendAudio(
                audio=_pcm_f32(),
                format="pcm_f32le",
                sample_rate_hz=16_000,
                is_speech=True,
            )
        )
        await harness.run(commands.Commit(create_response=False))
        await harness.run(
            commands.CreateResponse(
                options={
                    "instructions": "answer briefly",
                    "temperature": 0.2,
                    "max_tokens": 12,
                }
            )
        )

        assert len(harness.fallback_requests) == 1
        request = harness.fallback_requests[0].request
        assert request.response_config["instructions"] == "answer briefly"
        assert request.response_config["temperature"] == 0.2
        assert request.response_config["max_tokens"] == 12
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_initial_user_text_is_separated_from_fallback_history_snapshot() -> None:
    harness = await open_harness(initial_user_text="start with this")
    try:
        await harness.run(commands.CreateResponse())

        assert len(harness.fallback_requests) == 1
        request = harness.fallback_requests[0].request
        assert request.response_config["initial_user_text"] == "start with this"
        assert list(request.history).count({"role": "user", "content": "start with this"}) == 0
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_fallback_failure_closes_response_and_drops_late_output() -> None:
    harness = await open_harness()
    try:
        request = await harness.start_fallback_turn()
        events = await harness.send_fallback_failed(request)

        assert _event(events, "error").code == "provider_error"
        assert _event(events, "response.done").status == "failed"
        assert harness.session.active_response_id is None
        assert harness.session.active_request_id is None
        assert harness.runner.run.fallback_request_id is None
        assert harness.runner.plugin.data_plane.is_terminal(request.request.request_id)

        late_events = await harness.send_fallback_output(
            request,
            {"text": "late", "data_plane_request_id": request.request.request_id},
        )
        assert late_events == []
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_barge_in_cancels_fallback_and_drops_stale_output() -> None:
    harness = await open_harness()
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_started(request)
        old_epoch = harness.session.epoch

        events = await harness.run(commands.BargeIn())

        assert harness.session.epoch == old_epoch + 1
        assert harness.session.active_response_id is None
        assert [message.request_id for message in harness.fallback_cancels] == [request.request.request_id]
        assert harness.port.aborts == []
        assert _event(events, "response.done").status == "cancelled"

        late_events = await harness.send_fallback_output(
            request,
            {
                "text": "late",
                "data_plane_request_id": request.request.request_id,
                "end_of_turn": True,
            },
            epoch=old_epoch,
        )
        assert late_events == []
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_vad_turn_detected_cancels_fallback_and_marks_qwen_interrupted() -> None:
    harness = await open_harness()
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_started(request)

        cancelled = await harness.runner._cancel_active_response(None, reason="turn_detected")
        await harness.settle()

        assert cancelled is True
        assert harness.runner.model_state.last_turn_interrupted is True
        assert {"role": "system", "content": INTERRUPTION_NOTE} in harness.runner.plugin.fallback_policy_messages(
            harness.runner.model_state
        )
        assert [message.request_id for message in harness.fallback_cancels] == [request.request.request_id]
        assert harness.fallback_cancels[0].reason == "turn_detected"
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_committed_audio_fallback_consumes_pending_initial_user_item() -> None:
    harness = await open_harness(initial_user_text="start with this")
    try:
        request = await harness.start_fallback_turn()

        assert harness.session.unanswered_user_items() == 0
        assert [message["role"] for message in harness.session.history] == ["user", "user"]
        assert harness.session.history[0] == {"role": "user", "content": "start with this"}

        await harness.send_fallback_started(request)
        await harness.send_fallback_output(
            request,
            {"text": "answer", "data_plane_request_id": request.request.request_id, "end_of_turn": True},
        )
        events = await harness.run(commands.CreateResponse())

        assert not harness.fallback_requests
        assert _event(events, "error").code == "response_create_without_input"
        assert harness.session.history[0] == {"role": "user", "content": "start with this"}
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_close_cancels_an_active_qwen_fallback_request() -> None:
    harness = await open_harness()
    request = await harness.start_fallback_turn()

    await close_harness(harness)

    assert len(harness.fallback_cancels) == 1
    cancel = harness.fallback_cancels[0]
    assert cancel.request_id == request.request.request_id
    assert cancel.response_id == request.request.response_id
    assert cancel.epoch == request.request.epoch
    assert cancel.reason == "test_cleanup"


@pytest.mark.asyncio
async def test_expiry_cancels_an_active_qwen_fallback_request() -> None:
    harness = await open_harness()
    request = await harness.start_fallback_turn()
    harness.session.lease.last_activity = 0

    try:
        assert await harness.manager.reap_expired(now=301) == 1
        await harness.settle()

        assert len(harness.fallback_cancels) == 1
        cancel = harness.fallback_cancels[0]
        assert cancel.request_id == request.request.request_id
        assert cancel.response_id == request.request.response_id
        assert cancel.reason == "idle_ttl_expired"
    finally:
        await harness.manager.shutdown()
