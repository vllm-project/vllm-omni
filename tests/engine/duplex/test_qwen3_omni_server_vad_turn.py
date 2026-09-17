# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""L2: one server_vad turn through the real runner with the Qwen3-Omni plugin.

Uses the real ``SileroStreamingVAD`` endpoint rules on CPU with a scripted
frame scorer (same detector class as production; no ONNX artifact required)
and a recording fake stage port.
"""

from __future__ import annotations

import asyncio
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
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    DuplexOutputContext,
    DuplexRequestIdentity,
    DuplexStagePort,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
    duplex_ephemeral_stage_request_id,
)
from vllm_omni.engine.duplex.events import DuplexEvent
from vllm_omni.engine.duplex.messages import (
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    DuplexSessionEventMessage,
    OpenDuplexSessionMessage,
)
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION_ID = "qwen3-omni-server-vad-test"
FRAME_SAMPLES = 512


class ScriptedSpeechBackend:
    """CPU Silero frame contract with scripted probabilities."""

    frame_samples = FRAME_SAMPLES

    def __init__(self, scores: Sequence[float]) -> None:
        self._scores = iter(scores)

    def new_state(self) -> object:
        return object()

    def infer(self, frame: np.ndarray, state: object) -> tuple[float, object]:
        del frame
        try:
            return float(next(self._scores)), state
        except StopIteration:
            return 0.0, state


class ScriptedBackendProvider:
    def __init__(self, backend: ScriptedSpeechBackend) -> None:
        self._backend = backend

    def get(self) -> ScriptedSpeechBackend:
        return self._backend


class RecordingStagePort(DuplexStagePort):
    def __init__(self, *, stage_count: int = 3) -> None:
        self._stage_count = stage_count
        self.ensured: list[DuplexStageRequestContext] = []
        self.submissions: list[DuplexStageSubmission] = []

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
        del request_ids, abort

    async def abort_requests(self, request_ids: list[str]) -> None:
        del request_ids


@dataclass
class Harness:
    manager: DuplexSessionManager
    port: RecordingStagePort
    plugin: Qwen3OmniDuplexPlugin
    output: asyncio.Queue[Any]
    results: asyncio.Queue[Any]
    runner: DuplexSessionRunner
    events: list[DuplexEvent] = field(default_factory=list)

    def submit(self, command: DuplexCommand) -> None:
        self.manager.dispatch(DuplexSessionCommandMessage(session_id=SESSION_ID, command=command))

    async def settle(self, *, idle_s: float = 0.08, timeout_s: float = 5.0) -> list[DuplexEvent]:
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

    def deliver(self, output: object, *, stage_id: int) -> bool:
        session = self.runner.session
        context = DuplexOutputContext(
            identity=DuplexRequestIdentity(session_id=SESSION_ID, fence=session.fence),
            final_stage_id=self.port.stage_count - 1,
            segment_finished=bool(getattr(output, "finished", False)),
        )
        return self.runner.on_stage_output(
            stage_id, output, None, request_id=getattr(output, "request_id"), context=context
        )

    async def deliver_and_settle(self, output: object, *, stage_id: int) -> list[DuplexEvent]:
        self.deliver(output, stage_id=stage_id)
        return await self.settle()


def _encode_audio(audio: object, sample_rate: int, fmt: str, speed: float | None) -> str | None:
    del audio, sample_rate, fmt, speed
    return "ZmFrZQ=="


def pcm_f32(samples: int, *, value: float = 0.05) -> bytes:
    return struct.pack(f"<{samples}f", *([value] * samples))


def types(events: Sequence[DuplexEvent]) -> list[str]:
    return [event.type for event in events]


def fake_thinker(request_id: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        stage_id=0,
        finished=False,
        duplex_turn_id=0,
        outputs=[SimpleNamespace(text=text, cumulative_text=text, token_ids=[], multimodal_output={})],
        multimodal_output={},
    )


def fake_code2wav(request_id: str) -> SimpleNamespace:
    audio = np.zeros(16, dtype=np.float32)
    return SimpleNamespace(
        request_id=request_id,
        stage_id=2,
        finished=True,
        duplex_turn_id=0,
        outputs=[SimpleNamespace(text="", token_ids=[], multimodal_output={"audio": audio, "sr": 24000})],
        multimodal_output={"audio": audio, "sr": 24000},
    )


async def open_harness(scores: Sequence[float] | None = None, *, server_vad: bool = True) -> Harness:
    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    port = RecordingStagePort(stage_count=3)
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
    if scores is not None:
        manager.vad_backend_provider = ScriptedBackendProvider(ScriptedSpeechBackend(scores))
    payload: dict[str, object] = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "modalities": ["text", "audio"],
        "instructions": "You are a concise assistant.",
    }
    if server_vad:
        payload["turn_detection"] = {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 0,
            "silence_duration_ms": 100,
            "min_speech_duration_ms": 32,
            "create_response": True,
            "interrupt_response": True,
        }
    config = DuplexSessionConfig.from_realtime(payload)
    await manager.handle(OpenDuplexSessionMessage(control_id="c-open", session_id=SESSION_ID, session_config=config))
    result = await asyncio.wait_for(results.get(), timeout=2.0)
    assert isinstance(result, DuplexControlResultMessage) and result.ok, result
    harness = Harness(
        manager=manager, port=port, plugin=plugin, output=output, results=results, runner=manager.runners[SESSION_ID]
    )
    await harness.settle()
    return harness


@pytest.mark.asyncio
async def test_server_vad_turn_submits_one_ephemeral_qwen3_request() -> None:
    # Same probability sequence as the Silero endpoint-rule parity case
    # "one utterance": start then stop after trailing silence.
    scores = [0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    h = await open_harness(scores)
    try:
        assert [context.request_id for context in h.port.ensured] == [
            duplex_ephemeral_stage_request_id(DuplexFence(SESSION_ID, epoch=0, turn_id=0), stage_id=0)
        ]

        await h.run(
            commands.AppendAudio(
                audio=pcm_f32(FRAME_SAMPLES * len(scores)),
                format="pcm_f32le",
                sample_rate_hz=16000,
            )
        )
        event_types = types(h.events)
        assert "input_audio_buffer.speech_started" in event_types
        assert "input_audio_buffer.speech_stopped" in event_types
        assert "input_audio_buffer.committed" in event_types
        assert "response.created" in event_types
        assert len(h.port.submissions) == 1
        submission = h.port.submissions[0]
        assert submission.context.request_id.endswith(".r.stage0_t0")
        assert submission.already_submitted is False
        assert submission.resumable is False
        prompt = dict(submission.prompt)
        assert "<|audio_pad|>" in str(prompt.get("prompt", ""))
        assert "audio" in prompt.get("multi_modal_data", {})
        assert prompt.get("additional_information", {}).get("qwen3_duplex") is True

        request_id = submission.context.request_id
        # observe_stage_output projects Thinker text; the orchestrator
        # continues Talker/Code2Wav.
        forwarded = h.deliver(fake_thinker(request_id, "hello there"), stage_id=0)
        assert forwarded is False
        await h.settle()
        assert "response.output_audio_transcript.delta" in types(h.events)
        await h.deliver_and_settle(fake_code2wav(request_id), stage_id=2)
        event_types = types(h.events)
        assert "response.output_audio.delta" in event_types
        assert "response.done" in event_types
    finally:
        await h.manager.shutdown()


def _one_utterance_scores() -> list[float]:
    return [0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_server_vad_second_turn_rebinds_ephemeral_stage0() -> None:
    scores = _one_utterance_scores() + _one_utterance_scores()
    h = await open_harness(scores)
    try:
        await h.run(commands.AppendAudio(audio=pcm_f32(FRAME_SAMPLES * 10), format="pcm_f32le", sample_rate_hz=16000))
        assert len(h.port.submissions) == 1
        first = h.port.submissions[0]
        assert first.context.request_id.endswith(".r.stage0_t0")
        await h.deliver_and_settle(fake_thinker(first.context.request_id, "first"), stage_id=0)
        await h.deliver_and_settle(fake_code2wav(first.context.request_id), stage_id=2)
        assert "response.done" in types(h.events)
        assert h.runner.session.turn_id == 1

        await h.run(commands.AppendAudio(audio=pcm_f32(FRAME_SAMPLES * 10), format="pcm_f32le", sample_rate_hz=16000))
        assert len(h.port.submissions) == 2
        second = h.port.submissions[1]
        assert second.context.request_id.endswith(".r.stage0_t1")
        assert second.context.request_id != first.context.request_id
        assert second.already_submitted is False
        assert second.resumable is False
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_client_commit_submits_one_ephemeral_request() -> None:
    h = await open_harness(server_vad=False)
    try:
        await h.run(
            commands.AppendAudio(
                audio=pcm_f32(FRAME_SAMPLES * 4),
                format="pcm_f32le",
                sample_rate_hz=16000,
                is_speech=True,
            )
        )
        assert h.port.submissions == []
        events = await h.run(commands.Commit(create_response=True, is_speech=True))
        assert "input_audio_buffer.committed" in types(events)
        assert "response.created" in types(events)
        assert len(h.port.submissions) == 1
        submission = h.port.submissions[0]
        assert submission.context.request_id.endswith(".r.stage0_t0")
        assert submission.already_submitted is False
        assert submission.resumable is False
    finally:
        await h.manager.shutdown()
