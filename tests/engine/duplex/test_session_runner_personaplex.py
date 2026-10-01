# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The session runner driven by the real PersonaPlex plugin.

Same harness as ``test_session_runner.py`` (recording stage port, typed
commands through the manager, stage outputs pushed by hand), plugin swapped:
this pins the lockstep contract -- one Stage 0 submission per 80 ms frame,
prefill slots only on the first append of an epoch, auto-response implied by
``supports_client_commit=False``, cumulative Code2Wav output projected as
24 kHz deltas, and a cancel restarting the epoch.
"""

from __future__ import annotations

import asyncio
import base64
import struct
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.engine.duplex.test_session_runner import (
    SESSION_ID,
    Harness,
    RecordingStagePort,
    _fake_encode_audio,
    close_harness,
    find,
    types,
)
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.messages import DuplexControlResultMessage, OpenDuplexSessionMessage
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.model_executor.models.personaplex.duplex import stage0
from vllm_omni.model_executor.models.personaplex.duplex.config import FRAME_SIZE, SAMPLE_RATE
from vllm_omni.model_executor.models.personaplex.duplex.plugin import PersonaPlexDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PREFILL_SLOTS = 4


@pytest.fixture(autouse=True)
def _fake_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage0, "personaplex_prefill_slots", lambda model_path, voice, persona: PREFILL_SLOTS)


async def open_personaplex_harness(*, extra_body: dict[str, object] | None = None) -> Harness:
    plugin = PersonaPlexDuplexPlugin(_fake_encode_audio)
    port = RecordingStagePort(stage_count=2)
    output: asyncio.Queue[Any] = asyncio.Queue()
    results: asyncio.Queue[Any] = asyncio.Queue()
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=DuplexSessionRuntimeConfig(),
        model_config=SimpleNamespace(model="/models/personaplex-7b-v1"),
    )
    config = DuplexSessionConfig(
        model="nvidia/personaplex-7b-v1",
        modalities=["audio", "text"],
        instructions="Be concise.",
        voice="NATF2.pt",
        extra_body=dict(extra_body or {}),
    )
    await manager.handle(OpenDuplexSessionMessage(control_id="c-open", session_id=SESSION_ID, session_config=config))
    result = await asyncio.wait_for(results.get(), timeout=2.0)
    assert isinstance(result, DuplexControlResultMessage) and result.ok, result
    harness = Harness(manager=manager, port=port, output=output, results=results, runner=manager.runners[SESSION_ID])
    await harness.settle()
    return harness


def frame(samples: int = FRAME_SIZE, *, value: float = 0.05) -> commands.AppendAudio:
    return commands.AppendAudio(
        audio=struct.pack(f"<{samples}f", *([value] * samples)),
        format="pcm_f32le",
        sample_rate_hz=SAMPLE_RATE,
        is_speech=True,
    )


def code2wav_output(request_id: str, *, samples: int, text: str) -> SimpleNamespace:
    """A cumulative Stage 1 (Code2Wav) output the way the orchestrator hands it to the runner."""
    return SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[SimpleNamespace(text=text, token_ids=[], multimodal_output={})],
        multimodal_output={"audio": np.zeros(samples, dtype=np.float32), "sr": SAMPLE_RATE},
    )


def submitted_duplex(port: RecordingStagePort) -> list[dict[str, object]]:
    return [dict(sub.prompt["model_intermediate_buffer"]["duplex"]) for sub in port.submissions]


@pytest.mark.asyncio
async def test_open_advertises_the_lockstep_capabilities_and_reserves_stage0() -> None:
    h = await open_personaplex_harness()
    try:
        created = find(h.events, "session.created")
        capabilities = created.session["capabilities"]
        assert capabilities["chunk_period_ms"] == 80
        assert capabilities["supports_client_commit"] is False
        assert capabilities["supports_barge_in"] is False
        # Not on the wire: the api server reads it from the plugin to decide the chat route.
        assert h.session.capabilities.supports_chat_completions is False
        assert [context.request_id for context in h.port.ensured] == [h.stage0_request_id(epoch=0)]
        assert h.session.runtime_config["personaplex_prefill_slots"] == PREFILL_SLOTS
        assert h.session.runtime_config["personaplex_voice_prompt"] == "NATF2.pt"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_a_session_without_the_auto_response_flag_still_auto_responds() -> None:
    h = await open_personaplex_harness(extra_body={})
    try:
        assert h.runner.out.auto_responds() is True
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_each_frame_is_one_stage0_submission_with_prefill_only_on_the_first() -> None:
    h = await open_personaplex_harness()
    try:
        await h.run(frame())
        await h.run(frame())
        await h.run(frame())

        assert len(h.port.submissions) == 3
        assert [len(sub.prompt["prompt_token_ids"]) for sub in h.port.submissions] == [1 + PREFILL_SLOTS, 1, 1]
        duplex = submitted_duplex(h.port)
        assert [item["seq"] for item in duplex] == [1, 2, 3]
        assert all(item["epoch"] == 0 for item in duplex)
        assert all(item["session_id"] == SESSION_ID for item in duplex)
        assert all("incarnation" not in item for item in duplex)
        assert all(sub.context.request_id == h.stage0_request_id(epoch=0) for sub in h.port.submissions)
        assert [sub.already_submitted for sub in h.port.submissions] == [False, True, True]
        assert h.port.submissions[0].context.stage_sampling_params.max_tokens == 1
        assert not [event for event in h.events if event.type == "error"]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_half_frames_are_buffered_until_a_whole_frame_exists() -> None:
    h = await open_personaplex_harness()
    try:
        await h.run(frame(FRAME_SIZE // 2))
        assert h.port.submissions == []
        assert h.runner.model_state.audio_buffer.pending_byte_count == FRAME_SIZE * 2

        await h.run(frame(FRAME_SIZE // 2))
        assert len(h.port.submissions) == 1
        assert h.runner.model_state.audio_buffer.pending_byte_count == 0
        payload = submitted_duplex(h.port)[0]["payload"]
        assert isinstance(payload, dict)
        assert len(base64.b64decode(payload["audio"])) == FRAME_SIZE * 4
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_a_wrong_rate_append_is_refused_without_a_submission() -> None:
    h = await open_personaplex_harness()
    try:
        events = await h.run(
            commands.AppendAudio(audio=b"\x00" * 64, format="pcm_f32le", sample_rate_hz=16000, is_speech=None)
        )

        error = find(events, "error")
        assert error.code == "bad_event"
        assert "24000" in error.message
        assert h.port.submissions == []
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_cumulative_code2wav_output_becomes_24k_audio_and_text_deltas() -> None:
    h = await open_personaplex_harness()
    try:
        await h.run(frame())
        request_id = h.stage0_request_id(epoch=0)

        first = await h.deliver_and_settle(code2wav_output(request_id, samples=FRAME_SIZE, text="he"), stage_id=1)
        second = await h.deliver_and_settle(
            code2wav_output(request_id, samples=2 * FRAME_SIZE, text="hello"), stage_id=1
        )

        assert types(first)[0] == "response.created"
        delta = find(first, "response.output_audio.delta")
        assert delta.delta == f"wav-{FRAME_SIZE}"
        assert delta.sample_rate_hz == SAMPLE_RATE
        assert find(first, "response.output_audio_transcript.delta").delta == "he"
        assert h.session.active_response_id is not None

        [delta_2] = [event for event in second if event.type == "response.output_audio.delta"]
        assert delta_2.delta == f"wav-{FRAME_SIZE}"
        assert find(second, "response.output_audio_transcript.delta").delta == "llo"
        assert not [event for event in second if event.type == "response.created"]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_cancel_restarts_the_epoch_and_the_next_frame_replays_the_prefill() -> None:
    h = await open_personaplex_harness()
    try:
        await h.run(frame())
        old_request_id = h.stage0_request_id(epoch=0)
        await h.deliver_and_settle(code2wav_output(old_request_id, samples=FRAME_SIZE, text="he"), stage_id=1)

        cancelled = await h.run(commands.CancelResponse())
        await h.run(frame())

        assert h.session.epoch == 1
        assert "audio.cancelled" in types(cancelled) or "response.done" in types(cancelled)
        assert ([old_request_id], True) in h.port.cleanups
        assert h.port.submissions[-1].context.request_id == h.stage0_request_id(epoch=1)
        assert submitted_duplex(h.port)[-1]["seq"] == 1
        assert len(h.port.submissions[-1].prompt["prompt_token_ids"]) == 1 + PREFILL_SLOTS
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_close_aborts_the_stage0_request() -> None:
    h = await open_personaplex_harness()
    await h.run(frame())
    request_id = h.stage0_request_id(epoch=0)

    await h.run(commands.CloseSession(reason="client_close"))

    assert h.port.cleanups == [([request_id], True)]
    await close_harness(h)
