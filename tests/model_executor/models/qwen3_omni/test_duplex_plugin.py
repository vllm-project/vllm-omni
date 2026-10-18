# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import ast
import asyncio
import base64
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.engine.duplex.test_session_runner import (
    SESSION_ID,
    Harness,
    RecordingStagePort,
    _speech_then_stop,
    append_audio,
    tts_output,
)
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.commands import (
    AckPlayback,
    AppendAudio,
    CancelResponse,
    ClearInput,
    ClearOutputAudio,
    Commit,
    CreateResponse,
    UpdateSession,
)
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.messages import OpenDuplexSessionMessage
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.model_executor.models.qwen3_omni.duplex.input import QwenPcmBuffer
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import MAX_PROMPT_IMAGES, Qwen3OmniDuplexPlugin
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_buffer_commit_and_rollback_preserve_all_samples():
    buffer = QwenPcmBuffer()
    raw = np.ones(101, dtype="<f4").tobytes()
    payload = {"audio": base64.b64encode(raw).decode(), "format": "pcm_f32le", "sample_rate_hz": 16000}
    assert buffer.prepare_append(payload, operation_id="a", chunk_period_ms=1000, allow_emit=True) is None
    reservation = buffer.prepare_commit(operation_id="c", chunk_period_ms=1000)
    assert base64.b64decode(reservation.payload["audio"]) == raw
    reservation.rollback()
    assert buffer.pending_byte_count == len(raw)
    final = buffer.prepare_commit(operation_id="d", chunk_period_ms=1000)
    final.commit()
    assert not buffer.has_pending() and not buffer.has_reserved()


def test_buffer_refuses_video_frames_and_points_at_the_openai_interface():
    buffer = QwenPcmBuffer()
    payload = {
        "audio": base64.b64encode(np.ones(16, dtype="<f4").tobytes()).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "video_frames": ["a-frame"],
    }
    with pytest.raises(ValueError, match="input_image"):
        buffer.prepare_append(payload, operation_id="first", chunk_period_ms=200, allow_emit=True)
    # The refusal must not swallow the audio it was sent with.
    assert not buffer.has_pending()

    del payload["video_frames"]
    buffer.prepare_append(payload, operation_id="second", chunk_period_ms=200, allow_emit=True)
    committed = buffer.prepare_commit(operation_id="second-commit", chunk_period_ms=200)
    assert "video_frames" not in committed.payload
    committed.commit()


async def open_qwen():
    plugin = Qwen3OmniDuplexPlugin(lambda audio, *args: "AAAA")
    plugin.processor = SimpleNamespace(apply_chat_template=lambda messages, **kw: repr(messages))
    port = RecordingStagePort(stage_count=3)
    output: asyncio.Queue[Any] = asyncio.Queue()
    results: asyncio.Queue[Any] = asyncio.Queue()
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=DuplexSessionRuntimeConfig(),
        model_config=None,
    )
    config = DuplexSessionConfig(model="qwen", modalities=["text", "audio"], overlap_policy="barge_in_on_speech")
    await manager.handle(
        OpenDuplexSessionMessage(
            control_id="open", session_id=SESSION_ID, session_config=config, output_buffer=output_buffer
        )
    )
    result = await results.get()
    assert result.ok, result
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


@pytest.mark.asyncio
async def test_two_committed_turns_stream_text_and_audio_and_release_requests():
    h = await open_qwen()
    try:
        ids = []
        for i in range(2):
            await h.run(append_audio())
            assert len(h.port.submissions) == i
            await h.run(Commit(final=True, create_response=True))
            assert len(h.port.submissions) == i + 1, [e.to_realtime() for e in h.events]
            request_id = h.port.submissions[-1].context.request_id
            ids.append(request_id)
            assert h.session.active_request_id == request_id
            assert len(h.port.submissions[-1].prompt["multi_modal_data"]["audio"]) == i + 1
            text = OmniRequestOutput.from_stage_output(
                tts_output(request_id, samples=0, text="hello", finished=True),
                request_id=request_id,
                stage_id=None,
            )
            assert not h.deliver(text, stage_id=0), "Thinker output must still forward to Talker"
            await h.settle()
            assert text.finished, "projection must not mutate the output forwarded to Talker"
            assert h.session.active_response_id is not None, "Thinker finished must not end the response"
            response_id = h.session.active_response_id
            await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2)
            await h.run(AckPlayback(response_id=response_id, played_ms=1000))
            assert h.session.active_response_id is None
            assert h.port.cleanups[-1][0] == [request_id]
        assert len(set(ids)) == 2
        assert h.session.capabilities.as_dict()["implementation_level"] == "turn_based_duplex"
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("played_ms", [0, 500, 1000])
async def test_clear_completed_qwen_playback_allows_next_manual_turn(played_ms):
    h = await open_qwen()
    try:
        await h.run(UpdateSession(patch={"turn_detection": None}))
        assert h.session.config.overlap_policy == "listen_only"
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=False))
        await h.run(CreateResponse())
        old = h.port.submissions[-1].context
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(old.request_id, samples=0, text="first answer", finished=True), stage_id=0
        )
        await h.deliver_and_settle(tts_output(old.request_id, finished=True), stage_id=2)
        if played_ms:
            await h.run(AckPlayback(response_id=response_id, played_ms=played_ms, committed_ms=played_ms))
        assert h.session.active_response_id is None
        assert h.session.playback.sent_ms == 1000

        events = await h.run(ClearOutputAudio(response_id=response_id))
        cleared = [event for event in events if event.type == "output_audio_buffer.cleared"]
        assert len(cleared) == 1
        assert cleared[0].response_id == response_id
        assert h.session.playback.sent_ms == h.session.playback.committed_ms == 0
        # Repeated clear after the queue is empty still identifies the old turn.
        repeated = await h.run(ClearOutputAudio(response_id=response_id))
        assert [event.response_id for event in repeated if event.type == "output_audio_buffer.cleared"] == [response_id]

        await h.run(append_audio(value=-0.25))
        await h.run(Commit(final=True, create_response=False))
        await h.run(CreateResponse())
        assert len(h.port.submissions) == 2, [event.to_realtime() for event in h.events]
        latest = h.port.submissions[-1]
        assert latest.context.request_id != old.request_id
        assert latest.prompt["prompt"] == repr(
            [
                {"role": "user", "content": [{"type": "audio"}]},
                {"role": "assistant", "content": "first answer" if played_ms == 1000 else ""},
                {"role": "user", "content": [{"type": "audio"}]},
            ]
        )
        newest_audio, sample_rate = latest.prompt["multi_modal_data"]["audio"][-1]
        assert sample_rate == 16000
        np.testing.assert_array_equal(newest_audio, np.full(16000, -0.25, dtype=np.float32))

        new_response_id = h.session.active_response_id
        new_epoch = h.session.epoch
        await h.deliver_and_settle(tts_output(latest.context.request_id, finished=False), stage_id=2)
        new_playback = h.session.playback.as_dict()
        late_clear = await h.run(ClearOutputAudio(response_id=response_id))
        assert [event.response_id for event in late_clear if event.type == "output_audio_buffer.cleared"] == [
            response_id
        ]
        assert h.session.active_response_id == new_response_id
        assert h.session.active_request_id == latest.context.request_id
        assert h.session.epoch == new_epoch
        assert h.session.playback.as_dict() == new_playback
        terminals = [event for event in h.events if event.type == "response.done" and event.response_id == response_id]
        assert len(terminals) == 1 and terminals[0].status == "completed"
        assert not any(event.to_realtime().get("type") == "error" for event in h.events)
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("next_input", [False, True])
@pytest.mark.parametrize("delete_before_ack", [False, True])
async def test_audio_without_transcript_accepts_playback_ack(cancelled, next_input, delete_before_ack):
    h = await open_qwen()
    try:
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        request_id = h.port.submissions[-1].context.request_id
        response_id = h.session.active_response_id
        await h.deliver_and_settle(tts_output(request_id, samples=0, text="", finished=True), stage_id=0)
        audio = await h.deliver_and_settle(tts_output(request_id, text="", finished=not cancelled), stage_id=2)
        assert any(event.type == "response.output_audio.delta" for event in audio)
        if cancelled:
            await h.run(CancelResponse())
        assert h.session.active_response_id is None
        assert all(message["role"] == "user" for message in h.session.history)
        if next_input:
            await h.run(append_audio())
            await h.run(Commit(final=True, create_response=True))
        current_response = h.session.active_response_id
        history = h.session.history
        if delete_before_ack:
            assert h.session.delete_history_item(f"item_{response_id}") is True
            ack = await h.run(AckPlayback(response_id=response_id, played_ms=1000))
            assert ack[0].to_realtime()["error"]["code"] == "playback_item_not_found"
            assert response_id not in h.session._playback.by_response
            assert h.session.history == history
            return
        for played_ms in (500, 1000):
            ack = await h.run(AckPlayback(response_id=response_id, played_ms=played_ms))
            assert [event.type for event in ack] == ["playback.acknowledged"], [event.to_realtime() for event in ack]
            assert ack[0].details["history_committed"] is False
            assert ack[0].details["playback"]["played_ms"] == played_ms
            assert h.session.history == history
            assert h.session.active_response_id == current_response
        # Fully played audio must not leave response bookkeeping behind.
        assert response_id not in h.session._playback.by_response
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_drops_old_output_and_accepts_next_turn():
    h = await open_qwen()
    try:
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        old = h.port.submissions[-1].context
        await h.run(CancelResponse())
        await h.deliver_and_settle(tts_output(old.request_id, finished=True), stage_id=2, epoch=old.fence.epoch)
        assert h.session.active_response_id is None
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        assert len(h.port.submissions) == 2
        assert h.port.submissions[-1].context.request_id != old.request_id
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_vad_commits_and_speech_interrupts_active_qwen_response():
    h = await open_qwen()
    try:
        h.runner.control._detector = _speech_then_stop()
        await h.run(append_audio())
        await h.run(append_audio(is_speech=False, value=0.0))
        assert len(h.port.submissions) == 1
        old = h.port.submissions[-1].context
        h.runner.control._detector = _speech_then_stop()
        await h.run(append_audio())
        assert h.session.epoch > old.fence.epoch
        await h.run(append_audio(is_speech=False, value=0.0))
        assert len(h.port.submissions) == 2
        assert h.port.submissions[-1].context.request_id != old.request_id
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("generation_finished", [False, True])
async def test_vad_interrupts_queued_qwen_audio_and_keeps_the_new_utterance(generation_finished):
    h = await open_qwen()
    try:
        h.runner.control._detector = _speech_then_stop()
        await h.run(append_audio(value=0.125))
        await h.run(append_audio(is_speech=False, value=0.0))
        old = h.port.submissions[-1].context
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(old.request_id, samples=0, text="first answer", finished=True), stage_id=0
        )
        await h.deliver_and_settle(tts_output(old.request_id, samples=240000, finished=generation_finished), stage_id=2)
        await h.run(AckPlayback(response_id=response_id, played_ms=1000, committed_ms=1000))
        assert h.session.playback.sent_ms == 10000
        assert h.session.playback.committed_ms == 1000
        if generation_finished:
            assert h.session.active_response_id is None
            assert h.session.active_request_id is None

        h.runner.control._detector = _speech_then_stop()
        events = await h.run(append_audio(value=-0.25))
        if generation_finished:
            cleared = [event for event in events if event.type == "output_audio_buffer.cleared"]
            assert len(cleared) == 1, [event.to_realtime() for event in events]
            assert cleared[0].response_id == response_id
        else:
            assert any(event.type == "response.done" and event.status == "cancelled" for event in events)
        assert h.session.epoch > old.fence.epoch
        # Generation already completed is still completed: stopping playback
        # must not emit a second, cancelled terminal for the same response.
        terminals = [event for event in h.events if event.type == "response.done" and event.response_id == response_id]
        assert len(terminals) == 1
        assert terminals[0].status == ("completed" if generation_finished else "cancelled")

        await h.run(append_audio(is_speech=False, value=0.0))
        assert len(h.port.submissions) == 2
        latest = h.port.submissions[-1]
        assert latest.context.request_id != old.request_id
        assert latest.prompt["prompt"] == repr(
            [
                {"role": "user", "content": [{"type": "audio"}]},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": [{"type": "audio"}]},
            ]
        )
        newest_audio, sample_rate = latest.prompt["multi_modal_data"]["audio"][-1]
        assert sample_rate == 16000
        np.testing.assert_array_equal(newest_audio, np.full(16000, -0.25, dtype=np.float32))
    finally:
        await h.manager.shutdown()


def test_audio_projector_handles_delta_tensor_lists_and_sample_rate():
    import torch

    encoded: list[tuple[Any, object]] = []

    def _encode(audio: object, rate: int, *_args: object) -> str:
        encoded.append((audio, rate))
        return "encoded"

    plugin = Qwen3OmniDuplexPlugin(_encode)
    output = SimpleNamespace(
        request_id="r",
        stage_id=2,
        finished=True,
        outputs=[],
        multimodal_output={"audio": [torch.ones(8), torch.ones(16)], "sr": [torch.tensor(24000)]},
    )
    events = list(plugin.data_plane.project({"data_plane_outputs": [output]}, context={"modalities": ["audio"]}))
    assert len(encoded) == 1 and encoded[0][0].shape == (24,)
    assert events[0]["audio_duration_ms"] == 1
    assert events[0]["end_of_turn"] is True


def test_format_history_merges_user_parts_but_preserves_empty_assistant_boundary():
    text = {"type": "text", "text": "first question"}
    image = {"type": "image"}
    audio = {"type": "audio"}
    source = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": [image]},
        {"role": "user", "content": "first question"},
        {"role": "user", "content": [audio]},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "next question"},
        {"role": "user", "content": [image, audio]},
    ]
    before = repr(source)
    assert Qwen3OmniDuplexPlugin.format_history(source) == [
        source[0],
        {"role": "user", "content": [image, text, audio]},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": [image, audio, {"type": "text", "text": "next question"}]},
    ]
    assert repr(source) == before


def test_audio_byte_budget_evicts_whole_multimodal_turn():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    state = plugin.create_session_state()
    first = {"role": "user", "content": [{"type": "audio_url"}]}
    image = {"role": "user", "content": [{"type": "image_url"}]}
    source = [first, image]
    plugin.prepare_prompt_config({"conversation": source}, state=state, payload={"audio": "A" * (5 * 1024 * 1024)})
    source.extend(
        [
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": [{"type": "audio_url"}]},
        ]
    )
    current = {"audio": "B" * (4 * 1024 * 1024)}
    config = plugin.prepare_prompt_config({"conversation": source}, state=state, payload=current)
    assert config["qwen_messages"] == [{"role": "user", "audio_payload": current}]
    assert len(state.audio_history) == 1
    assert source[1] is image


def test_audio_history_is_bounded_and_does_not_cross_sessions():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    state, other = plugin.create_session_state(), plugin.create_session_state()
    history = []
    for i in range(6):
        history.append({"role": "user", "content": [{"type": "audio_url"}]})
        config = plugin.prepare_prompt_config({"conversation": history}, state=state, payload={"audio": str(i)})
        history.append({"role": "assistant", "content": f"reply-{i}"})
    assert len(state.audio_history) == 4
    assert [m["audio_payload"]["audio"] for m in config["qwen_messages"] if "audio_payload" in m] == [
        "2",
        "3",
        "4",
        "5",
    ]
    assert other.audio_history == []


@pytest.mark.parametrize("assistant_text", ["", "previous answer"])
def test_dropped_audio_removes_its_image_and_assistant(assistant_text):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    state = plugin.create_session_state()
    image = {"role": "user", "content": [{"type": "image_url"}]}
    history = [image]
    for i in range(6):
        history.append({"role": "user", "content": [{"type": "audio_url"}]})
        config = plugin.prepare_prompt_config({"conversation": history}, state=state, payload={"audio": str(i)})
        history.append({"role": "assistant", "content": assistant_text})
    assert config["qwen_messages"] == [
        {"role": "user", "audio_payload": {"audio": "2"}},
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "audio_payload": {"audio": "3"}},
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "audio_payload": {"audio": "4"}},
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "audio_payload": {"audio": "5"}},
    ]


@pytest.mark.asyncio
async def test_late_playback_ack_keeps_answer_before_new_user_input():
    h = await open_qwen()
    try:
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        request_id = h.port.submissions[-1].context.request_id
        response_id = h.session.active_response_id
        await h.deliver_and_settle(tts_output(request_id, samples=0, text="first answer", finished=True), stage_id=0)
        await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2)
        # The unplayed answer reserves its turn without exposing unheard text.
        assert [m["role"] for m in h.session.history] == ["user", "assistant"]
        assert h.session.history[1]["content"] == ""
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        assert h.port.submissions[-1].prompt["prompt"] == repr(
            [
                {"role": "user", "content": [{"type": "audio"}]},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": [{"type": "audio"}]},
            ]
        )
        current_response = h.session.active_response_id
        events = await h.run(AckPlayback(response_id=response_id, played_ms=1000))
        assert not any(e.to_realtime().get("type") == "error" for e in events)
        assert [m["role"] for m in h.session.history] == ["user", "assistant", "user"]
        assert h.session.history[1]["content"] == "first answer"
        assert h.session.active_response_id == current_response
    finally:
        await h.manager.shutdown()


@pytest.fixture
def scripted_turn_detectors(monkeypatch):
    from vllm_omni.engine.duplex.turn_detection import SemanticTurnDetector, TurnDetectionConfig

    def build_detector(config, backend_provider=None, semantic_backend_provider=None):
        detector = (
            SemanticTurnDetector(config, turn_scorer=lambda _: 0.1)
            if config.type == "semantic_vad"
            else original_build(config, backend_provider, semantic_backend_provider)
        )
        detector.vad._frame_scorer = lambda frame: 0.95 if np.abs(frame).max() > 0.1 else 0.02
        return detector

    original_build = TurnDetectionConfig.build_detector
    monkeypatch.setattr(TurnDetectionConfig, "build_detector", build_detector)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mode", ["semantic_vad", "server_vad"])
@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.asyncio
async def test_identical_vad_update_preserves_pending_audio(scripted_turn_detectors, nested, mode, paused):
    h = await open_qwen()
    try:
        await h.run(UpdateSession(patch={"turn_detection": {"type": mode}}))
        detector = h.runner.control._detector
        events = await h.run(append_audio(value=0.2, is_speech=None))
        if paused:
            pause_samples = 16000 if mode == "semantic_vad" else 3200
            events += await h.run(append_audio(samples=pause_samples, value=0.0, is_speech=None))
        assert h.runner.model_state.audio_buffer.has_pending()
        assert not h.port.submissions
        # Re-send the normalized public echo, including defaults, through either spelling.
        turn_detection = dict(h.session.as_public_dict()["turn_detection"])
        patch = (
            {"audio": {"input": {"turn_detection": turn_detection}}} if nested else {"turn_detection": turn_detection}
        )
        updated = await h.run(UpdateSession(patch=patch))
        assert [event.type for event in updated] == ["session.updated"]
        for _ in range(10):
            events += await h.run(append_audio(value=0.0, is_speech=None))
        assert sum(event.type == "input_audio_buffer.committed" for event in events) == 1
        assert sum(event.type == "input_audio_buffer.speech_started" for event in events) == 1
        assert sum(event.type == "input_audio_buffer.speech_stopped" for event in events) == 1
        assert h.runner.control._detector is detector
        assert len(h.port.submissions) == 1
        audio, rate = h.port.submissions[0].prompt["multi_modal_data"]["audio"][-1]
        assert rate == 16000
        assert np.count_nonzero(audio == np.float32(0.2)) == 16000
        expected_end_ms = 5024 if mode == "semantic_vad" else 1536
        assert audio.size == expected_end_ms * 16, "the update must not restart the silence deadline"
    finally:
        await h.manager.shutdown()


@pytest.mark.parametrize(
    "turn_detection",
    [
        {"type": "semantic_vad", "eagerness": "high"},
        {"type": "semantic_vad", "create_response": False},
        {"type": "server_vad"},
        None,
    ],
)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.asyncio
async def test_vad_changes_reject_pending_input_atomically(scripted_turn_detectors, turn_detection, nested):
    h = await open_qwen()
    try:
        await h.run(UpdateSession(patch={"turn_detection": {"type": "semantic_vad"}}))
        await h.run(append_audio(value=0.2, is_speech=None))
        await h.run(append_audio(value=0.0, is_speech=None))
        detector = h.runner.control._detector
        config, runtime = h.session.config, dict(h.session.runtime_config)
        generation = h.session.config_generation
        pending_bytes = h.runner.model_state.audio_buffer.pending_byte_count

        def update():
            patch = (
                {"audio": {"input": {"turn_detection": turn_detection}}}
                if nested
                else {"turn_detection": turn_detection}
            )
            return UpdateSession(patch={**patch, "temperature": 0.3}, event_id="vad-update")

        rejected = await h.run(update())
        assert [event.type for event in rejected] == ["error"]
        assert rejected[0].code == "unsupported_turn_detection"
        assert rejected[0].related_event_id == "vad-update"
        assert "commit or clear" in rejected[0].message
        assert h.session.config is config and h.session.runtime_config == runtime
        assert h.session.config_generation == generation
        assert h.runner.control._detector is detector
        assert h.runner.model_state.audio_buffer.pending_byte_count == pending_bytes

        events = []
        for _ in range(10):
            events += await h.run(append_audio(value=0.0, is_speech=None))
        assert sum(event.type == "input_audio_buffer.committed" for event in events) == 1
        assert len(h.port.submissions) == 1
        audio, _ = h.port.submissions[0].prompt["multi_modal_data"]["audio"][-1]
        assert audio.size == 5024 * 16
        assert np.count_nonzero(audio == np.float32(0.2)) == 16000

        # Once the old input is committed, the same requested change can succeed.
        accepted = await h.run(update())
        assert [event.type for event in accepted] == ["session.updated"]
        assert h.session.config.temperature == 0.3
        assert h.runner.control._detector is not detector
        if turn_detection is None:
            assert h.runner.control._detector is None
        else:
            for key, value in turn_detection.items():
                assert accepted[0].session["turn_detection"][key] == value
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_enabling_vad_requires_pending_manual_input_to_be_cleared(scripted_turn_detectors):
    h = await open_qwen()
    try:
        await h.run(UpdateSession(patch={"turn_detection": None}))
        await h.run(append_audio(value=0.2))
        pending_bytes = h.runner.model_state.audio_buffer.pending_byte_count
        rejected = await h.run(UpdateSession(patch={"turn_detection": {"type": "semantic_vad"}}))
        assert [event.type for event in rejected] == ["error"]
        assert rejected[0].code == "unsupported_turn_detection"
        assert h.runner.control._detector is None
        assert h.runner.model_state.audio_buffer.pending_byte_count == pending_bytes

        await h.run(ClearInput())
        accepted = await h.run(UpdateSession(patch={"turn_detection": {"type": "semantic_vad"}}))
        assert [event.type for event in accepted] == ["session.updated"]
        events = await h.run(append_audio(value=0.4, is_speech=None))
        for _ in range(6):
            events += await h.run(append_audio(value=0.0, is_speech=None))
        assert sum(event.type == "input_audio_buffer.committed" for event in events) == 1
        audio, _ = h.port.submissions[0].prompt["multi_modal_data"]["audio"][-1]
        assert np.count_nonzero(audio == np.float32(0.4)) == 16000
        assert not np.any(audio == np.float32(0.2))
    finally:
        await h.manager.shutdown()


@pytest.mark.parametrize("mode", ["server_vad", "semantic_vad"])
@pytest.mark.parametrize("overlap_policy", ["listen_only", "barge_in_on_speech"])
@pytest.mark.parametrize("queued_appends", [False, True])
@pytest.mark.asyncio
async def test_large_vad_append_submits_separate_audio_turns(mode, overlap_policy, queued_appends):
    from vllm_omni.engine.duplex.turn_detection import SemanticTurnDetector, TurnDetectionConfig

    h = await open_qwen()
    try:
        await h.run(UpdateSession(patch={"overlap_policy": overlap_policy}))
        config = TurnDetectionConfig(type=mode)
        detector = (
            SemanticTurnDetector(config, turn_scorer=lambda _: 0.1)
            if mode == "semantic_vad"
            else config.build_detector()
        )
        detector.vad._frame_scorer = lambda frame: 0.95 if np.abs(frame).max() > 0.1 else 0.02
        h.runner.control._detector = detector
        audio = np.concatenate([np.full(16000, 0.2), np.zeros(80000), np.full(16000, 0.4), np.zeros(80000)])
        for chunk in np.split(audio, 2) if queued_appends else [audio]:
            h.submit(AppendAudio(audio=chunk.astype("<f4").tobytes(), format="pcm_f32le", sample_rate_hz=16000))
        events = await h.settle()
        assert sum(event.type == "input_audio_buffer.committed" for event in events) == 2
        assert not any(event.type == "error" for event in events)
        if overlap_policy == "listen_only":
            assert len(h.port.submissions) == 1
            first, _ = h.port.submissions[0].prompt["multi_modal_data"]["audio"][-1]
            assert np.count_nonzero(first == np.float32(0.2)) == 16000
            assert not np.any(first == np.float32(0.4))
            request_id = h.port.submissions[0].context.request_id
            await h.deliver_and_settle(tts_output(request_id, samples=0, text="first", finished=True), stage_id=0)
            await h.deliver_and_settle(tts_output(request_id, samples=0, finished=True), stage_id=2)
            assert len(h.port.submissions) == 2
        else:
            assert len(h.port.submissions) in (1, 2)
            if len(h.port.submissions) == 2:
                first, _ = h.port.submissions[0].prompt["multi_modal_data"]["audio"][-1]
                assert np.count_nonzero(first == np.float32(0.2)) == 16000
                assert not np.any(first == np.float32(0.4))
        # With barge-in, the second turn can cancel the first before its
        # background submission runs. Its audio must still be a separate turn.
        second, _ = h.port.submissions[-1].prompt["multi_modal_data"]["audio"][-1]
        assert np.count_nonzero(second == np.float32(0.4)) == 16000
        assert not np.any(second == np.float32(0.2))
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming_vad", [False, True])
async def test_finished_qwen_response_does_not_repeat_during_silence(streaming_vad):
    h = await open_qwen()
    try:
        if streaming_vad:
            from vllm_omni.engine.duplex.turn_detection import ServerTurnDetector, TurnDetectionConfig

            detector = ServerTurnDetector(TurnDetectionConfig())
            # Exercise framing, endpointing and reset across the entire response.
            # A deterministic scorer avoids requiring an ONNX artifact in CPU tests.
            detector.vad._frame_scorer = lambda frame: float(np.max(np.abs(frame)) > 0.01)
            h.runner.control._detector = detector
        else:
            h.runner.control._detector = _speech_then_stop()
        await h.run(append_audio())
        await h.run(append_audio(is_speech=False, value=0.0))
        if not streaming_vad:
            h.runner.control._detector = None
        assert len(h.port.submissions) == 1
        request_id = h.port.submissions[0].context.request_id
        response_id = h.session.active_response_id
        await h.deliver_and_settle(tts_output(request_id, samples=0, text="story", finished=True), stage_id=0)
        for _ in range(3):
            await h.run(append_audio(is_speech=False, value=0.0))
        await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2)
        for _ in range(3):
            await h.run(append_audio(is_speech=False, value=0.0))
        await h.run(AckPlayback(response_id=response_id, played_ms=1000))
        await h.settle()
        assert len(h.port.submissions) == 1
        assert not h.runner.model_state.audio_buffer.has_pending()
        assert h.runner.model_state.committed_audio_payload is None
    finally:
        await h.manager.shutdown()


def test_current_audio_is_not_replaced_by_history_ending_with_assistant():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    plugin.processor = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: repr(messages))
    state = plugin.create_session_state()
    first = {"audio": base64.b64encode(np.full(160, 0.1, dtype="<f4").tobytes()).decode()}
    current = {"audio": base64.b64encode(np.full(320, 0.2, dtype="<f4").tobytes()).decode()}
    history = [{"role": "user", "content": [{"type": "audio_url"}]}]
    plugin.prepare_prompt_config({"conversation": history}, state=state, payload=first)
    history.append({"role": "assistant", "content": "Previous answer"})
    config = plugin.prepare_prompt_config({"conversation": history}, state=state, payload=current)
    plan = plugin.plan_append(
        request_id="new",
        fence=None,
        session_config=config,
        runtime_config={},
        seq=1,
        turn_seq=1,
        payload=current,
        final=True,
        sampling_params=None,
    )
    audios = plan.prompt["multi_modal_data"]["audio"]
    assert len(audios) == 2
    np.testing.assert_allclose(audios[-1][0], np.full(320, 0.2, dtype="<f4"))
    assert config["qwen_messages"][-1]["audio_payload"] is current


@pytest.mark.parametrize("current_in_history", [False, True])
def test_empty_assistant_keeps_turn_boundary_in_audio_prompt(current_in_history):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    plugin.processor = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: repr(messages))
    state = plugin.create_session_state()
    first = {"audio": base64.b64encode(np.full(160, 0.1, dtype="<f4").tobytes()).decode()}
    current = {"audio": base64.b64encode(np.full(320, 0.2, dtype="<f4").tobytes()).decode()}
    history = [{"role": "user", "content": [{"type": "audio_url"}]}]
    plugin.prepare_prompt_config({"conversation": history}, state=state, payload=first)
    # Playback was interrupted before any unaligned Thinker text was committed.
    # The assistant turn still separates the old request from the new utterance.
    history.append({"role": "assistant", "content": ""})
    if current_in_history:
        history.append({"role": "user", "content": [{"type": "audio_url"}]})
    config = plugin.prepare_prompt_config({"conversation": history}, state=state, payload=current)
    plan = plugin.plan_append(
        request_id="after-interruption",
        fence=None,
        session_config=config,
        runtime_config={},
        seq=1,
        turn_seq=1,
        payload=current,
        final=True,
        sampling_params=None,
    )
    assert plan.prompt["prompt"] == repr(
        [
            {"role": "user", "content": [{"type": "audio"}]},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": [{"type": "audio"}]},
        ]
    )
    audios = plan.prompt["multi_modal_data"]["audio"]
    assert len(audios) == 2
    np.testing.assert_allclose(audios[0][0], np.full(160, 0.1, dtype="<f4"))
    np.testing.assert_allclose(audios[1][0], np.full(320, 0.2, dtype="<f4"))


@pytest.mark.asyncio
async def test_partial_playback_does_not_commit_unaligned_thinker_text():
    h = await open_qwen()
    try:
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        request_id = h.port.submissions[-1].context.request_id
        response_id = h.session.active_response_id
        await h.deliver_and_settle(tts_output(request_id, samples=0, text="entire answer", finished=True), stage_id=0)
        await h.deliver_and_settle(tts_output(request_id, samples=24000, finished=False), stage_id=2)
        await h.run(AckPlayback(response_id=response_id, played_ms=1000))
        assert not any(m["role"] == "assistant" for m in h.session.history)
        await h.deliver_and_settle(tts_output(request_id, samples=216000, finished=True), stage_id=2)
        await h.run(AckPlayback(response_id=response_id, played_ms=1000))
        assert [m for m in h.session.history if m["role"] == "assistant"] == [{"role": "assistant", "content": ""}]
        await h.run(AckPlayback(response_id=response_id, played_ms=10000))
        assert h.session.history[-1]["content"] == "entire answer"
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_prompt_preparation_runs_off_loop_and_cancellation_prevents_submission():
    import threading

    h = await open_qwen()
    entered, release = threading.Event(), threading.Event()
    thread_ids = []
    original = h.manager.plugin.plan_append

    def slow_plan(**kwargs):
        thread_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return original(**kwargs)

    h.manager.plugin.plan_append = slow_plan
    try:
        await h.run(append_audio())
        commit = asyncio.create_task(h.run(Commit(final=True, create_response=True)))
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        assert thread_ids != [threading.get_ident()]
        await h.run(CancelResponse())
        release.set()
        await commit
        await h.settle()
        assert not h.port.submissions
    finally:
        release.set()
        await h.manager.shutdown()


def camera_frame():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def pcm16_base64(samples=16000):
    """Loud PCM16, so the projection reads it as speech rather than silence."""
    pcm = (np.sin(np.arange(samples) * 0.05) * 20000).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode()


def image_item(item_id="camera"):
    return {
        "id": item_id,
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + camera_frame()},
            {"type": "input_text", "text": "What color is this?"},
        ],
    }


@pytest.mark.asyncio
async def test_image_item_is_context_until_response_create_and_delete_removes_it():
    from vllm_omni.engine.duplex.commands import CreateResponse, DeleteItem

    h = await open_qwen()
    try:
        from vllm_omni.engine.duplex.realtime_commands import translate_realtime_command

        await h.run(translate_realtime_command({"type": "conversation.item.create", "item": image_item()}))
        assert not h.port.submissions
        await h.run(CreateResponse())
        assert len(h.port.submissions) == 1
        prompt = h.port.submissions[-1].prompt
        assert len(prompt["multi_modal_data"]["image"]) == 1
        assert "audio" not in prompt["multi_modal_data"]
        assert "What color is this?" in prompt["prompt"]
        await h.run(CancelResponse())
        await h.run(DeleteItem(item_id="camera"))
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=False))
        assert len(h.port.submissions) == 1
        await h.run(CreateResponse())
        assert "image" not in h.port.submissions[-1].prompt["multi_modal_data"]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_image_context_reaches_audio_turn_and_survives_commit():
    from vllm_omni.engine.duplex.commands import CreateItem

    h = await open_qwen()
    try:
        await h.run(CreateItem(item=image_item()))
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        mm = h.port.submissions[-1].prompt["multi_modal_data"]
        assert len(mm["image"]) == 1 and len(mm["audio"]) == 1
        assert any(
            isinstance(m["content"], list) and any(p.get("type") == "image_url" for p in m["content"])
            for m in h.session.history
        )
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_multiple_commits_before_response_keep_one_combined_audio_in_history():
    from vllm_omni.engine.duplex.commands import CreateItem

    h = await open_qwen()
    try:
        await h.run(append_audio(value=0.1))
        await h.run(Commit(final=True, create_response=False))
        await h.run(CreateItem(item=image_item()))
        await h.run(append_audio(value=0.2))
        await h.run(Commit(final=True, create_response=False))
        await h.run(CreateResponse())
        assert len(h.port.submissions) == 1, [e.to_realtime() for e in h.events]
        first = h.port.submissions[-1]
        audio = first.prompt["multi_modal_data"]["audio"]
        assert len(audio) == 1
        np.testing.assert_allclose(audio[0][0], np.repeat(np.array([0.1, 0.2], dtype=np.float32), 16000))
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(first.context.request_id, samples=0, text="combined answer", finished=True), stage_id=0
        )
        await h.deliver_and_settle(tts_output(first.context.request_id, finished=True), stage_id=2)
        await h.run(AckPlayback(response_id=response_id, played_ms=1000, committed_ms=1000))
        await h.run(append_audio(value=0.3))
        await h.run(Commit(final=True, create_response=True))
        next_audio = h.port.submissions[-1].prompt["multi_modal_data"]["audio"]
        assert len(next_audio) == 2
        np.testing.assert_array_equal(next_audio[0][0], audio[0][0])
        np.testing.assert_allclose(next_audio[1][0], 0.3)
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("image_after_commit", [False, True])
async def test_multimodal_turn_is_merged_retained_and_evicted_with_its_answer(image_after_commit):
    from vllm_omni.engine.duplex.commands import CreateItem

    h = await open_qwen()
    try:
        for turn in range(6):
            if turn == 0 and not image_after_commit:
                await h.run(CreateItem(item=image_item()))
            await h.run(append_audio(value=(turn + 1) / 10))
            await h.run(Commit(final=True, create_response=False))
            assert len(h.port.submissions) == turn
            if turn == 0 and image_after_commit:
                await h.run(CreateItem(item=image_item()))
            await h.run(CreateResponse())
            submission = h.port.submissions[-1]
            prompt = submission.prompt
            messages = ast.literal_eval(prompt["prompt"])
            first_retained = max(0, turn - 3)
            assert [m["role"] for m in messages] == ["user", "assistant"] * (turn - first_retained) + ["user"]
            assert [m["content"] for m in messages if m["role"] == "assistant"] == [
                f"answer-{i}" for i in range(first_retained, turn)
            ]
            audios = prompt["multi_modal_data"]["audio"]
            assert len(audios) == turn - first_retained + 1
            for index, (audio, rate) in enumerate(audios, start=first_retained):
                assert rate == 16000
                np.testing.assert_allclose(audio, (index + 1) / 10)
            assert len(prompt["multi_modal_data"].get("image", [])) == int(first_retained == 0)
            if first_retained == 0:
                parts = messages[0]["content"]
                assert sum(p["type"] == "image" for p in parts) == 1
                assert sum(p["type"] == "audio" for p in parts) == 1
                assert {"type": "text", "text": "What color is this?"} in parts

            response_id = h.session.active_response_id
            request_id = submission.context.request_id
            await h.deliver_and_settle(
                tts_output(request_id, samples=0, text=f"answer-{turn}", finished=True), stage_id=0
            )
            await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2)
            await h.run(AckPlayback(response_id=response_id, played_ms=1000, committed_ms=1000))
            assert h.session.history[-1]["content"] == f"answer-{turn}"
        # Prompt eviction leaves source items individually addressable by the client.
        assert "camera" in h.session.history_item_ids
        assert not any(e.to_realtime().get("type") == "error" for e in h.events)
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_image_only_item_reaches_the_prompt_the_way_the_camera_sends_it():
    """The camera sends each frame as its own item, with no text part.

    A user item used to be forwarded to the session only when it carried text
    or audio, so an image-only item was acknowledged to the client and then
    dropped. Nothing reported it and every prompt that followed was blind.
    """
    from vllm_omni.engine.duplex.realtime_commands import translate_realtime_command

    h = await open_qwen()
    try:
        item = {
            "id": "camera_1",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": "data:image/jpeg;base64," + camera_frame()}],
        }
        await h.run(translate_realtime_command({"type": "conversation.item.create", "item": item}))
        assert [
            part
            for message in h.session.history
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image_url"
        ], "an image-only item never reached the session history"

        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        mm = h.port.submissions[-1].prompt["multi_modal_data"]
        assert len(mm["image"]) == 1 and len(mm["audio"]) == 1
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_one_item_carrying_both_speech_and_a_picture_keeps_both():
    """Audio and images leave the same item by different routes.

    The audio becomes buffered appends that a commit seals under this item's
    id; registering the spoken message there would overwrite an image stored
    under the same id, so the picture has to become its own item.
    """
    from vllm_omni.engine.duplex.commands import CreateResponse
    from vllm_omni.engine.duplex.realtime_commands import translate_realtime_command

    h = await open_qwen()
    try:
        item = {
            "id": "mixed_1",
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + camera_frame()},
                {"type": "input_audio", "audio": pcm16_base64(), "format": "pcm16", "sample_rate_hz": 16000},
            ],
        }
        await h.run(translate_realtime_command({"type": "conversation.item.create", "item": item}))
        await h.run(CreateResponse())
        mm = h.port.submissions[-1].prompt["multi_modal_data"]
        assert len(mm["image"]) == 1, "the picture was dropped on its way out of a mixed item"
        assert len(mm["audio"]) == 1
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_image_limit_rejects_atomically_and_delete_reclaims_capacity():
    from vllm_omni.engine.duplex.commands import CreateItem, DeleteItem

    h = await open_qwen()
    try:
        for i in range(8):
            await h.run(CreateItem(item=image_item(f"camera_{i}")))
        events = await h.run(CreateItem(item=image_item("overflow")))
        assert any(e.to_realtime()["type"] == "error" for e in events)
        assert len(h.session.history) == 8
        await h.run(DeleteItem(item_id="camera_0"))
        await h.run(CreateItem(item=image_item("replacement")))
        assert len(h.session.history) == 8
        assert not h.port.submissions
    finally:
        await h.manager.shutdown()


def test_prompt_holds_stored_images_to_the_stage_limit():
    """The prompt is the last guard before ``limit_mm_per_prompt``.

    A dropped image has to take its ``{"type": "image"}`` placeholder with
    it, because the processor pairs placeholders with ``mm["image"]``
    positionally.
    """
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    plugin.processor = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: repr(messages))
    url = "data:image/jpeg;base64," + camera_frame()
    history: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": f"i{i}"}],
        }
        for i in range(MAX_PROMPT_IMAGES + 2)
    ]
    audio = base64.b64encode(np.full(160, 0.1, dtype="<f4").tobytes()).decode()
    history.append({"role": "user", "audio_payload": {"audio": audio}})

    plan = plugin.plan_append(
        request_id="r",
        fence=None,
        session_config={"qwen_messages": history},
        runtime_config={},
        seq=1,
        turn_seq=1,
        payload={"audio": audio},
        final=True,
        sampling_params=None,
    )

    prompt = plan.prompt
    assert len(prompt["multi_modal_data"]["image"]) == MAX_PROMPT_IMAGES
    assert prompt["prompt"].count("{'type': 'image'}") == MAX_PROMPT_IMAGES
    # Only the image part is retired; the message keeps what it said.
    assert "'i0'" in prompt["prompt"]


def test_text_only_turn_without_prepared_messages_is_refused_not_crashed():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    plugin.processor = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: repr(messages))
    with pytest.raises(DuplexRuntimeConfigError):
        plugin.plan_append(
            request_id="r",
            fence=None,
            # An empty prepared list is not a missing one: it must not fall
            # back to a payload that has no audio to read.
            session_config={"qwen_messages": []},
            runtime_config={},
            seq=1,
            turn_seq=1,
            payload={"is_speech": True},
            final=True,
            sampling_params=None,
        )


@pytest.mark.asyncio
async def test_commit_all_on_done_keeps_unaligned_text_the_client_never_acks():
    h = await open_qwen()
    try:
        # The runner starts every session ack-gated; this client opts out.
        await h.run(UpdateSession(patch={"playback_commit_policy": "commit_all_on_done"}))
        assert h.session.config.playback_commit_policy == "commit_all_on_done"
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        request_id = h.port.submissions[-1].context.request_id
        await h.deliver_and_settle(tts_output(request_id, samples=0, text="entire answer", finished=True), stage_id=0)
        await h.deliver_and_settle(tts_output(request_id, samples=24000, finished=True), stage_id=2)
        # Qwen text is unaligned to its audio, but waiting for an ACK this
        # client will never send would drop the turn from history entirely.
        assert h.session.active_response_id is None
        assert h.session.history[-1]["content"] == "entire answer"
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_the_bound_request_id_is_the_one_the_stage_received():
    h = await open_qwen()
    try:
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        submitted = h.port.submissions[-1].context.request_id
        # Qwen has no resumable core request, so every turn gets its own stage
        # id. The runner names that id before the append runs and the model
        # channel names it again when it submits; they have to agree.
        assert submitted.endswith(".r.stage0-turn0"), submitted
        assert h.session.active_request_id == submitted
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_visual_capture_records_submitted_pixels_and_request(tmp_path, monkeypatch):
    import json

    from PIL import Image

    from vllm_omni.engine.duplex.commands import CreateItem

    monkeypatch.setenv("VLLM_OMNI_QWEN_VISUAL_DEBUG_DIR", str(tmp_path))
    h = await open_qwen()
    try:
        await h.run(CreateItem(item=image_item()))
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        submission = h.port.submissions[-1]
        captures = list(tmp_path.glob("*/input.json"))
        assert len(captures) == 1
        metadata = json.loads(captures[0].read_text())
        assert metadata["request_id"] == submission.context.request_id
        assert metadata["audio_count"] == 1
        assert metadata["prompt"] == submission.prompt["prompt"]
        assert len(metadata["images"]) == 1
        with Image.open(captures[0].parent / metadata["images"][0]) as saved:
            actual = submission.prompt["multi_modal_data"]["image"][0]
            assert saved.size == actual.size
            assert saved.tobytes() == actual.tobytes()
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_listen_audio_commit_does_not_use_video_deferred_response():
    h = await open_qwen()
    try:
        await h.run(UpdateSession(patch={"turn_detection": None}))
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        request_id = h.port.submissions[-1].context.request_id
        response_id = h.session.active_response_id
        await h.run(replace(append_audio(), hints={"overlap_action": "listen"}))
        await h.run(Commit(final=True, create_response=True))
        assert not h.runner.model_state.deferred_response_create
        assert h.runner.model_state.committed_audio_payload is None
        assert h.runner.model_state.audio_buffer.has_pending()
        await h.deliver_and_settle(tts_output(request_id, text="Hello", finished=True), stage_id=0)
        await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2)
        await h.run(AckPlayback(response_id=response_id, played_ms=1000))
        assert len(h.port.submissions) == 1
    finally:
        await h.manager.shutdown()
