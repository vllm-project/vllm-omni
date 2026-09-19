# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
import base64
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
from vllm_omni.engine.duplex.commands import AckPlayback, CancelResponse, Commit, UpdateSession
from vllm_omni.engine.duplex.config import DuplexSessionConfig
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
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=DuplexSessionRuntimeConfig(),
        model_config=None,
    )
    config = DuplexSessionConfig(model="qwen", modalities=["text", "audio"], overlap_policy="barge_in_on_speech")
    await manager.handle(OpenDuplexSessionMessage(control_id="open", session_id=SESSION_ID, session_config=config))
    result = await results.get()
    assert result.ok, result
    harness = Harness(manager, port, output, results, manager.runners[SESSION_ID])
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
        # Playback has not drained yet. A new utterance commits first.
        assert [m["role"] for m in h.session.history] == ["user"]
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        current_response = h.session.active_response_id
        events = await h.run(AckPlayback(response_id=response_id, played_ms=1000))
        assert not any(e.to_realtime().get("type") == "error" for e in events)
        assert [m["role"] for m in h.session.history] == ["user", "assistant", "user"]
        assert h.session.history[1]["content"] == "first answer"
        assert h.session.active_response_id == current_response
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
        assert not any(m["role"] == "assistant" for m in h.session.history)
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
