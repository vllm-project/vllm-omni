# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import io
import threading
from types import SimpleNamespace

import pytest
from PIL import Image
from vllm.sampling_params import SamplingParams

from tests.engine.duplex.test_session_runner import SESSION_ID, Harness, RecordingStagePort, tts_output
from tests.engine.test_duplex_orchestrator import SESSION_ID as ORCHESTRATOR_SESSION_ID
from tests.engine.test_duplex_orchestrator import _build, _close, _open, _submit
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig, load_deploy_config, merge_pipeline_deploy
from vllm_omni.engine.duplex.commands import (
    AckPlayback,
    AppendAudio,
    CancelResponse,
    ClearInput,
    CloseSession,
    Commit,
    CreateItem,
    CreateResponse,
)
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.messages import DuplexControlResultMessage, OpenDuplexSessionMessage
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.engine.duplex.realtime_commands import translate_realtime_command
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.model_executor.models.aura_omni.duplex.capabilities import aura_duplex_capabilities
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.input import JoyAIFrameBuffer
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.plugin import JoyAIDuplexPlugin
from vllm_omni.model_executor.models.joyai_vl_interaction.pipeline import JOYAI_VL_INTERACTION_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class Processor:
    merge_size = 2
    image_token = "<|image_pad|>"

    def __init__(self):
        self.image_processor = self
        self.image_size_calls = 0
        self.vision_calls = 0

    def get_number_of_image_patches(self, height, width, images_kwargs):
        self.image_size_calls += 1
        return 256 * self.merge_size**2

    def apply_chat_template(self, messages, **kwargs):
        return repr(messages).replace("{'type': 'image'}", self.image_token)

    def tokenizer(self, *, text):
        return {"input_ids": [[0] * len(text[0].replace(self.image_token, "@"))]}

    def __call__(self, *, text, images):
        self.vision_calls += 1
        length = len(self.tokenizer(text=text)["input_ids"][0]) + 255 * len(images or [])
        return {"input_ids": [[0] * length]}


def frame(color="red"):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


async def open_joyai(*, modalities=("text", "audio")):
    plugin = JoyAIDuplexPlugin(lambda audio, *args: "AAAA")
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    port = RecordingStagePort(stage_count=3)
    output: asyncio.Queue[object] = asyncio.Queue()
    results: asyncio.Queue[DuplexControlResultMessage] = asyncio.Queue()
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=DuplexSessionRuntimeConfig(),
        model_config=None,
    )
    await manager.handle(
        OpenDuplexSessionMessage(
            control_id="open",
            session_id=SESSION_ID,
            session_config=DuplexSessionConfig(
                model="joyai", instructions="Watch the pot.", modalities=list(modalities)
            ),
            output_buffer=output_buffer,
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


async def observe(h):
    count = len(h.port.submissions)
    await h.run(AppendAudio(video_frames=(frame(),)))
    await h.run(Commit(final=True, create_response=True))
    assert len(h.port.submissions) == count + 1, [event.to_realtime() for event in h.events]
    return h.port.submissions[-1].context.request_id


def test_frame_buffer_latest_snapshot_rollback_and_clear():
    buffer = JoyAIFrameBuffer()
    buffer.prepare_append({"video_frames": ["old", "latest"]}, operation_id="a", chunk_period_ms=1000, allow_emit=True)
    assert buffer.pending_byte_count == len("latest")
    reservation = buffer.prepare_commit(operation_id="b", chunk_period_ms=1000)
    assert reservation.payload["video_frames"] == ["latest"]
    assert reservation.byte_count == len("latest")
    assert not buffer.has_pending() and buffer.has_reserved()
    reservation.rollback()
    assert buffer.has_pending() and not buffer.has_reserved()
    reservation = buffer.prepare_commit(operation_id="c", chunk_period_ms=1000)
    buffer.prepare_append({"video_frames": ["newer"]}, operation_id="d", chunk_period_ms=1000, allow_emit=False)
    reservation.rollback()
    assert buffer.frames == ["newer"]
    reservation = buffer.prepare_commit(operation_id="e", chunk_period_ms=1000)
    buffer.clear()
    reservation.rollback()
    assert buffer.pending_byte_count == 0 and not buffer.has_reserved()


def test_opt_in_profile_preserves_turn_pipeline():
    normal = load_deploy_config(get_deploy_config_path("joyai_vl_interaction.yaml"))
    duplex = load_deploy_config(get_deploy_config_path("joyai_vl_interaction_duplex.yaml"))
    assert normal.session_mode == "turn"
    assert duplex.session_mode == "duplex"
    assert normal.stages == duplex.stages
    assert normal.connectors == duplex.connectors
    assert not duplex.async_chunk
    stages = merge_pipeline_deploy(JOYAI_VL_INTERACTION_PIPELINE, duplex)
    assert len(stages) == 3
    assert all(stage.session_mode == "duplex" for stage in stages)


def test_capabilities_and_configuration_refuse_audio_and_tools():
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    caps = plugin.capabilities(max_sessions=2)
    assert caps.allows_video_without_audio()
    assert caps.validate_append_modalities(has_audio=False, has_video=True) is None
    assert caps.validate_append_modalities(has_audio=True, has_video=False)
    assert caps.supports_text_only_turn and caps.supports_multi_session
    for extra in ({"realtime_tools": [{"name": "search"}]}, {"auto_response": True}):
        with pytest.raises(DuplexRuntimeConfigError):
            plugin.runtime_config_for_update(DuplexSessionConfig(extra_body=extra), {})


@pytest.mark.asyncio
async def test_silent_observation_finishes_request_and_keeps_session_open():
    h = await open_joyai()
    try:
        request_id = await observe(h)
        prompt = h.port.submissions[-1].prompt
        assert "Watch the pot." in prompt["prompt"]
        assert "audio" not in prompt["multi_modal_data"]
        assert len(prompt["multi_modal_data"]["image"]) == 1
        consumed = h.deliver(
            tts_output(request_id, text="</silence>", finished=True), stage_id=0, segment_finished=True
        )
        assert consumed, "Silent observations must stop before TTS"
        await h.settle()
        assert h.session.active_response_id is None
        assert h.port.cleanups[-1][0] == [request_id]
        next_id = await observe(h)
        assert next_id != request_id
        await h.run(CloseSession())
        assert not h.runner.plugin.data_plane.requests
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_spoken_action_forwards_once_and_audio_completes_response():
    h = await open_joyai()
    try:
        request_id = await observe(h)
        response_id = h.session.active_response_id
        consumed = h.deliver(
            tts_output(request_id, text="</response> The pot is smoking.", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        assert not consumed, "Spoken actions must enter the existing TTS stage"
        await h.settle()
        assert h.session.active_response_id == response_id
        await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2, segment_finished=True)
        await h.run(AckPlayback(response_id=response_id, played_ms=1000))
        assert h.session.active_response_id is None
        events = [event.to_realtime() for event in h.events]
        assert any(e.get("delta") == "The pot is smoking." for e in events)
        assert any(e["type"] == "response.output_audio.delta" for e in events)
        assert h.port.cleanups[-1][0] == [request_id]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_repeated_response_is_silent_and_decision_is_pure():
    h = await open_joyai()
    try:
        h.runner.model_state.last_response_text = "The pot is smoking."
        request_id = await observe(h)
        output = tts_output(request_id, text="</response> The pot is smoking.", finished=True)
        plugin = h.runner.plugin
        decision = plugin.decide_output(
            stage_id=0,
            final_stage_id=2,
            segment_finished=True,
            segment_token_ids=(),
            segment_output_metadata={},
            output=output,
        )
        assert decision is not None
        assert h.runner.model_state.last_response_text == "The pot is smoking."
        assert h.deliver(output, stage_id=0, segment_finished=True)
        await h.settle()
        assert h.runner.model_state.last_response_text == "The pot is smoking."
        assert h.session.active_response_id is None
        request_id = await observe(h)
        assert h.deliver(
            tts_output(request_id, text="</response> The pot is smoking.", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        await h.settle()
        assert h.runner.model_state.last_response_text == "The pot is smoking."
        request_id = await observe(h)
        await h.deliver_and_settle(
            tts_output(request_id, text="</silence>", finished=True), stage_id=0, segment_finished=True
        )
        assert h.runner.model_state.last_response_text == ""
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_releases_request_and_late_output_cannot_speak():
    h = await open_joyai()
    try:
        request_id = await observe(h)
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> Smoke!", finished=True), stage_id=0, segment_finished=True
        )
        await h.run(CancelResponse())
        assert h.runner.plugin.data_plane.is_terminal(request_id)
        assert not any(message.get("content") == "Smoke!" for message in h.session.history)
        before = len(h.events)
        await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2, epoch=0)
        assert not any(event.to_realtime()["type"] == "response.output_audio.delta" for event in h.events[before:])
        assert h.runner.model_state.last_response_text == ""
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["delegation", "missing_audio", "empty_audio"])
async def test_failed_output_releases_resources_and_next_observation_runs(failure):
    h = await open_joyai()
    try:
        request_id = await observe(h)
        action = "</delegation> Ask the agent" if failure == "delegation" else "</response> Smoke!"
        await h.deliver_and_settle(
            tts_output(request_id, text=action, finished=True), stage_id=0, segment_finished=True
        )
        if failure != "delegation":
            output = tts_output(request_id, finished=True)
            output.multimodal_output = {} if failure == "missing_audio" else {"audio": []}
            await h.deliver_and_settle(output, stage_id=2, segment_finished=True)
        assert h.session.active_response_id is None
        assert h.runner.plugin.data_plane.is_terminal(request_id)
        assert h.port.cleanups[-1][0] == [request_id]
        events = [event.to_realtime() for event in h.events]
        failures = [event for event in events if event["type"] == "response.done"]
        assert len(failures) == 1 and failures[0]["response"]["status"] == "failed"
        assert await observe(h) != request_id
        await h.run(CloseSession())
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_text_item_uses_same_session_without_audio():
    h = await open_joyai()
    try:
        await h.run(
            translate_realtime_command(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": "instruction",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Tell me when smoke appears."}],
                    },
                }
            )
        )
        await h.run(CreateResponse())
        assert len(h.port.submissions) == 1
        prompt = h.port.submissions[0].prompt
        assert "Tell me when smoke appears." in prompt["prompt"]
        assert prompt["multi_modal_data"] == {}
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["</silence>", "</response> Repeat", "</delegation> Ask the agent"])
async def test_intermediate_terminal_allows_text_followup(action):
    h = await open_joyai()
    try:
        h.runner.model_state.last_response_text = "Repeat"
        request_id = await observe(h)
        await h.deliver_and_settle(
            tts_output(request_id, text=action, finished=True), stage_id=0, segment_finished=True
        )
        assert h.session.active_response_id is None
        assert h.runner.run.stream_request_id is None
        assert h.port.cleanups[-1][0] == [request_id]
        await h.run(
            translate_realtime_command(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": "followup",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "What do you see now?"}],
                    },
                }
            )
        )
        await h.run(CreateResponse())
        assert len(h.port.submissions) == 2
        assert h.port.submissions[-1].context.request_id != request_id
        assert "What do you see now?" in h.port.submissions[-1].prompt["prompt"]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("modalities", [("text",), ("text", "audio")])
async def test_response_history_requires_playback_only_for_audio(modalities):
    h = await open_joyai(modalities=modalities)
    try:
        await h.run(
            CreateItem(
                item={
                    "id": "question",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What do you see?"}],
                }
            )
        )
        request_id = await observe(h)
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> Smoke!", finished=True), stage_id=0, segment_finished=True
        )
        spoken = "audio" in modalities
        if spoken:
            await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2, segment_finished=True)
        assert h.session.active_response_id is None
        assert any(message.get("content") == "Smoke!" for message in h.session.history) is not spoken
        assert (h.session.playback.sent_ms > 0) is spoken
        assert h.session.playback.text_requires_complete_audio is spoken
        if spoken:
            await h.run(AckPlayback(response_id=response_id, played_ms=1000))
            assert any(message.get("content") == "Smoke!" for message in h.session.history)
        else:
            assert not any(event.to_realtime()["type"] == "response.output_audio.delta" for event in h.events)
        next_request = await observe(h)
        assert next_request != request_id
        assert "</response> Smoke!" in h.port.submissions[-1].prompt["prompt"]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_camera_input_during_output_waits_for_previous_turn():
    h = await open_joyai()
    try:
        request_id = await observe(h)
        await h.run(AppendAudio(video_frames=(frame("blue"),)))
        await h.run(Commit(final=True, create_response=True))
        assert len(h.port.submissions) == 1
        await h.deliver_and_settle(
            tts_output(request_id, text="</silence>", finished=True), stage_id=0, segment_finished=True
        )
        assert len(h.port.submissions) == 2
        latest = h.port.submissions[-1]
        assert latest.context.request_id != request_id
        assert latest.prompt["multi_modal_data"]["image"][-1].getpixel((0, 0)) == (0, 0, 255)
        await h.run(CloseSession())
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("retained_speech", [False, True])
async def test_aura_video_commit_keeps_existing_ack_path(retained_speech):
    h = await open_joyai()
    try:
        await observe(h)
        # Exercise shared dispatch with AURA's actual modality/concurrency contract.
        h.session.capabilities = aura_duplex_capabilities()
        previous_payload = {"audio": "PCM", "format": "pcm_f32le"} if retained_speech else None
        if previous_payload is not None:
            h.runner.model_state.retain_committed_audio(previous_payload, operation_id="speech")
        await h.run(AppendAudio(video_frames=(frame("blue"),)))
        await h.run(Commit(final=True, create_response=True))
        assert len(h.port.submissions) == 1
        assert not h.runner.model_state.deferred_response_create
        assert h.runner.model_state.committed_audio_payload is previous_payload
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("early_ack", [False, True])
async def test_camera_commit_after_done_starts_on_final_playback_ack(early_ack):
    h = await open_joyai()
    try:
        await h.run(
            CreateItem(
                item={
                    "id": "question",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What do you see?"}],
                }
            )
        )
        request_id = await observe(h)
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> The pot is smoking.", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        await h.deliver_and_settle(
            tts_output(request_id, samples=24000, finished=True), stage_id=2, segment_finished=True
        )
        assert h.session.active_response_id is None
        if early_ack:
            await h.run(AckPlayback(response_id=response_id, played_ms=100))
        await h.run(AppendAudio(video_frames=(frame("blue"),)))
        await h.run(Commit(final=True, create_response=True))
        assert len(h.port.submissions) == 1
        await h.run(AckPlayback(response_id=response_id, played_ms=500))
        assert len(h.port.submissions) == 1
        await h.run(AckPlayback(response_id=response_id, played_ms=1000))
        assert len(h.port.submissions) == 2
        assert h.port.submissions[-1].prompt["multi_modal_data"]["image"][-1].getpixel((0, 0)) == (0, 0, 255)
        assert "The pot is smoking." in h.port.submissions[-1].prompt["prompt"]
        await h.run(AckPlayback(response_id=response_id, played_ms=1000))
        assert len(h.port.submissions) == 2
        assert not any(event.to_realtime()["type"] == "error" for event in h.events)
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_new_camera_commit_supersedes_queued_promotion():
    h = await open_joyai()
    try:
        request_id = await observe(h)
        await h.run(AppendAudio(video_frames=(frame("blue"),)))
        await h.run(Commit(final=True, create_response=True))
        assert h.runner.model_state.deferred_response_create
        # Queue C behind A's terminal before processing either. The terminal
        # appends B's promotion behind C, exercising the real mailbox ordering.
        h.deliver(tts_output(request_id, text="</silence>", finished=True), stage_id=0, segment_finished=True)
        h.submit(AppendAudio(video_frames=(frame("green"),)))
        h.submit(Commit(final=True, create_response=True))
        await h.settle()
        assert len(h.port.submissions) == 2
        latest = h.port.submissions[-1]
        assert latest.prompt["multi_modal_data"]["image"][-1].getpixel((0, 0)) == (0, 128, 0)
        assert not any(abort for _, abort in h.port.cleanups)
        assert not h.port.aborts
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(latest.context.request_id, text="</silence>", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        events = [event.to_realtime() for event in h.events]
        assert any(
            event["type"] == "response.done"
            and event["response"]["id"] == response_id
            and event["response"]["status"] == "completed"
            for event in events
        )
        assert not any(event["type"] == "error" for event in events)
        assert h.session.active_response_id is None
        assert h.runner.model_state.committed_audio_payload is None
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [ClearInput(), CloseSession()], ids=["clear", "close"])
async def test_invalidated_camera_commit_cannot_be_promoted(command):
    h = await open_joyai()
    try:
        request_id = await observe(h)
        await h.run(AppendAudio(video_frames=(frame("blue"),)))
        await h.run(Commit(final=True, create_response=True))
        h.deliver(tts_output(request_id, text="</silence>", finished=True), stage_id=0, segment_finished=True)
        h.submit(command)
        await h.settle()
        assert len(h.port.submissions) == 1
        assert h.session.active_response_id is None
        assert h.runner.model_state.committed_audio_payload is None
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_during_prompt_preparation_cannot_submit(monkeypatch):
    h = await open_joyai()
    entered, released = threading.Event(), threading.Event()
    original = h.runner.plugin.plan_append

    def blocked_plan(**kwargs):
        entered.set()
        assert released.wait(timeout=5)
        return original(**kwargs)

    monkeypatch.setattr(h.runner.plugin, "plan_append", blocked_plan)
    try:
        h.submit(AppendAudio(video_frames=(frame(),)))
        h.submit(Commit(final=True, create_response=True))
        assert await asyncio.to_thread(entered.wait, 3)
        await h.run(CancelResponse())
        released.set()
        await h.settle()
        assert not h.port.submissions
        assert not h.runner.plugin.data_plane.requests
        assert h.runner.model_state.frames == []
        assert h.runner.model_state.conversation_images == ()
    finally:
        released.set()
        await h.run(CloseSession())
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_conversation_images_and_camera_frames_share_input_order():
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    state = plugin.create_session_state()
    conversation: list[dict[str, object]] = []
    config = {"conversation": conversation, "modalities": ["text", "audio"]}
    runtime = plugin.runtime_config_for_update(DuplexSessionConfig(), {})

    async def prepare(frames=()):
        payload = {"video_frames": list(frames)}
        prompt_config = plugin.prepare_prompt_config(config, state=state, payload=payload)
        plan = await plugin.prepare_append_plan(
            request_id="r",
            fence=DuplexFence("s"),
            session_config=prompt_config,
            runtime_config=runtime,
            seq=1,
            turn_seq=1,
            payload=payload,
            final=True,
            sampling_params=SamplingParams(max_tokens=64),
        )
        plugin.commit_append_plan("r")
        plugin.data_plane.close_stream("r")
        return [image.getpixel((0, 0)) for image in plan.prompt["multi_modal_data"]["image"]]

    red, blue, green = (255, 0, 0), (0, 0, 255), (0, 128, 0)
    assert await prepare([frame()] * 4) == [red] * 4
    image_message: dict[str, object] = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": frame("blue")}}],
    }
    conversation.append(image_message)
    assert await prepare() == [red, red, red, blue]
    assert await prepare([frame("green")]) == [red, red, blue, green]
    conversation.append({"role": "user", "content": "What changed?"})
    assert await prepare() == [red, red, blue, green]
    conversation.append(image_message.copy())
    assert await prepare() == [red, blue, green, blue]
    config["conversation"] = [{"role": "user", "content": "New scene"}]
    assert await prepare([frame("green")]) == [green]


def test_prompt_budget_counts_visual_tokens_and_keeps_instruction():
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.processor = Processor()
    plugin.max_model_len = 1500
    config = DuplexSessionConfig(instructions="Watch the pot.")
    runtime = plugin.runtime_config_for_update(config, {})
    kwargs = dict(
        request_id="r",
        fence=DuplexFence("s"),
        runtime_config=runtime,
        seq=1,
        turn_seq=1,
        payload={"video_frames": [frame()]},
        final=True,
        sampling_params=SamplingParams(max_tokens=64),
    )
    session_config = {
        "joyai_frames": [frame()] * 3 + [frame("blue")],
        "conversation": [
            {"role": "user", "content": "old" * 1000},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "Any smoke?"},
        ],
    }
    plan = plugin.plan_append(session_config=session_config, **kwargs)
    assert plugin.processor.vision_calls == 0
    assert plugin.processor.image_size_calls == 4
    prompt = plan.prompt
    assert "Watch the pot." in prompt["prompt"] and "Any smoke?" in prompt["prompt"]
    assert "old answer" not in prompt["prompt"]
    assert prompt["multi_modal_data"]["image"][-1].getpixel((0, 0)) == (0, 0, 255)
    assert (
        len(plugin.processor(text=[prompt["prompt"]], images=prompt["multi_modal_data"]["image"])["input_ids"][0])
        <= 1436
    )
    plugin.max_model_len = 10
    with pytest.raises(DuplexRuntimeConfigError, match="context budget"):
        plugin.plan_append(session_config=session_config, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["image", "budget", "submit"])
async def test_failed_append_preserves_visual_history_and_next_observation_runs(failure):
    h = await open_joyai()
    try:
        state = h.runner.model_state
        state.frames = [frame("red")]
        state.conversation_images = (frame("green"),)
        before = (list(state.frames), state.conversation_images)
        failed_frame = frame("blue")
        if failure == "image":
            failed_frame = base64.b64encode(b"not an image").decode()
        elif failure == "budget":
            h.runner.plugin.max_model_len = 10
        else:
            h.port.fail_submit = RuntimeError("stage unavailable")
        await h.run(AppendAudio(video_frames=(failed_frame,)))
        await h.run(Commit(final=True, create_response=True))
        assert not h.port.submissions
        assert not h.runner.plugin.data_plane.requests
        assert (state.frames, state.conversation_images) == before
        assert any(event.to_realtime()["type"] == "error" for event in h.events)

        h.runner.plugin.max_model_len = 16384
        h.port.fail_submit = None
        await h.run(AppendAudio(video_frames=(frame("blue"),)))
        events = await h.run(Commit(final=True, create_response=True))
        assert not any(event.to_realtime()["type"] == "error" for event in events)
        assert len(h.port.submissions) == 1
        assert state.frames[-1] == frame("blue")
        assert len(h.runner.plugin.data_plane.requests) == 1
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_buffered_video_keeps_pending_append_failure_until_fresh_input():
    h = await open_joyai()
    h.port.submit_gate = asyncio.Event()
    try:
        h.submit(AppendAudio(video_frames=(frame(),)))
        h.submit(Commit(final=True, create_response=True))
        await asyncio.wait_for(h.port.submit_started.wait(), timeout=3)
        pending = h.runner.tasks.append_tail
        assert pending is not None and not pending.done()

        h.submit(AppendAudio(video_frames=(frame("blue"),)))
        await h.settle(timeout_s=0.1)
        assert h.runner.model_state.audio_buffer.frames == [frame("blue")]
        assert h.runner.tasks.append_tail is pending

        pending.cancel()
        await h.settle()
        h.port.submit_gate = None
        events = await h.run(Commit(final=True, create_response=True))
        assert any(getattr(event, "code", None) == "commit_aborted" for event in events)
        assert not h.port.submissions

        await h.run(AppendAudio(video_frames=(frame("green"),)))
        events = await h.run(Commit(final=True, create_response=True))
        assert not any(event.to_realtime()["type"] == "error" for event in events)
        assert len(h.port.submissions) == 1
        assert h.runner.model_state.frames == [frame("green")]
    finally:
        if h.port.submit_gate is not None:
            h.port.submit_gate.set()
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [CancelResponse(), CloseSession()], ids=["cancel", "close"])
async def test_cancel_or_close_during_submission_preserves_visual_history(command, mocker):
    h = await open_joyai()
    h.port.submit_gate = asyncio.Event()
    commit_plan = mocker.spy(h.runner.plugin, "commit_append_plan")
    try:
        h.submit(AppendAudio(video_frames=(frame(),)))
        h.submit(Commit(final=True, create_response=True))
        await asyncio.wait_for(h.port.submit_started.wait(), timeout=3)
        assert h.runner.model_state.frames == []
        pending = h.runner.tasks.append_tail
        assert pending is not None and not pending.done()
        request_id = h.port.ensured[-1].request_id
        submissions_before_cancel = len(h.port.submissions)
        await h.run(command)
        await h.settle()
        assert pending.cancelled()
        assert len(h.port.submissions) == submissions_before_cancel
        assert any(request_id in ids for ids, _ in h.port.cleanups)
        commit_plan.assert_not_called()
        assert h.session.input_seq == 0
        assert h.runner.model_state.frames == []
        assert h.runner.model_state.conversation_images == ()
        assert not h.runner.plugin.data_plane.requests
        assert not any(event.to_realtime()["type"] == "error" for event in h.events)
    finally:
        h.port.submit_gate.set()
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_session", [False, True], ids=["cancel", "close"])
async def test_teardown_aborts_accepted_submission_before_ack(close_session, monkeypatch, mocker):
    plugin = JoyAIDuplexPlugin(lambda audio, *args: "AAAA")
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    orchestrator, clients, rpc_q, _ = _build(stages=3, plugin=plugin)
    accepted, ack = asyncio.Event(), asyncio.Event()
    add_request = clients[0].add_request_async

    async def build_request(request_id, *args, **kwargs):
        return SimpleNamespace(request_id=request_id, mm_features=None)

    async def accept_then_wait(*args, **kwargs):
        await add_request(*args, **kwargs)
        accepted.set()
        await ack.wait()

    # Keep the real stage port, pool ownership, and teardown path. Only model
    # preprocessing and engine IPC are replaced; the engine records acceptance
    # before the submission acknowledgement is allowed to return.
    monkeypatch.setattr(orchestrator, "_build_next_stage_request", build_request)
    monkeypatch.setattr(clients[0], "add_request_async", accept_then_wait)
    commit_plan = mocker.spy(plugin, "commit_append_plan")
    try:
        assert (await _open(orchestrator, rpc_q, extra_body={"auto_response": False})).ok
        runner = orchestrator.session_manager.runners[ORCHESTRATOR_SESSION_ID]
        runner.submit(AppendAudio(video_frames=(frame(),)))
        runner.submit(Commit(final=True, create_response=True))
        await asyncio.wait_for(accepted.wait(), timeout=3)
        request_id = clients[0].add_request_calls[-1][0].request_id
        pending = runner.tasks.append_tail
        assert pending is not None and not pending.done()
        assert not runner.session.stage_request_submitted(0, request_id)
        if close_session:
            assert (await _close(orchestrator, rpc_q)).ok
        else:
            await _submit(orchestrator, CancelResponse())
        assert pending.cancelled()
        assert len(clients[0].add_request_calls) == 1
        assert any(request_id in ids for ids in clients[0].abort_calls)
        assert request_id not in orchestrator.request_states
        assert not runner.session.resource_request_ids()
        assert not plugin.data_plane.requests
        commit_plan.assert_not_called()
        assert runner.session.input_seq == 0
        assert runner.model_state.frames == []
        assert runner.model_state.conversation_images == ()
    finally:
        ack.set()
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_late_submission_ack_does_not_commit_visual_history(mocker):
    h = await open_joyai()
    h.port.submit_gate = asyncio.Event()
    commit_plan = mocker.spy(h.runner.plugin, "commit_append_plan")
    cleanup = mocker.spy(h.port, "cleanup")
    try:
        h.submit(AppendAudio(video_frames=(frame(),)))
        h.submit(Commit(final=True, create_response=True))
        await asyncio.wait_for(h.port.submit_started.wait(), timeout=3)
        request_id = h.port.ensured[-1].request_id
        pending = h.runner.tasks.append_tail
        assert pending is not None and not pending.done()
        # Exercise the post-submit fence check separately from task cancellation.
        # Retain cancellation's resource cleanup while allowing the ACK to arrive.
        cancelled_fence = h.session.fence
        h.session.barge_in()
        assert await h.runner.model.signal_cancel_fence(cancelled_fence)
        cleanup.assert_awaited_once_with([request_id], abort=True)
        h.port.submit_gate.set()
        await h.settle()
        assert pending.done() and not pending.cancelled()
        assert len(h.port.submissions) == 1
        assert cleanup.await_args_list == [mocker.call([request_id], abort=True), mocker.call([request_id])]
        commit_plan.assert_not_called()
        assert h.session.input_seq == 0
        assert h.runner.model_state.frames == []
        assert h.runner.model_state.conversation_images == ()
        assert not h.runner.plugin.data_plane.requests
    finally:
        h.port.submit_gate.set()
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("response_override", [False, True])
async def test_text_only_response_finishes_without_tts(response_override):
    h = await open_joyai(modalities=("text", "audio") if response_override else ("text",))
    try:
        await h.run(
            CreateItem(
                item={
                    "id": "question",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What do you see?"}],
                }
            )
        )
        await h.run(CreateResponse(options={"modalities": ["text"]} if response_override else {}))
        request_id = h.port.submissions[-1].context.request_id
        consumed = h.deliver(
            tts_output(request_id, text="</response> Smoke!", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        await h.settle()
        assert consumed  # DIRECT_RESPONSE prevents Stage 1/2 forwarding.
        assert h.session.active_response_id is None
        assert h.runner.model_state.last_response_text == "Smoke!"
        assert any(message.get("content") == "Smoke!" for message in h.session.history)
        # A late downstream output cannot turn a completed text response into an error.
        await h.deliver_and_settle(tts_output(request_id, samples=0, finished=True), stage_id=2)
        events = [event.to_realtime() for event in h.events]
        assert sum(event["type"] == "response.done" for event in events) == 1
        assert not any(event["type"] in {"error", "response.output_audio.delta"} for event in events)
    finally:
        await h.manager.shutdown()


@pytest.mark.parametrize("sizes", [[(32, 32)], [(448, 224), (224, 448), (448, 448)]])
def test_prompt_budget_matches_qwen3_vl_processor(sizes):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

    backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"],
    )
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(patch_size=16),
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=(
            "{% for message in messages %}{{message['role']}}: "
            "{% if message['content'] is string %}{{message['content']}}"
            "{% else %}{% for part in message['content'] %}"
            "{% if part['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
            "{% else %}{{part['text']}}{% endif %}{% endfor %}{% endif %}\n"
            "{% endfor %}assistant: "
        ),
    )
    frames = []
    for size in sizes:
        buffer = io.BytesIO()
        Image.new("RGB", size).save(buffer, format="PNG")
        frames.append(base64.b64encode(buffer.getvalue()).decode())
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.processor = processor
    plugin.max_model_len = 16384
    kwargs = dict(
        request_id="r",
        fence=DuplexFence("s"),
        session_config={"joyai_frames": frames},
        runtime_config=plugin.runtime_config_for_update(DuplexSessionConfig(), {}),
        seq=1,
        turn_seq=1,
        payload={"video_frames": frames},
        final=True,
        sampling_params=SamplingParams(max_tokens=64),
    )
    plan = plugin.plan_append(**kwargs)
    length = len(
        processor(
            text=[plan.prompt["prompt"]],
            images=plan.prompt["multi_modal_data"]["image"],
        )["input_ids"][0]
    )
    # The exact budget fits; one token less must trim an older frame or fail.
    plugin.max_model_len = length + 64
    assert len(plugin.plan_append(**kwargs).prompt["multi_modal_data"]["image"]) == len(sizes)
    plugin.max_model_len -= 1
    if len(sizes) == 1:
        with pytest.raises(DuplexRuntimeConfigError, match="context budget"):
            plugin.plan_append(**kwargs)
    else:
        trimmed = plugin.plan_append(**kwargs)
        assert len(trimmed.prompt["multi_modal_data"]["image"]) == len(sizes) - 1
        assert (
            len(
                processor(
                    text=[trimmed.prompt["prompt"]],
                    images=trimmed.prompt["multi_modal_data"]["image"],
                )["input_ids"][0]
            )
            <= plugin.max_model_len - 64
        )
