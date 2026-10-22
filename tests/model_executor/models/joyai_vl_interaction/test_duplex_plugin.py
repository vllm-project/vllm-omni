# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import io
import struct
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image
from vllm.sampling_params import SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from tests.engine.duplex.test_session_runner import (
    SESSION_ID,
    Harness,
    RecordingStagePort,
    _ScriptedDetector,
    _speech_then_stop,
    append_audio,
    tts_output,
)
from tests.engine.test_duplex_orchestrator import SESSION_ID as ORCHESTRATOR_SESSION_ID
from tests.engine.test_duplex_orchestrator import _build, _close, _open, _submit
from tests.engine.test_orchestrator import _processed_request
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
    DeleteItem,
)
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence, DuplexOutputContext, DuplexRequestIdentity
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.messages import DuplexControlResultMessage, OpenDuplexSessionMessage
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.engine.duplex.realtime_commands import translate_realtime_command
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.turn_detection import TurnDetectionResult
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.model_executor.models.aura_omni.duplex.capabilities import aura_duplex_capabilities
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.input import JoyAIFrameBuffer
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.plugin import JoyAIDuplexPlugin
from vllm_omni.model_executor.models.joyai_vl_interaction.pipeline import (
    JOYAI_VL_INTERACTION_ASR_PIPELINE,
    JOYAI_VL_INTERACTION_PIPELINE,
)
from vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction import asr_to_joyai

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


async def open_joyai(*, modalities=("text", "audio"), asr=False, runtime_config=None):
    plugin = JoyAIDuplexPlugin(lambda audio, *args: "AAAA")
    plugin.configure_pipeline(JOYAI_VL_INTERACTION_ASR_PIPELINE if asr else JOYAI_VL_INTERACTION_PIPELINE)
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    port = RecordingStagePort(stage_count=4 if asr else 3)
    output: asyncio.Queue[object] = asyncio.Queue()
    results: asyncio.Queue[DuplexControlResultMessage] = asyncio.Queue()
    runtime_config = runtime_config or DuplexSessionRuntimeConfig()
    output_buffer = DuplexOutputBuffer(
        max_bytes=runtime_config.max_pending_output_bytes_per_session,
        max_events=runtime_config.max_pending_output_events_per_session,
    )
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=runtime_config,
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


@pytest.mark.asyncio
async def test_quiet_pcm_commit_keeps_client_identity_without_creating_response():
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        command = translate_realtime_command(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(struct.pack("<h", 1) * 3200).decode(),
                "format": "pcm16",
                "sample_rate_hz": 16000,
            }
        )
        assert command.audio and command.is_speech is False
        await h.run(command)
        events = await h.run(
            translate_realtime_command(
                {
                    "type": "input_audio_buffer.commit",
                    "event_id": "quiet-commit",
                    "final": True,
                    "create_response": True,
                }
            )
        )
        committed = next(event.to_realtime() for event in events if event.type == "input_audio_buffer.committed")
        assert committed["event"]["no_response"] is True
        assert committed["event"]["realtime_event_id"] == "quiet-commit"
        assert not h.port.submissions
        assert not any(event.type in {"error", "response.created"} for event in events)
        await observe(h)  # A later camera observation still opens a response.
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["listen_only", "barge_in_on_speech"])
async def test_short_overlap_commit_keeps_identity_and_listen_only_speech(policy):
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        h.session.config.overlap_policy = policy
        await h.run(append_audio(samples=8000))
        await h.run(Commit(final=True, create_response=True, event_id="first-commit"))
        original = h.port.submissions[-1]
        item_id = original.prompt["additional_information"]["realtime_item_id"]
        h.runner.on_stage_output(
            0,
            tts_output(original.context.request_id, finished=True, text="Hello."),
            None,
            request_id=original.context.request_id,
            context=DuplexOutputContext(
                identity=DuplexRequestIdentity(session_id=SESSION_ID, fence=original.context.fence),
                final_stage_id=3,
                segment_finished=True,
            ),
            input_transcript="Hello.",
            input_item_id=item_id,
        )
        await h.settle()
        # An in-progress VAD utterance defers barge-in until commit.
        await h.run(replace(append_audio(samples=8000), hints={"vad": {"speech_started": False}}))
        events = await h.run(Commit(final=True, create_response=True, event_id="second-commit"))
        committed = next(e.to_realtime() for e in events if e.type == "input_audio_buffer.committed")
        assert committed["item_id"] != item_id
        assert not any(e.type == "conversation.item.input_audio_transcription.completed" for e in events)
        assert not h.runner._require_projector().pending_commit_item_ids
        assert len(h.port.submissions) == 1
        if policy == "listen_only":
            assert committed["event"].get("overlap_deferred") is True
            await h.deliver_and_settle(
                tts_output(original.context.request_id, finished=True, text="</silence>"),
                stage_id=1,
                segment_finished=True,
            )
            assert len(h.port.submissions) == 2
            assert h.port.submissions[-1].prompt["additional_information"]["realtime_item_id"] == committed["item_id"]
        else:
            assert committed["event"].get("no_response") is True
            assert committed["event"]["realtime_event_id"] == "second-commit"
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("asr", [False, True])
async def test_full_tts_waveform_fits_joyai_deployment_output_budget(asr):
    from vllm_omni.entrypoints.duplex.audio_encoding import encode_audio

    filename = "joyai_vl_interaction_asr_duplex.yaml" if asr else "joyai_vl_interaction_duplex.yaml"
    deploy = load_deploy_config(get_deploy_config_path(filename))
    h = await open_joyai(asr=asr, runtime_config=deploy.duplex_session)
    h.runner.plugin.data_plane.encode_audio = encode_audio
    try:
        await h.run(translate_realtime_command({"type": "session.update", "session": {"output_audio_format": "pcm16"}}))
        request_id = await observe(h)
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> A long spoken answer.", finished=True),
            stage_id=int(asr),
            segment_finished=True,
        )
        # One complete waveform, conservatively covering 4096 codec frames at 12 Hz.
        events = await h.deliver_and_settle(
            tts_output(request_id, samples=4096 * 2000, finished=True),
            stage_id=2 + int(asr),
            segment_finished=True,
        )
        assert not [e.to_realtime() for e in events if e.type in {"error", "session.closed"}]
        audio = [e.to_realtime()["delta"] for e in events if e.type == "response.output_audio.delta"]
        assert sum(len(base64.b64decode(part)) for part in audio) == 4096 * 2000 * 2
        assert any(e.type == "response.done" and e.to_realtime()["response"]["status"] == "completed" for e in events)
        assert all("audio" not in e.to_realtime()["response"]["metadata"] for e in events if e.type == "response.done")
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_deferred_speech_reply_survives_previous_response_continuation():
    h = await open_joyai(asr=True)
    try:
        await h.run(append_audio(samples=8000))
        await h.run(Commit(final=True, create_response=True))
        first = h.port.submissions[-1].context.request_id
        await h.deliver_and_settle(
            tts_output(first, text="</response> First reply.", finished=True), stage_id=1, segment_finished=True
        )
        await h.run(append_audio(samples=8000))
        await h.run(Commit(final=True, create_response=True))
        await h.deliver_and_settle(tts_output(first, finished=True), stage_id=3, segment_finished=True)
        assert len(h.port.submissions) == 2
        second = h.port.submissions[-1].context.request_id
        assert first != second
        assert h.session.active_response_id is not None
        assert sum(e.type == "response.done" for e in h.events) == 1
        await h.deliver_and_settle(
            tts_output(second, text="</response> Second reply.", finished=True), stage_id=1, segment_finished=True
        )
        await h.deliver_and_settle(tts_output(second, finished=True), stage_id=3, segment_finished=True)
        assert sum(e.type == "response.done" for e in h.events) == 2
        assert sum(e.type == "response.output_audio.delta" for e in h.events) == 2
    finally:
        await h.manager.shutdown()


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


def test_asr_profile_and_sampling_keep_joyai_ownership():
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.configure_pipeline(JOYAI_VL_INTERACTION_ASR_PIPELINE)
    assert plugin.joyai_stage_id == 1 and plugin.audio_stage_id == 3
    assert plugin.input_stage_id({"audio": "pcm"}) == 0
    assert plugin.input_stage_id({"video_frames": ["frame"]}) == 1
    assert plugin.capabilities(max_sessions=1).accepts_input_modality("audio")
    params = plugin.configure_sampling_params(
        runtime_config={"max_tokens": 80, "temperature": 0.2},
        defaults=tuple(SamplingParams(max_tokens=256, temperature=0.0) for _ in range(4)),
    )
    assert params[0].max_tokens == 256 and params[0].temperature == 0.0
    assert params[1].max_tokens == 80 and params[1].temperature == 0.2
    assert plugin.user_transcript(stage_id=0, output=SimpleNamespace(outputs=[]), prompt={}, finished=True) is None
    output = SimpleNamespace(outputs=[SimpleNamespace(text="language English<asr_text>Watch the pot.")])
    assert plugin.user_transcript(stage_id=0, output=output, prompt={}, finished=True) == "Watch the pot."
    assert plugin.user_transcript(stage_id=0, output=output, prompt={}, finished=False) is None
    normal = load_deploy_config(get_deploy_config_path("joyai_vl_interaction_asr.yaml"))
    duplex = load_deploy_config(get_deploy_config_path("joyai_vl_interaction_asr_duplex.yaml"))
    assert normal.stages == duplex.stages and duplex.session_mode == "duplex"


def test_speech_buffer_rolls_back_audio_but_keeps_newest_frame():
    buffer = JoyAIFrameBuffer(accepts_audio=True)
    raw = struct.pack("<ff", 0.25, 0.5)
    payload = {"audio": base64.b64encode(raw).decode(), "format": "pcm_f32le", "sample_rate_hz": 16000}
    buffer.prepare_append({**payload, "video_frames": ["old"]}, operation_id="a", chunk_period_ms=1, allow_emit=True)
    reservation = buffer.prepare_commit(operation_id="b", chunk_period_ms=1)
    assert reservation.byte_count == len(raw) + len("old")
    buffer.prepare_append({**payload, "video_frames": ["new"]}, operation_id="c", chunk_period_ms=1, allow_emit=True)
    reservation.rollback()
    assert buffer.frames == ["new"] and buffer.audio == raw + raw
    reservation = buffer.prepare_commit(operation_id="d", chunk_period_ms=1)
    assert base64.b64decode(reservation.payload["audio"]) == raw + raw
    reservation.commit()
    assert not buffer.has_pending() and not buffer.has_reserved()
    buffer.prepare_append(payload, operation_id="e", chunk_period_ms=1, allow_emit=True)
    reservation = buffer.prepare_commit(operation_id="f", chunk_period_ms=1)
    buffer.clear()
    reservation.rollback()
    assert not buffer.has_pending()


@pytest.mark.parametrize("chunk_period_ms", [250, 1000])
def test_speech_buffer_bounds_idle_preroll_and_preserves_speech(chunk_period_ms):
    buffer = JoyAIFrameBuffer(accepts_audio=True)
    samples = 16000 * chunk_period_ms // 1000
    idle = struct.pack("<f", 0.01) * samples
    speech = struct.pack("<f", 0.25) * (samples * 2)

    def append(raw, is_speech, frame_name):
        buffer.prepare_append(
            {
                "audio": base64.b64encode(raw).decode(),
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
                "is_speech": is_speech,
                "video_frames": [frame_name],
            },
            operation_id="append",
            chunk_period_ms=chunk_period_ms,
            allow_emit=False,
        )

    for i in range(10):
        append(idle, False, str(i))
    assert buffer.audio == idle
    assert buffer.frames == ["9"]
    assert buffer.pending_byte_count == len(idle) + 1
    append(speech, True, "speech")
    append(idle, False, "tail")
    expected = idle + speech + idle
    assert buffer.audio == expected
    reservation = buffer.prepare_commit(operation_id="commit", chunk_period_ms=chunk_period_ms)
    assert base64.b64decode(reservation.payload["audio"]) == expected
    assert reservation.byte_count == len(expected) + len("tail")
    reservation.rollback()
    append(idle, False, "new")
    assert buffer.audio == expected + idle  # A failed speech commit still owns the full utterance.
    buffer.clear()
    append(idle, False, "camera")
    reservation = buffer.prepare_commit(operation_id="camera", chunk_period_ms=chunk_period_ms)
    assert reservation.payload == {"video_frames": ["camera"], "is_speech": False}
    assert reservation.byte_count == len(idle) + len("camera")
    reservation.commit()
    assert not buffer.has_pending() and not buffer.has_reserved()


@pytest.mark.asyncio
async def test_idle_vad_camera_stream_stays_bounded_and_accepts_speech():
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        h.manager.runtime_config = DuplexSessionRuntimeConfig(max_pending_input_bytes_per_session=256 * 1024)
        idle_turns = 10
        h.runner.control._detector = _ScriptedDetector(
            [TurnDetectionResult(is_speech=False, speech_active=False)] * idle_turns
            + [
                TurnDetectionResult(is_speech=True, speech_active=True, speech_started=True),
                TurnDetectionResult(is_speech=False, speech_active=False, speech_stopped=True, should_commit=True),
            ]
        )
        silent_pcm = bytes(16000 * 2)
        red, blue = frame("red"), frame("blue")
        for i in range(idle_turns):
            latest = blue if i == idle_turns - 1 else red
            events = await h.run(
                AppendAudio(audio=silent_pcm, format="pcm16", sample_rate_hz=16000, video_frames=(latest,))
            )
            assert not any(getattr(event, "code", None) == "input_backpressure" for event in events)
        buffer = h.runner.model_state.audio_buffer
        assert buffer.frames == [blue]
        assert len(buffer.audio) <= 16000 * 4
        assert h.session.pending_input_bytes == buffer.pending_byte_count
        assert not h.port.submissions
        await h.run(AppendAudio(audio=struct.pack("<h", 1000) * 16000, format="pcm16", sample_rate_hz=16000))
        await h.run(AppendAudio(audio=silent_pcm, format="pcm16", sample_rate_hz=16000, video_frames=(blue,)))
        assert len(h.port.submissions) == 1
        assert h.port.submissions[0].context.stage_id == 0
        assert "audio" in h.port.submissions[0].prompt["multi_modal_data"]
        assert h.session.pending_input_bytes == 0
        assert not buffer.has_pending()
    finally:
        await h.manager.shutdown()


def test_rejected_idle_append_restores_preroll_and_latest_frame():
    buffer = JoyAIFrameBuffer(accepts_audio=True)
    idle = struct.pack("<f", 0.00001) * 16000
    payload = {
        "audio": base64.b64encode(idle).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "is_speech": False,
        "video_frames": ["red"],
    }
    buffer.prepare_append(payload, operation_id="first", chunk_period_ms=1000, allow_emit=False).commit()
    before = (bytes(buffer.audio), buffer.frames, buffer.version, buffer._had_speech)
    payload.update(audio=base64.b64encode(bytes(8000 * 4)).decode(), video_frames=["blue"])
    reservation = buffer.prepare_append(payload, operation_id="rejected", chunk_period_ms=1000, allow_emit=False)
    assert buffer.frames == ["blue"] and len(buffer.audio) == len(idle)
    reservation.rollback()
    assert (bytes(buffer.audio), buffer.frames, buffer.version, buffer._had_speech) == before
    assert not reservation.active


@pytest.mark.asyncio
async def test_rejected_mixed_append_preserves_buffer_and_byte_accounting():
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        image = frame()
        h.manager.runtime_config = DuplexSessionRuntimeConfig(
            max_pending_input_bytes_per_session=8000 + len(image) // 2
        )
        pcm = struct.pack("<h", 1000) * 1000
        h.runner.control._detector = _ScriptedDetector([TurnDetectionResult(is_speech=True, speech_active=True)] * 4)
        await h.run(AppendAudio(audio=pcm, format="pcm16", sample_rate_hz=16000))
        buffer = h.runner.model_state.audio_buffer
        before = (bytes(buffer.audio), list(buffer.frames), buffer.version, buffer._had_speech)
        for _ in range(2):
            events = await h.run(AppendAudio(audio=pcm, format="pcm16", sample_rate_hz=16000, video_frames=(image,)))
            assert any(getattr(event, "code", None) == "input_backpressure" for event in events)
            assert (bytes(buffer.audio), buffer.frames, buffer.version, buffer._had_speech) == before
            assert h.session.pending_input_bytes == buffer.pending_byte_count == 4000
        await h.run(AppendAudio(audio=pcm, format="pcm16", sample_rate_hz=16000))
        assert h.session.pending_input_bytes == buffer.pending_byte_count == 8000
        await h.run(Commit(final=True, create_response=True))
        assert len(h.port.submissions) == 1
        assert h.session.pending_input_bytes == 0
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_silent_audio_camera_commit_uses_visual_entry_and_releases_bytes():
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        await h.run(AppendAudio(audio=bytes(16000 * 2), format="pcm16", sample_rate_hz=16000, video_frames=(frame(),)))
        await h.run(Commit(final=True, create_response=True))
        assert len(h.port.submissions) == 1
        assert h.port.submissions[0].context.stage_id == 1
        assert "audio" not in h.port.submissions[0].prompt["multi_modal_data"]
        assert h.session.pending_input_bytes == 0
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred", [False, True])
async def test_camera_commit_preserves_retained_speech_and_transcript_owner(deferred):
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        active_request = await observe(h) if deferred else None
        pcm = struct.pack("<hh", 1000, 2000) * 4000
        await h.run(
            translate_realtime_command(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode(),
                    "format": "pcm16",
                    "sample_rate_hz": 16000,
                    "video_frames": [frame("green")],
                }
            )
        )
        events = await h.run(
            translate_realtime_command({"type": "input_audio_buffer.commit", "final": True, "create_response": False})
        )
        speech_item_id = next(e.to_realtime()["item_id"] for e in events if e.type == "input_audio_buffer.committed")
        assert len(h.port.submissions) == int(deferred)
        assert h.runner.model_state.committed_audio_payload["audio"]
        await h.run(translate_realtime_command({"type": "input_audio_buffer.append", "video_frames": [frame("blue")]}))
        events = await h.run(
            translate_realtime_command({"type": "input_audio_buffer.commit", "final": True, "create_response": True})
        )
        camera_item_id = next(e.to_realtime()["item_id"] for e in events if e.type == "input_audio_buffer.committed")
        assert camera_item_id != speech_item_id
        if deferred:
            assert len(h.port.submissions) == 1
            await h.deliver_and_settle(
                tts_output(active_request, finished=True, text="</silence>"), stage_id=1, segment_finished=True
            )
        assert len(h.port.submissions) == int(deferred) + 1
        submission = h.port.submissions[-1]
        assert submission.context.stage_id == 0
        audio, sample_rate = submission.prompt["multi_modal_data"]["audio"]
        assert sample_rate == 16000
        assert audio.tobytes() == struct.pack("<ff", 1000 / 32768, 2000 / 32768) * 4000
        info = submission.prompt["additional_information"]
        assert info["realtime_item_id"] == speech_item_id
        expected_colors = ([(255, 0, 0)] if deferred else []) + [(0, 128, 0), (0, 0, 255)]
        assert [image.getpixel((0, 0)) for image in info["deferred_multi_modal_data"]["image"]] == expected_colors
        transcript = "Describe the blue scene."
        h.runner.on_stage_output(
            0,
            tts_output(submission.context.request_id, finished=True, text=transcript),
            None,
            request_id=submission.context.request_id,
            context=DuplexOutputContext(
                identity=DuplexRequestIdentity(session_id=SESSION_ID, fence=submission.context.fence),
                final_stage_id=3,
                segment_finished=True,
            ),
            input_transcript=transcript,
            input_item_id=info["realtime_item_id"],
        )
        await h.settle()
        transcriptions = [
            e.to_realtime() for e in h.events if e.type == "conversation.item.input_audio_transcription.completed"
        ]
        assert [(e["item_id"], e["transcript"]) for e in transcriptions] == [(speech_item_id, transcript)]
        assert h.session.pending_input_bytes == 0
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("first_transcript", ["Remember the red cup.", ""])
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("count", [2, 3])
async def test_batched_commits_keep_separate_audio_and_transcript_owners(cancel, first_transcript, interleave, count):
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        item_ids = []
        for sample in range(1000, (count + 1) * 1000, 1000):
            await h.run(
                translate_realtime_command(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(struct.pack("<h", sample) * 8000).decode(),
                        "format": "pcm16",
                        "sample_rate_hz": 16000,
                        "video_frames": [frame("red" if sample == 1000 else "blue")],
                    }
                )
            )
            events = await h.run(
                translate_realtime_command({"type": "input_audio_buffer.commit", "create_response": False})
            )
            item_ids.append(
                next(e.to_realtime()["item_id"] for e in events if e.type == "input_audio_buffer.committed")
            )
        assert not h.port.submissions
        h.submit(CreateResponse())
        await h.settle(timeout_s=0.2)
        if interleave:
            events = await h.run(
                translate_realtime_command({"type": "session.update", "session": {"temperature": 0.2}})
            )
            assert any(e.type == "session.updated" for e in events)
            await h.run(AppendAudio(audio=struct.pack("<h", 4000) * 8000, format="pcm16", sample_rate_hz=16000))
            events = await h.run(
                translate_realtime_command({"type": "input_audio_buffer.commit", "create_response": False})
            )
            assert any(e.type == "input_audio_buffer.committed" for e in events)
            retained = h.runner.model_state.committed_audio_payload
            assert base64.b64decode(retained["audio"]) == struct.pack("<f", 4000 / 32768) * 8000
        expected_transcripts = (first_transcript, "Now describe the blue plate.", "Keep the answer short.")[:count]
        for index, transcript in enumerate(expected_transcripts):
            assert len(h.port.submissions) == index + 1
            submission = h.port.submissions[index]
            info = submission.prompt["additional_information"]
            assert info["realtime_item_id"] == item_ids[index]
            audio, rate = submission.prompt["multi_modal_data"]["audio"]
            assert rate == 16000
            assert audio.tobytes() == struct.pack("<f", (index + 1) * 1000 / 32768) * 8000
            if index < count - 1:
                assert submission.context.final_stage_id == 0
            else:
                assert submission.context.final_stage_id == 3
                images = info["deferred_multi_modal_data"]["image"]
                assert [image.getpixel((0, 0)) for image in images] == [(255, 0, 0)] + [(0, 0, 255)] * (count - 1)
                history = info["joyai_prompt_context"]["history"]
                for previous in expected_transcripts[:-1]:
                    assert ({"role": "user", "content": previous} in history) is bool(previous)
            if cancel:
                await h.run(CancelResponse())
            h.runner.on_stage_output(
                0,
                tts_output(submission.context.request_id, finished=True, text=transcript),
                None,
                request_id=submission.context.request_id,
                context=DuplexOutputContext(
                    identity=DuplexRequestIdentity(session_id=SESSION_ID, fence=submission.context.fence),
                    final_stage_id=submission.context.final_stage_id,
                    segment_finished=True,
                ),
                input_transcript=transcript,
                input_item_id=item_ids[index],
            )
            await h.settle(timeout_s=0.2)
            if cancel:
                assert len(h.port.submissions) == 1
                assert not h.runner.model._input_transcriptions
                assert not any(e.type == "conversation.item.input_audio_transcription.completed" for e in h.events)
                assert h.session.pending_input_bytes == 0
                return
        transcripts = [
            e.to_realtime() for e in h.events if e.type == "conversation.item.input_audio_transcription.completed"
        ]
        assert [(e["item_id"], e["transcript"]) for e in transcripts] == [
            (item_id, transcript) for item_id, transcript in zip(item_ids, expected_transcripts) if transcript
        ]
        created = [e.to_realtime()["response"]["id"] for e in h.events if e.type == "response.created"]
        assert len(created) == 1
        events = await h.deliver_and_settle(
            tts_output(submission.context.request_id, finished=True, text="</response> Both noted."),
            stage_id=1,
            segment_finished=True,
        )
        done = [e.to_realtime()["response"] for e in events if e.type == "response.done"]
        assert [(response["id"], response["status"]) for response in done] == [(created[0], "completed")]
        assert len(h.port.submissions) == count
        assert not any(e.type == "error" for e in h.events)
        await h.run(DeleteItem(item_id=item_ids[-1]))
        assert h.session.history_item_ids[item_ids[0]].get("transcript", "") == first_transcript
        assert item_ids[-1] not in h.session.history_item_ids
        assert (h.session.pending_input_bytes > 0) is interleave
        if interleave:
            assert h.runner.model_state.committed_audio_payload is retained
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_batched_commits_do_not_transcribe_recreated_pending_item():
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        item_ids = []
        for sample in (1000, 2000):
            await h.run(AppendAudio(audio=struct.pack("<h", sample) * 8000, format="pcm16", sample_rate_hz=16000))
            events = await h.run(Commit(final=True, create_response=False))
            item_ids.append(
                next(e.to_realtime()["item_id"] for e in events if e.type == "input_audio_buffer.committed")
            )
        await h.run(CreateResponse())
        original_part = h.session.input_audio_part(item_ids[1])
        await h.run(DeleteItem(item_id=item_ids[1]))
        await h.run(
            translate_realtime_command(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": item_ids[1],
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "audio": base64.b64encode(struct.pack("<h", 3000) * 8000).decode(),
                                "format": "pcm16",
                                "sample_rate_hz": 16000,
                            }
                        ],
                    },
                }
            )
        )
        replacement_part = h.session.input_audio_part(item_ids[1])
        assert replacement_part == original_part and replacement_part is not original_part
        for index, transcript in enumerate(("First original audio.", "Second original audio.")):
            submission = h.port.submissions[index]
            h.runner.on_stage_output(
                0,
                tts_output(submission.context.request_id, finished=True, text=transcript),
                None,
                request_id=submission.context.request_id,
                context=DuplexOutputContext(
                    identity=DuplexRequestIdentity(session_id=SESSION_ID, fence=submission.context.fence),
                    final_stage_id=submission.context.final_stage_id,
                    segment_finished=True,
                ),
                input_transcript=transcript,
                input_item_id=item_ids[index],
            )
            await h.settle()
        transcripts = [
            e.to_realtime() for e in h.events if e.type == "conversation.item.input_audio_transcription.completed"
        ]
        assert [(e["item_id"], e["transcript"]) for e in transcripts] == [(item_ids[0], "First original audio.")]
        assert "transcript" not in replacement_part
        assert "transcript" not in h.session.history_item_ids[item_ids[1]]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_batched_commits_release_continuation_on_asr_submission_failure():
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        for sample in (1000, 2000):
            await h.run(AppendAudio(audio=struct.pack("<h", sample) * 8000, format="pcm16", sample_rate_hz=16000))
            await h.run(Commit(final=True, create_response=False))
        h.port.fail_submit = RuntimeError("ASR submission failed")
        events = await h.run(CreateResponse())
        assert not h.port.submissions
        assert not h.runner.model._input_transcriptions
        assert not h.runner.plugin.data_plane.requests
        assert h.session.pending_input_bytes == 0
        done = [e.to_realtime() for e in events if e.type == "response.done"]
        assert len(done) == 1 and done[0]["response"]["status"] == "failed"
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_batched_commits_wait_for_submission_acceptance_before_continuing():
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        for sample in (1000, 2000):
            await h.run(AppendAudio(audio=struct.pack("<h", sample) * 8000, format="pcm16", sample_rate_hz=16000))
            await h.run(Commit(final=True, create_response=False))
        gate = asyncio.Event()
        h.port.submit_gate = gate
        h.submit(CreateResponse())
        await asyncio.wait_for(h.port.submit_started.wait(), timeout=2)
        context = h.port.ensured[-1]
        first_item_id = next(iter(h.session.history_item_ids))
        h.runner.on_stage_output(
            0,
            tts_output(context.request_id, finished=True, text="Remember the red cup."),
            None,
            request_id=context.request_id,
            context=DuplexOutputContext(
                identity=DuplexRequestIdentity(session_id=SESSION_ID, fence=context.fence),
                final_stage_id=0,
                segment_finished=True,
            ),
            input_transcript="Remember the red cup.",
            input_item_id=first_item_id,
        )
        await h.settle(timeout_s=0.05)
        assert not h.port.submissions
        assert not h.port.cleanups
        gate.set()
        await h.settle()
        assert len(h.port.submissions) == 2
        first, second = h.port.submissions
        assert first.context.final_stage_id == 0
        assert second.context.final_stage_id == 3
        assert first.context.request_id != second.context.request_id
        assert not h.runner.model._input_transcriptions
        assert (0, first.context.request_id) not in h.session.request_resources
        assert first.context.request_id not in h.runner.plugin.data_plane.requests
        assert sum(e.type == "response.created" for e in h.events) == 1
        assert not any(e.type in {"error", "response.done"} for e in h.events)
    finally:
        if h.port.submit_gate is not None:
            h.port.submit_gate.set()
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "cancel"])
@pytest.mark.parametrize("failure_at", ["engine-send", "append-commit"])
@pytest.mark.parametrize("finished", [False, True], ids=["partial-asr", "finished-asr"])
async def test_batched_commits_cleanup_asr_output_when_submission_ack_fails(failure, failure_at, finished, monkeypatch):
    from tests.engine.test_duplex_orchestrator import _settle

    plugin = JoyAIDuplexPlugin(lambda audio, *args: "AAAA")
    plugin.configure_pipeline(JOYAI_VL_INTERACTION_ASR_PIPELINE)
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    orchestrator, clients, rpc_q, _ = _build(stages=4, plugin=plugin)
    accepted, release = asyncio.Event(), asyncio.Event()
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    add_request = clients[0].add_request_async
    commit_plan = plugin.commit_append_plan

    def fail_ack(*args):
        if failure == "cancel":
            raise asyncio.CancelledError()
        raise RuntimeError("submission acknowledgement failed after ASR output")

    async def build_request(request_id, *args, **kwargs):
        return SimpleNamespace(request_id=request_id, mm_features=None)

    async def accept_then_fail(*args, **kwargs):
        await add_request(*args, **kwargs)
        accepted.set()
        await release.wait()
        if failure_at == "engine-send":
            fail_ack()

    # Exercise real stage ownership and cleanup, with engine acceptance before
    # its submission acknowledgement fails underneath an already queued output.
    monkeypatch.setattr(orchestrator, "_build_next_stage_request", build_request)
    monkeypatch.setattr(clients[0], "add_request_async", accept_then_fail)
    if failure_at == "append-commit":
        monkeypatch.setattr(plugin, "commit_append_plan", fail_ack)
    try:
        assert (await _open(orchestrator, rpc_q, extra_body={"auto_response": False}, output_buffer=output_buffer)).ok
        runner = orchestrator.session_manager.runners[ORCHESTRATOR_SESSION_ID]
        for sample in (1000, 2000):
            await _submit(
                orchestrator,
                AppendAudio(audio=struct.pack("<h", sample) * 8000, format="pcm16", sample_rate_hz=16000),
            )
            await _submit(orchestrator, Commit(final=True, create_response=False))
        runner.submit(CreateResponse())
        await asyncio.wait_for(accepted.wait(), timeout=3)
        request_id = clients[0].add_request_calls[-1][0].request_id
        request = orchestrator.request_states[request_id]
        runner.on_stage_output(
            0,
            tts_output(request_id, finished=finished, text="Remember the red cup."),
            None,
            request_id=request_id,
            context=DuplexOutputContext(
                identity=DuplexRequestIdentity(session_id=ORCHESTRATOR_SESSION_ID, fence=request.fence),
                final_stage_id=0,
                segment_finished=finished,
            ),
            input_transcript="Remember the red cup.",
            input_item_id=next(iter(runner.session.history_item_ids)),
        )
        await _settle(orchestrator, timeout_s=0.05)
        assert runner.model.is_input_transcription(request_id)
        release.set()
        await _settle(orchestrator)
        assert not runner.model._input_transcriptions
        assert not plugin.data_plane.requests
        assert request_id not in orchestrator.request_states
        assert not runner.session.resource_request_ids()
        assert runner.session.pending_input_bytes == 0
        assert runner.session.active_response_id is None
        assert len(clients[0].add_request_calls) == 1  # Do not start the remaining segment.
        if not finished:
            assert any(request_id in ids for ids in clients[0].abort_calls)
        events = [await output_buffer.get() for _ in range(output_buffer.pending_events)]
        done = [event.to_realtime() for event in events if event.type == "response.done"]
        assert len(done) == 1 and done[0]["response"]["status"] == "failed"
        monkeypatch.setattr(clients[0], "add_request_async", add_request)
        monkeypatch.setattr(plugin, "commit_append_plan", commit_plan)
        await _submit(
            orchestrator,
            AppendAudio(audio=struct.pack("<h", 3000) * 8000, format="pcm16", sample_rate_hz=16000),
        )
        await _submit(orchestrator, Commit(final=True, create_response=True))
        assert len(clients[0].add_request_calls) == 2
        assert clients[0].add_request_calls[-1][0].request_id != request_id
    finally:
        release.set()
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_batched_commits_cleanup_finished_asr_without_local_resource(monkeypatch):
    from tests.engine.test_duplex_orchestrator import _settle

    plugin = JoyAIDuplexPlugin(lambda audio, *args: "AAAA")
    plugin.configure_pipeline(JOYAI_VL_INTERACTION_ASR_PIPELINE)
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    orchestrator, clients, rpc_q, _ = _build(stages=4, plugin=plugin)

    async def build_request(request_id, *args, **kwargs):
        return SimpleNamespace(request_id=request_id, mm_features=None)

    monkeypatch.setattr(orchestrator, "_build_next_stage_request", build_request)
    try:
        assert (await _open(orchestrator, rpc_q, extra_body={"auto_response": False})).ok
        runner = orchestrator.session_manager.runners[ORCHESTRATOR_SESSION_ID]
        for sample in (1000, 2000):
            await _submit(
                orchestrator,
                AppendAudio(audio=struct.pack("<h", sample) * 8000, format="pcm16", sample_rate_hz=16000),
            )
            await _submit(orchestrator, Commit(final=True, create_response=False))
        await _submit(orchestrator, CreateResponse())
        request_id = clients[0].add_request_calls[-1][0].request_id
        request = orchestrator.request_states[request_id]
        # A local binding is not the lifetime of the orchestrator request.
        # Completing ASR must release both owners even if that binding was pruned.
        runner.session.request_resources.pop((0, request_id))
        runner.on_stage_output(
            0,
            tts_output(request_id, finished=True, text="Remember the red cup."),
            None,
            request_id=request_id,
            context=DuplexOutputContext(
                identity=DuplexRequestIdentity(session_id=ORCHESTRATOR_SESSION_ID, fence=request.fence),
                final_stage_id=0,
                segment_finished=True,
            ),
        )
        await _settle(orchestrator)
        assert request_id not in orchestrator.request_states
        assert request_id not in orchestrator.session_manager._request_index
        assert not runner.model._input_transcriptions
        assert len(clients[0].add_request_calls) == 2
        assert clients[0].add_request_calls[-1][0].request_id != request_id
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_asr_transcript_history_and_camera_bypass_are_fenced(cancel):
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        pcm = struct.pack("<hh", 1000, 2000)
        await h.run(AppendAudio(audio=pcm, format="pcm16", sample_rate_hz=16000, video_frames=(frame(),)))
        await h.run(Commit(final=True, create_response=True))
        submission = h.port.submissions[-1]
        assert submission.context.stage_id == 0 and submission.context.final_stage_id == 3
        info = submission.prompt["additional_information"]
        assert len(info["deferred_multi_modal_data"]["image"]) == 1
        assert "Watch the pot." in info["joyai_prompt_context"]["system"]
        assert info["joyai_prompt_budget"] == h.runner.plugin.max_model_len - 8
        request_id = submission.context.request_id
        context = DuplexOutputContext(
            identity=DuplexRequestIdentity(session_id=SESSION_ID, fence=submission.context.fence),
            final_stage_id=3,
            segment_finished=True,
        )
        if cancel:
            await h.run(CancelResponse())
        transcript = "Tell me when the red light changes."
        for _ in range(2):
            h.runner.on_stage_output(
                0,
                tts_output(request_id, finished=True, text=transcript),
                None,
                request_id=request_id,
                context=context,
                input_transcript=transcript,
                input_item_id=info["realtime_item_id"],
            )
        await h.settle()
        events = [e.to_realtime() for e in h.events]
        transcriptions = [e for e in events if e["type"] == "conversation.item.input_audio_transcription.completed"]
        assert len(transcriptions) == (0 if cancel else 1)
        if not cancel:
            await h.deliver_and_settle(
                tts_output(request_id, finished=True, text="</silence>"), stage_id=1, segment_finished=True
            )
            assert any(message.get("transcript") == transcript for message in h.session.history)
        await observe(h)
        assert h.port.submissions[-1].context.stage_id == 1
        assert (transcript in h.port.submissions[-1].prompt["prompt"]) is (not cancel)
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_asr_visuals_stay_on_raw_prompt_across_engine_wire(mocker, monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    plugin = JoyAIDuplexPlugin(lambda audio, *args: "AAAA")
    plugin.configure_pipeline(JOYAI_VL_INTERACTION_ASR_PIPELINE)
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    orchestrator, clients, rpc_q, _ = _build(stages=4, plugin=plugin)

    async def process_inputs(stage_id, *, request_id, params, **kwargs):
        return _processed_request(request_id, params)

    mocker.patch.object(orchestrator, "_process_stage_inputs", side_effect=process_inputs)
    mocker.patch(
        "vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction.cached_processor_from_config",
        return_value=plugin.processor,
    )
    try:
        assert (await _open(orchestrator, rpc_q, extra_body={"auto_response": False})).ok
        await _submit(orchestrator, AppendAudio(audio=struct.pack("<hh", 1000, 2000), video_frames=(frame(),)))
        await _submit(orchestrator, Commit(final=True, create_response=True))
        request = clients[0].add_request_calls[-1][0]
        restored = MsgpackDecoder(type(request)).decode(MsgpackEncoder().encode(request))
        info = deserialize_additional_information(restored.additional_information)
        assert "deferred_multi_modal_data" not in info
        raw_prompt = orchestrator.request_states[request.request_id].prompt
        raw_info = raw_prompt["additional_information"]
        assert info["joyai_prompt_context"] == raw_info["joyai_prompt_context"]
        assert info["tts_language"] == raw_info["tts_language"]
        image = raw_info["deferred_multi_modal_data"]["image"][0]
        assert isinstance(image, Image.Image) and image.getpixel((0, 0)) == (255, 0, 0)
        output = SimpleNamespace(outputs=[SimpleNamespace(text="Describe this image.")])
        (joyai_prompt,) = asr_to_joyai([output], raw_prompt)
        assert joyai_prompt["multi_modal_data"]["image"][0] is image
        assert "Describe this image." in joyai_prompt["prompt"]
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_late_asr_does_not_transcribe_recreated_audio_item():
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        pcm = struct.pack("<h", 1000) * 12800  # 800 ms exceeds the listen-only overlap threshold.
        await h.run(AppendAudio(audio=pcm, format="pcm16", sample_rate_hz=16000, video_frames=(frame(),)))
        await h.run(Commit(final=True, create_response=True))
        original = h.port.submissions[-1]
        item_id = original.prompt["additional_information"]["realtime_item_id"]
        original_part = h.session.history_item_ids[item_id]["content"][0]

        await h.run(translate_realtime_command({"type": "conversation.item.delete", "item_id": item_id}))
        await h.run(
            translate_realtime_command(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "audio": base64.b64encode(struct.pack("<h", 2000) * 12800).decode(),
                                "format": "pcm16",
                                "sample_rate_hz": 16000,
                            }
                        ],
                    },
                }
            )
        )
        replacement_part = h.session.history_item_ids[item_id]["content"][0]
        assert replacement_part == original_part
        assert replacement_part is not original_part
        assert h.session.epoch == original.context.fence.epoch
        assert h.session.active_request_id == original.context.request_id
        assert len(h.port.submissions) == 1  # The new audio is retained behind the old response.

        def deliver_transcript(submission, text):
            request_id = submission.context.request_id
            output = tts_output(request_id, finished=True, text=text)
            h.runner.on_stage_output(
                0,
                output,
                None,
                request_id=request_id,
                context=DuplexOutputContext(
                    identity=DuplexRequestIdentity(session_id=SESSION_ID, fence=submission.context.fence),
                    final_stage_id=3,
                    segment_finished=True,
                ),
                input_transcript=h.runner.plugin.user_transcript(
                    stage_id=0, output=output, prompt=submission.prompt, finished=True
                ),
                input_item_id=submission.prompt["additional_information"]["realtime_item_id"],
            )

        deliver_transcript(original, "Old spoken instruction.")
        events = await h.settle()
        assert "transcript" not in h.session.history_item_ids[item_id]
        assert "transcript" not in replacement_part
        assert not any(event.type == "conversation.item.input_audio_transcription.completed" for event in events)

        await h.deliver_and_settle(
            tts_output(original.context.request_id, finished=True, text="</silence>"),
            stage_id=1,
            segment_finished=True,
        )
        await h.run(CreateResponse())
        replacement = h.port.submissions[-1]
        assert replacement.context.stage_id == 0
        assert replacement.context.request_id != original.context.request_id
        deliver_transcript(replacement, "New spoken instruction.")
        events = await h.settle()
        transcriptions = [
            event for event in events if event.type == "conversation.item.input_audio_transcription.completed"
        ]
        assert len(transcriptions) == 1
        assert transcriptions[0].transcript == "New spoken instruction."
        assert h.session.history_item_ids[item_id]["transcript"] == "New spoken instruction."
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_shared_vad_commits_joyai_speech_once():
    h = await open_joyai(asr=True)
    try:
        h.runner.control._detector = _speech_then_stop()
        await h.run(append_audio())
        await h.run(append_audio(is_speech=False, value=0.0))
        assert len(h.port.submissions) == 1
        assert h.port.submissions[0].context.stage_id == 0
        assert "audio" in h.port.submissions[0].prompt["multi_modal_data"]
    finally:
        await h.manager.shutdown()


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


@pytest.mark.parametrize("transcript", ["current question " * 50, ""], ids=["long-transcript", "empty-transcript"])
def test_asr_handoff_rebudgets_history_and_handles_empty_transcript(monkeypatch, transcript):
    from vllm_omni.model_executor.stage_input_processors import joyai_vl_interaction as bridge

    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.configure_pipeline(JOYAI_VL_INTERACTION_ASR_PIPELINE)
    plugin.processor = Processor()
    plugin.max_model_len = 2600
    monkeypatch.setattr(bridge, "cached_processor_from_config", lambda config: plugin.processor, raising=False)
    monkeypatch.setattr(
        bridge,
        "cached_tokenizer_from_config",
        lambda config: SimpleNamespace(
            encode=lambda text, **kwargs: plugin.processor.tokenizer(text=[text])["input_ids"][0]
        ),
    )
    history = (
        [
            {"role": "user", "content": "outdated " * 130},
            {"role": "assistant", "content": "old answer"},
        ]
        if transcript
        else [
            {"role": "user", "content": "What color is the car?"},
            {"role": "assistant", "content": "The car is red."},
        ]
    )
    runtime = plugin.runtime_config_for_update(DuplexSessionConfig(instructions="Watch the pot."), {})
    if not transcript:
        runtime["initial_user_text"] = "Describe what you see."
    plan = plugin.plan_append(
        request_id="r",
        fence=DuplexFence("s"),
        seq=1,
        turn_seq=1,
        final=True,
        runtime_config=runtime,
        session_config={"joyai_frames": [frame("blue")], "conversation": history},
        payload={
            "audio": base64.b64encode(struct.pack("<ff", 0.1, 0.2)).decode(),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
        },
        sampling_params=SamplingParams(max_tokens=64),
    )
    output = SimpleNamespace(outputs=[SimpleNamespace(text="language English<asr_text>" + transcript)])
    (result,) = bridge.asr_to_joyai([output], plan.prompt)
    assert "Watch the pot." in result["prompt"]
    assert result["multi_modal_data"]["image"][-1].getpixel((0, 0)) == (0, 0, 255)
    if transcript:
        assert transcript.strip() in result["prompt"]
        assert "outdated" not in result["prompt"]
    else:
        assert "Observe the current scene and choose whether to speak or stay silent." in result["prompt"]
        assert "[User Query" not in result["prompt"]
        assert result["prompt"].count("What color is the car?") == 1
        assert result["prompt"].count("Describe what you see.") == 1
    assert (
        len(plugin.processor(text=[result["prompt"]], images=result["multi_modal_data"]["image"])["input_ids"][0])
        <= 2536
    )


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
        assert cleanup.await_args_list == [mocker.call([request_id], abort=True), mocker.call([request_id], abort=True)]
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
