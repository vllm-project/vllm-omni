# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contract tests for AURA duplex plugin seams."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    duplex_ephemeral_stage_request_id,
    duplex_resource_request_id,
)
from vllm_omni.engine.duplex.plugin import load_duplex_plugin
from vllm_omni.model_executor.models.aura_omni.duplex.data_plane import (
    AuraDataPlaneContext,
    AuraDataPlaneSession,
)
from vllm_omni.model_executor.models.aura_omni.duplex.history import (
    drop_session_history,
    get_or_create_session_history,
)
from vllm_omni.model_executor.models.aura_omni.duplex.input import AuraPcmAppendBuffer
from vllm_omni.model_executor.models.aura_omni.duplex.plugin import (
    AURA_SILENT_TOKEN_ID,
    AuraDuplexPlugin,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import SILENT_TEXT

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encode_audio(audio: object, sample_rate: int, fmt: str, speed: float | None) -> str | None:
    del audio, sample_rate, fmt, speed
    return "ZmFrZQ=="


def test_load_aura_duplex_plugin_and_sampling_arity() -> None:
    plugin = load_duplex_plugin(
        "vllm_omni.model_executor.models.aura_omni.duplex.plugin.AuraDuplexPlugin",
        _encode_audio,
    )
    assert isinstance(plugin, AuraDuplexPlugin)
    assert plugin.plugin_id == "aura"
    defaults = (
        SamplingParams(max_tokens=16),
        SamplingParams(max_tokens=16),
        SamplingParams(max_tokens=4096),
        SamplingParams(max_tokens=16),
    )
    configured = plugin.configure_sampling_params(runtime_config={}, defaults=defaults)
    assert len(configured) == 4
    assert configured[2].stop_token_ids == [2150]
    assert configured[2].max_tokens == 4096
    stage1_stops = set(configured[1].stop_token_ids or [])
    assert {151669, 151645}.issubset(stage1_stops)
    assert 248070 not in stage1_stops
    caps = plugin.capabilities(max_sessions=1)
    assert caps.supports_turn_commit_only is True
    assert caps.supports_core_resumable_request is False
    assert caps.supports_concurrent_turn_requests is True
    assert caps.required_input_modalities == frozenset({"video"})
    assert caps.optional_input_modalities == frozenset({"audio"})
    assert caps.allows_video_without_audio() is True
    assert caps.validate_append_modalities(has_audio=True, has_video=False) == (
        "This duplex model requires video_frames on input_audio_buffer.append"
    )
    assert caps.validate_append_modalities(has_audio=True, has_video=True) is None
    assert caps.validate_append_modalities(has_audio=False, has_video=True) is None


def test_commit_only_buffer_emits_on_commit() -> None:
    buf = AuraPcmAppendBuffer()
    samples = np.zeros(1600, dtype="<f4")
    payload = {
        "type": "audio",
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "audio": base64.b64encode(samples.tobytes()).decode("ascii"),
        "is_speech": True,
        "video_frames": ["aGVsbG8="],
    }
    reservation = buf.prepare_append(
        payload,
        operation_id="op1",
        chunk_period_ms=1000,
        allow_emit=True,
    )
    assert reservation is None
    assert buf.has_pending()
    commit = buf.prepare_commit(operation_id="c1", chunk_period_ms=1000)
    assert commit.payload is not None
    assert commit.payload.get("final") is True
    assert commit.payload.get("aura_turn_commit") is True
    assert commit.payload.get("video_frames") == ["aGVsbG8="]
    commit.commit()


def test_commit_reservation_includes_video_bytes() -> None:
    buf = AuraPcmAppendBuffer()
    pcm = np.zeros(1600, dtype="<f4").tobytes()
    frames = ["aGVsbG8=", "d29ybGQ="]
    payload = {
        "type": "audio",
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "audio": base64.b64encode(pcm).decode("ascii"),
        "is_speech": True,
        "video_frames": frames,
    }
    assert (
        buf.prepare_append(
            payload,
            operation_id="op1",
            chunk_period_ms=1000,
            allow_emit=True,
        )
        is None
    )
    pending = buf.pending_byte_count
    assert pending == len(pcm) + sum(len(frame) for frame in frames)
    commit = buf.prepare_commit(operation_id="c1", chunk_period_ms=1000)
    assert buf.pending_byte_count == 0
    assert commit.byte_count == pending
    commit.commit()
    assert buf.pending_byte_count == 0
    assert not buf.has_reserved()


def test_plan_append_commit_builds_stage0_prompt() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    samples = np.zeros(800, dtype="<f4")
    payload = {
        "type": "audio",
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "audio": base64.b64encode(samples.tobytes()).decode("ascii"),
        "final": True,
        "aura_turn_commit": True,
        "is_speech": True,
    }
    fence = DuplexFence("sess-1", epoch=2, turn_id=5)
    plan = plugin.plan_append(
        request_id="req",
        fence=fence,
        session_config={},
        runtime_config={"aura_system_prompt": "sys"},
        seq=1,
        turn_seq=1,
        payload=payload,
        final=True,
        sampling_params=SamplingParams(max_tokens=8),
    )
    info = plan.prompt["additional_information"]
    assert info["aura_duplex"] is True
    assert info["session_id"] == "sess-1"
    assert "audio" in plan.prompt["multi_modal_data"]
    assert "prompt_token_ids" not in plan.prompt


def test_video_frames_to_mm_packs_one_video() -> None:
    """Two JPEG frames are one video clip, not two image items."""
    from io import BytesIO

    from PIL import Image

    from vllm_omni.model_executor.models.aura_omni.duplex.plugin import _video_frames_to_mm

    frames: list[str] = []
    for color in ((255, 0, 0), (0, 255, 0)):
        buf = BytesIO()
        Image.new("RGB", (8, 8), color).save(buf, format="JPEG")
        frames.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    mm = _video_frames_to_mm(frames)
    assert list(mm.keys()) == ["video"]
    assert len(mm["video"]) == 1
    video, metadata = mm["video"][0]
    assert np.asarray(video).shape == (2, 8, 8, 3)
    assert metadata["total_num_frames"] == 2
    assert metadata["do_sample_frames"] is False


def test_plan_append_vision_empty_audio_never_leaves_empty_prompt() -> None:
    """Frames + empty/near-silent audio must still get a Stage0 ASR pad."""
    from io import BytesIO

    from PIL import Image

    plugin = AuraDuplexPlugin(_encode_audio)
    img = Image.new("RGB", (32, 32), (8, 16, 24))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    frame = base64.b64encode(buf.getvalue()).decode("ascii")
    fence = DuplexFence("sess-v", epoch=0, turn_id=1)
    near = np.linspace(1e-5, -1e-5, 80, dtype="<f4")
    cases = [
        ("empty_speech_true", True, ""),
        ("empty_speech_false", False, ""),
        ("near_silent", False, base64.b64encode(near.tobytes()).decode("ascii")),
    ]
    for name, is_speech, audio in cases:
        payload = {
            "type": "audio",
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
            "audio": audio,
            "final": True,
            "aura_turn_commit": True,
            "is_speech": is_speech,
            "video_frames": [frame],
        }
        plan = plugin.plan_append(
            request_id="req",
            fence=fence,
            session_config={},
            runtime_config={},
            seq=1,
            turn_seq=1,
            payload=payload,
            final=True,
            sampling_params=SamplingParams(max_tokens=8),
        )
        assert plan.prompt.get("prompt"), f"{name}: Stage0 prompt must be non-empty"
        assert "audio" in plan.prompt["multi_modal_data"], f"{name}: padded audio required"
        assert "deferred_multi_modal_data" in plan.prompt["additional_information"]


def test_project_intermediate_output_targets_stage1_only() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    assert plugin.project_intermediate_output(stage_id=1, output=object(), context=object()) is True
    assert plugin.project_intermediate_output(stage_id=0, output=object(), context=object()) is False
    assert plugin.project_intermediate_output(stage_id=3, output=object(), context=object()) is False


def test_release_concurrent_turn_requests_on_stage1_final() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    assert plugin.release_concurrent_turn_requests(stage_id=1, segment_finished=True, output=object(), context=object())
    assert not plugin.release_concurrent_turn_requests(
        stage_id=1, segment_finished=False, output=object(), context=object()
    )
    finished = type("Out", (), {"finished": True})()
    assert plugin.release_concurrent_turn_requests(
        stage_id=1, segment_finished=False, output=finished, context=object()
    )
    assert not plugin.release_concurrent_turn_requests(
        stage_id=2, segment_finished=True, output=object(), context=object()
    )


def test_configure_sampling_keeps_silent_stop_visible() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    defaults = (
        SamplingParams(max_tokens=16),
        SamplingParams(max_tokens=16),
        SamplingParams(max_tokens=4096),
        SamplingParams(max_tokens=16),
    )
    configured = plugin.configure_sampling_params(runtime_config={}, defaults=defaults)
    stage1 = configured[1]
    assert isinstance(stage1, SamplingParams)
    assert stage1.include_stop_str_in_output is True
    assert stage1.skip_special_tokens is False
    assert AURA_SILENT_TOKEN_ID in (stage1.stop_token_ids or [])


def test_configure_sampling_keeps_explicit_talker_limits() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    defaults = (
        SamplingParams(max_tokens=16),
        SamplingParams(max_tokens=16),
        SamplingParams(max_tokens=500, stop_token_ids=[7]),
        SamplingParams(max_tokens=16),
    )
    configured = plugin.configure_sampling_params(runtime_config={}, defaults=defaults)
    stage2 = configured[2]
    assert isinstance(stage2, SamplingParams)
    assert stage2.max_tokens == 500
    assert stage2.stop_token_ids == [7]
    unset = plugin.configure_sampling_params(
        runtime_config={},
        defaults=(
            SamplingParams(max_tokens=16),
            SamplingParams(max_tokens=16),
            SamplingParams(max_tokens=None),
            SamplingParams(max_tokens=16),
        ),
    )
    assert unset[2].max_tokens == 240
    assert unset[2].stop_token_ids == [2150]


def test_decide_output_silent_short_circuits() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)

    class _Completion:
        text = SILENT_TEXT
        token_ids = [AURA_SILENT_TOKEN_ID]
        finished = True

    class _Output:
        outputs = [_Completion()]

    decision = plugin.decide_output(
        stage_id=1,
        final_stage_id=3,
        segment_finished=True,
        segment_token_ids=(AURA_SILENT_TOKEN_ID,),
        segment_output_metadata={},
        output=_Output(),
    )
    assert decision is not None
    assert decision.metadata.get("model_listen") is True


def test_decide_output_chinese_silence_is_not_special_token() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)

    class _Completion:
        text = "[沉默]"
        token_ids = [58, 107107, 60]
        finished = True

    class _Output:
        outputs = [_Completion()]

    assert (
        plugin.decide_output(
            stage_id=1,
            final_stage_id=3,
            segment_finished=True,
            segment_token_ids=(58, 107107, 60),
            segment_output_metadata={},
            output=_Output(),
        )
        is None
    )


def test_ephemeral_request_id_includes_turn() -> None:
    fence = DuplexFence("s", epoch=1, turn_id=7)
    resumable = duplex_resource_request_id(fence, "stage0")
    ephemeral = duplex_ephemeral_stage_request_id(fence, stage_id=0)
    assert ephemeral.endswith("stage0-turn7")
    assert resumable != ephemeral
    fence2 = DuplexFence("s", epoch=1, turn_id=8)
    assert duplex_ephemeral_stage_request_id(fence, stage_id=0) != duplex_ephemeral_stage_request_id(fence2, stage_id=0)


def test_stage_submission_defaults_resumable_true() -> None:
    ctx = DuplexStageRequestContext(
        request_id="r",
        session_id="s",
        fence=DuplexFence("s"),
        stage_id=0,
        final_stage_id=3,
        config_generation=0,
        sampling_params=(SamplingParams(max_tokens=1),),
    )
    submission = DuplexStageSubmission(
        context=ctx,
        prompt={"prompt_token_ids": [0]},
        already_submitted=False,
    )
    assert submission.resumable is True


def test_data_plane_close_session_drops_history() -> None:
    from vllm_omni.model_executor.models.aura_omni.duplex.data_plane import AuraDataPlaneSession
    from vllm_omni.model_executor.models.aura_omni.duplex.history import get_or_create_session_history

    drop_session_history("close-hist")
    get_or_create_session_history("close-hist").begin_user_turn("hi")
    AuraDataPlaneSession(encode_audio=_encode_audio).close_session("close-hist")
    from vllm_omni.model_executor.models.aura_omni.duplex.history import _STORE

    assert "close-hist" not in _STORE


def test_session_history_strips_oldest_silent_videos() -> None:
    drop_session_history("hist-ut")
    history = get_or_create_session_history("hist-ut")
    history.max_video_rounds = 2
    history.video_rounds_to_remove = 1
    for index in range(3):
        history.begin_user_turn("", video=(f"clip-{index}",))
        history.commit_turn(SILENT_TEXT)
    kept = [message["video"] for message in history.messages if message.get("video") is not None]
    assert kept == [("clip-1",), ("clip-2",)]
    assert not any(message.get("video") == ("clip-0",) for message in history.messages)
    drop_session_history("hist-ut")


def test_pipeline_declares_aura_duplex_plugin() -> None:
    from vllm_omni.model_executor.models.aura_omni.pipeline import AURA_OMNI_PIPELINE

    assert AURA_OMNI_PIPELINE.duplex_plugin == (
        "vllm_omni.model_executor.models.aura_omni.duplex.plugin.AuraDuplexPlugin"
    )


def test_project_output_uses_requested_format_and_chunk_duration() -> None:
    seen: dict[str, object] = {}

    def encode(audio: object, sample_rate: int, fmt: str, speed: float | None) -> str:
        seen["fmt"] = fmt
        seen["rate"] = sample_rate
        seen["speed"] = speed
        del audio
        return "AAAA"

    plane = AuraDataPlaneSession(encode)

    class _Output:
        request_id = "aura-turn"
        finished = False
        stage_id = 3
        multimodal_output = {"audio": np.zeros(4800, dtype=np.float32), "sr": 24000}

    events = list(plane.project_output(_Output(), context=AuraDataPlaneContext(response_format="pcm16", speed=1.0)))
    assert seen["fmt"] == "pcm16"
    assert events[0]["audio_format"] == "pcm16"
    assert events[0]["audio_duration_ms"] == 200
    assert events[0]["sample_rate_hz"] == 24000

    def encode_wav(audio: object, sample_rate: int, fmt: str, speed: float | None) -> str:
        del audio, sample_rate, speed
        assert fmt == "wav"
        return "V0FW"

    plane_wav = AuraDataPlaneSession(encode_wav)
    wav_events = list(plane_wav.project_output(_Output(), context=AuraDataPlaneContext(response_format="wav")))
    assert wav_events[0]["audio_format"] == "wav"


def test_instructions_update_replaces_effective_prompt() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    current = {"aura_system_prompt": "creation default", "instructions": "creation default"}
    updated = plugin.runtime_config_for_update(
        DuplexSessionConfig(model="aurateam/AURA", instructions="be brief"),
        current,
    )
    assert updated["aura_system_prompt"] == "be brief"
    assert updated["instructions"] == "be brief"
    kept = plugin.runtime_config_for_update(
        DuplexSessionConfig(
            model="aurateam/AURA",
            extra_body={"aura_system_prompt": "explicit"},
            instructions="ignored when explicit",
        ),
        updated,
    )
    assert kept["aura_system_prompt"] == "explicit"


def test_plan_partial_stage_output_hands_a_sentence_to_talker() -> None:
    def aura2tts() -> None:
        return None

    class _Pool:
        stage_client = SimpleNamespace(custom_process_input_func=aura2tts)

    orchestrator = SimpleNamespace(
        stage_pools={2: _Pool()},
        _stage_receives_async_chunks=lambda stage_id: False,
    )
    req_state = SimpleNamespace(
        session_owned=True,
        final_stage_id=3,
        request_id="req-1",
        prompt={"additional_information": {}},
        streaming=SimpleNamespace(bridge_states={}),
    )
    sentence = "这是一段用来直接测试句子级语音交接而且长度已经超过三十个字的内容。"
    output = SimpleNamespace(
        finished=False,
        request_id="req-1",
        outputs=[SimpleNamespace(cumulative_text=sentence)],
    )

    plan = AuraDuplexPlugin(_encode_audio).plan_partial_stage_output(orchestrator, 1, 0, output, req_state)

    assert plan is not None
    assert plan.close_only is False
    assert plan.is_final_update is False
    assert plan.output.text == sentence.strip()
    assert req_state.prompt["additional_information"]["aura_tts_partial"] is True
    assert AuraDuplexPlugin(_encode_audio).plan_partial_stage_output(orchestrator, 0, 0, output, req_state) is None


def _sentence_plan_fixture() -> tuple[SimpleNamespace, SimpleNamespace]:
    def aura2tts() -> None:
        return None

    class _Pool:
        stage_client = SimpleNamespace(custom_process_input_func=aura2tts)

    orchestrator = SimpleNamespace(
        stage_pools={2: _Pool()},
        _stage_receives_async_chunks=lambda stage_id: False,
    )
    req_state = SimpleNamespace(
        session_owned=True,
        final_stage_id=3,
        request_id="req-1",
        prompt={"additional_information": {}},
        streaming=SimpleNamespace(bridge_states={}),
    )
    return orchestrator, req_state


def _stage1_output(text: str, *, finished: bool) -> SimpleNamespace:
    return SimpleNamespace(
        finished=finished,
        request_id="req-1",
        outputs=[SimpleNamespace(cumulative_text=text)],
    )


def test_finished_remainder_stays_resumable_until_close_sentinel() -> None:
    """A second sentence at Stage1 finish must not be the Talker end sentinel.

    The first sentence is already a resumable Talker request. Marking the
    remainder ``is_final_update`` makes ``StreamingUpdate.from_request``
    return None, which aborts that in-flight sentence before Code2Wav has a
    full codec group.
    """
    orchestrator, req_state = _sentence_plan_fixture()
    plugin = AuraDuplexPlugin(_encode_audio)
    first = "我看到你身后是一个很明亮的办公室环境，天花板上能看到白色的管道和通风口，"
    full = first + "远处还有办公桌和绿植。"
    opened = plugin.plan_partial_stage_output(orchestrator, 1, 0, _stage1_output(first, finished=False), req_state)
    assert opened is not None
    assert opened.is_final_update is False
    assert opened.queue_close_after is False
    assert req_state.prompt["additional_information"]["aura_tts_close_only"] is False

    tail = plugin.plan_partial_stage_output(orchestrator, 1, 0, _stage1_output(full, finished=True), req_state)
    assert tail is not None
    assert tail.close_only is False
    assert tail.queue_close_after is True
    assert tail.is_final_update is False
    assert tail.output.text == "远处还有办公桌和绿植。"
    assert req_state.prompt["additional_information"]["aura_tts_close_only"] is False

    followup = plugin.partial_stage_followup(tail, req_state)
    assert followup is not None
    assert followup.is_final_update is True
    assert followup.close_only is True
    assert followup.output.text == ""
    assert req_state.prompt["additional_information"]["aura_tts_close_only"] is True


def test_single_finished_sentence_is_one_non_resumable_submit() -> None:
    orchestrator, req_state = _sentence_plan_fixture()
    plugin = AuraDuplexPlugin(_encode_audio)
    sentence = "你好，我看到一个戴眼镜、穿黑T恤的男生正对着镜头说话呢。"
    plan = plugin.plan_partial_stage_output(orchestrator, 1, 0, _stage1_output(sentence, finished=True), req_state)
    assert plan is not None
    assert plan.queue_close_after is False
    assert plan.is_final_update is True
    assert plan.close_only is False
    assert plugin.partial_stage_followup(plan, req_state) is None


def test_aura_draining_stages_are_declared_by_the_plugin() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    assert plugin.draining_stage_ids(stage_count=4) == frozenset({2, 3})
    assert plugin.draining_stage_ids(stage_count=2) == frozenset()


def test_spoken_stage0_text_is_the_user_transcript() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    output = SimpleNamespace(
        finished=True,
        outputs=[
            SimpleNamespace(text="language Chinese<asr_text>你好", cumulative_text="language Chinese<asr_text>你好")
        ],
    )
    prompt = {"additional_information": {"is_speech": True}}
    assert plugin.user_transcript(stage_id=0, output=output, prompt=prompt, finished=True) == "你好"


def test_vision_follow_stage0_text_is_not_shown_as_user_speech() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    output = SimpleNamespace(finished=True, text="嗯")
    prompt = {"additional_information": {"is_speech": False}}
    assert plugin.user_transcript(stage_id=0, output=output, prompt=prompt, finished=True) is None
    assert plugin.user_transcript(stage_id=1, output=output, prompt=prompt, finished=True) is None
    assert plugin.user_transcript(stage_id=0, output=output, prompt=prompt, finished=False) is None
