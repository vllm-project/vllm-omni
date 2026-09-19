# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contract tests for AURA duplex plugin seams."""

from __future__ import annotations

import base64

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
    assert 2150 in (configured[2].stop_token_ids or [])
    assert configured[2].max_tokens == 240
    stage1_stops = set(configured[1].stop_token_ids or [])
    assert {151669, 151645}.issubset(stage1_stops)
    assert 248070 not in stage1_stops
    caps = plugin.capabilities(max_sessions=1)
    assert caps.supports_turn_commit_only is True
    assert caps.supports_core_resumable_request is False
    assert caps.supports_overlapped_commit is True
    assert caps.required_input_modalities == frozenset({"video"})
    assert caps.optional_input_modalities == frozenset({"audio"})
    assert caps.allows_video_without_audio() is True


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


def test_release_overlapped_commit_on_stage1_final() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    assert plugin.release_overlapped_commit(stage_id=1, segment_finished=True, output=object(), context=object())
    assert not plugin.release_overlapped_commit(stage_id=1, segment_finished=False, output=object(), context=object())
    finished = type("Out", (), {"finished": True})()
    assert plugin.release_overlapped_commit(stage_id=1, segment_finished=False, output=finished, context=object())
    assert not plugin.release_overlapped_commit(stage_id=2, segment_finished=True, output=object(), context=object())


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
    get_or_create_session_history("close-hist", max_turns=2).begin_user_turn("hi")
    AuraDataPlaneSession(encode_audio=_encode_audio).close_session("close-hist")
    from vllm_omni.model_executor.models.aura_omni.duplex.history import _STORE

    assert "close-hist" not in _STORE


def test_session_history_commit_and_prune() -> None:
    drop_session_history("hist-ut")
    history = get_or_create_session_history("hist-ut", max_turns=2)
    history.begin_user_turn("hi")
    history.commit_turn("hello")
    history.begin_user_turn("again")
    history.commit_turn("world")
    history.begin_user_turn("third")
    history.commit_turn(SILENT_TEXT)
    assert len(history.messages) <= 4
    prefix = history.render_prefix()
    assert "<|im_start|>user" in prefix
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
