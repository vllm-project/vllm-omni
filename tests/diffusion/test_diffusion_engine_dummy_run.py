# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion import io_support
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_dummy_run_num_frames_uses_explicit_model_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    class JointAudioVideoModel:
        dummy_run_num_frames = 2

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: JointAudioVideoModel,
    )

    assert io_support.get_dummy_run_num_frames("joint_audio_video", supports_audio_input=False) == 2


def test_dummy_run_num_frames_keeps_audio_output_default(monkeypatch: pytest.MonkeyPatch) -> None:
    class AudioOutputModel:
        support_audio_output = True

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: AudioOutputModel,
    )

    assert io_support.get_dummy_run_num_frames("audio_output", supports_audio_input=False) == 2


def test_dummy_run_num_frames_defaults_to_single_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    class VideoOnlyModel:
        pass

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: VideoOnlyModel,
    )

    assert io_support.get_dummy_run_num_frames("video_only", supports_audio_input=False) == 1


def test_dummy_run_num_frames_uses_audio_input_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: None,
    )

    assert io_support.get_dummy_run_num_frames("unknown", supports_audio_input=True) == 2


def test_dummy_run_image_count_resolves_hunyuan_architecture_alias() -> None:
    assert io_support.get_dummy_run_num_image_inputs("HunyuanImage3ForCausalMM") == 3
    assert io_support.get_dummy_run_num_image_inputs("unknown") == 1


def test_dense_mode_does_not_build_kv_profile_request() -> None:
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY)
    engine._make_dummy_request = Mock(side_effect=AssertionError("dense mode must not profile"))

    assert engine._prepare_diffusion_kv_profile_request() is None
    engine._make_dummy_request.assert_not_called()


def test_paged_kv_profile_request_is_preprocessed_without_scheduler_state() -> None:
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        model_class_name="HunyuanImage3ForCausalMM",
    )
    request = OmniDiffusionRequest(
        prompt="profile",
        request_id="profile-request",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )
    engine._make_dummy_request = Mock(return_value=request)
    prepared_layout = object()
    scheduler_kv_state = (object(),)

    def preprocess(req: OmniDiffusionRequest) -> OmniDiffusionRequest:
        req.prepared_layout = prepared_layout
        req.diffusion_kv_requests = scheduler_kv_state
        return req

    engine._prepare_request_for_admission = Mock(side_effect=preprocess)

    result = engine._prepare_diffusion_kv_profile_request()

    assert result is request
    assert result.prepared_layout is prepared_layout
    assert result.diffusion_kv_requests is None
    engine._make_dummy_request.assert_called_once_with(
        height=1024,
        width=1024,
        guidance_scale=5.0,
        num_image_inputs=3,
    )
    engine._prepare_request_for_admission.assert_called_once_with(request)
