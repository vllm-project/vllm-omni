# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion import io_support
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
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

    assert engine._prepare_diffusion_kv_profile_requests() is None
    engine._make_dummy_request.assert_not_called()


@pytest.mark.parametrize(
    ("execution_mode", "uses_dlo_dp", "max_num_seqs", "expected_profile_requests"),
    [
        (DiffusionExecutionMode.STEP_BATCH, False, 2, 2),
        (DiffusionExecutionMode.REQUEST_BATCH, True, 4, 1),
        (DiffusionExecutionMode.STEP_BATCH, True, 3, 3),
    ],
)
def test_paged_kv_profile_requests_match_per_rank_batch(
    execution_mode: DiffusionExecutionMode,
    uses_dlo_dp: bool,
    max_num_seqs: int,
    expected_profile_requests: int,
) -> None:
    engine = object.__new__(DiffusionEngine)
    engine.execution_mode = execution_mode
    engine.od_config = SimpleNamespace(
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        model_class_name="HunyuanImage3ForCausalMM",
        max_num_seqs=max_num_seqs,
        parallel_config=SimpleNamespace(data_parallel_size=max_num_seqs if uses_dlo_dp else 1),
        enable_distributed_layerwise_offload=uses_dlo_dp,
        dlo_use_allgather=True,
    )
    request = OmniDiffusionRequest(
        prompt="profile",
        request_id="profile-request",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )
    engine._make_dummy_request = Mock(return_value=request)
    prepared_layout = object()
    scheduler_kv_state = (
        DiffusionKVRequest(
            "profile-request/diffusion-kv/0",
            sequence_id=0,
            prefix_len=3,
            target_len=8,
            seq_len=12,
        ),
        DiffusionKVRequest(
            "profile-request/diffusion-kv/1",
            sequence_id=1,
            prefix_len=4,
            target_len=8,
            seq_len=13,
        ),
    )

    def preprocess(req: OmniDiffusionRequest) -> OmniDiffusionRequest:
        req.prepared_layout = prepared_layout
        req.diffusion_kv_requests = scheduler_kv_state
        return req

    engine._prepare_request_for_admission = Mock(side_effect=preprocess)

    result = engine._prepare_diffusion_kv_profile_requests()

    assert result is not None
    assert len(result) == expected_profile_requests
    assert len({profile_request.request_id for profile_request in result}) == expected_profile_requests
    assert all(profile_request.is_dummy_run() for profile_request in result)
    assert all(profile_request.prepared_layout is prepared_layout for profile_request in result)
    assert all(profile_request.diffusion_kv_requests is None for profile_request in result)
    assert len({id(profile_request.sampling_params) for profile_request in result}) == expected_profile_requests
    assert engine._diffusion_kv_profile_limits == (2, 13, 8)
    assert len(result) * engine._diffusion_kv_profile_limits[0] == expected_profile_requests * 2
    engine._make_dummy_request.assert_called_once_with(
        height=1024,
        width=1024,
        guidance_scale=5.0,
        num_image_inputs=3,
    )
    engine._prepare_request_for_admission.assert_called_once_with(request)


@pytest.mark.parametrize(
    ("num_sequences", "seq_len", "target_len"),
    [
        (3, 16, 8),
        (1, 17, 8),
        (1, 16, 9),
    ],
)
def test_paged_kv_admission_rejects_shape_beyond_profile_envelope(
    num_sequences: int,
    seq_len: int,
    target_len: int,
) -> None:
    engine = object.__new__(DiffusionEngine)
    engine._diffusion_kv_profile_limits = (2, 16, 8)
    engine.pre_process_func = lambda request: request
    request = OmniDiffusionRequest(
        prompt="too large",
        request_id="too-large",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        diffusion_kv_requests=tuple(
            DiffusionKVRequest(
                f"too-large/diffusion-kv/{sequence_id}",
                sequence_id=sequence_id,
                prefix_len=0,
                target_len=target_len,
                seq_len=seq_len,
            )
            for sequence_id in range(num_sequences)
        ),
    )

    with pytest.raises(ValueError, match="exceeds the startup memory-profile envelope"):
        engine._prepare_request_for_admission(request)


def test_paged_kv_admission_accepts_shape_at_profile_envelope() -> None:
    engine = object.__new__(DiffusionEngine)
    engine._diffusion_kv_profile_limits = (2, 16, 8)
    engine.pre_process_func = lambda request: request
    request = OmniDiffusionRequest(
        prompt="fits",
        request_id="fits",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        diffusion_kv_requests=tuple(
            DiffusionKVRequest(
                f"fits/diffusion-kv/{sequence_id}",
                sequence_id=sequence_id,
                prefix_len=8,
                target_len=8,
                seq_len=16,
            )
            for sequence_id in range(2)
        ),
    )

    assert engine._prepare_request_for_admission(request) is request
