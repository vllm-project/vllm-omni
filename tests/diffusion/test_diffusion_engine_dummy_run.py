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


def test_dummy_run_recipe_returns_none_without_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    class PlainModel:
        pass

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: PlainModel,
    )

    assert io_support.get_dummy_run_recipe("plain") is None


def test_dummy_run_recipe_returns_copy_of_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    declared = {"task": "t2va", "duration": 4.0}

    class RecipeModel:
        dummy_run_recipe = declared

    monkeypatch.setattr(
        io_support.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: RecipeModel,
    )

    recipe = io_support.get_dummy_run_recipe("recipe")
    assert recipe == declared
    assert recipe is not declared


def test_minimax_h3_declares_a_valid_dlo_warmup_recipe() -> None:
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MINIMAX_H3_MAX_OUTPUT_SECONDS,
        MiniMaxH3Pipeline,
    )

    recipe = MiniMaxH3Pipeline.dummy_run_recipe
    assert recipe is not None
    assert recipe["task"] == "t2va"
    assert recipe["aspect_ratio"] == "16:9"
    # Maximum duration makes the learned peak dominate every production
    # request at the fixed short edge, so no real request re-learns.
    assert float(recipe["duration"]) == MINIMAX_H3_MAX_OUTPUT_SECONDS
    assert MiniMaxH3Pipeline.dummy_run_num_frames == 0  # generic warmup stays opted out


def _dlo_recipe_engine(enable_dlo: bool) -> DiffusionEngine:
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(
        model_class_name="RecipeModel",
        enable_distributed_layerwise_offload=enable_dlo,
    )
    return engine


def _opt_out_generic_warmup(
    monkeypatch: pytest.MonkeyPatch,
    recipe: dict | None,
    *,
    supports_image_input: bool = False,
) -> None:
    """Make the generic dummy run opt out and stub the declared recipe."""

    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.supports_multimodal_input",
        lambda od_config: (supports_image_input, False),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.get_dummy_run_num_frames",
        lambda model_class_name, supports_audio_input: 0,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.get_dummy_run_recipe",
        lambda model_class_name: recipe,
    )


def test_make_dummy_request_applies_recipe_without_dlo(monkeypatch: pytest.MonkeyPatch) -> None:
    # The recipe is feature-neutral: a model whose geometry needs it warms up
    # whatever features are enabled, not only under DLO.
    _opt_out_generic_warmup(monkeypatch, {"task": "t2va", "duration": 4.0, "aspect_ratio": "16:9"})

    request = _dlo_recipe_engine(enable_dlo=False)._make_dummy_request(height=512, width=512, guidance_scale=0.0)

    assert request is not None
    assert request.sampling_params.extra_args["task"] == "t2va"


def test_make_dummy_request_builds_recipe_request_when_generic_opted_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _opt_out_generic_warmup(
        monkeypatch,
        {
            "task": "t2va",
            "duration": 4.0,
            "aspect_ratio": "16:9",
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
        },
    )

    request = _dlo_recipe_engine(enable_dlo=True)._make_dummy_request(height=512, width=512, guidance_scale=0.0)

    assert request is not None
    assert request.is_dummy_run()
    assert request.sampling_params.num_inference_steps == 2
    assert request.sampling_params.guidance_scale == 1.0
    extra = request.sampling_params.extra_args
    assert extra["task"] == "t2va"
    assert extra["duration"] == 4.0
    assert extra["aspect_ratio"] == "16:9"
    assert extra["cfg_text_scale"] == 1.0


def test_make_dummy_request_recipe_stays_text_only_with_image_capable_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # H3 advertises image input for fl2va, but the warmup recipe names t2va,
    # which rejects image conditions; the recipe request must not inherit the
    # generic dummy run's synthetic multimodal data.
    _opt_out_generic_warmup(
        monkeypatch,
        {"task": "t2va", "duration": 4.0, "aspect_ratio": "16:9"},
        supports_image_input=True,
    )

    request = _dlo_recipe_engine(enable_dlo=True)._make_dummy_request(height=512, width=512, guidance_scale=0.0)

    assert request is not None
    assert not request.prompt.get("multi_modal_data")


def test_dummy_run_sends_recipe_request_when_generic_opted_out(monkeypatch: pytest.MonkeyPatch) -> None:
    _opt_out_generic_warmup(monkeypatch, {"task": "t2va", "duration": 15.0, "aspect_ratio": "16:9"})
    engine = _dlo_recipe_engine(enable_dlo=True)
    engine.add_req_and_wait_for_response = Mock(return_value=SimpleNamespace(error=None, request_id="dlo-dummy"))

    engine._dummy_run()

    sent = engine.add_req_and_wait_for_response.call_args.args[0]
    assert sent.is_dummy_run()
    assert sent.sampling_params.extra_args["task"] == "t2va"


def test_dummy_run_still_skips_without_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    _opt_out_generic_warmup(monkeypatch, None)
    engine = _dlo_recipe_engine(enable_dlo=True)
    engine.add_req_and_wait_for_response = Mock()

    engine._dummy_run()

    engine.add_req_and_wait_for_response.assert_not_called()


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
