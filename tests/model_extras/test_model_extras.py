# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
from PIL import Image

from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    build_image_to_image_prompt,
    build_text_to_image_prompt,
    get_extra_body_params,
    get_extra_output_params,
    should_init_extra_args_for_non_diffusion_stages,
)


@pytest.mark.core_model
@pytest.mark.cpu
def test_bagel_extra_registry_declares_request_and_response_params() -> None:
    assert get_extra_body_params("BagelPipeline") == frozenset(
        {
            "cfg_text_scale",
            "cfg_img_scale",
            "cfg_interval",
            "cfg_renorm_type",
            "cfg_renorm_min",
            "negative_prompt",
            "think",
            "max_think_tokens",
            "do_sample",
            "text_temperature",
            "timestep_shift",
        }
    )
    assert get_extra_output_params("BagelPipeline") == frozenset({"text_output", "think_text"})
    assert should_init_extra_args_for_non_diffusion_stages("BagelPipeline") is True


@pytest.mark.core_model
@pytest.mark.cpu
def test_sensenova_extra_registry_declares_request_and_response_params() -> None:
    assert get_extra_body_params("SenseNovaU1Pipeline") == frozenset(
        {
            "think",
            "cfg_scale",
            "cfg_norm",
            "timestep_shift",
            "t_eps",
            "img_cfg_scale",
            "max_tokens",
        }
    )
    assert get_extra_output_params("SenseNovaU1Pipeline") == frozenset({"think_text"})
    assert should_init_extra_args_for_non_diffusion_stages("SenseNovaU1Pipeline") is False


@pytest.mark.core_model
@pytest.mark.cpu
def test_helios_extra_registry_declares_request_and_response_params() -> None:
    expected_body = frozenset(
        {
            "is_enable_stage2",
            "pyramid_num_stages",
            "pyramid_num_inference_steps_list",
            "is_amplify_first_chunk",
            "is_skip_first_chunk",
            "use_cfg_zero_star",
            "use_zero_init",
            "zero_steps",
            "image",
            "video",
            "add_noise_to_image_latents",
            "image_noise_sigma_min",
            "image_noise_sigma_max",
            "add_noise_to_video_latents",
            "video_noise_sigma_min",
            "video_noise_sigma_max",
        }
    )
    # Both the base and pyramid class names resolve to the same declaration.
    for cls in ("HeliosPipeline", "HeliosPyramidPipeline"):
        assert get_extra_body_params(cls) == expected_body
        assert get_extra_output_params(cls) == frozenset()
        assert should_init_extra_args_for_non_diffusion_stages(cls) is False


@pytest.mark.core_model
@pytest.mark.cpu
def test_unknown_pipeline_has_empty_extra_registry() -> None:
    assert get_extra_body_params("UnknownPipeline") == frozenset()
    assert get_extra_output_params("UnknownPipeline") == frozenset()
    assert should_init_extra_args_for_non_diffusion_stages("UnknownPipeline") is False


@pytest.mark.core_model
@pytest.mark.cpu
def test_bagel_text_to_image_prompt_builder() -> None:
    assert build_text_to_image_prompt(
        "BagelPipeline",
        prompt="a cat",
        negative_prompt="blurry",
        height=512,
        width=768,
    ) == {
        "prompt": "<|im_start|>a cat<|im_end|>",
        "modalities": ["image"],
        "mm_processor_kwargs": {
            "target_h": 512,
            "target_w": 768,
            "modalities": ["image"],
        },
        "negative_prompt": "blurry",
    }


@pytest.mark.core_model
@pytest.mark.cpu
def test_bagel_image_to_image_prompt_builder() -> None:
    dummy_image = Image.new("RGB", (64, 64))
    result = build_image_to_image_prompt(
        "BagelPipeline",
        prompt="paint it",
        negative_prompt="ugly",
        input_image=dummy_image,
        height=256,
        width=256,
    )
    assert result["prompt"] == "<|fim_middle|><|im_start|>paint it<|im_end|>"
    assert result["modalities"] == ["img2img"]
    assert result["multi_modal_data"]["img2img"] is dummy_image
    assert result["mm_processor_kwargs"]["target_h"] == 256
    assert result["mm_processor_kwargs"]["target_w"] == 256
    assert result["negative_prompt"] == "ugly"


@pytest.mark.core_model
@pytest.mark.cpu
def test_unknown_pipeline_uses_default_text_to_image_prompt() -> None:
    assert build_text_to_image_prompt(
        "UnknownPipeline",
        prompt="a cat",
        negative_prompt=None,
        height=512,
        width=512,
    ) == {"prompt": "a cat"}


@pytest.mark.core_model
@pytest.mark.cpu
def test_unknown_pipeline_uses_default_image_to_image_prompt() -> None:
    dummy_image = Image.new("RGB", (64, 64))
    result = build_image_to_image_prompt(
        "UnknownPipeline",
        prompt="edit",
        negative_prompt=None,
        input_image=dummy_image,
    )
    assert result == {
        "prompt": "edit",
        "multi_modal_data": {"image": dummy_image},
    }


@pytest.mark.core_model
@pytest.mark.cpu
def test_declared_extra_args_apply_to_existing_sampling_params() -> None:
    params = OmniDiffusionSamplingParams(extra_args={"existing": 1})

    declared_extra_params: frozenset[str] = frozenset({"cfg_text_scale", "think"})
    apply_declared_extra_args(
        params,
        declared_extra_params,
        {
            "cfg_text_scale": 4.0,
            "think": False,
            "unknown": "ignored",
        },
    )

    assert params.extra_args == {
        "existing": 1,
        "cfg_text_scale": 4.0,
        "think": False,
    }
