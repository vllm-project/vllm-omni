# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
from PIL import Image

from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    build_image_to_image_prompt,
    build_image_to_video_prompt,
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
def test_cosmos3_extra_registry_declares_request_and_response_params() -> None:
    assert get_extra_body_params("Cosmos3OmniDiffusersPipeline") == frozenset(
        {
            "flow_shift",
            "max_sequence_length",
            "use_resolution_template",
            "use_duration_template",
            "use_system_prompt",
            "system_prompt",
            "negative_prompt",
            "guardrails",
            "condition_frame_indexes_vision",
            "condition_video_keep",
            "generate_sound",
            "sound_gen",
            "sound_duration",
            "audio_duration",
            "action_mode",
            "action",
            "domain_name",
            "domain_id",
            "raw_action_dim",
            "action_chunk_size",
            "action_space",
            "action_fps",
            "image_height",
            "image_width",
            "history_length",
            "conditioning_fps",
            "resolution",
            "image_size",
            "use_state",
            "observation",
            "robot_obs",
            "deterministic_seed",
            "session_id",
        }
    )
    assert get_extra_output_params("Cosmos3OmniDiffusersPipeline") == frozenset(
        {
            "action",
            "raw_action_dim",
            "domain_id",
            "action_mode",
        }
    )
    assert should_init_extra_args_for_non_diffusion_stages("Cosmos3OmniDiffusersPipeline") is False


@pytest.mark.core_model
@pytest.mark.cpu
def test_magi_human_extra_registry_declares_request_and_response_params() -> None:
    assert get_extra_body_params("MagiHumanPipeline") == frozenset(
        {
            "seconds",
            "audio_path",
            "image_path",
            "sr_height",
            "sr_width",
            "sr_num_inference_steps",
        }
    )
    assert get_extra_output_params("MagiHumanPipeline") == frozenset()
    assert should_init_extra_args_for_non_diffusion_stages("MagiHumanPipeline") is False


@pytest.mark.core_model
@pytest.mark.cpu
def test_cosmos3_text_to_image_prompt_builder_selects_image_modality() -> None:
    assert build_text_to_image_prompt(
        "Cosmos3OmniDiffusersPipeline",
        prompt="a red sports car at golden hour",
        negative_prompt="blurry, distorted",
        height=1024,
        width=1024,
    ) == {
        "prompt": "a red sports car at golden hour",
        "modalities": ["image"],
        "negative_prompt": "blurry, distorted",
    }
    assert build_text_to_image_prompt(
        "Cosmos3OmniDiffusersPipeline",
        prompt="a red sports car",
        negative_prompt=None,
    ) == {"prompt": "a red sports car", "modalities": ["image"]}


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
def test_vace_extra_registry_has_no_pipeline_params() -> None:
    assert get_extra_body_params("WanVACEPipeline") == frozenset()
    assert get_extra_output_params("WanVACEPipeline") == frozenset()
    assert should_init_extra_args_for_non_diffusion_stages("WanVACEPipeline") is False


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


def _build_vace_prompt(media_inputs: dict[str, object], *, num_frames: int = 5) -> dict:
    return build_image_to_video_prompt(
        "WanVACEPipeline",
        prompt="a bird flying",
        negative_prompt=None,
        media_inputs=media_inputs,
        height=16,
        width=320,
        num_frames=num_frames,
    )


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize(
    "media_inputs",
    [
        {"image": Image.new("RGB", (320, 16), "red")},
        {"last_image": Image.new("RGB", (320, 16), "blue")},
        {
            "image": Image.new("RGB", (320, 16), "red"),
            "last_image": Image.new("RGB", (320, 16), "blue"),
        },
        {"image": Image.new("RGB", (320, 16), "red"), "mask": Image.new("L", (320, 16), 0)},
        {"reference_images": [Image.new("RGB", (64, 64), "red")]},
    ],
    ids=["i2v", "v2lf", "flf2v", "inpaint", "r2v"],
)
def test_vace_image_to_video_prompt_builder(media_inputs: dict[str, object]) -> None:
    result = _build_vace_prompt(media_inputs)
    mmd = result["multi_modal_data"]
    if "reference_images" in mmd:
        assert mmd["reference_images"] is media_inputs["reference_images"]
    else:
        assert len(mmd["video"]) == len(mmd["mask"]) == 5


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize(
    ("media_inputs", "message"),
    [
        ({}, "requires a conditioning media input"),
        ({"mask": Image.new("L", (320, 16))}, "mask input requires an image"),
        ({"control_image": Image.new("RGB", (320, 16))}, "Unsupported VACE media input"),
    ],
)
def test_vace_rejects_invalid_media_combinations(media_inputs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _build_vace_prompt(media_inputs)


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


@pytest.mark.core_model
@pytest.mark.cpu
def test_mammothmoda2_extra_registry_declares_request_and_response_params() -> None:
    assert get_extra_body_params("MammothModa2DiTPipeline") == frozenset(
        {
            "text_guidance_scale",
            "cfg_range",
            "num_inference_steps",
        }
    )
    assert get_extra_output_params("MammothModa2DiTPipeline") == frozenset()
    assert should_init_extra_args_for_non_diffusion_stages("MammothModa2DiTPipeline") is True


@pytest.mark.core_model
@pytest.mark.cpu
def test_mammothmoda2_text_to_image_prompt_builder() -> None:
    # Image dims are converted to the AR grid (width/16 x height/16); the negative
    # prompt is ignored (MammothModa2 t2i uses CFG, not an explicit negative path).
    assert build_text_to_image_prompt(
        "MammothModa2DiTPipeline",
        prompt="a cat",
        negative_prompt="blurry",
        height=512,
        width=768,
    ) == {
        "prompt": (
            "<|im_start|>system\nYou are a helpful image generator.<|im_end|>\n"
            "<|im_start|>user\na cat<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<|image start|>48*32<|image token|>"
        ),
        "additional_information": {
            "omni_task": ["t2i"],
            "ar_width": [48],
            "ar_height": [32],
            "image_height": [512],
            "image_width": [768],
        },
    }
