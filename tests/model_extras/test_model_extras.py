# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    build_image_to_image_prompt,
    build_image_to_video_prompt,
    build_text_to_image_prompt,
    build_x_to_text_prompt,
    get_extra_body_params,
    get_extra_output_params,
    get_output_tensor_range,
    get_x_to_text_model_family,
    should_init_extra_args_for_non_diffusion_stages,
    should_preserve_reference_image_size,
)


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model_type": "bagel"}, "bagel"),
        ({"architectures": ["HunyuanImage3ForConditionalGeneration"]}, "hunyuan_image3"),
        ({"model_type": "mammothmoda2"}, "mammoth_moda2"),
        ({"model_type": "qwen2_vl"}, "generic"),
    ],
)
def test_x_to_text_model_family(monkeypatch: pytest.MonkeyPatch, config: dict[str, object], expected: str) -> None:
    monkeypatch.setattr("vllm.transformers_utils.config.get_hf_file_to_dict", lambda *args: config)
    assert get_x_to_text_model_family("unused") == expected


@pytest.mark.core_model
@pytest.mark.cpu
def test_bagel_x_to_text_prompt_builder() -> None:
    prompt, stop_token_ids = build_x_to_text_prompt("bagel", "unused", "Describe it.", has_image=True)
    assert prompt == {
        "prompt": "<|im_start|>user\n<|image_pad|>\nDescribe it.<|im_end|>\n<|im_start|>assistant\n",
        "modalities": ["text"],
    }
    assert stop_token_ids is None


@pytest.mark.core_model
@pytest.mark.cpu
def test_mammoth_x_to_text_prompt_builder() -> None:
    prompt, stop_token_ids = build_x_to_text_prompt("mammoth_moda2", "unused", "Hello.", has_image=False)
    assert prompt == {
        "prompt": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nHello.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "modalities": ["text"],
        "additional_information": {"omni_task": ["chat"]},
    }
    assert stop_token_ids is None


@pytest.mark.core_model
@pytest.mark.cpu
def test_generic_x_to_text_prompt_builder() -> None:
    assert build_x_to_text_prompt("generic", "unused", "Hello.", has_image=False) == (
        {"prompt": "Hello.", "modalities": ["text"]},
        None,
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
def test_lingbot_extra_registry_declares_request_params() -> None:
    assert get_extra_body_params("LingBotVideoPipeline") == frozenset(
        {
            "batch_cfg",
            "duration",
            "flow_shift",
            "negative_prompt",
            "null_cond_clone_zero",
            "offload_vae_during_denoise",
            "output_type",
            "refiner_sigma_tail_steps",
            "resolution",
            "ratio",
            "shift",
            "t_thresh",
        }
    )
    assert get_extra_output_params("LingBotVideoPipeline") == frozenset()
    assert get_output_tensor_range("LingBotVideoPipeline") == "zero_to_one"
    assert get_output_tensor_range("WanPipeline") == "negative_one_to_one"
    assert should_init_extra_args_for_non_diffusion_stages("LingBotVideoPipeline") is False


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("pipeline_name", ["Cosmos3OmniDiffusersPipeline", "Cosmos3OmniPipeline"])
def test_cosmos3_extra_registry_declares_request_and_response_params(pipeline_name: str) -> None:
    assert get_extra_body_params(pipeline_name) == frozenset(
        {
            "flow_shift",
            "max_sequence_length",
            "use_resolution_template",
            "use_duration_template",
            "negative_metadata_mode",
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
    assert get_extra_output_params(pipeline_name) == frozenset(
        {
            "action",
            "raw_action_dim",
            "domain_id",
            "action_mode",
        }
    )
    assert should_init_extra_args_for_non_diffusion_stages(pipeline_name) is False


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
def test_ltx_extra_registry_declares_official_guidance_params() -> None:
    expected = frozenset(
        {
            "video_cfg_scale",
            "audio_cfg_scale",
            "video_cfg_guidance_scale",
            "audio_cfg_guidance_scale",
            "video_stg_scale",
            "audio_stg_scale",
            "video_stg_guidance_scale",
            "audio_stg_guidance_scale",
            "video_modality_scale",
            "audio_modality_scale",
            "a2v_guidance_scale",
            "v2a_guidance_scale",
            "video_rescale_scale",
            "audio_rescale_scale",
            "video_stg_blocks",
            "audio_stg_blocks",
            "sigmas",
            "stage_1_sigmas",
            "stage_2_sigmas",
            "image_crf",
        }
    )

    for pipeline_name in (
        "LTX2Pipeline",
        "LTX2TwoStagePipeline",
        "LTX2DistilledOneStagePipeline",
        "LTX2DistilledTwoStagePipeline",
        "LTX2DistilledPipeline",
    ):
        assert get_extra_body_params(pipeline_name) == expected
        assert get_extra_output_params(pipeline_name) == frozenset()


@pytest.mark.parametrize(
    ("model_version", "expected"),
    [("2", False), ("2.3", False), ("2.5", True)],
)
def test_ltx_reference_image_size_policy(tmp_path, model_version: str, expected: bool) -> None:
    (tmp_path / "model_index.json").write_text(
        '{"_class_name": "LTX2Pipeline", "model_version": "' + model_version + '"}'
    )

    assert (
        should_preserve_reference_image_size(
            None,
            model=str(tmp_path),
        )
        is expected
    )


def test_reference_image_size_policy_threads_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_policy(*, model, revision=None):
        captured.update(model=model, revision=revision)
        return True

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.ltx2.ltx2_components.preserves_reference_image_size",
        fake_policy,
    )

    assert should_preserve_reference_image_size(
        "LTX2Pipeline",
        model="org/model",
        revision="pinned-revision",
    )
    assert captured == {"model": "org/model", "revision": "pinned-revision"}


@pytest.mark.core_model
@pytest.mark.cpu
def test_ming_flash_omni_extra_registry_declares_request_and_response_params() -> None:
    assert get_extra_body_params("MingImagePipeline") == frozenset(
        {
            "height",
            "width",
            "steps",
            "cfg",
            "seed",
            "byte5_text",
            "negative_prompt",
        }
    )
    assert get_extra_output_params("MingImagePipeline") == frozenset()
    assert should_init_extra_args_for_non_diffusion_stages("MingImagePipeline") is True


@pytest.mark.core_model
@pytest.mark.cpu
def test_ming_flash_omni_declared_extra_args_route_into_sampling_params() -> None:
    params = OmniDiffusionSamplingParams()
    declared = get_extra_body_params("MingImagePipeline")
    apply_declared_extra_args(
        params,
        declared,
        {
            "steps": 6,
            "cfg": 1.5,
            "byte5_text": ["理解与生成统一"],
            "negative_prompt": "ugly, blurry",
            "unknown": "ignored",
        },
    )
    assert params.extra_args == {
        "steps": 6,
        "cfg": 1.5,
        "byte5_text": ["理解与生成统一"],
        "negative_prompt": "ugly, blurry",
    }


@pytest.mark.core_model
@pytest.mark.cpu
def test_ming_flash_omni_text_to_image_prompt_builder() -> None:
    assert build_text_to_image_prompt(
        "MingImagePipeline",
        {
            "prompt": "Please draw a cute cat.",
            "modalities": ["image"],
            "negative_prompt": "ugly, blurry",
        },
        height=512,
        width=768,
    ) == {
        "prompt": "Please draw a cute cat.",
        "modalities": ["image"],
        "mm_processor_kwargs": {
            "modalities": ["image"],
            "target_h": 512,
            "target_w": 768,
        },
        "negative_prompt": "ugly, blurry",
    }
    # target_h/w are omitted when height/width are not supplied so the
    # pipeline's 1024x1024 default still applies.
    assert build_text_to_image_prompt(
        "MingImagePipeline",
        {"prompt": "Draw a poster.", "modalities": ["image"]},
    ) == {
        "prompt": "Draw a poster.",
        "modalities": ["image"],
        "mm_processor_kwargs": {"modalities": ["image"]},
    }


@pytest.mark.core_model
@pytest.mark.cpu
def test_ming_flash_omni_image_to_image_prompt_builder() -> None:
    dummy_image = Image.new("RGB", (64, 64))
    result = build_image_to_image_prompt(
        "MingImagePipeline",
        prompt="Change the background to a sandy beach at sunset.",
        negative_prompt=None,
        input_image=dummy_image,
        height=256,
        width=256,
    )
    assert result["modalities"] == ["img2img"]
    assert result["multi_modal_data"]["img2img"] is dummy_image
    assert result["mm_processor_kwargs"] == {
        "modalities": ["img2img"],
        "target_h": 256,
        "target_w": 256,
    }
    assert "negative_prompt" not in result


@pytest.mark.core_model
@pytest.mark.cpu
def test_audiox_extra_registry_declares_request_and_response_params() -> None:
    assert get_extra_body_params("AudioXPipeline") == frozenset(
        {
            "audiox_task",
            "seconds_start",
            "seconds_total",
            "sigma_min",
            "sigma_max",
            "cfg_rescale",
            "video_path",
            "audio_path",
        }
    )
    assert get_extra_output_params("AudioXPipeline") == frozenset({"audiox_task"})
    assert should_init_extra_args_for_non_diffusion_stages("AudioXPipeline") is False


@pytest.mark.core_model
@pytest.mark.cpu
def test_audiox_declared_extra_args_route_into_sampling_params() -> None:
    params = OmniDiffusionSamplingParams()
    declared = get_extra_body_params("AudioXPipeline")
    apply_declared_extra_args(
        params,
        declared,
        {
            "audiox_task": "t2a",
            "seconds_total": 10.0,
            "sigma_min": 0.03,
            "unknown": "ignored",
        },
    )
    assert params.extra_args == {
        "audiox_task": "t2a",
        "seconds_total": 10.0,
        "sigma_min": 0.03,
    }


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
    assert (
        should_preserve_reference_image_size(
            "UnknownPipeline",
            model="org/model",
        )
        is False
    )
    assert should_init_extra_args_for_non_diffusion_stages("UnknownPipeline") is False


@pytest.mark.core_model
@pytest.mark.cpu
def test_bagel_text_to_image_prompt_builder() -> None:
    assert build_text_to_image_prompt(
        "BagelPipeline",
        {
            "prompt": "a cat",
            "modalities": ["image"],
            "negative_prompt": "blurry",
        },
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
def test_unknown_pipeline_preserves_canonical_text_to_image_prompt() -> None:
    canonical_prompt = {"prompt": "a cat", "modalities": ["image"]}
    assert (
        build_text_to_image_prompt(
            "UnknownPipeline",
            canonical_prompt,
            height=512,
            width=512,
        )
        is canonical_prompt
    )


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


def _build_vace_prompt(media_inputs: dict[str, Any], *, num_frames: int = 5) -> dict:
    return build_image_to_video_prompt(
        "WanVACEPipeline",
        {
            "prompt": "a bird flying",
            "modalities": ["video"],
            "multi_modal_data": media_inputs,
        },
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
def test_vace_image_to_video_prompt_builder(media_inputs: dict[str, Any]) -> None:
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
def test_vace_rejects_invalid_media_combinations(media_inputs: dict[str, Any], message: str) -> None:
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
    for model_class_name in (
        "MammothModa2DiTPipeline",
        "MammothModa2ForConditionalGeneration",
        "Mammothmoda2Model",
    ):
        assert get_extra_body_params(model_class_name) == frozenset(
            {
                "text_guidance_scale",
                "cfg_range",
                "num_inference_steps",
            }
        )
        assert get_extra_output_params(model_class_name) == frozenset()
        assert should_init_extra_args_for_non_diffusion_stages(model_class_name) is True

    wrapper_prompt = build_text_to_image_prompt(
        "MammothModa2ForConditionalGeneration",
        {"prompt": "a cat", "modalities": ["image"]},
        height=256,
        width=256,
    )
    assert wrapper_prompt["additional_information"]["omni_task"] == ["t2i"]


@pytest.mark.core_model
@pytest.mark.cpu
def test_mammothmoda2_text_to_image_prompt_builder() -> None:
    # Image dims are converted to the AR grid (width/16 x height/16); the negative
    # prompt is ignored (MammothModa2 t2i uses CFG, not an explicit negative path).
    assert build_text_to_image_prompt(
        "MammothModa2DiTPipeline",
        {
            "prompt": "a cat",
            "modalities": ["image"],
            "negative_prompt": "blurry",
        },
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
            "eol_token_id": [152064],
            "visual_token_start_id": [152072],
            "visual_token_end_id": [168456],
        },
    }
