# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
from PIL import Image

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.io_support import get_dummy_run_num_frames
from vllm_omni.diffusion.models.skyreels_v3.pipeline_skyreels_v3_r2v import (
    SkyReelsV3R2VPipeline,
    get_skyreels_v3_r2v_pre_process_func,
    _resolve_guidance_scales,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import build_image_to_video_prompt, get_extra_body_params

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def _write_wan_like_model(tmp_path: Path, transformer_class_name: str) -> str:
    model_dir = tmp_path / transformer_class_name
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "WanPipeline"}), encoding="utf-8"
    )
    (transformer_dir / "config.json").write_text(
        json.dumps({"_class_name": transformer_class_name}), encoding="utf-8"
    )
    return str(model_dir)


def _make_request(prompt, **sampling_kwargs) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(**sampling_kwargs),
        request_id="skyreels-r2v-test",
    )


def test_skyreels_v3_r2v_model_resolution_overrides_wan_pipeline(tmp_path: Path):
    model = _write_wan_like_model(tmp_path, "SkyReelsC1WanI2v3DModel")

    assert resolve_model_class_name(model) == "SkyReelsV3R2VPipeline"

    config = OmniDiffusionConfig(model=model)
    config.enrich_config()
    assert config.model_class_name == "SkyReelsV3R2VPipeline"
    assert config.supports_multimodal_inputs
    assert config.max_multimodal_image_inputs == 4


def test_skyreels_v3_r2v_uses_default_diffusion_stage_config(tmp_path: Path):
    model = _write_wan_like_model(tmp_path, "SkyReelsC1WanI2v3DModel")

    StageConfigFactory.get_hf_config.cache_clear()
    StageConfigFactory.try_infer_model_type.cache_clear()

    assert StageConfigFactory.try_infer_model_type(model, trust_remote_code=False) == "skyreels_v3_r2v"
    assert StageConfigFactory.get_pipeline_config(model, trust_remote_code=False) is None


def test_skyreels_v3_r2v_declares_small_dummy_warmup():
    assert get_dummy_run_num_frames("SkyReelsV3R2VPipeline", supports_audio_input=False) == 5
    assert SkyReelsV3R2VPipeline.dummy_run_num_frames > 1


def test_skyreels_v3_r2v_guidance_accepts_warmup_cfg_aliases():
    request = _make_request(
        {"prompt": "warmup", "multi_modal_data": {"image": Image.new("RGB", (8, 8))}},
        guidance_scale=0.0,
    )

    text_scale, image_scale = _resolve_guidance_scales(
        request.sampling_params,
        {"cfg_text_scale": 1.0, "cfg_img_scale": 1.0},
    )

    assert text_scale == 1.0
    assert image_scale == 1.0


def test_non_r2v_wan_pipeline_is_not_routed_to_skyreels_r2v(tmp_path: Path):
    model = _write_wan_like_model(tmp_path, "SkyReelsA2WanI2v3DModel")

    assert resolve_model_class_name(model) == "WanPipeline"


def test_skyreels_v3_r2v_preprocess_resizes_reference_alias():
    image = Image.new("RGB", (320, 200), color=(10, 20, 30))
    request = _make_request(
        {"prompt": "animate the reference", "multi_modal_data": {"reference_images": image}},
        extra_args={"resolution": "480P"},
    )
    preprocess = get_skyreels_v3_r2v_pre_process_func(OmniDiffusionConfig())

    processed = preprocess(request)

    assert processed.sampling_params.height is not None
    assert processed.sampling_params.width is not None
    assert processed.sampling_params.height % 16 == 0
    assert processed.sampling_params.width % 16 == 0
    processed_images = processed.prompt["multi_modal_data"]["image"]
    assert len(processed_images) == 1
    assert processed_images[0].size == (
        processed.sampling_params.width,
        processed.sampling_params.height,
    )


def test_skyreels_v3_r2v_preprocess_rejects_too_many_reference_images():
    image = Image.new("RGB", (64, 64), color=(255, 255, 255))
    request = _make_request(
        {"prompt": "animate the reference", "multi_modal_data": {"image": [image] * 5}},
    )
    preprocess = get_skyreels_v3_r2v_pre_process_func(OmniDiffusionConfig())

    with pytest.raises(ValueError, match="at most 4 reference images"):
        preprocess(request)


def test_skyreels_v3_r2v_image_to_video_prompt_builder_accepts_references():
    images = [Image.new("RGB", (64, 64), color=(idx, idx, idx)) for idx in range(2)]

    prompt = build_image_to_video_prompt(
        "SkyReelsV3R2VPipeline",
        "animate the references",
        "",
        {"reference_images": images},
    )

    assert prompt["multi_modal_data"]["reference_images"] is images
    assert "guidance_scale_img" in get_extra_body_params("SkyReelsV3R2VPipeline")
