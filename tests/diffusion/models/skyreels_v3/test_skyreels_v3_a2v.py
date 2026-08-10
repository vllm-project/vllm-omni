# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.io_support import get_dummy_run_num_frames
from vllm_omni.diffusion.models.skyreels_v3.pipeline_skyreels_v3_a2v import (
    DEFAULT_SKYREELS_A2V_FRAMES,
    SkyReelsV3A2VPipeline,
    get_skyreels_v3_a2v_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import build_image_to_video_prompt, get_extra_body_params

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _write_a2v_like_model(tmp_path: Path) -> str:
    model_dir = tmp_path / "SkyReels-V3-A2V-19B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "i2v",
                "dim": 5120,
                "num_heads": 40,
                "num_layers": 40,
                "audio_window": 5,
                "intermediate_dim": 512,
                "context_tokens": 32,
                "vae_scale": 4,
            }
        ),
        encoding="utf-8",
    )
    for filename in (
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "Wan2.1_VAE.pth",
    ):
        (model_dir / filename).write_bytes(b"")
    for dirname in ("google/umt5-xxl", "xlm-roberta-large", "chinese-wav2vec2-base"):
        (model_dir / dirname).mkdir(parents=True)
    return str(model_dir)


def _make_request(prompt, **sampling_kwargs) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(**sampling_kwargs),
        request_id="skyreels-a2v-test",
    )


def test_skyreels_v3_a2v_model_resolution_uses_a2v_pipeline(tmp_path: Path):
    model = _write_a2v_like_model(tmp_path)

    assert resolve_model_class_name(model) == "SkyReelsV3A2VPipeline"

    config = OmniDiffusionConfig(model=model)
    config.enrich_config()
    assert config.model_class_name == "SkyReelsV3A2VPipeline"
    assert config.supports_multimodal_inputs
    assert config.max_multimodal_image_inputs == 1


def test_skyreels_v3_a2v_uses_default_diffusion_stage_config(tmp_path: Path):
    model = _write_a2v_like_model(tmp_path)

    StageConfigFactory.get_hf_config.cache_clear()
    StageConfigFactory.try_infer_model_type.cache_clear()

    assert StageConfigFactory.try_infer_model_type(model, trust_remote_code=False) == "skyreels_v3_a2v"
    assert StageConfigFactory.get_pipeline_config(model, trust_remote_code=False) is None


def test_skyreels_v3_a2v_declares_audio_image_warmup():
    assert get_dummy_run_num_frames("SkyReelsV3A2VPipeline", supports_audio_input=True) == 5
    assert SkyReelsV3A2VPipeline.dummy_run_num_frames > 1


def test_skyreels_v3_a2v_preprocess_resizes_image_and_keeps_audio_array():
    image = Image.new("RGB", (320, 480), color=(10, 20, 30))
    audio = np.zeros(16000, dtype=np.float32)
    request = _make_request(
        {"prompt": "a person is talking", "multi_modal_data": {"image": image, "audio": audio}},
        extra_args={"resolution": "480P"},
    )
    preprocess = get_skyreels_v3_a2v_pre_process_func(OmniDiffusionConfig())

    processed = preprocess(request)

    assert processed.sampling_params.height is not None
    assert processed.sampling_params.width is not None
    assert processed.sampling_params.height % 16 == 0
    assert processed.sampling_params.width % 16 == 0
    assert processed.prompt["multi_modal_data"]["image"].size == (
        processed.sampling_params.width,
        processed.sampling_params.height,
    )
    assert processed.prompt["multi_modal_data"]["audio"].shape == audio.shape
    assert processed.prompt["additional_information"]["audio_sample_rate"] == 16000


def test_skyreels_v3_a2v_preprocess_rejects_missing_audio():
    image = Image.new("RGB", (64, 64), color=(255, 255, 255))
    request = _make_request({"prompt": "talk", "multi_modal_data": {"image": image}})
    preprocess = get_skyreels_v3_a2v_pre_process_func(OmniDiffusionConfig())

    with pytest.raises(ValueError, match="requires driving audio"):
        preprocess(request)


def test_skyreels_v3_a2v_preprocess_rejects_short_audio():
    image = Image.new("RGB", (64, 64), color=(255, 255, 255))
    audio = np.zeros(1000, dtype=np.float32)
    request = _make_request({"prompt": "talk", "multi_modal_data": {"image": image, "audio": audio}})
    preprocess = get_skyreels_v3_a2v_pre_process_func(OmniDiffusionConfig())

    with pytest.raises(ValueError, match="too short"):
        preprocess(request)


def test_skyreels_v3_a2v_image_to_video_prompt_builder_accepts_audio():
    image = Image.new("RGB", (64, 64), color=(1, 2, 3))
    audio = np.zeros(16000, dtype=np.float32)

    prompt = build_image_to_video_prompt(
        "SkyReelsV3A2VPipeline",
        "talk",
        "",
        {"image": image, "audio": audio},
    )

    assert prompt["multi_modal_data"]["image"] is image
    assert prompt["multi_modal_data"]["audio"] is audio
    assert "audio_guide_scale" in get_extra_body_params("SkyReelsV3A2VPipeline")
    assert DEFAULT_SKYREELS_A2V_FRAMES == 81
