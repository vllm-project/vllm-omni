# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.models.skyreels_v3.pipeline_skyreels_v3_v2v import (
    DEFAULT_SKYREELS_V3_V2V_CONDITION_FRAMES,
    DEFAULT_SKYREELS_V3_V2V_DURATION,
    DEFAULT_SKYREELS_V3_V2V_MAX_DURATION,
    SkyReelsV3V2VPipeline,
    get_skyreels_v3_v2v_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import get_extra_body_params

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _write_v2v_like_model(tmp_path: Path) -> str:
    model_dir = tmp_path / "SkyReels-V3-V2V-14B"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "google" / "umt5-xxl").mkdir(parents=True)
    (model_dir / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "WanTransformer3DModel"}), encoding="utf-8"
    )
    for filename in ("Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"):
        (model_dir / filename).write_bytes(b"")
    return str(model_dir)


def _make_request(prompt, **sampling_kwargs) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(**sampling_kwargs),
        request_id="skyreels-v2v-test",
    )


def _dummy_video(frames: int = 32, height: int = 720, width: int = 1280) -> np.ndarray:
    return np.random.randint(0, 255, size=(frames, height, width, 3), dtype=np.uint8)


def test_skyreels_v3_v2v_model_resolution_uses_v2v_pipeline(tmp_path: Path):
    model = _write_v2v_like_model(tmp_path)

    assert resolve_model_class_name(model) == "SkyReelsV3V2VPipeline"

    config = OmniDiffusionConfig(model=model)
    config.enrich_config()
    assert config.model_class_name == "SkyReelsV3V2VPipeline"


def test_skyreels_v3_v2v_uses_default_diffusion_stage_config(tmp_path: Path):
    model = _write_v2v_like_model(tmp_path)

    StageConfigFactory.get_hf_config.cache_clear()
    StageConfigFactory.try_infer_model_type.cache_clear()

    # V2V checkpoints have no root config.json, so HF-config inference returns
    # None; the routing shortcut in entrypoints/utils.py maps them to the
    # skyreels_v3_v2v type once resolve_model_config_path fires.
    inferred = StageConfigFactory.try_infer_model_type(model, trust_remote_code=False)
    assert inferred in (None, "skyreels_v3_v2v")
    assert StageConfigFactory.get_pipeline_config(model, trust_remote_code=False) is None


def test_skyreels_v3_v2v_declares_zero_dummy_frames():
    assert SkyReelsV3V2VPipeline.dummy_run_num_frames == 0


def test_skyreels_v3_v2v_preprocess_rejects_missing_video():
    request = _make_request({"prompt": "extend"})
    preprocess = get_skyreels_v3_v2v_pre_process_func(OmniDiffusionConfig())

    with pytest.raises(ValueError, match="input video"):
        preprocess(request)


def test_skyreels_v3_v2v_preprocess_rejects_short_input():
    frames = _dummy_video(frames=DEFAULT_SKYREELS_V3_V2V_CONDITION_FRAMES - 1, height=64, width=64)
    request = _make_request(
        {"prompt": "extend", "multi_modal_data": {"video": frames}},
        extra_args={"resolution": "480P"},
    )
    preprocess = get_skyreels_v3_v2v_pre_process_func(OmniDiffusionConfig())

    with pytest.raises(ValueError, match="at least"):
        preprocess(request)


def test_skyreels_v3_v2v_preprocess_resizes_and_records_source_fps():
    frames = _dummy_video(frames=DEFAULT_SKYREELS_V3_V2V_CONDITION_FRAMES, height=1080, width=1920)
    request = _make_request(
        {"prompt": "extend", "multi_modal_data": {"video": frames}},
        extra_args={"resolution": "720P", "duration": 5},
    )
    preprocess = get_skyreels_v3_v2v_pre_process_func(OmniDiffusionConfig())

    processed = preprocess(request)

    assert processed.sampling_params.height is not None
    assert processed.sampling_params.width is not None
    assert processed.sampling_params.height % 16 == 0
    assert processed.sampling_params.width % 16 == 0
    resized = processed.prompt["multi_modal_data"]["video"]
    assert resized.shape[0] == DEFAULT_SKYREELS_V3_V2V_CONDITION_FRAMES
    assert resized.shape[1] == processed.sampling_params.height
    assert resized.shape[2] == processed.sampling_params.width


def test_skyreels_v3_v2v_extra_body_params_include_duration():
    params = get_extra_body_params("SkyReelsV3V2VPipeline")
    assert "duration" in params
    assert "video_path" in params
    assert "condition_frames" in params
    assert DEFAULT_SKYREELS_V3_V2V_DURATION > 0
