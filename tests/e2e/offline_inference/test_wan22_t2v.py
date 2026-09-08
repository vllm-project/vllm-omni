# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OfflineOmniClient
from vllm_omni.config.omni_config import VllmOmniDiffusionStageConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
PROMPT = "Two anthropomorphic cats in boxing gear on a spotlighted stage."
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards={"cuda": 1, "rocm": 1})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_structured_diffusion_config_reaches_runtime(omni_runner) -> None:
    """The default WAN resolver launches the typed diffusion config itself."""
    stage_configs = omni_runner.omni.engine.stage_configs
    assert len(stage_configs) == 1
    stage_config = stage_configs[0]

    assert isinstance(stage_config, VllmOmniDiffusionStageConfig)
    assert stage_config.stage_id == 0
    assert stage_config.model_stage == "dit"
    assert stage_config.model_config.model == MODEL
    assert stage_config.diffusion_config.model_class_name == "WanPipeline"
    assert stage_config.final_output_type == "video"


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards={"cuda": 1, "rocm": 1})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_text_to_video_001(offline_client: OfflineOmniClient):
    sampling = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_frames=8,
        fps=8,
        num_inference_steps=2,
        guidance_scale=4.0,
        seed=42,
    )
    request_config = {
        "model": MODEL,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "sampling_params": sampling,
    }
    offline_client.send_diffusion_request(request_config)
