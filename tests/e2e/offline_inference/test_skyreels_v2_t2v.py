# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OfflineOmniClient
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "Skywork/SkyReels-V2-T2V-14B-540P-Diffusers"
PROMPT = "A serene lake surrounded by mountains, with birds gliding across the water."
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards={"cuda": 1, "rocm": 1})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_text_to_video_001(offline_client: OfflineOmniClient):
    sampling = OmniDiffusionSamplingParams(
        height=544,
        width=960,
        num_frames=9,
        fps=24,
        num_inference_steps=2,
        guidance_scale=6.0,
        seed=42,
    )
    request_config = {
        "model": MODEL,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "sampling_params": sampling,
    }
    offline_client.send_diffusion_request(request_config)
