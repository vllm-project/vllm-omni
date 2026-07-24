# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end test for DreamX-World-5B-Cam camera-controlled I2V (WanCameraPipeline)."""

import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "GD-ML/DreamX-World-5B-Cam"
PROMPT = "A serene blocky landscape at sunset, a cliffside overlooking a calm ocean, smooth camera motion."
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"

HEIGHT = 480
WIDTH = 832
# Must satisfy the 1+4k VAE pattern and num_frames >= len(action_seq) + 1.
NUM_FRAMES = 9


def _start_image() -> Image.Image:
    """Create a deterministic test image for I2V tests."""
    return Image.fromarray(generate_synthetic_image(WIDTH, HEIGHT, seed=42)["np_array"])


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_image_to_video_camera_001(omni_runner_handler: OmniRunnerHandler):
    sampling = OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        fps=16,
        num_inference_steps=2,
        guidance_scale=3.0,
        seed=42,
        extra_args={"action_seq": ["w", "wj"], "action_speed_list": [4, 6]},
    )
    request_config = {
        "model": MODEL,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "images": _start_image(),
        "sampling_params": sampling,
    }
    omni_runner_handler.send_diffusion_request(request_config)
