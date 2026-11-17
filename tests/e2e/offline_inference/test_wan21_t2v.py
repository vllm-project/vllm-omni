# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan2.1 text-to-video with denoise-step graph replay."""

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OfflineOmniClient
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
PROMPT = "Two anthropomorphic cats in boxing gear on a spotlighted stage."
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"
_REPEAT_MAX_MEAN_ABSOLUTE_ERROR = 0.015


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "xpu": "B60"})
@pytest.mark.parametrize("omni_runner", [(MODEL, None, {"diffusion_use_graph": True})], indirect=True)
def test_wan21_t2v_denoise_graph(offline_client: OfflineOmniClient):
    def generate() -> np.ndarray:
        sampling = OmniDiffusionSamplingParams(
            height=320,
            width=512,
            num_frames=9,
            num_inference_steps=4,
            guidance_scale=5.0,
            seed=42,
        )
        response = offline_client.send_diffusion_request(
            {"model": MODEL, "prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT, "sampling_params": sampling}
        )
        frames = np.asarray(response.images[0], dtype=np.float32)
        return frames * 255.0 if frames.max() <= 1.0 else frames

    # The first request captures the cond and uncond steps; the repeat replays them.
    first = generate()
    repeat = generate()
    assert first.shape == repeat.shape
    assert first.shape[-4:] == (9, 320, 512, 3)
    assert first.std() > 1.0
    deviation = np.abs(first - repeat)
    mean_absolute_error = float(deviation.mean() / 255.0)
    print(f"repeat request: MAE={mean_absolute_error:.6f} maxdiff={deviation.max():.0f}")
    assert mean_absolute_error < _REPEAT_MAX_MEAN_ABSOLUTE_ERROR
