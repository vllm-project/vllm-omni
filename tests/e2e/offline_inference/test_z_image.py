# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OfflineOmniClient
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A high-detail studio photo of an orange tabby cat sitting on a laptop keyboard."
_REPEAT_MAX_MEAN_ABSOLUTE_ERROR = 0.015


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards={"cuda": 1, "rocm": 1, "xpu": 2})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_zimage(offline_client: OfflineOmniClient):
    # high resolution may cause OOM on L4
    sampling = OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_inference_steps=2,
        guidance_scale=0.0,
        seed=42,
        num_outputs_per_prompt=2,
    )
    request_config = {
        "model": MODEL,
        "prompt": PROMPT,
        "sampling_params": sampling,
    }
    offline_client.send_diffusion_request(request_config)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "xpu": "B60"}, num_cards={"cuda": 1, "xpu": 2})
@pytest.mark.parametrize("omni_runner", [(MODEL, None, {"diffusion_use_graph": True})], indirect=True)
def test_zimage_denoise_graph(offline_client: OfflineOmniClient):
    def generate(prompt: str):
        sampling = OmniDiffusionSamplingParams(
            height=256,
            width=256,
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=42,
            num_outputs_per_prompt=1,
        )
        response = offline_client.send_diffusion_request(
            {"model": MODEL, "prompt": prompt, "sampling_params": sampling}
        )
        return response.images[0]

    # First request captures, the repeat replays the same graph; a longer prompt changes the
    # caption length, so it captures again.
    first = np.asarray(generate(PROMPT), dtype=np.float32)
    repeat = np.asarray(generate(PROMPT), dtype=np.float32)
    assert first.shape == repeat.shape
    # XPU kernels are not bitwise deterministic even with graphs off, so compare with a tolerance.
    deviation = np.abs(first - repeat)
    mean_absolute_error = float(deviation.mean() / 255.0)
    print(
        f"repeat request: MAE={mean_absolute_error:.6f} maxdiff={deviation.max():.0f} "
        f"differing={float((deviation > 0).mean()):.4f}"
    )
    assert mean_absolute_error < _REPEAT_MAX_MEAN_ABSOLUTE_ERROR
    generate(PROMPT + " Soft window light, shallow depth of field, a coffee mug beside the laptop.")
