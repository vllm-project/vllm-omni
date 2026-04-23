# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A high-detail studio photo of an orange tabby cat sitting on a laptop keyboard."


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards={"cuda": 1, "rocm": 1, "xpu": 2})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_zimage(omni_runner_handler: OmniRunnerHandler):
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
    omni_runner_handler.send_diffusion_request(request_config)
