# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline inference smoke test for HiDream-O1-Image-Dev.

Verifies that the pipeline loads, runs a minimal denoising loop, and returns
a valid image tensor without GPU errors. No reference image comparison —
quality validation belongs in the L4 expansion tests.
"""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "HiDream-ai/HiDream-O1-Image-Dev"
PROMPT = "A golden retriever running through a field of sunflowers."

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", [(MODEL, None)], indirect=True)
def test_hidream_o1_t2i(omni_runner_handler: OmniRunnerHandler):
    """Offline text-to-image smoke for HiDream-O1-Image-Dev."""
    sampling = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=42,
    )
    request_config = {
        "model": MODEL,
        "prompt": PROMPT,
        "sampling_params": sampling,
    }
    omni_runner_handler.send_diffusion_request(request_config)
