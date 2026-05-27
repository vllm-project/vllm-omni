# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for Flux1 Schnell."""

import pytest

from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "black-forest-labs/FLUX.1-schnell"

_OMNI_RUNNER_PARAM = (MODEL, None)

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


def test_flux_schnell_text_to_image(omni_runner_handler: OmniRunnerHandler):
    """Test FLUX.1-schnell text-to-image generation."""
    request_config = {
        "model": MODEL,
        "prompt": "A photo of a cat sitting on a laptop",
        "sampling_params": OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            seed=42,
        ),
    }
    omni_runner_handler.send_diffusion_request(request_config)
