# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end tests for Krea 2 text-to-image generation.

Tests both the base (Krea-2-Raw, 28-step) and distilled (Krea-2-Turbo, 8-step)
checkpoints via the OmniRunner offline path.
"""

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import DiffusionResponse, OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

MODEL_RAW = "krea/Krea-2-Raw"
MODEL_TURBO = "krea/Krea-2-Turbo"

_HEIGHT = 512
_WIDTH = 512


def _images_from_response(response: DiffusionResponse) -> list:
    if isinstance(response.images[0], list):
        return [f for fr in response.images for f in fr]
    return list(response.images)


# --- Base model (Krea-2-Raw) ---

_RAW_RUNNER_PARAM = (MODEL_RAW, {"model_config": {}})

pytestmark_raw = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
]


@pytest.mark.parametrize("omni_runner", [_RAW_RUNNER_PARAM], indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.full_model
@pytest.mark.diffusion
def test_krea2_raw_text_to_image(omni_runner_handler: OmniRunnerHandler):
    """Test Krea-2-Raw base model text-to-image generation."""
    omni_runner_handler.send_diffusion_request(
        {
            "model": MODEL_RAW,
            "prompt": "a serene Vermont mountain lake at dawn",
            "sampling_params": OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=28,
                guidance_scale=4.5,
                seed=42,
            ),
        }
    )


# --- Distilled model (Krea-2-Turbo) ---

_TURBO_RUNNER_PARAM = (MODEL_TURBO, {"model_config": {"is_distilled": True}})


@pytest.mark.parametrize("omni_runner", [_TURBO_RUNNER_PARAM], indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.full_model
@pytest.mark.diffusion
def test_krea2_turbo_text_to_image(omni_runner_handler: OmniRunnerHandler):
    """Test Krea-2-Turbo distilled model text-to-image generation."""
    omni_runner_handler.send_diffusion_request(
        {
            "model": MODEL_TURBO,
            "prompt": "a cup of coffee on the table",
            "sampling_params": OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=8,
                guidance_scale=0.0,
                seed=42,
            ),
        }
    )


@pytest.mark.parametrize("omni_runner", [_TURBO_RUNNER_PARAM], indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.full_model
@pytest.mark.diffusion
def test_krea2_deterministic_seed(omni_runner_handler: OmniRunnerHandler):
    """Same seed must produce pixel-identical output."""
    seed = 12345

    def _generate():
        gen = torch.Generator(current_omni_platform.device_type).manual_seed(seed)
        return omni_runner_handler.send_diffusion_request(
            {
                "model": MODEL_TURBO,
                "prompt": "a red flower in a green field",
                "sampling_params": OmniDiffusionSamplingParams(
                    height=_HEIGHT,
                    width=_WIDTH,
                    num_inference_steps=8,
                    guidance_scale=0.0,
                    generator=gen,
                    num_outputs_per_prompt=1,
                ),
            }
        )

    r1 = _generate()
    r2 = _generate()

    images1 = _images_from_response(r1)
    images2 = _images_from_response(r2)

    assert list(images1[0].getdata()) == list(images2[0].getdata()), (
        "Same input with same seed should produce identical output."
    )
