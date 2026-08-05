# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test for Qwen-Image AutoRound W4A16 quantized inference.

Verifies that the W4A16 quantized Qwen-Image checkpoint loads end-to-end
(quant_config propagated correctly) and produces a valid image.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

QUANTIZED_MODEL = os.environ.get(
    "QWEN_IMAGE_AUTOROUND_MODEL",
    "INC4AI/Qwen-Image-AutoRound-W4A16",
)

HEIGHT = 512
WIDTH = 512
NUM_STEPS = 2


def _sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        true_cfg_scale=1.0,
        generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
    )


def _first_request_images(outputs) -> list:
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput) and hasattr(req_out, "images")
    return req_out.images


@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"})
def test_qwen_image_autoround_w4a16_load():
    """Load the W4A16 quantized Qwen-Image model and run a minimal generation.

    Verifies that quant_config is propagated to all transformer blocks
    and produces a valid, non-blank image.
    """
    with OmniRunner(QUANTIZED_MODEL, enforce_eager=True) as runner:
        outputs = runner.omni.generate(
            "a cup of coffee on a table",
            _sampling_params(),
        )
        images = _first_request_images(outputs)
        assert len(images) >= 1, "Expected at least one generated image"
        img = images[0]
        assert img.width == WIDTH and img.height == HEIGHT
        arr = np.array(img)
        assert arr.std() > 1.0, "Generated image appears blank (std ≈ 0)"


@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"})
def test_qwen_image_autoround_w4a16_generate():
    """Full generation: 512×512, 20 steps, CFG=5.0.

    Validates that realistic quantized inference (multi-step with CFG
    negative prompts and longer denoising) produces a valid non-blank image.
    """
    params = OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=20,
        true_cfg_scale=5.0,
        generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
    )
    with OmniRunner(QUANTIZED_MODEL, enforce_eager=True) as runner:
        outputs = runner.omni.generate(
            "a cup of coffee on a wooden table, morning sunlight, photorealistic",
            params,
        )
        images = _first_request_images(outputs)
        assert len(images) >= 1, "Expected at least one generated image"
        img = images[0]

        # Save for manual inspection
        output_path = Path(__file__).resolve().parents[4] / "ar_coffee_full.png"
        img.save(str(output_path))

        assert img.width == WIDTH and img.height == HEIGHT
        arr = np.array(img)
        assert arr.std() > 1.0, "Generated image appears blank (std ≈ 0)"
