# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E smoke test for Stable Diffusion 3.5 medium with online FP8 quantization.

This test intentionally uses the upstream BF16 checkpoint plus
``quantization="fp8"`` so it can be run on a single CUDA GPU without requiring
an extra pre-quantized SD3 checkpoint.
"""

import os as _os

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

MODEL = _os.environ.get("SD35_MODEL", "stabilityai/stable-diffusion-3.5-medium")
HEIGHT = 256
WIDTH = 256
NUM_STEPS = 2


def _sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=4.5,
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
def test_sd35_fp8_load_and_generate():
    """Load SD3.5-medium with online FP8 quantization and generate one image."""
    with OmniRunner(MODEL, enforce_eager=True, quantization="fp8") as runner:
        outputs = runner.omni.generate(
            "a cozy reading corner with a chair, lamp, and books",
            _sampling_params(),
        )
        images = _first_request_images(outputs)
        assert len(images) >= 1, "Expected at least one generated image"
        img = images[0]
        assert img.width == WIDTH and img.height == HEIGHT
        arr = np.array(img)
        assert arr.std() > 1.0, "Generated image appears blank (std ≈ 0)"
