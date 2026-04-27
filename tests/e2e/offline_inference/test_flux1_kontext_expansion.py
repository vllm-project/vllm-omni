# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for FluxKontext model pipeline.

FluxKontext is a text-to-image and image-to-image diffusion model that supports:
- Text-to-image generation
- Image editing with text guidance
"""

from __future__ import annotations

import pytest
from vllm.assets.image import ImageAsset

from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "black-forest-labs/FLUX.1-Kontext-dev"

_OMNI_RUNNER_PARAM = (
    MODEL,
    None,
    {
        "parallel_config": DiffusionParallelConfig(tensor_parallel_size=2),
        "enable_cpu_offload": False,
    },
)

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


def _sampling_512() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=2,
        seed=42,
    )


def test_flux_kontext_text_to_image(omni_runner_handler: OmniRunnerHandler):
    """Test FluxKontext text-to-image generation with real model."""
    request_config = {
        "model": MODEL,
        "prompt": "A photo of a cat sitting on a laptop",
        "sampling_params": _sampling_512(),
    }
    omni_runner_handler.send_diffusion_request(request_config)


def test_flux_kontext_image_edit(omni_runner_handler: OmniRunnerHandler):
    """Test FluxKontext image-to-image editing with real model."""
    input_image = ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")
    request_config = {
        "prompt": "Transform this image into a Vincent van Gogh style painting",
        "multi_modal_data": {"img2img": input_image},
        "modalities": ["img2img"],
        "sampling_params": _sampling_512(),
    }
    omni_runner_handler.send_diffusion_request(request_config)
