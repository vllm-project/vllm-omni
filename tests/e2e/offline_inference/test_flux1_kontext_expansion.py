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
from PIL import Image
from vllm.assets.image import ImageAsset

from tests.helpers.runtime import OmniRunner
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


def _extract_images_from_output(output) -> list | None:
    if output.images:
        return output.images
    if hasattr(output, "request_output") and output.request_output:
        for stage_out in output.request_output:
            if hasattr(stage_out, "images") and stage_out.images:
                return stage_out.images
    return None


def test_flux_kontext_text_to_image(omni_runner: OmniRunner):
    """Test FluxKontext text-to-image generation with real model."""
    omni_outputs = list(
        omni_runner.omni.generate(
            prompts=["A photo of a cat sitting on a laptop"],
            sampling_params_list=_sampling_512(),
        )
    )

    assert len(omni_outputs) > 0
    images = _extract_images_from_output(omni_outputs[0])

    assert images is not None
    assert len(images) > 0
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (512, 512)


def test_flux_kontext_image_edit(omni_runner: OmniRunner):
    """Test FluxKontext image-to-image editing with real model."""
    input_image = ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")
    omni_outputs = list(
        omni_runner.omni.generate(
            prompts=[
                {
                    "prompt": "Transform this image into a Vincent van Gogh style painting",
                    "multi_modal_data": {"img2img": input_image},
                    "modalities": ["img2img"],
                }
            ],
            sampling_params_list=_sampling_512(),
        )
    )

    assert len(omni_outputs) > 0
    images = _extract_images_from_output(omni_outputs[0])

    assert images is not None
    assert len(images) > 0
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (512, 512)
