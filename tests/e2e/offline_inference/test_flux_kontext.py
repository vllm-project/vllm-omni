# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for FluxKontext model pipeline.

FluxKontext is a text-to-image and image-to-image diffusion model that supports:
- Text-to-image generation
- Image editing with text guidance
"""

import pytest
from PIL import Image
from vllm.assets.image import ImageAsset

from tests.conftest import OmniRunner
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "black-forest-labs/FLUX.1-Kontext-dev"


@pytest.mark.core_model
@pytest.mark.diffusion
def test_flux_kontext_text_to_image():
    """Test FluxKontext text-to-image generation with real model."""
    with OmniRunner(
        MODEL,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=2,
        ),
        enable_cpu_offload=False,
    ) as runner:
        omni_outputs = list(
            runner.omni.generate(
                prompts=["A photo of a cat sitting on a laptop"],
                sampling_params_list=OmniDiffusionSamplingParams(
                    height=512,
                    width=512,
                    num_inference_steps=2,
                    seed=42,
                ),
            )
        )

    assert len(omni_outputs) > 0
    output = omni_outputs[0]
    images = None
    if output.images:
        images = output.images
    elif hasattr(output, "request_output") and output.request_output:
        for stage_out in output.request_output:
            if hasattr(stage_out, "images") and stage_out.images:
                images = stage_out.images
                break

    assert images is not None
    assert len(images) > 0
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (512, 512)


@pytest.mark.core_model
@pytest.mark.diffusion
def test_flux_kontext_image_edit():
    """Test FluxKontext image-to-image editing with real model."""
    input_image = ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")
    with OmniRunner(
        MODEL,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=2,
        ),
        enable_cpu_offload=False,
    ) as runner:
        omni_outputs = list(
            runner.omni.generate(
                prompts=[
                    {
                        "prompt": "Transform this image into a Vincent van Gogh style painting",
                        "multi_modal_data": {"img2img": input_image},
                        "modalities": ["img2img"],
                    }
                ],
                sampling_params_list=OmniDiffusionSamplingParams(
                    height=512,
                    width=512,
                    num_inference_steps=2,
                    seed=42,
                ),
            )
        )

    assert len(omni_outputs) > 0
    output = omni_outputs[0]
    images = None
    if output.images:
        images = output.images
    elif hasattr(output, "request_output") and output.request_output:
        for stage_out in output.request_output:
            if hasattr(stage_out, "images") and stage_out.images:
                images = stage_out.images
                break

    assert images is not None
    assert len(images) > 0
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (512, 512)
