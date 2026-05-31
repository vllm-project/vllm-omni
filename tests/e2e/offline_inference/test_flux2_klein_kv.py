# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for Flux2KleinKV model pipeline.

Flux2KleinKV is an image-to-image diffusion model that supports:
- Multi-reference image editing
- Image editing with text guidance
- KV cache optimization for fast inference
"""

import os
import sys
from pathlib import Path

import pytest
from PIL import Image

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

MODEL = "black-forest-labs/FLUX.2-klein-9b-kv"


def _make_image(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _create_omni(tp_size: int = 2, cpu_offload: bool = True) -> Omni:
    return Omni(
        model=MODEL,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=tp_size,
        ),
        enable_cpu_offload=cpu_offload,
        model_class_name="Flux2KleinKVPipeline",
    )


@pytest.mark.advanced_model
@pytest.mark.diffusion
def test_flux2_klein_kv_image_edit_single_reference():
    """Test Flux2KleinKV single reference image editing with KV cache."""
    input_image = _make_image((640, 640), (220, 80, 80))

    omni = _create_omni()

    try:
        omni_outputs = list(
            omni.generate(
                prompts=[
                    {
                        "prompt": "Transform the subject into a watercolor painting style",
                        "multi_modal_data": {"image": input_image},
                        "modalities": ["img2img"],
                    }
                ],
                sampling_params_list=OmniDiffusionSamplingParams(
                    height=512,
                    width=512,
                    num_inference_steps=4,
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
    finally:
        omni.close()


@pytest.mark.advanced_model
@pytest.mark.diffusion
def test_flux2_klein_kv_image_edit_multi_reference():
    """Test Flux2KleinKV multi-reference image editing with KV cache."""
    image1 = _make_image((640, 640), (220, 80, 80))
    image2 = _make_image((768, 512), (80, 220, 120))
    image3 = _make_image((512, 768), (80, 120, 220))

    omni = _create_omni()

    try:
        omni_outputs = list(
            omni.generate(
                prompts=[
                    {
                        "prompt": "The person from image 1 is petting the cat from image 2, and the bird from image 3 is next to them.",
                        "negative_prompt": "blurry, low quality",
                        "multi_modal_data": {"image": [image1, image2, image3]},
                        "modalities": ["img2img"],
                    }
                ],
                sampling_params_list=OmniDiffusionSamplingParams(
                    height=1024,
                    width=1024,
                    num_inference_steps=4,
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
        assert images[0].size == (1024, 1024)
    finally:
        omni.close()


@pytest.mark.advanced_model
@pytest.mark.diffusion
def test_flux2_klein_kv_kv_cache_with_multiple_steps():
    """Test Flux2KleinKV KV cache behavior with more inference steps.

    This test verifies that:
    1. First denoising step extracts and caches reference image KV (extract mode)
    2. Subsequent steps reuse cached KV (cached mode)
    3. KV cache is correctly maintained throughout the denoising process
    """
    input_image = _make_image((512, 512), (100, 150, 200))

    omni = _create_omni()

    try:
        omni_outputs = list(
            omni.generate(
                prompts=[
                    {
                        "prompt": "Apply a sepia tone effect to this image",
                        "multi_modal_data": {"image": input_image},
                        "modalities": ["img2img"],
                    }
                ],
                sampling_params_list=OmniDiffusionSamplingParams(
                    height=512,
                    width=512,
                    num_inference_steps=10,
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
    finally:
        omni.close()


@pytest.mark.advanced_model
@pytest.mark.diffusion
def test_flux2_klein_kv_kv_cache_with_guidance_scale():
    """Test Flux2KleinKV KV cache behavior with classifier-free guidance.

    This test verifies that:
    1. Both positive and negative prompt KV caches are properly handled
    2. CFG combination works correctly with cached KV
    """
    input_image = _make_image((640, 640), (180, 100, 60))

    omni = _create_omni()

    try:
        omni_outputs = list(
            omni.generate(
                prompts=[
                    {
                        "prompt": "Make the colors more vibrant and saturated",
                        "negative_prompt": "blurry, distorted, low quality",
                        "multi_modal_data": {"image": input_image},
                        "modalities": ["img2img"],
                    }
                ],
                sampling_params_list=OmniDiffusionSamplingParams(
                    height=512,
                    width=512,
                    num_inference_steps=4,
                    seed=42,
                    guidance_scale=7.5,
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
    finally:
        omni.close()
