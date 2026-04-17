# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end test for Flux2 Klein inpainting.

"""

# ruff: noqa: E402

import os
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

MODEL = "black-forest-labs/FLUX.2-klein-4B"

_HEIGHT = 512
_WIDTH = 512
_NUM_INFERENCE_STEPS = 4


def _create_test_image(width: int = _WIDTH, height: int = _HEIGHT, color: tuple = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def _create_test_mask(width: int = _WIDTH, height: int = _HEIGHT) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([width // 4, height // 4, width * 3 // 4, height * 3 // 4], fill=255)
    return mask


def _create_test_inputs(color: tuple = (100, 150, 200)):
    return _create_test_image(_WIDTH, _HEIGHT, color), _create_test_mask(_WIDTH, _HEIGHT)


def _extract_images_from_output(outputs: list) -> list[Image.Image]:
    images = []
    for req_output in outputs:
        if hasattr(req_output, "images") and req_output.images:
            images.extend(req_output.images)
        elif hasattr(req_output, "request_output") and req_output.request_output:
            stage_out = req_output.request_output
            if isinstance(stage_out, OmniRequestOutput) and hasattr(stage_out, "images"):
                images.extend(stage_out.images)
            elif isinstance(stage_out, list):
                for s in stage_out:
                    if hasattr(s, "images") and s.images:
                        images.extend(s.images)
    return images


@pytest.mark.core_model
@pytest.mark.diffusion
def test_flux2_klein_inpaint_basic():
    m = None
    try:
        m = Omni(model=MODEL)
        input_image, mask_image = _create_test_inputs()

        outputs = m.generate(
            prompts=[
                {
                    "prompt": "Fill in the masked area with a beautiful garden",
                    "multi_modal_data": {"image": input_image, "mask_image": mask_image},
                }
            ],
            sampling_params_list=OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                guidance_scale=0.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
                num_outputs_per_prompt=1,
            ),
        )

        images = _extract_images_from_output(list(outputs))
        assert len(images) == 1
        assert images[0].size == (_WIDTH, _HEIGHT)
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


@pytest.mark.diffusion
def test_flux2_klein_inpaint_deterministic():
    m = None
    try:
        m = Omni(model=MODEL)
        input_image, mask_image = _create_test_inputs()
        seed = 12345

        gen1 = torch.Generator(current_omni_platform.device_type).manual_seed(seed)
        gen2 = torch.Generator(current_omni_platform.device_type).manual_seed(seed)

        outputs1 = m.generate(
            prompts=[
                {
                    "prompt": "A red flower in a field",
                    "multi_modal_data": {"image": input_image, "mask_image": mask_image},
                }
            ],
            sampling_params_list=OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                guidance_scale=0.0,
                generator=gen1,
                num_outputs_per_prompt=1,
            ),
        )

        outputs2 = m.generate(
            prompts=[
                {
                    "prompt": "A red flower in a field",
                    "multi_modal_data": {"image": input_image, "mask_image": mask_image},
                }
            ],
            sampling_params_list=OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                guidance_scale=0.0,
                generator=gen2,
                num_outputs_per_prompt=1,
            ),
        )

        images1 = _extract_images_from_output(list(outputs1))
        images2 = _extract_images_from_output(list(outputs2))

        assert len(images1) == 1
        assert len(images2) == 1

        assert list(images1[0].getdata()) == list(images2[0].getdata()), (
            "Same input with same seed should produce identical output. "
            "This is critical for offline/online consistency."
        )
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()


@pytest.mark.diffusion
def test_flux2_klein_inpaint_different_seeds_different_output():
    m = None
    try:
        m = Omni(model=MODEL)
        input_image, mask_image = _create_test_inputs()

        gen1 = torch.Generator(current_omni_platform.device_type).manual_seed(42)
        gen2 = torch.Generator(current_omni_platform.device_type).manual_seed(99999)

        outputs1 = m.generate(
            prompts=[
                {
                    "prompt": "A beautiful landscape",
                    "multi_modal_data": {"image": input_image, "mask_image": mask_image},
                }
            ],
            sampling_params_list=OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                guidance_scale=0.0,
                generator=gen1,
                num_outputs_per_prompt=1,
            ),
        )

        outputs2 = m.generate(
            prompts=[
                {
                    "prompt": "A beautiful landscape",
                    "multi_modal_data": {"image": input_image, "mask_image": mask_image},
                }
            ],
            sampling_params_list=OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                guidance_scale=0.0,
                generator=gen2,
                num_outputs_per_prompt=1,
            ),
        )

        images1 = _extract_images_from_output(list(outputs1))
        images2 = _extract_images_from_output(list(outputs2))

        assert len(images1) == 1
        assert len(images2) == 1

        different_pixel_count = sum(1 for p1, p2 in zip(images1[0].getdata(), images2[0].getdata()) if p1 != p2)
        assert different_pixel_count > 0, "Different seeds should produce different outputs"
    finally:
        if m is not None and hasattr(m, "close"):
            m.close()
