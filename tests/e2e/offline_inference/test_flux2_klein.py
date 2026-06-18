# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end tests for Flux2 Klein text-to-image and inpainting.

Text-to-image tests use ``send_diffusion_request`` with a prompt and
``OmniDiffusionSamplingParams``. Inpainting tests additionally pass
``multi_modal_data`` containing ``image`` and ``mask_image``; see
:meth:`OmniRunnerHandler.send_diffusion_request` in ``tests.helpers.runtime``.
"""

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from tests.helpers.assertions import assert_image_valid
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import DiffusionResponse, OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

MODEL = "black-forest-labs/FLUX.2-klein-4B"

_OMNI_RUNNER_PARAM = (MODEL, None)

pytestmark = [pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)]

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


def _images_from_response(response: DiffusionResponse) -> list[Image.Image]:
    if isinstance(response.images[0], list):
        return [f for fr in response.images for f in fr]
    return list(response.images)


def _send_text2img_request(
    omni_runner_handler: OmniRunnerHandler,
    prompt: str,
    seed: int,
    num_outputs_per_prompt: int = 1,
) -> DiffusionResponse:
    return omni_runner_handler.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": prompt,
            "sampling_params": OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                seed=seed,
                num_outputs_per_prompt=num_outputs_per_prompt,
            ),
        }
    )


def _send_inpaint_with_generator(
    omni_runner_handler: OmniRunnerHandler, prompt: str, input_image, mask_image, generator: torch.Generator
) -> DiffusionResponse:
    return omni_runner_handler.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": prompt,
            "multi_modal_data": {"image": input_image, "mask_image": mask_image},
            "sampling_params": OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                guidance_scale=0.0,
                generator=generator,
                num_outputs_per_prompt=1,
            ),
        }
    )


# Regression test for https://github.com/vllm-project/vllm-omni/issues/3097
@pytest.mark.advanced_model
@pytest.mark.diffusion
def test_flux2_klein_can_accept_text_inputs(omni_runner_handler: OmniRunnerHandler):
    omni_runner_handler.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": "a cup of coffee on the table",
            "sampling_params": OmniDiffusionSamplingParams(num_inference_steps=2, seed=42),
        }
    )


@pytest.mark.advanced_model
@pytest.mark.diffusion
def test_flux2_klein_text_to_image(omni_runner_handler: OmniRunnerHandler):
    response = _send_text2img_request(omni_runner_handler, "A cat on a laptop", seed=42)
    images = _images_from_response(response)
    assert len(images) > 0, "No images in response"
    for img in images:
        assert_image_valid(img, width=_WIDTH, height=_HEIGHT)


@pytest.mark.advanced_model
@pytest.mark.diffusion
def test_flux2_klein_text_to_image_deterministic(omni_runner_handler: OmniRunnerHandler):
    prompt = "A mountain landscape at sunset"
    seed = 12345

    r1 = _send_text2img_request(omni_runner_handler, prompt, seed=seed)
    r2 = _send_text2img_request(omni_runner_handler, prompt, seed=seed)

    images1 = _images_from_response(r1)
    images2 = _images_from_response(r2)

    arr1 = np.array(images1[0])
    arr2 = np.array(images2[0])
    assert arr1.shape == arr2.shape, "Images should have the same shape"
    assert np.array_equal(arr1, arr2), "Same prompt with same seed should produce identical output."


@pytest.mark.advanced_model
@pytest.mark.diffusion
def test_flux2_klein_text_to_image_different_seeds(omni_runner_handler: OmniRunnerHandler):
    prompt = "A beautiful landscape"

    r1 = _send_text2img_request(omni_runner_handler, prompt, seed=42)
    r2 = _send_text2img_request(omni_runner_handler, prompt, seed=99999)

    images1 = _images_from_response(r1)
    images2 = _images_from_response(r2)

    arr1 = np.array(images1[0])
    arr2 = np.array(images2[0])
    assert arr1.shape == arr2.shape, "Images should have the same shape"
    assert not np.array_equal(arr1, arr2), "Different seeds should produce different outputs"


@pytest.mark.advanced_model
@pytest.mark.diffusion
def test_flux2_klein_text_to_image_multi_output(omni_runner_handler: OmniRunnerHandler):
    response = _send_text2img_request(omni_runner_handler, "A red rose in a garden", seed=42, num_outputs_per_prompt=2)
    images = _images_from_response(response)
    assert len(images) == 2, f"Expected 2 images, got {len(images)}"
    for img in images:
        assert_image_valid(img, width=_WIDTH, height=_HEIGHT)
    arr0 = np.array(images[0])
    arr1 = np.array(images[1])
    assert arr0.shape == arr1.shape, "Multi-output images should have the same shape"
    assert not np.array_equal(arr0, arr1), "Multi-output images should be different"


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.xfail(reason="Missing input validation: #3702")
def test_flux2_klein_rejects_empty_prompt(omni_runner_handler: OmniRunnerHandler):
    with pytest.raises((ValueError, RuntimeError)):
        omni_runner_handler.send_diffusion_request(
            {
                "model": MODEL,
                "prompt": "",
                "sampling_params": OmniDiffusionSamplingParams(
                    height=_HEIGHT,
                    width=_WIDTH,
                    num_inference_steps=_NUM_INFERENCE_STEPS,
                    seed=42,
                ),
            }
        )


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.xfail(reason="Missing input validation: #3703")
def test_flux2_klein_rejects_zero_inference_steps(omni_runner_handler: OmniRunnerHandler):
    with pytest.raises((ValueError, RuntimeError)):
        omni_runner_handler.send_diffusion_request(
            {
                "model": MODEL,
                "prompt": "A cat on a laptop",
                "sampling_params": OmniDiffusionSamplingParams(
                    height=_HEIGHT,
                    width=_WIDTH,
                    num_inference_steps=0,
                    seed=42,
                ),
            }
        )


@pytest.mark.advanced_model
@pytest.mark.diffusion
def test_flux2_klein_handles_non_divisible_dimensions(omni_runner_handler: OmniRunnerHandler):
    response = omni_runner_handler.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": "A cat on a laptop",
            "sampling_params": OmniDiffusionSamplingParams(
                height=513,
                width=513,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                seed=42,
            ),
        }
    )
    images = _images_from_response(response)
    assert len(images) > 0, "Non-divisible dimensions should be handled gracefully"


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux2_klein_inpaint_basic(omni_runner_handler: OmniRunnerHandler):
    input_image, mask_image = _create_test_inputs()
    omni_runner_handler.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": "Fill in the masked area with a beautiful garden",
            "multi_modal_data": {"image": input_image, "mask_image": mask_image},
            "sampling_params": OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                guidance_scale=0.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
                num_outputs_per_prompt=1,
            ),
        }
    )


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux2_klein_inpaint_deterministic(omni_runner_handler: OmniRunnerHandler):
    input_image, mask_image = _create_test_inputs()
    seed = 12345

    gen1 = torch.Generator(current_omni_platform.device_type).manual_seed(seed)
    gen2 = torch.Generator(current_omni_platform.device_type).manual_seed(seed)

    r1 = _send_inpaint_with_generator(omni_runner_handler, "A red flower in a field", input_image, mask_image, gen1)
    r2 = _send_inpaint_with_generator(omni_runner_handler, "A red flower in a field", input_image, mask_image, gen2)

    images1 = _images_from_response(r1)
    images2 = _images_from_response(r2)

    arr1 = np.array(images1[0])
    arr2 = np.array(images2[0])
    assert arr1.shape == arr2.shape, "Images should have the same shape"
    assert np.array_equal(arr1, arr2), (
        "Same input with same seed should produce identical output. This is critical for offline/online consistency."
    )


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_flux2_klein_inpaint_different_seeds_different_output(omni_runner_handler: OmniRunnerHandler):
    input_image, mask_image = _create_test_inputs()

    gen1 = torch.Generator(current_omni_platform.device_type).manual_seed(42)
    gen2 = torch.Generator(current_omni_platform.device_type).manual_seed(99999)

    r1 = _send_inpaint_with_generator(omni_runner_handler, "A beautiful landscape", input_image, mask_image, gen1)
    r2 = _send_inpaint_with_generator(omni_runner_handler, "A beautiful landscape", input_image, mask_image, gen2)

    images1 = _images_from_response(r1)
    images2 = _images_from_response(r2)

    arr1 = np.array(images1[0])
    arr2 = np.array(images2[0])
    assert arr1.shape == arr2.shape, "Images should have the same shape"
    assert not np.array_equal(arr1, arr2), "Different seeds should produce different outputs"
