# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end tests for FLUX.2-dev text-to-image offline inference.

FLUX.2-dev is a 32B parameter diffusion transformer that requires CPU offload
on a single H100. Tests cover happy-path generation (dimensions, determinism,
seed sensitivity, multi-output, CFG) and negative-path validation (invalid
dimensions, empty prompt, bad inference steps).
"""

import numpy as np
import pytest

from tests.helpers.assertions import assert_image_valid
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import DiffusionResponse, OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "black-forest-labs/FLUX.2-dev"

_OMNI_RUNNER_PARAM = (
    MODEL,
    None,
    {
        "enable_cpu_offload": True,
    },
)

pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.diffusion,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]

_HEIGHT = 512
_WIDTH = 512
_NUM_INFERENCE_STEPS = 2


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


def _images_from_response(response: DiffusionResponse) -> list:
    if isinstance(response.images[0], list):
        return [f for fr in response.images for f in fr]
    return list(response.images)


@hardware_test(res={"cuda": "H100"})
def test_flux2_dev_text_to_image(omni_runner_handler: OmniRunnerHandler):
    response = _send_text2img_request(omni_runner_handler, "A cat on a laptop", seed=42)
    images = _images_from_response(response)
    assert len(images) > 0, "No images in response"
    for img in images:
        assert_image_valid(img, width=_WIDTH, height=_HEIGHT)


@hardware_test(res={"cuda": "H100"})
def test_flux2_dev_text_to_image_deterministic(omni_runner_handler: OmniRunnerHandler):
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


@hardware_test(res={"cuda": "H100"})
def test_flux2_dev_text_to_image_different_seeds(omni_runner_handler: OmniRunnerHandler):
    prompt = "A beautiful landscape"

    r1 = _send_text2img_request(omni_runner_handler, prompt, seed=42)
    r2 = _send_text2img_request(omni_runner_handler, prompt, seed=99999)

    images1 = _images_from_response(r1)
    images2 = _images_from_response(r2)

    arr1 = np.array(images1[0])
    arr2 = np.array(images2[0])
    assert arr1.shape == arr2.shape, "Images should have the same shape"
    assert not np.array_equal(arr1, arr2), "Different seeds should produce different outputs"


@hardware_test(res={"cuda": "H100"})
def test_flux2_dev_text_to_image_multi_output(omni_runner_handler: OmniRunnerHandler):
    response = _send_text2img_request(omni_runner_handler, "A red rose in a garden", seed=42, num_outputs_per_prompt=2)
    images = _images_from_response(response)
    assert len(images) == 2, f"Expected 2 images, got {len(images)}"
    for img in images:
        assert_image_valid(img, width=_WIDTH, height=_HEIGHT)
    arr0 = np.array(images[0])
    arr1 = np.array(images[1])
    assert arr0.shape == arr1.shape, "Multi-output images should have the same shape"
    assert not np.array_equal(arr0, arr1), "Multi-output images should be different"


@hardware_test(res={"cuda": "H100"})
def test_flux2_dev_text_to_image_with_cfg(omni_runner_handler: OmniRunnerHandler):
    response = omni_runner_handler.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": "A cinematic mountain landscape at sunrise",
            "sampling_params": OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                seed=42,
                true_cfg_scale=4.0,
            ),
        }
    )
    images = _images_from_response(response)
    assert len(images) > 0, "No images in response"
    for img in images:
        assert_image_valid(img, width=_WIDTH, height=_HEIGHT)


@hardware_test(res={"cuda": "H100"})
@pytest.mark.xfail(reason="Missing input validation: #3702")
def test_flux2_dev_rejects_empty_prompt(omni_runner_handler: OmniRunnerHandler):
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


@hardware_test(res={"cuda": "H100"})
@pytest.mark.xfail(reason="Missing input validation: #3703")
def test_flux2_dev_rejects_zero_inference_steps(omni_runner_handler: OmniRunnerHandler):
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


@hardware_test(res={"cuda": "H100"})
def test_flux2_dev_handles_non_divisible_dimensions(omni_runner_handler: OmniRunnerHandler):
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
