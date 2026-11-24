# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
End-to-end test for Flux2 Klein inpainting.

Inpainting uses ``offline_client.send_diffusion_request`` with
``multi_modal_data`` containing ``image`` and ``mask_image``; see
:meth:`OfflineOmniClient.send_diffusion_request` in ``tests.helpers.runtime``.
"""

import numpy as np
import pytest
from PIL import Image, ImageChops, ImageDraw

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import DiffusionResponse, OfflineOmniClient
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

# NOTE: Most of the flux2klein tests have migrated to the common diffusion tests.
# This model will also use the tiny model builder when run at the core model level.
MODEL = "black-forest-labs/FLUX.2-klein-4B"

_OMNI_RUNNER_PARAM = (MODEL, None)

pytestmark = [pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)]

_HEIGHT = 512
_WIDTH = 512
_NUM_INFERENCE_STEPS = 4
_ROCM_REPEAT_MAX_MEAN_ABSOLUTE_ERROR = 0.015


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
    images = response.images
    assert images is not None
    if images and isinstance(images[0], list):
        return [frame for frames in images for frame in frames]
    return list(images)


def _image_difference_bbox(first: Image.Image, second: Image.Image) -> tuple[int, int, int, int] | None:
    """Compare generated images without materializing every pixel as Python objects."""
    assert first.mode == second.mode, f"Image modes differ: {first.mode} != {second.mode}"
    assert first.size == second.size, f"Image sizes differ: {first.size} != {second.size}"
    return ImageChops.difference(first, second).getbbox(alpha_only=False)


def _image_difference_metrics(first: Image.Image, second: Image.Image) -> tuple[float, float, float]:
    """Return normalized MAE, maximum channel delta, and changed-channel ratio."""
    assert first.mode == second.mode, f"Image modes differ: {first.mode} != {second.mode}"
    assert first.size == second.size, f"Image sizes differ: {first.size} != {second.size}"
    deviation = np.abs(np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32))
    return (
        float(deviation.mean() / 255.0),
        float(deviation.max()),
        float((deviation > 0).mean()),
    )


def _send_inpaint_with_seed(
    offline_client: OfflineOmniClient, prompt: str, input_image, mask_image, seed: int
) -> DiffusionResponse:
    """Exercise the serializable request seed contract across the worker boundary."""
    return offline_client.send_diffusion_request(
        {
            "model": MODEL,
            "prompt": prompt,
            "multi_modal_data": {"image": input_image, "mask_image": mask_image},
            "sampling_params": OmniDiffusionSamplingParams(
                height=_HEIGHT,
                width=_WIDTH,
                num_inference_steps=_NUM_INFERENCE_STEPS,
                guidance_scale=0.0,
                seed=seed,
                num_outputs_per_prompt=1,
            ),
        }
    )


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4", "rocm": "MI300"}, num_cards=1)
def test_flux2_klein_inpaint_deterministic(offline_client: OfflineOmniClient):
    input_image, mask_image = _create_test_inputs()
    seed = 12345

    r1 = _send_inpaint_with_seed(offline_client, "A red flower in a field", input_image, mask_image, seed)
    r2 = _send_inpaint_with_seed(offline_client, "A red flower in a field", input_image, mask_image, seed)

    images1 = _images_from_response(r1)
    images2 = _images_from_response(r2)

    if current_omni_platform.is_rocm():
        mean_absolute_error, max_difference, differing_ratio = _image_difference_metrics(images1[0], images2[0])
        print(
            f"same-seed repeat: MAE={mean_absolute_error:.6f} "
            f"maxdiff={max_difference:.0f} differing={differing_ratio:.4f}"
        )
        assert mean_absolute_error < _ROCM_REPEAT_MAX_MEAN_ABSOLUTE_ERROR, (
            "Same input with the same seed diverged materially on ROCm: "
            f"MAE={mean_absolute_error:.6f}, maxdiff={max_difference:.0f}, differing={differing_ratio:.4f}"
        )
    else:
        assert _image_difference_bbox(images1[0], images2[0]) is None, (
            "Same input with the same seed should produce identical output. "
            "This is critical for offline/online consistency."
        )


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4", "rocm": "MI300"}, num_cards=1)
def test_flux2_klein_inpaint_different_seeds_different_output(offline_client: OfflineOmniClient):
    input_image, mask_image = _create_test_inputs()

    r1 = _send_inpaint_with_seed(offline_client, "A beautiful landscape", input_image, mask_image, 42)
    r2 = _send_inpaint_with_seed(offline_client, "A beautiful landscape", input_image, mask_image, 99999)

    images1 = _images_from_response(r1)
    images2 = _images_from_response(r2)

    assert _image_difference_bbox(images1[0], images2[0]) is not None, (
        "Different seeds should produce different outputs"
    )
