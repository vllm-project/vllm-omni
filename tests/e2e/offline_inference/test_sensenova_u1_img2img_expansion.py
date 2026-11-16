# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
End-to-end test for SenseNova-U1 img2img generation.

Weekly smoke coverage checks that image editing succeeds with the expected size.
Pixel accuracy runs in a separate process in the weekly H100 job with a pinned
model revision and reviewed references. Run it explicitly:

    pytest -sv tests/e2e/offline_inference/test_sensenova_u1_img2img_expansion.py::test_sensenova_u1_img2img_pixel_accuracy \
        -m "slow and diffusion and cuda and H100 and cards_1" --run-level full_model

Equivalent to running:
    python examples/offline_inference/image_to_image/image_edit.py \
        --model SenseNova/SenseNova-U1-8B-MoT \
        --prompt "Turn this into an oil painting" \
        --image 2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg \
        --resolution 2048 \
        --seed 42 --num-inference-steps 50 \
        --cfg-scale 4.0 \
        --extra-args '{"think": true, "img_cfg_scale": 1.0, "cfg_norm": "none", "timestep_shift": 3.0}'
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
from PIL import Image
from vllm.assets.image import ImageAsset

from tests.e2e.accuracy import sensenova_u1 as accuracy
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Preserve the historical sampling positions; RGB values live with capture provenance.
PIXEL_POSITIONS = [
    (100, 100),
    (400, 50),
    (700, 100),
    (150, 400),
    (512, 336),
    (700, 400),
    (100, 600),
    (400, 600),
    (700, 600),
    (256, 256),
]

DEFAULT_PROMPT = "Turn this into an oil painting"

EXPECTED_OUTPUT_SIZE = (2560, 1664)


def _load_input_image() -> Image.Image:
    """Load the test input image via vllm's ImageAsset."""
    return ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")


def _build_sampling_params() -> OmniDiffusionSamplingParams:
    """Build sampling parameters for SenseNova-U1 img2img generation."""
    return OmniDiffusionSamplingParams(
        height=1669,
        width=2560,
        seed=42,
        num_inference_steps=50,
        extra_args={
            "cfg_scale": 4.0,
            "img_cfg_scale": 1.0,
            "cfg_norm": "none",
            "timestep_shift": 3.0,
            "cfg_interval": (0.0, 1.0),
            "batch_size": 1,
            "think": True,
            "t_eps": 0.02,
        },
    )


def _extract_generated_image(omni_outputs: list) -> Image.Image | None:
    """Extract the generated image from Omni outputs (single-stage DiT)."""
    for req_output in omni_outputs:
        if images := getattr(req_output, "images", None):
            return images[0]
    return None


def _generate_sensenova_u1_img2img(
    omni: Omni,
    input_image: Image.Image,
    prompt: str = DEFAULT_PROMPT,
) -> Image.Image:
    """Generate an edited image using SenseNova-U1 model with img2img pipeline.

    Keep the existing direct Omni path to share exactly the same request between
    smoke and pixel accuracy, including model-specific output dimensions.

    Args:
        omni: The Omni instance to use for generation.
        input_image: The input PIL Image for img2img.
        prompt: The text prompt for image editing.

    Returns:
        The generated PIL Image.

    Raises:
        AssertionError: If no image is generated or size is incorrect.
    """
    sampling_params = _build_sampling_params()

    omni_outputs = list(
        omni.generate(
            prompts={
                "prompt": prompt,
                "multi_modal_data": {"image": input_image},
                "modalities": ["img2img"],
            },
            sampling_params_list=sampling_params,
        )
    )

    generated_image = _extract_generated_image(omni_outputs)
    assert generated_image is not None, "No images generated"
    assert generated_image.size == EXPECTED_OUTPUT_SIZE, f"Expected {EXPECTED_OUTPUT_SIZE}, got {generated_image.size}"

    return generated_image


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
def test_sensenova_u1_img2img():
    """Smoke-test think-mode image editing and output dimensions."""
    input_image = _load_input_image()
    with OmniRunner("SenseNova/SenseNova-U1-8B-MoT") as runner:
        _generate_sensenova_u1_img2img(runner.omni, input_image)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
def test_sensenova_u1_img2img_pixel_accuracy():
    """Compare pinned H100 output with reviewed goldens in the weekly job."""
    input_image = _load_input_image()
    with OmniRunner(accuracy.accuracy_model_path()) as runner:
        generated_image = _generate_sensenova_u1_img2img(runner.omni, input_image)
        record = accuracy.record_accuracy(
            generated_image,
            case="i2i",
            prompt=DEFAULT_PROMPT,
            sampling_params=_build_sampling_params(),
            pixel_positions=PIXEL_POSITIONS,
            input_image=input_image,
        )
        accuracy.assert_accuracy(record)


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "H100"})
def test_sensenova_u1_img2img_cache_dit():
    """Test SenseNova-U1 img2img with Cache-DiT enabled."""
    input_image = _load_input_image()
    with OmniRunner(
        "SenseNova/SenseNova-U1-8B-MoT",
        cache_backend="cache_dit",
    ) as runner:
        _generate_sensenova_u1_img2img(runner.omni, input_image)
