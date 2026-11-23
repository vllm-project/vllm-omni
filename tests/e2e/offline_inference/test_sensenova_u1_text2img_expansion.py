# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
End-to-end test for SenseNova-U1 text2img generation.

Weekly smoke coverage checks that generation succeeds with the expected size.
Pixel accuracy runs in a separate process in the weekly H100 job with a pinned
model revision and reviewed references. Run it explicitly:

    pytest -sv tests/e2e/offline_inference/test_sensenova_u1_text2img_expansion.py::test_sensenova_u1_text2img_pixel_accuracy \
        -m "slow and diffusion and cuda and H100 and cards_1" --run-level full_model

Equivalent to running:
    python SenseNova-U1/examples/t2i/inference.py \
        --model_path SenseNova/SenseNova-U1-8B-MoT \
        --prompt "Close portrait of an elderly woman ..." \
        --width 1536 --height 2720 \
        --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 \
        --num_steps 50 --seed 42 --think
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
from PIL import Image

from tests.e2e.accuracy import sensenova_u1 as accuracy
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Preserve the historical sampling positions; RGB values live with capture provenance.
PIXEL_POSITIONS = [
    (100, 100),
    (768, 200),
    (400, 600),
    (1200, 2000),
    (750, 500),
    (300, 1360),
    (1000, 1800),
    (500, 2400),
    (768, 1360),
    (200, 900),
]

DEFAULT_PROMPT = (
    "Close portrait of an elderly woman by a farmhouse window, textured skin, "
    "gentle smile, warm natural light, emotional documentary look. The portrait "
    "should feel polished and natural, with sharp eyes, realistic skin texture, "
    "accurate facial anatomy, and premium lighting that keeps the face as the "
    "main focus."
)

EXPECTED_OUTPUT_SIZE = (1536, 2720)


def _build_sampling_params() -> OmniDiffusionSamplingParams:
    """Build sampling parameters for SenseNova-U1 text2img generation."""
    return OmniDiffusionSamplingParams(
        height=EXPECTED_OUTPUT_SIZE[1],
        width=EXPECTED_OUTPUT_SIZE[0],
        seed=42,
        num_inference_steps=50,
        extra_args={
            "cfg_scale": 4.0,
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


def _generate_sensenova_u1_image(
    omni: Omni,
    prompt: str = DEFAULT_PROMPT,
) -> Image.Image:
    """Generate an image using SenseNova-U1 model with configured parameters.

    Keep the existing direct Omni path to share exactly the same request between
    smoke and pixel accuracy, including model-specific output dimensions.

    Args:
        omni: The Omni instance to use for generation.
        prompt: The text prompt for image generation.

    Returns:
        The generated PIL Image.

    Raises:
        AssertionError: If no image is generated or size is incorrect.
    """
    sampling_params = _build_sampling_params()

    omni_outputs = list(
        omni.generate(
            prompts={"prompt": prompt, "modalities": ["image"]},
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
def test_sensenova_u1_text2img():
    """Smoke-test think-mode text2img generation and output dimensions."""
    with OmniRunner("SenseNova/SenseNova-U1-8B-MoT") as runner:
        _generate_sensenova_u1_image(runner.omni)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
def test_sensenova_u1_text2img_pixel_accuracy():
    """Compare pinned H100 output with reviewed goldens in the weekly job."""
    with OmniRunner(accuracy.accuracy_model_path()) as runner:
        generated_image = _generate_sensenova_u1_image(runner.omni)
        record = accuracy.record_accuracy(
            generated_image,
            case="t2i",
            prompt=DEFAULT_PROMPT,
            sampling_params=_build_sampling_params(),
            pixel_positions=PIXEL_POSITIONS,
        )
        accuracy.assert_accuracy(record)


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.cache
@hardware_test(res={"cuda": "H100"})
def test_sensenova_u1_text2img_cache_dit():
    """Test SenseNova-U1 text2img with Cache-DiT enabled."""
    with OmniRunner(
        "SenseNova/SenseNova-U1-8B-MoT",
        cache_backend="cache_dit",
    ) as runner:
        _generate_sensenova_u1_image(runner.omni)
