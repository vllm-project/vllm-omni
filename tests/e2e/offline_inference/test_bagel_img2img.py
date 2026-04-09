# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end test for Bagel img2img generation.

This test validates that the Bagel model generates images from an input image
and text prompt that match expected reference pixel values within a ±10 tolerance.

Equivalent to running:
    python3 examples/offline_inference/bagel/end2end.py \
        --prompts "Change the grass color to red" \
        --modality img2img --step 15 \
        --image-path 2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg
"""

import socket
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from vllm.assets.image import ImageAsset

from tests.conftest import modify_stage_config
from tests.utils import hardware_test
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.platforms import current_omni_platform

# Reference pixel data extracted from the known-good output image
# Generated with seed=52, num_inference_steps=15,
# prompt='Change the grass color to red',
# input image: 2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg
REFERENCE_PIXELS = [
    {"position": (100, 100), "rgb": (157, 172, 217)},
    {"position": (400, 50), "rgb": (105, 144, 218)},
    {"position": (700, 100), "rgb": (118, 159, 233)},
    {"position": (150, 400), "rgb": (195, 34, 60)},
    {"position": (512, 336), "rgb": (222, 214, 193)},
    {"position": (700, 400), "rgb": (197, 15, 43)},
    {"position": (100, 600), "rgb": (105, 13, 18)},
    {"position": (400, 600), "rgb": (169, 33, 44)},
    {"position": (700, 600), "rgb": (101, 86, 93)},
    {"position": (256, 256), "rgb": (181, 202, 222)},
]

if current_omni_platform.is_rocm():
    REFERENCE_PIXELS = [
        {"position": (100, 100), "rgb": (156, 172, 215)},
        {"position": (400, 50), "rgb": (106, 144, 216)},
        {"position": (700, 100), "rgb": (118, 158, 231)},
        {"position": (150, 400), "rgb": (183, 23, 48)},
        {"position": (512, 336), "rgb": (218, 215, 191)},
        {"position": (700, 400), "rgb": (194, 14, 42)},
        {"position": (100, 600), "rgb": (105, 10, 16)},
        {"position": (400, 600), "rgb": (167, 33, 46)},
        {"position": (700, 600), "rgb": (102, 86, 92)},
        {"position": (256, 256), "rgb": (181, 201, 220)},
    ]

PIXEL_TOLERANCE = 10

DEFAULT_PROMPT = "<|fim_middle|><|im_start|>Change the grass color to red<|im_end|>"

EXPECTED_OUTPUT_SIZE = (1024, 672)


def _load_input_image() -> Image.Image:
    """Load the test input image via vllm's ImageAsset."""
    return ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")


def _find_free_port() -> int:
    """Find and return a free ephemeral port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def _configure_sampling_params(omni: Omni, num_inference_steps: int = 15) -> list:
    """Configure sampling parameters for Bagel img2img generation.

    Args:
        omni: The Omni instance to get default params from.
        num_inference_steps: Number of inference steps for the diffusion stage.

    Returns:
        Configured sampling params list.
    """
    params_list = omni.default_sampling_params_list
    if len(params_list) > 1:
        params_list[1].num_inference_steps = num_inference_steps  # type: ignore
        params_list[1].extra_args = {  # type: ignore
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.5,
        }
    return params_list


def _extract_generated_image(omni_outputs: list) -> Image.Image | None:
    """Extract the generated image from Omni outputs.

    Args:
        omni_outputs: List of outputs from omni.generate().

    Returns:
        The first generated PIL Image, or None if no image found.
    """
    for req_output in omni_outputs:
        if images := getattr(req_output, "images", None):
            return images[0]
        if hasattr(req_output, "request_output") and req_output.request_output:
            stage_out = req_output.request_output
            if hasattr(stage_out, "images") and stage_out.images:
                return stage_out.images[0]
    return None


def _validate_pixels(
    image: Image.Image,
    reference_pixels: list[dict[str, Any]] = REFERENCE_PIXELS,
    tolerance: int = PIXEL_TOLERANCE,
) -> None:
    """Validate that image pixels match expected reference values.

    Args:
        image: The PIL Image to validate.
        reference_pixels: List of dicts with 'position' (x, y) and 'rgb' (R, G, B).
        tolerance: Maximum allowed difference per color channel.

    Raises:
        AssertionError: If any pixel differs beyond tolerance.
    """
    for ref in reference_pixels:
        x, y = ref["position"]
        expected = ref["rgb"]
        actual = image.getpixel((x, y))[:3]
        assert all(abs(a - e) <= tolerance for a, e in zip(actual, expected)), (
            f"Pixel mismatch at ({x}, {y}): expected {expected}, got {actual}"
        )


def _generate_bagel_img2img(
    omni: Omni,
    input_image: Image.Image,
    prompt: str = DEFAULT_PROMPT,
) -> Image.Image:
    """Generate an image using Bagel model with img2img pipeline.

    Args:
        omni: The Omni instance to use for generation.
        input_image: The input PIL Image for img2img.
        prompt: The text prompt for image editing.

    Returns:
        The generated PIL Image.

    Raises:
        AssertionError: If no image is generated or size is incorrect.
    """
    params_list = _configure_sampling_params(omni)

    omni_outputs = list(
        omni.generate(
            prompts=[
                {
                    "prompt": prompt,
                    "multi_modal_data": {"img2img": input_image},
                    "modalities": ["img2img"],
                }
            ],
            sampling_params_list=params_list,
        )
    )

    generated_image = _extract_generated_image(omni_outputs)
    assert generated_image is not None, "No images generated"
    assert generated_image.size == EXPECTED_OUTPUT_SIZE, f"Expected {EXPECTED_OUTPUT_SIZE}, got {generated_image.size}"

    return generated_image


def _resolve_stage_config(config_path: str, run_level: str) -> str:
    """Resolve stage config based on run level.

    For advanced_model (real weights), strip load_format: dummy so the model
    falls back to loading real weights from HuggingFace.
    """
    if run_level == "advanced_model":
        return modify_stage_config(
            config_path,
            deletes={
                "stage_args": {
                    0: ["engine_args.load_format"],
                    1: ["engine_args.load_format"],
                }
            },
        )
    return config_path


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_bagel_img2img_shared_memory_connector(run_level):
    """Test Bagel img2img with shared memory connector."""
    input_image = _load_input_image()
    config_path = str(Path(__file__).parent / "stage_configs" / "bagel_sharedmemory_ci.yaml")
    config_path = _resolve_stage_config(config_path, run_level)
    omni = Omni(model="ByteDance-Seed/BAGEL-7B-MoT", stage_configs_path=config_path, stage_init_timeout=300)

    try:
        generated_image = _generate_bagel_img2img(omni, input_image)
        if run_level == "advanced_model":
            _validate_pixels(generated_image)
    finally:
        omni.close()
