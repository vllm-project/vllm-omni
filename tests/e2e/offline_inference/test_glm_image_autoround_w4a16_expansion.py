# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for GLM-Image AutoRound W4A16 quantized inference.

These tests cover text-to-image and image-to-image generation with
the W4A16 quantized GLM-Image model.

Requirements:
  - 2 CUDA GPUs (H100 or equivalent)
  - The quantized model checkpoint (Intel/GLM-Image-int4-AutoRound)
"""

import gc
import math
import os

import numpy as np
import pytest
from PIL import Image
from vllm import SamplingParams

from tests.helpers.env import DeviceMemoryMonitor
from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniRunnerHandler
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

QUANTIZED_MODEL = os.environ.get("GLM_IMAGE_AUTOROUND_MODEL", "Intel/GLM-Image-int4-AutoRound")

# Small resolution to keep GPU memory & time manageable
HEIGHT = 256
WIDTH = 256
NUM_STEPS = 2  # minimal for smoke-test

# GLM-Image AR generation config (from generation_config.json)
GLM_IMAGE_EOS_TOKEN_ID = 16385
GLM_IMAGE_VISION_VOCAB_SIZE = 16512

_CI_DEPLOY = get_deploy_config_path("glm_image.yaml")


def _get_stage_config():
    """Build a CI-friendly stage config with eager mode for testing."""
    return modify_stage_config(
        _CI_DEPLOY,
        updates={
            "stages": {
                0: {"enforce_eager": True},
                1: {"enforce_eager": True},
            },
        },
    )


stage_config = _get_stage_config()

# (model, stage_config_path) for ``omni_runner`` indirect parametrize
_OMNI_RUNNER_PARAM = (QUANTIZED_MODEL, stage_config)


def compute_max_tokens(height: int, width: int, factor: int = 32) -> int:
    """Compute max_new_tokens for GLM-Image AR text-to-image generation."""
    token_h = height // factor
    token_w = width // factor
    large_tokens = token_h * token_w

    ratio = token_h / token_w if token_w > 0 else 1.0
    small_token_h = max(1, int(math.sqrt(ratio) * (factor // 2)))
    small_token_w = max(1, int(math.sqrt(1 / ratio) * (factor // 2)))
    small_tokens = small_token_h * small_token_w

    return small_tokens + large_tokens + 1


def _ar_sampling_params(max_tokens: int, height: int, width: int, seed: int = 42) -> SamplingParams:
    """Build AR stage SamplingParams for GLM-Image."""
    return SamplingParams(
        temperature=0.9,
        top_p=0.75,
        top_k=GLM_IMAGE_VISION_VOCAB_SIZE,
        max_tokens=max_tokens,
        stop_token_ids=[GLM_IMAGE_EOS_TOKEN_ID],
        seed=seed,
        detokenize=False,
        extra_args={
            "target_h": height,
            "target_w": width,
        },
    )


def _diffusion_sampling_params(
    height: int = HEIGHT,
    width: int = WIDTH,
    num_steps: int = NUM_STEPS,
    seed: int = 42,
) -> OmniDiffusionSamplingParams:
    """Build Diffusion stage OmniDiffusionSamplingParams."""
    return OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=0.0,
        seed=seed,
    )


pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
]


# ------------------------------------------------------------------
# Test: text-to-image generation produces a valid image (quantized)
# ------------------------------------------------------------------


@pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_glm_image_autoround_w4a16_generates_image(omni_runner_handler: OmniRunnerHandler):
    """Load the W4A16 quantized GLM-Image model and verify it produces a valid image."""
    gc.collect()
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    current_omni_platform.reset_peak_memory_stats()
    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()

    prompt_dict = {
        "prompt": "A photo of a cat sitting on a laptop keyboard",
        "height": HEIGHT,
        "width": WIDTH,
        "mm_processor_kwargs": {
            "target_h": HEIGHT,
            "target_w": WIDTH,
        },
    }
    ar_params = _ar_sampling_params(
        max_tokens=compute_max_tokens(HEIGHT, WIDTH),
        height=HEIGHT,
        width=WIDTH,
        seed=42,
    )
    diffusion_params = _diffusion_sampling_params(
        height=HEIGHT,
        width=WIDTH,
        num_steps=NUM_STEPS,
        seed=42,
    )

    outputs = omni_runner_handler.runner.generate(
        [prompt_dict],
        [ar_params, diffusion_params],
    )

    monitor.stop()

    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput) and hasattr(req_out, "images")
    images = req_out.images

    assert len(images) >= 1, "Expected at least one generated image"
    img = images[0]
    assert isinstance(img, Image.Image)
    assert img.width == WIDTH, f"Expected width {WIDTH}, got {img.width}"
    assert img.height == HEIGHT, f"Expected height {HEIGHT}, got {img.height}"

    # Sanity: image should not be blank (all-zero)
    arr = np.array(img)
    assert arr.std() > 1.0, "Generated image appears blank (std ≈ 0)"

    gc.collect()
    current_omni_platform.empty_cache()


# ------------------------------------------------------------------
# Test: image-to-image generation (quantized)
# ------------------------------------------------------------------


@pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_glm_image_autoround_w4a16_image_to_image(omni_runner_handler: OmniRunnerHandler):
    """Load the W4A16 quantized GLM-Image and verify image-to-image generation works."""
    ref_image_arr = generate_synthetic_image(WIDTH, HEIGHT)["np_array"]

    gc.collect()
    current_omni_platform.empty_cache()
    current_omni_platform.reset_peak_memory_stats()

    prompt_dict = {
        "prompt": "Make it look like winter",
        "multi_modal_data": {"image": ref_image_arr},
        "height": HEIGHT,
        "width": WIDTH,
        "mm_processor_kwargs": {
            "target_h": HEIGHT,
            "target_w": WIDTH,
        },
    }
    ar_params = _ar_sampling_params(
        max_tokens=compute_max_tokens(HEIGHT, WIDTH),
        height=HEIGHT,
        width=WIDTH,
        seed=42,
    )
    diffusion_params = _diffusion_sampling_params(
        height=HEIGHT,
        width=WIDTH,
        num_steps=NUM_STEPS,
        seed=42,
    )

    outputs = omni_runner_handler.runner.generate(
        [prompt_dict],
        [ar_params, diffusion_params],
    )

    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput) and hasattr(req_out, "images")
    images = req_out.images

    assert len(images) >= 1, "Expected at least one generated image"
    img = images[0]
    assert isinstance(img, Image.Image)
    assert img.width == WIDTH
    assert img.height == HEIGHT

    gc.collect()
    current_omni_platform.empty_cache()
