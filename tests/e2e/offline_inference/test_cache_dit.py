# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
System test for cache-dit backend.

This test verifies that cache-dit acceleration works correctly with diffusion models.
It uses minimal settings to keep test time short for CI.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

# Use random weights model for testing
models = ["riverclouds/qwen_image_random"]


@pytest.mark.parametrize("model_name", models)
def test_cache_dit(model_name: str):
    """Test cache-dit backend with diffusion model."""
    # Configure cache-dit with minimal settings for fast testing
    cache_config = {
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 2,  # Minimal warmup for fast test
        "residual_diff_threshold": 0.24,
        "max_continuous_cached_steps": 3,
    }

    m = Omni(
        model=model_name,
        cache_backend="cache_dit",
        cache_config=cache_config,
    )

    # Use minimal settings for fast testing
    height = 256
    width = 256
    num_inference_steps = 4  # Minimal steps for fast test

    outputs = m.generate(
        "a photo of a cat sitting on a laptop keyboard",
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(42),
        num_outputs_per_prompt=1,  # Single output for speed
    )
    # Extract images from request_output[0]['images']
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output[0]
    if not isinstance(req_out, dict) or "images" not in req_out:
        raise ValueError("Invalid request_output structure or missing 'images' key")

    images = req_out["images"]

    # Verify generation succeeded
    assert images is not None
    assert len(images) == 1
    # Check image size
    assert images[0].width == width
    assert images[0].height == height
