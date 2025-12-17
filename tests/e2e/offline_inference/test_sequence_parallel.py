# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
System test for Ulysses sequence parallel backend.

This test verifies that Ulysses-SP (DeepSpeed Ulysses Sequence Parallel) works
correctly with diffusion models. It uses minimal settings to keep test time
short for CI.
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
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.distributed.parallel_state import device_count
from vllm_omni.diffusion.envs import get_device_name

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

# Use random weights model for testing
models = ["riverclouds/qwen_image_random"]


@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("ulysses_degree", [1, 2])
@pytest.mark.parametrize("ring_degree", [1])
def test_sequence_parallel(model_name: str, ulysses_degree: int, ring_degree: int):
    """Test SP (Ulysses-SP + Ring-SP) backend with diffusion model."""
    # Skip if not enough GPUs available
    if device_count() < ulysses_degree:
        pytest.skip(f"Test requires {ulysses_degree} GPUs but only {device_count()} available")

    # Configure sequence parallel with DiffusionParallelConfig
    parallel_config = DiffusionParallelConfig(ulysses_degree=ulysses_degree, ring_degree=ring_degree)

    m = Omni(
        model=model_name,
        parallel_config=parallel_config,
    )

    # Use minimal settings for fast testing
    height = 256
    width = 256
    num_inference_steps = 4  # Minimal steps for fast test

    images = m.generate(
        "a photo of a cat sitting on a laptop keyboard",
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
        generator=torch.Generator(get_device_name()).manual_seed(42),
        num_outputs_per_prompt=1,  # Single output for speed
    )

    # Verify generation succeeded
    assert images is not None
    assert len(images) == 1
    # Check image size
    assert images[0].width == width
    assert images[0].height == height
