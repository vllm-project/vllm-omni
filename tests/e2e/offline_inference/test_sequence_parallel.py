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
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image

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

PROMPT = "a photo of a cat sitting on a laptop keyboard"


def _pil_to_float_rgb_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to float32 RGB tensor in [0, 1] with shape [H, W, 3]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _diff_metrics(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    """Return (mean_abs_diff, max_abs_diff) over RGB pixels in [0, 1]."""
    ta = _pil_to_float_rgb_tensor(a)
    tb = _pil_to_float_rgb_tensor(b)
    assert ta.shape == tb.shape, f"Image shapes differ: {ta.shape} vs {tb.shape}"
    abs_diff = torch.abs(ta - tb)
    return abs_diff.mean().item(), abs_diff.max().item()


def _get_images(output):
    """Extract images from output, handling both dict and SimpleNamespace types.

    The output structure varies depending on serialization path:
    - Direct memory: SimpleNamespace with .images attribute
    - SHM serialization: dict with "images" key (dataclass converted via asdict)
    - Wrapped output: SimpleNamespace(output=...) which needs unwrapping
    """
    # Check if output has direct images attribute (diffusion mode)
    if hasattr(output, "images") and output.images:
        return output.images

    # Check request_output for pipeline mode
    if output.request_output is None:
        return None

    if isinstance(output.request_output, list) and len(output.request_output) == 0:
        return None

    item = output.request_output[0]

    # Handle wrapped SimpleNamespace (e.g. from omni_stage.py)
    # Some items are wrapped as SimpleNamespace(request_id=..., output=...)
    while hasattr(item, "output") and not hasattr(item, "images"):
        item = item.output
        if item is None:
            return None

    # Handle both dict (from SHM serialization) and object (direct) types
    if isinstance(item, dict):
        return item.get("images")
    return getattr(item, "images", None)


@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("ulysses_degree", [1, 2])
@pytest.mark.parametrize("ring_degree", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])  # Only test bfloat16 to reduce CI time
@pytest.mark.parametrize("attn_backend", ["sdpa"])
def test_sequence_parallel(
    model_name: str,
    ulysses_degree: int,
    ring_degree: int,
    dtype: torch.dtype,
    attn_backend: str,
):
    """Compare baseline (ulysses_degree=1) vs SP (ulysses_degree>1) outputs."""
    if ulysses_degree <= 1 and ring_degree <= 1:
        pytest.skip(
            "This test compares ulysses_degree * ring_degree = 1 vs ulysses_degree * ring_degree > 1; provide ulysses_degree or ring_degree>1."
        )

    # Skip if not enough GPUs available for SP run
    if device_count() < ulysses_degree * ring_degree:
        pytest.skip(f"Test requires {ulysses_degree * ring_degree} GPUs but only {device_count()} available")

    # Use minimal settings for fast testing
    height = 256
    width = 256
    num_inference_steps = 4  # Minimal steps for fast test
    seed = 42

    # Step 1: Baseline (no Ulysses sequence parallel)
    baseline_parallel_config = DiffusionParallelConfig(ulysses_degree=1, ring_degree=1)
    baseline = Omni(
        model=model_name,
        parallel_config=baseline_parallel_config,
        dtype=dtype,
        attention_backend=attn_backend,
    )
    try:
        outputs = baseline.generate(
            PROMPT,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            generator=torch.Generator(get_device_name()).manual_seed(seed),
            num_outputs_per_prompt=1,
        )
        baseline_images = list(outputs)[0].request_output[0].images
    finally:
        baseline.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
            os.environ.pop(key, None)
        time.sleep(2)  # Wait for resources to release

    assert baseline_images is not None
    assert len(baseline_images) == 1
    assert baseline_images[0].width == width
    assert baseline_images[0].height == height

    # Step 2: SP (Ulysses-SP + Ring-SP)
    sp_parallel_config = DiffusionParallelConfig(ulysses_degree=ulysses_degree, ring_degree=ring_degree)
    sp = Omni(
        model=model_name,
        parallel_config=sp_parallel_config,
        dtype=dtype,
        attention_backend=attn_backend,
    )
    try:
        outputs = sp.generate(
            PROMPT,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            generator=torch.Generator(get_device_name()).manual_seed(seed),
            num_outputs_per_prompt=1,
        )
        sp_images = list(outputs)[0].request_output[0].images
    finally:
        sp.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
            os.environ.pop(key, None)
        time.sleep(2)

    assert sp_images is not None
    assert len(sp_images) == 1
    assert sp_images[0].width == width
    assert sp_images[0].height == height

    # Step 3: Compare outputs
    mean_abs_diff, max_abs_diff = _diff_metrics(baseline_images[0], sp_images[0])

    # FP16/BF16 may differ slightly due to different computation order under parallelism.
    if dtype in (torch.float16, torch.bfloat16):
        mean_threshold = 2e-2
        max_threshold = 2e-1
    else:
        mean_threshold = 1e-2
        max_threshold = 1e-1

    print(
        "Image diff stats (baseline ulysses_degree*ring_degree=1 vs SP): "
        f"mean_abs_diff={mean_abs_diff:.6e}, max_abs_diff={max_abs_diff:.6e}; "
        f"thresholds: mean<={mean_threshold:.6e}, max<={max_threshold:.6e}; "
        f"ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, dtype={dtype}"
    )

    assert mean_abs_diff <= mean_threshold and max_abs_diff <= max_threshold, (
        f"Image diff exceeded threshold: mean_abs_diff={mean_abs_diff:.6e}, max_abs_diff={max_abs_diff:.6e} "
        f"(thresholds: mean<={mean_threshold:.6e}, max<={max_threshold:.6e}); "
        f"ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, dtype={dtype}"
    )
