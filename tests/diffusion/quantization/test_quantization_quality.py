# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Quantization quality gate — validates that quantized models stay within
acceptable perceptual distance (LPIPS) of the BF16 baseline.

Developers adding a new quantization method should:
1. Add their method + model to QUALITY_CONFIGS below
2. Set a max_lpips threshold (use 0.15 for image, 0.20 for video as defaults)
3. Run: pytest tests/diffusion/quantization/test_quantization_quality.py -v -m ""
4. Paste the output table into their PR description

The test generates outputs with both BF16 and the quantized method using the
same seed, computes LPIPS, and fails if it exceeds the threshold.

Requirements:
    pip install lpips

Example — run only FP8 tests:
    pytest tests/diffusion/quantization/test_quantization_quality.py -v -m "" -k "fp8"

Example — run a specific model:
    pytest tests/diffusion/quantization/test_quantization_quality.py -v -m "" -k "z_image"
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tests.utils import hardware_marks

# ---------------------------------------------------------------------------
# Configuration — add new quantization methods / models here
# ---------------------------------------------------------------------------


@dataclass
class QualityTestConfig:
    """Defines a single quantization quality test case."""

    id: str  # pytest ID, e.g. "fp8_z_image"
    model: str  # HF model name
    quantization: str  # quantization method, e.g. "fp8"
    task: str  # "t2i" or "t2v"
    prompt: str  # generation prompt
    max_lpips: float  # fail threshold — higher = more lenient
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 20  # keep low for CI speed
    num_frames: int = 5  # only for t2v
    seed: int = 42
    gpu: str = "H100"  # minimum GPU requirement


# Add new quantization methods / models here.
# Developers: copy a config, change quantization + max_lpips, run the test.
QUALITY_CONFIGS = [
    QualityTestConfig(
        id="fp8_z_image",
        model="Tongyi-MAI/Z-Image-Turbo",
        quantization="fp8",
        task="t2i",
        prompt="a cup of coffee on a wooden table, morning light",
        max_lpips=0.10,
        num_inference_steps=20,
    ),
    QualityTestConfig(
        id="fp8_flux",
        model="black-forest-labs/FLUX.1-dev",
        quantization="fp8",
        task="t2i",
        prompt="a cup of coffee on a wooden table, morning light",
        max_lpips=0.20,
        num_inference_steps=10,
    ),
    QualityTestConfig(
        id="fp8_qwen_image",
        model="Qwen/Qwen-Image",
        quantization="fp8",
        task="t2i",
        prompt="a cup of coffee on a wooden table, morning light",
        max_lpips=0.35,
        seed=142,
        num_inference_steps=20,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_image(omni, config: QualityTestConfig):
    """Generate a single image, return (PIL.Image, peak_mem_gib)."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(
        device=current_omni_platform.device_type,
    ).manual_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()

    outputs = omni.generate(
        {"prompt": config.prompt},
        OmniDiffusionSamplingParams(
            height=config.height,
            width=config.width,
            generator=generator,
            num_inference_steps=config.num_inference_steps,
        ),
    )

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    first = outputs[0]
    req_out = first.request_output[0] if hasattr(first, "request_output") else first
    return req_out.images[0], peak_mem


def _generate_video(omni, config: QualityTestConfig):
    """Generate a video, return (np.ndarray [F,H,W,C], peak_mem_gib)."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(
        device=current_omni_platform.device_type,
    ).manual_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()

    outputs = omni.generate(
        {"prompt": config.prompt, "negative_prompt": ""},
        OmniDiffusionSamplingParams(
            height=config.height,
            width=config.width,
            generator=generator,
            num_inference_steps=config.num_inference_steps,
            num_frames=config.num_frames,
        ),
    )

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    first = outputs[0]
    if hasattr(first, "request_output") and isinstance(first.request_output, list):
        inner = first.request_output[0]
        if isinstance(inner, OmniRequestOutput) and hasattr(inner, "images"):
            frames = inner.images[0] if inner.images else None
        else:
            frames = inner
    elif hasattr(first, "images") and first.images:
        frames = first.images
    else:
        raise ValueError("Could not extract video frames from output.")

    if isinstance(frames, torch.Tensor):
        video = frames.detach().cpu()
        if video.dim() == 5:
            video = video[0]
        if video.dim() == 4 and video.shape[0] in (3, 4):
            video = video.permute(1, 2, 3, 0)
        if video.is_floating_point():
            video = video.clamp(-1, 1) * 0.5 + 0.5
        return video.float().numpy(), peak_mem

    return np.asarray(frames), peak_mem


def _compute_lpips(baseline, quantized, task: str) -> float:
    """Compute LPIPS between baseline and quantized outputs."""
    from benchmarks.diffusion.quantization_quality import (
        compute_lpips_images,
        compute_lpips_video,
    )

    if task == "t2i":
        return compute_lpips_images([baseline], [quantized])[0]
    return compute_lpips_video(baseline, quantized)


def _unload(omni):
    del omni
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_marks = hardware_marks(res={"cuda": "H100"})


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "config",
    [pytest.param(c, id=c.id, marks=_marks) for c in QUALITY_CONFIGS],
)
def test_quantization_quality(config: QualityTestConfig):
    """Validate that quantized output stays within LPIPS threshold of BF16."""
    from vllm_omni.entrypoints.omni import Omni

    generate_fn = _generate_video if config.task == "t2v" else _generate_image

    # --- BF16 baseline ---
    omni_bl = Omni(model=config.model)
    baseline_out, bl_mem = generate_fn(omni_bl, config)
    _unload(omni_bl)

    # --- Quantized ---
    omni_qt = Omni(model=config.model, quantization_config=config.quantization)
    quant_out, qt_mem = generate_fn(omni_qt, config)
    _unload(omni_qt)

    # --- LPIPS ---
    lpips_score = _compute_lpips(baseline_out, quant_out, config.task)

    # --- Report ---
    mem_reduction = (bl_mem - qt_mem) / bl_mem * 100 if bl_mem > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"Quantization Quality: {config.id}")
    print(f"{'=' * 60}")
    print(f"  Model:         {config.model}")
    print(f"  Method:        {config.quantization}")
    print(f"  LPIPS:         {lpips_score:.4f}  (threshold: {config.max_lpips})")
    print(f"  BF16 memory:   {bl_mem:.2f} GiB")
    print(f"  Quant memory:  {qt_mem:.2f} GiB  ({mem_reduction:.0f}% reduction)")
    print(f"  Result:        {'PASS' if lpips_score <= config.max_lpips else 'FAIL'}")
    print(f"{'=' * 60}\n")

    assert lpips_score <= config.max_lpips, (
        f"LPIPS {lpips_score:.4f} exceeds threshold {config.max_lpips} for {config.quantization} on {config.model}"
    )
