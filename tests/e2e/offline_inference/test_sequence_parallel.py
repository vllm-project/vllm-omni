# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
System test for Sequence Parallel (SP) backends: Ulysses and Ring attention.

Tests verify that SP inference produces correct outputs compared to baseline.
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image

from tests.utils import hardware_test
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Test configuration
MODELS = ["riverclouds/qwen_image_random"]
PROMPT = "a photo of a cat sitting on a laptop keyboard"
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_SEED = 42
DEFAULT_STEPS = 4
DIFF_MEAN_THRESHOLD = 2e-2
DIFF_MAX_THRESHOLD = 2e-1


class InferenceResult(NamedTuple):
    """Result of an inference run."""

    images: list[Image.Image]
    elapsed_ms: float


def _cleanup_distributed():
    """Clean up distributed environment and GPU resources."""
    if dist.is_initialized():
        dist.destroy_process_group()

    for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        os.environ.pop(key, None)

    gc.collect()
    if current_omni_platform.is_available():
        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()

    time.sleep(5)


def _diff_metrics(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    """Return (mean_abs_diff, max_abs_diff) over RGB pixels in [0, 1]."""
    ta = torch.from_numpy(np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0)
    tb = torch.from_numpy(np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0)
    assert ta.shape == tb.shape, f"Image shapes differ: {ta.shape} vs {tb.shape}"
    abs_diff = torch.abs(ta - tb)
    return abs_diff.mean().item(), abs_diff.max().item()


def _run_inference(
    model_name: str,
    dtype: torch.dtype,
    attn_backend: str,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    seed: int = DEFAULT_SEED,
    warmup: bool = True,
) -> InferenceResult:
    """Run inference with specified configuration.

    Args:
        warmup: If True, run one warmup iteration before the timed run.
    """
    parallel_config = DiffusionParallelConfig(ulysses_degree=ulysses_degree, ring_degree=ring_degree)
    omni = Omni(
        model=model_name,
        parallel_config=parallel_config,
        dtype=dtype,
        attention_backend=attn_backend,
    )

    try:
        # Warmup run (not timed)
        if warmup:
            _ = omni.generate(
                PROMPT,
                OmniDiffusionSamplingParams(
                    height=height,
                    width=width,
                    num_inference_steps=DEFAULT_STEPS,
                    guidance_scale=0.0,
                    generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed + 1000),
                    num_outputs_per_prompt=1,
                ),
            )

        # Timed run
        start = time.time()
        outputs = omni.generate(
            PROMPT,
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=DEFAULT_STEPS,
                guidance_scale=0.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
                num_outputs_per_prompt=1,
            ),
        )
        elapsed_ms = (time.time() - start) * 1000

        return InferenceResult(
            images=outputs[0].request_output[0].images,
            elapsed_ms=elapsed_ms,
        )
    finally:
        omni.close()
        _cleanup_distributed()


# =============================================================================
# Correctness & Performance Tests
# =============================================================================

# SP configurations: (ulysses_degree, ring_degree, height, width, warmup, is_perf_test)
# - warmup: whether to run warmup for this SP config
# - is_perf_test: whether this is a performance test (show speedup metrics)
SP_CONFIGS = [
    # Ulysses-2 - performance test
    (2, 1, DEFAULT_HEIGHT, DEFAULT_WIDTH, True, True),
    (1, 2, DEFAULT_HEIGHT, DEFAULT_WIDTH, True, True),  # Ring-2 - performance test
    # Hybrid - correctness only
    (2, 2, DEFAULT_HEIGHT, DEFAULT_WIDTH, False, False),
    (4, 1, 272, 272, False, False),  # Ulysses-4 - shape and correctness
]


def _get_sp_mode(ulysses_degree: int, ring_degree: int) -> str:
    """Get SP mode name for logging."""
    if ulysses_degree > 1 and ring_degree == 1:
        return f"ulysses-{ulysses_degree}"
    elif ring_degree > 1 and ulysses_degree == 1:
        return f"ring-{ring_degree}"
    else:
        return f"hybrid-{ulysses_degree}x{ring_degree}"


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
@pytest.mark.parametrize("model_name", MODELS)
def test_sp_correctness(model_name: str):
    """Test that SP inference produces correct outputs and measure performance.

    Runs baseline once per unique (height, width), then tests all SP configs.

    Note: Run with `pytest -v -s` to see detailed output.
    """
    device_count = current_omni_platform.get_device_count()

    # Cache baseline results by (height, width)
    # Key: (height, width), Value: (result, warmup_used)
    baseline_cache: dict[tuple[int, int], InferenceResult] = {}

    # Collect results for summary
    results: list[dict] = []

    print("\n" + "=" * 70)
    print(f"Sequence Parallel Test - Model: {model_name}")
    print(f"Available GPUs: {device_count}")
    print("=" * 70)

    for ulysses_degree, ring_degree, height, width, sp_warmup, is_perf_test in SP_CONFIGS:
        sp_size = ulysses_degree * ring_degree
        sp_mode = _get_sp_mode(ulysses_degree, ring_degree)

        if device_count < sp_size:
            print(f"\n[{sp_mode}] SKIPPED (requires {sp_size} GPUs)")
            continue

        # Determine baseline warmup: only for default size (performance tests)
        cache_key = (height, width)
        baseline_warmup = height == DEFAULT_HEIGHT and width == DEFAULT_WIDTH

        # Get or compute baseline for this (height, width)
        if cache_key not in baseline_cache:
            print(f"\n--- Running baseline {height}x{width} (warmup={baseline_warmup}) ---")
            baseline = _run_inference(
                model_name,
                torch.bfloat16,
                "sdpa",
                height=height,
                width=width,
                warmup=baseline_warmup,
            )
            assert len(baseline.images) == 1
            baseline_cache[cache_key] = baseline
            print(f"[baseline] {height}x{width}: {baseline.elapsed_ms:.0f}ms")
        else:
            baseline = baseline_cache[cache_key]

        # Run SP
        print(f"\n--- Running {sp_mode} (warmup={sp_warmup}) ---")
        sp_result = _run_inference(
            model_name,
            torch.bfloat16,
            "sdpa",
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            height=height,
            width=width,
            warmup=sp_warmup,
        )
        assert len(sp_result.images) == 1

        # Compare outputs (correctness)
        mean_diff, max_diff = _diff_metrics(baseline.images[0], sp_result.images[0])

        # Build result entry
        result = {
            "mode": sp_mode,
            "sp_size": sp_size,
            "height": height,
            "width": width,
            "baseline_ms": baseline.elapsed_ms,
            "sp_ms": sp_result.elapsed_ms,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "is_perf_test": is_perf_test,
        }
        results.append(result)

        # Output based on test type
        if is_perf_test:
            speedup = baseline.elapsed_ms / sp_result.elapsed_ms if sp_result.elapsed_ms > 0 else 0
            result["speedup"] = speedup
            print(
                f"[{sp_mode}] {sp_size} GPUs | "
                f"baseline: {baseline.elapsed_ms:.0f}ms, sp: {sp_result.elapsed_ms:.0f}ms, "
                f"speedup: {speedup:.2f}x"
            )
        else:
            print(f"[{sp_mode}] {sp_size} GPUs | sp: {sp_result.elapsed_ms:.0f}ms (correctness only)")

        print(f"[{sp_mode}] diff: mean={mean_diff:.6e}, max={max_diff:.6e}")

        # Assert correctness
        assert mean_diff <= DIFF_MEAN_THRESHOLD and max_diff <= DIFF_MAX_THRESHOLD, (
            f"[{sp_mode}] SP output differs from baseline: mean={mean_diff:.6e}, max={max_diff:.6e}"
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<15} {'GPUs':<6} {'Size':<10} {'Baseline':<12} {'SP':<12} {'Speedup':<10} {'Status'}")
    print("-" * 70)
    for r in results:
        speedup_str = f"{r['speedup']:.2f}x" if r.get("speedup") else "N/A"
        baseline_str = f"{r['baseline_ms']:.0f}ms" if r["is_perf_test"] else "N/A"
        status = "PASS" if r["mean_diff"] <= DIFF_MEAN_THRESHOLD else "FAIL"
        print(
            f"{r['mode']:<15} {r['sp_size']:<6} {r['height']}x{r['width']:<5} "
            f"{baseline_str:<12} {r['sp_ms']:.0f}ms{'':<7} {speedup_str:<10} {status}"
        )
    print("=" * 70)
