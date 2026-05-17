# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
System test for Expert Parallel (EP).

Tests verify that EP inference produces correct outputs compared to baseline.
"""

import gc
import os
import time
from typing import NamedTuple

import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

# Test configuration
MODELS = ["tencent/HunyuanImage-3.0"]
PROMPT = "A brown and white dog is running on the grass"
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_SEED = 1234
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 0.0
COS_SIM_MEAN_THRESHOLD = 1e-2
COS_SIM_MAX_THRESHOLD = 1e-2
MSE_THRESHOLD = 1e-2
DIFF_MEAN_THRESHOLD = 5e-2
DIFF_MAX_THRESHOLD = 1
TOTAL_CARD_NUM = 8


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


def _diff_metrics(a: Image.Image, b: Image.Image) -> tuple[float, float, float, float, float]:
    """Return (mean_abs_diff, max_abs_diff) over RGB pixels in [0, 1]."""
    ta = torch.from_numpy(np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0)
    tb = torch.from_numpy(np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0)
    assert ta.shape == tb.shape, f"Image shapes differ: {ta.shape} vs {tb.shape}"
    abs_diff = torch.abs(ta - tb)

    vec_a = ta.reshape(-1)
    vec_b = tb.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=0, eps=1e-8)
    mse = torch.nn.functional.mse_loss(ta, tb)
    return abs_diff.mean().item(), abs_diff.max().item(), cos_sim.mean().item(), cos_sim.max().item(), mse.item()


def _run_inference(
    model_name: str,
    tensor_parallel_size: int,
    enable_expert_parallel: bool,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    seed: int = DEFAULT_SEED,
    warmup: bool = True,
) -> InferenceResult:
    """Run inference with specified configuration.

    Args:
        warmup: If True, run one warmup iteration before the timed run.
    """
    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
    )
    try:
        with OmniRunner(model_name, parallel_config=parallel_config) as runner:
            omni = runner.omni
            # Warmup run (not timed)
            if warmup:
                _ = omni.generate(
                    PROMPT,
                    OmniDiffusionSamplingParams(
                        height=height,
                        width=width,
                        num_inference_steps=DEFAULT_STEPS,
                        guidance_scale=guidance_scale,
                        generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
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
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
                    num_outputs_per_prompt=1,
                ),
            )
            elapsed_ms = (time.time() - start) * 1000

            return InferenceResult(
                images=outputs[0].images,
                elapsed_ms=elapsed_ms,
            )
    finally:
        _cleanup_distributed()


# EP configurations: (tensor_parallel_size, enable_ep, height, width, is_perf_test)
EP_TEST_CONFIG = [
    (TOTAL_CARD_NUM, True, DEFAULT_HEIGHT, DEFAULT_WIDTH, True),
]


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": TOTAL_CARD_NUM})
@pytest.mark.parametrize("model_name", MODELS)
def test_ep(model_name):
    device_count = current_omni_platform.get_device_count()
    if device_count < TOTAL_CARD_NUM:
        pytest.skip(f"Requires at least {TOTAL_CARD_NUM} GPUs, got {device_count}")

    baseline_cache: dict[str, InferenceResult] = {}
    results: list[dict] = []
    # baseline_key = PROMPT + "_" + str(DEFAULT_HEIGHT) + "*" + str(DEFAULT_WIDTH) + "_" + str(DEFAULT_SEED) + "_" + str(DEFAULT_STEPS)
    # baseline_cache[baseline_key] = _run_inference(
    #     model_name=model_name,
    #     tensor_parallel_size=TOTAL_CARD_NUM,
    #     enable_expert_parallel=False,
    #     warmup=True,
    # )

    print("\n" + "=" * 90)
    print(f"Sequence Parallel Test - Model: {model_name}")
    print(f"Available GPUs: {device_count}")
    print("=" * 90)

    for tensor_parallel_size, enable_ep, height, width, is_perf_test in EP_TEST_CONFIG:
        cache_key = PROMPT + "_" + str(height) + "*" + str(width) + "_" + str(DEFAULT_SEED) + "_" + str(DEFAULT_STEPS)
        if cache_key not in baseline_cache:
            print(
                f"\n--- Running baseline{{ prompt: {PROMPT}, resolution: {height}x{width}, seed: {DEFAULT_SEED}, steps: {DEFAULT_STEPS} }}---"
            )
            baseline = _run_inference(
                model_name=model_name,
                tensor_parallel_size=TOTAL_CARD_NUM,
                enable_expert_parallel=False,
                warmup=True,
            )
            assert len(baseline.images) == 1
            baseline_cache[cache_key] = baseline
            print(f"[baseline] {height}x{width}: {baseline.elapsed_ms:.0f}ms")
        else:
            baseline = baseline_cache[cache_key]

        print(f"\n--- Running test: {{ enable_ep: {enable_ep} }} ---")
        infer_result = _run_inference(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            enable_expert_parallel=enable_ep,
            warmup=is_perf_test,
        )
        assert len(infer_result.images) == 1
        mean_diff, max_diff, cos_sim_mean, cos_sim_max, mse = _diff_metrics(baseline.images[0], infer_result.images[0])
        print(f"{cos_sim_mean}, {cos_sim_max}, {mse}")

        # Build result entry
        result = {
            "prompt": PROMPT,
            "enable_ep": enable_ep,
            "height": height,
            "width": width,
            "baseline_ms": baseline.elapsed_ms,
            "ep_ms": infer_result.elapsed_ms,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "cos_sim_mean": cos_sim_mean,
            "cos_sim_max": cos_sim_max,
            "mse": mse,
            "is_perf_test": is_perf_test,
        }
        results.append(result)

        # Output based on test type
        if is_perf_test:
            speedup = baseline.elapsed_ms / infer_result.elapsed_ms if infer_result.elapsed_ms > 0 else 0
            result["speedup"] = speedup
            print(
                f"[enable_ep: {enable_ep}] {tensor_parallel_size} GPUs | "
                f"baseline: {baseline.elapsed_ms:.0f}ms, ep: {infer_result.elapsed_ms:.0f}ms, "
                f"speedup: {speedup:.2f}x"
            )
        else:
            print(
                f"[enable_ep: {enable_ep}] {tensor_parallel_size} GPUs | ep: {infer_result.elapsed_ms:.0f}ms (correctness only)"
            )

        print(
            f"[enable_ep: {enable_ep}] diff: [mean={mean_diff:.6e}, max={max_diff:.6e}], cos_sim: [mean={cos_sim_mean:.6e}, max={cos_sim_max:.6e}], mse: {mse:.6e}"
        )
        assert mean_diff <= DIFF_MEAN_THRESHOLD and max_diff <= DIFF_MAX_THRESHOLD, (
            f"[enable_ep: {enable_ep}] output differs from baseline: mean={mean_diff:.6e}, max={max_diff:.6e}"
        )

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Mode':<15} {'GPUs':<6} {'Size':<10} {'Baseline':<12} {'EP':<12} {'Speedup':<10} {'Status'}")
    print("-" * 90)
    for r in results:
        speedup_str = f"{r['speedup']:.2f}x" if r.get("speedup") else "N/A"
        baseline_str = f"{r['baseline_ms']:.0f}ms" if r["is_perf_test"] else "N/A"
        status = "PASS" if r["mean_diff"] <= DIFF_MEAN_THRESHOLD else "FAIL"
        print(
            f"{r['prompt']:<15.10} {r['enable_ep']:<6} {r['height']}x{r['width']:<5} "
            f"{baseline_str:<12} {r['ep_ms']:.0f}ms{'':<7} {speedup_str:<10} {status}"
        )
    print("=" * 90)
