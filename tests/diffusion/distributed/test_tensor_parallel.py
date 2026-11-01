# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""DiT tensor-parallel parity and perf checks for diffusion models.

Compares TP=2 against TP=1 on output correctness, latency, and peak memory.
Uses ``Tongyi-MAI/Z-Image-Turbo`` as the parity model.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.monitor import DeviceMemoryMonitor
from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

PROMPT = "a photo of a cat sitting on a laptop keyboard"
DEFAULT_MODEL = "Tongyi-MAI/Z-Image-Turbo"


def _pil_to_float_rgb_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _diff_metrics(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    ta = _pil_to_float_rgb_tensor(a)
    tb = _pil_to_float_rgb_tensor(b)
    assert ta.shape == tb.shape, f"Image shapes differ: {ta.shape} vs {tb.shape}"
    abs_diff = torch.abs(ta - tb)
    p99_abs_diff = torch.quantile(abs_diff.flatten(), 0.99).item()
    return abs_diff.mean().item(), p99_abs_diff


def _extract_single_image(outputs) -> Image.Image:
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if not isinstance(first_output, OmniRequestOutput) or not first_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")

    images = req_out.images
    if images is None or len(images) != 1:
        raise ValueError(f"Expected 1 image, got {0 if images is None else len(images)}")
    return images[0]


def _run_generate(
    *,
    tp_size: int,
    height: int,
    width: int,
    num_inference_steps: int,
    seed: int,
    enforce_eager: bool,
    num_requests: int = 4,
) -> tuple[Image.Image, float, float]:
    if num_requests < 2:
        raise ValueError("num_requests must be >= 2 (1 warmup + >=1 timed)")

    gc.collect()
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    with current_omni_platform.device(device_index):
        free_bytes, total_bytes = current_omni_platform.mem_get_info()
    initial_used_mb = (total_bytes - free_bytes) / (1024**2)

    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()
    try:
        with OmniRunner(
            DEFAULT_MODEL,
            parallel_config=DiffusionParallelConfig(tensor_parallel_size=tp_size),
            enforce_eager=enforce_eager,
        ) as runner:
            gen = runner.omni.generate(
                [PROMPT] * num_requests,
                OmniDiffusionSamplingParams(
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=0.0,
                    seed=seed,
                    num_outputs_per_prompt=1,
                ),
                py_generator=True,
            )

            warmup_output = next(gen)

            t_prev = time.perf_counter()
            per_request_times_s: list[float] = []
            last_output = warmup_output
            for _ in range(num_requests - 1):
                last_output = next(gen)
                t_now = time.perf_counter()
                per_request_times_s.append(t_now - t_prev)
                t_prev = t_now

            for _ in gen:
                pass

            median_time_s = float(np.median(per_request_times_s))
            # DeviceMemoryMonitor samples absolute device usage. Normalize by
            # this run's starting usage so compiler and allocator caches retained
            # by the preceding TP=1 run are not charged to the TP=2 run.
            peak_memory_mb = max(0.0, monitor.peak_used_mb - initial_used_mb)
            return _extract_single_image([last_output]), median_time_s, peak_memory_mb
    finally:
        monitor.stop()


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
def test_tensor_parallel_tp2(tmp_path: Path):
    if current_omni_platform.is_npu():
        pytest.skip("Tensor parallel parity test is only supported on CUDA and ROCm for now.")
    if not current_omni_platform.is_available() or current_omni_platform.device_count() < 2:
        pytest.skip("Tensor parallel TP=2 requires >= 2 devices.")

    model_name = DEFAULT_MODEL
    enforce_eager = False
    height = 512
    width = 512
    num_inference_steps = 2
    seed = 42

    tp1_img, tp1_time_s, tp1_peak_mem = _run_generate(
        tp_size=1,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=seed,
        enforce_eager=enforce_eager,
    )
    tp2_img, tp2_time_s, tp2_peak_mem = _run_generate(
        tp_size=2,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=seed,
        enforce_eager=enforce_eager,
    )

    tp1_path = tmp_path / "tp1.png"
    tp2_path = tmp_path / "tp2.png"
    tp1_img.save(tp1_path)
    tp2_img.save(tp2_path)

    assert tp1_img.width == width and tp1_img.height == height
    assert tp2_img.width == width and tp2_img.height == height

    mean_abs_diff, p99_abs_diff = _diff_metrics(tp1_img, tp2_img)
    mean_threshold = 3e-2
    p99_threshold = 2.5e-1
    print(
        f"{model_name} TP image diff (TP=1 vs TP=2): "
        f"mean_abs_diff={mean_abs_diff:.6e}, p99_abs_diff={p99_abs_diff:.6e}; "
        f"thresholds: mean<={mean_threshold:.6e}, p99<={p99_threshold:.6e}; "
        f"tp1_img={tp1_path}, tp2_img={tp2_path}"
    )
    assert mean_abs_diff <= mean_threshold and p99_abs_diff <= p99_threshold, (
        f"Image diff exceeded threshold: mean_abs_diff={mean_abs_diff:.6e}, p99_abs_diff={p99_abs_diff:.6e} "
        f"(thresholds: mean<={mean_threshold:.6e}, p99<={p99_threshold:.6e})"
    )

    print(f"TP perf (lower is better): tp1_time_s={tp1_time_s:.6f}, tp2_time_s={tp2_time_s:.6f}")
    if not current_omni_platform.is_rocm():
        assert tp2_time_s < tp1_time_s, f"Expected TP=2 to be faster than TP=1 (tp1={tp1_time_s}, tp2={tp2_time_s})"

    print(f"TP peak memory (MB): tp1_peak_mem={tp1_peak_mem:.2f}, tp2_peak_mem={tp2_peak_mem:.2f}")
    assert tp2_peak_mem < tp1_peak_mem, (
        f"Expected TP=2 to use less peak memory than TP=1 (tp1={tp1_peak_mem}, tp2={tp2_peak_mem})"
    )
