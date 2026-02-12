# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Z-Image end-to-end tests for diffusion parallelism.

This file currently covers:
- DiT tensor parallelism (TP=2) vs TP=1.
- VAE patch parallelism (vae_patch_parallel_size=2) vs baseline on TP=2.

Note: CUDA-only (>=2 GPUs). We use `enforce_eager=False` (default) to enable
`torch.compile`.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from tests.utils import GPUMemoryMonitor, hardware_test
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

PROMPT = "a photo of a cat sitting on a laptop keyboard"


def _get_zimage_model() -> str:
    # Allow overriding the model for local/offline environments.
    # Can be either a HuggingFace repo id or a local path.
    return os.environ.get("VLLM_TEST_ZIMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")


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


def _extract_single_image(outputs) -> Image.Image:
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output[0]
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")

    images = req_out.images
    if images is None or len(images) != 1:
        raise ValueError(f"Expected 1 image, got {0 if images is None else len(images)}")
    return images[0]


def _run_zimage_generate(
    *,
    tp_size: int,
    height: int,
    width: int,
    num_inference_steps: int,
    seed: int,
    enforce_eager: bool,
    vae_use_tiling: bool = False,
    vae_patch_parallel_size: int = 1,
    num_requests: int = 4,
) -> tuple[Image.Image, float, float]:
    if num_requests < 2:
        raise ValueError("num_requests must be >= 2 (1 warmup + >=1 timed)")

    torch.cuda.empty_cache()
    device_index = torch.cuda.current_device()
    monitor = GPUMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()
    m = Omni(
        model=_get_zimage_model(),
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=tp_size,
            vae_patch_parallel_size=vae_patch_parallel_size,
        ),
        enforce_eager=enforce_eager,
        vae_use_tiling=vae_use_tiling,
    )
    try:
        # NOTE: Omni closes itself when a generate() call is exhausted.
        # To avoid measuring teardown time (process shutdown, memory cleanup),
        # we measure the latency to produce *subsequent* outputs within a single
        # generator run.
        #
        # This also serves as a warmup: the first output may include extra
        # compilation/caching overhead, while later outputs are closer to
        # steady-state inference.
        gen = m.generate(
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

        # Ensure the generator is fully consumed so it can clean up.
        for _ in gen:
            pass

        median_time_s = float(np.median(per_request_times_s))

        peak_memory_mb = monitor.peak_used_mb

        return _extract_single_image([last_output]), median_time_s, peak_memory_mb
    finally:
        monitor.stop()
        m.close()
        cleanup_dist_env_and_memory()


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
def test_zimage_tensor_parallel_tp2(tmp_path: Path):
    if current_omni_platform.is_npu() or current_omni_platform.is_rocm():
        pytest.skip("Z-Image TP e2e test is only supported on CUDA for now.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Z-Image TP=2 requires >= 2 CUDA devices.")

    enforce_eager = False

    height = 512
    width = 512
    num_inference_steps = 2
    seed = 42

    tp1_img, tp1_time_s, tp1_peak_mem = _run_zimage_generate(
        tp_size=1,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=seed,
        enforce_eager=enforce_eager,
    )
    tp2_img, tp2_time_s, tp2_peak_mem = _run_zimage_generate(
        tp_size=2,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=seed,
        enforce_eager=enforce_eager,
    )

    tp1_path = tmp_path / "zimage_tp1.png"
    tp2_path = tmp_path / "zimage_tp2.png"
    tp1_img.save(tp1_path)
    tp2_img.save(tp2_path)

    assert tp1_img.width == width and tp1_img.height == height
    assert tp2_img.width == width and tp2_img.height == height

    mean_abs_diff, max_abs_diff = _diff_metrics(tp1_img, tp2_img)
    mean_threshold = 3e-2
    max_threshold = 5e-1
    print(
        "Z-Image TP image diff stats (TP=1 vs TP=2): "
        f"mean_abs_diff={mean_abs_diff:.6e}, max_abs_diff={max_abs_diff:.6e}; "
        f"thresholds: mean<={mean_threshold:.6e}, max<={max_threshold:.6e}; "
        f"tp1_img={tp1_path}, tp2_img={tp2_path}"
    )
    assert mean_abs_diff <= mean_threshold and max_abs_diff <= max_threshold, (
        f"Image diff exceeded threshold: mean_abs_diff={mean_abs_diff:.6e}, max_abs_diff={max_abs_diff:.6e} "
        f"(thresholds: mean<={mean_threshold:.6e}, max<={max_threshold:.6e})"
    )

    print(f"Z-Image TP perf (lower is better): tp1_time_s={tp1_time_s:.6f}, tp2_time_s={tp2_time_s:.6f}")
    assert tp2_time_s < tp1_time_s, f"Expected TP=2 to be faster than TP=1 (tp1={tp1_time_s}, tp2={tp2_time_s})"

    print(f"Z-Image TP peak memory (MB): tp1_peak_mem={tp1_peak_mem:.2f}, tp2_peak_mem={tp2_peak_mem:.2f}")
    assert tp2_peak_mem < tp1_peak_mem, (
        f"Expected TP=2 to use less peak memory than TP=1 (tp1={tp1_peak_mem}, tp2={tp2_peak_mem})"
    )


@pytest.mark.integration
def test_zimage_vae_patch_parallel_tp2(tmp_path: Path):
    if current_omni_platform.is_npu() or current_omni_platform.is_rocm():
        pytest.skip("Z-Image VAE patch parallel e2e test is only supported on CUDA for now.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Z-Image VAE patch parallel TP=2 requires >= 2 CUDA devices.")

    enforce_eager = False

    # Use a larger image to ensure there are multiple VAE tiles.
    height = 1152
    width = 1152
    num_inference_steps = 2
    seed = 42

    baseline_img, _baseline_time_s, _baseline_peak_mem = _run_zimage_generate(
        tp_size=2,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=seed,
        enforce_eager=enforce_eager,
        vae_use_tiling=True,
        vae_patch_parallel_size=1,
        num_requests=2,
    )
    pp2_img, _pp2_time_s, _pp2_peak_mem = _run_zimage_generate(
        tp_size=2,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=seed,
        enforce_eager=enforce_eager,
        vae_use_tiling=True,
        vae_patch_parallel_size=2,
        num_requests=2,
    )

    baseline_path = tmp_path / "zimage_tp2_vae_pp1.png"
    pp2_path = tmp_path / "zimage_tp2_vae_pp2.png"
    baseline_img.save(baseline_path)
    pp2_img.save(pp2_path)

    mean_abs_diff, max_abs_diff = _diff_metrics(baseline_img, pp2_img)
    mean_threshold = 5e-3
    max_threshold = 1e-1
    print(
        "Z-Image VAE patch parallel image diff stats (TP=2, pp=1 vs pp=2): "
        f"mean_abs_diff={mean_abs_diff:.6e}, max_abs_diff={max_abs_diff:.6e}; "
        f"thresholds: mean<={mean_threshold:.6e}, max<={max_threshold:.6e}; "
        f"pp1_img={baseline_path}, pp2_img={pp2_path}"
    )
    assert mean_abs_diff <= mean_threshold and max_abs_diff <= max_threshold, (
        f"Image diff exceeded threshold: mean_abs_diff={mean_abs_diff:.6e}, max_abs_diff={max_abs_diff:.6e} "
        f"(thresholds: mean<={mean_threshold:.6e}, max<={max_threshold:.6e})"
    )
