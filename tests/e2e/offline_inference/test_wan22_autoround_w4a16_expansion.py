# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for Wan2.2 AutoRound W4A16 quantized inference.

These tests cover I2V (image-to-video) and T2V (text-to-video) generation
with quantized weights.

Requirements:
  - CUDA GPU (H100 or equivalent, ~36 GiB for quantized model)
  - The quantized model checkpoint (Intel/Wan2.2-I2V-A14B-Diffusers-int4-AutoRound,
    Intel/Wan2.2-T2V-A14B-Diffusers-int4-AutoRound)
"""

import gc
import os as _os

import numpy as np
import pytest
import torch
from PIL import Image

from tests.helpers.env import DeviceMemoryMonitor
from tests.helpers.mark import hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

_os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

QUANTIZED_MODEL_I2V = "Intel/Wan2.2-I2V-A14B-Diffusers-int4-AutoRound"
BASELINE_MODEL_I2V = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
QUANTIZED_MODEL_T2V = "Intel/Wan2.2-T2V-A14B-Diffusers-int4-AutoRound"
BASELINE_MODEL_T2V = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

QUANTIZED_MODEL_I2V = _os.environ.get("WAN22_I2V_AUTOROUND_MODEL", QUANTIZED_MODEL_I2V)
BASELINE_MODEL_I2V = _os.environ.get("WAN22_I2V_BASELINE_MODEL", BASELINE_MODEL_I2V)
QUANTIZED_MODEL_T2V = _os.environ.get("WAN22_T2V_AUTOROUND_MODEL", QUANTIZED_MODEL_T2V)
BASELINE_MODEL_T2V = _os.environ.get("WAN22_T2V_BASELINE_MODEL", BASELINE_MODEL_T2V)

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
]

# Small resolution to keep GPU memory & time manageable
HEIGHT = 480
WIDTH = 640
NUM_FRAMES = 5  # must satisfy num_frames % 4 == 1 for Wan2.2
NUM_STEPS = 2  # minimal for smoke-test

# Parametrise: (model, stage_config_path=None, extra_omni_kwargs)
# When stage_config_path is None, the engine auto-resolves from the model's own config.
quant_i2v_params = [(QUANTIZED_MODEL_I2V, None, {"enforce_eager": True})]
baseline_i2v_params = [(BASELINE_MODEL_I2V, None, {"enforce_eager": True})]
quant_t2v_params = [(QUANTIZED_MODEL_T2V, None, {"enforce_eager": True})]
baseline_t2v_params = [(BASELINE_MODEL_T2V, None, {"enforce_eager": True})]

# Module-level storage for peak memory results across tests
_memory_results: dict[str, float] = {}


def _sampling_params_i2v() -> OmniDiffusionSamplingParams:
    """Create sampling params for I2V generation."""
    return OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        guidance_scale=5.0,
        guidance_scale_2=6.0,
        boundary_ratio=0.875,
        generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
    )


def _sampling_params_t2v() -> OmniDiffusionSamplingParams:
    """Create sampling params for T2V generation."""
    return OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        guidance_scale=4.0,
        generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
    )


def _create_test_image(width: int = WIDTH, height: int = HEIGHT) -> Image.Image:
    """Create a deterministic test image for I2V tests."""
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _generate_i2v_video(omni_runner_handler, prompt: str = "A cat sitting on a table, smooth motion") -> tuple:
    """Generate one I2V video, return (frames, peak_memory_mb)."""
    gc.collect()
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    current_omni_platform.reset_peak_memory_stats()
    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()

    image = _create_test_image()
    response = omni_runner_handler.send_diffusion_request(
        {
            "prompt": prompt,
            "images": image,
            "sampling_params": _sampling_params_i2v(),
        },
    )

    peak = monitor.peak_used_mb
    monitor.stop()

    assert response.success, f"Request failed: {response.error_message}"
    assert response.images is not None and len(response.images) > 0, "Expected image output"
    frames = response.images[0]

    gc.collect()
    current_omni_platform.empty_cache()

    return frames, peak


def _generate_t2v_video(omni_runner_handler, prompt: str = "A cat sitting on a table") -> tuple:
    """Generate one T2V video, return (frames, peak_memory_mb)."""
    gc.collect()
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    current_omni_platform.reset_peak_memory_stats()
    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()

    response = omni_runner_handler.send_diffusion_request(
        {
            "prompt": prompt,
            "sampling_params": _sampling_params_t2v(),
        },
    )

    peak = monitor.peak_used_mb
    monitor.stop()

    assert response.success, f"Request failed: {response.error_message}"
    assert response.images is not None and len(response.images) > 0, "Expected image output"
    frames = response.images[0]

    gc.collect()
    current_omni_platform.empty_cache()

    return frames, peak


# ------------------------------------------------------------------
# Test: I2V quantized model generates valid video
# ------------------------------------------------------------------


@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_runner", quant_i2v_params, indirect=True)
def test_wan22_i2v_autoround_w4a16_generates_video(omni_runner, omni_runner_handler):
    """Load the W4A16 quantized Wan2.2 I2V model and verify it produces a valid video."""
    frames, _ = _generate_i2v_video(omni_runner_handler)

    assert frames is not None, "Expected video frames output"
    assert hasattr(frames, "shape"), "Expected frames to have a shape attribute"

    # frames shape: (batch, num_frames, height, width, channels)
    assert frames.shape[1] == NUM_FRAMES, f"Expected {NUM_FRAMES} frames, got {frames.shape[1]}"
    assert frames.shape[2] == HEIGHT, f"Expected height {HEIGHT}, got {frames.shape[2]}"
    assert frames.shape[3] == WIDTH, f"Expected width {WIDTH}, got {frames.shape[3]}"

    # Sanity: video should not be blank (frames are [0, 1] floats)
    arr = np.asarray(frames)
    assert arr.std() > 0.01, "Generated video appears blank (std ≈ 0)"


# ------------------------------------------------------------------
# Test: T2V quantized model generates valid video
# ------------------------------------------------------------------


@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_runner", quant_t2v_params, indirect=True)
def test_wan22_t2v_autoround_w4a16_generates_video(omni_runner, omni_runner_handler):
    """Load the W4A16 quantized Wan2.2 T2V model and verify it produces a valid video."""
    frames, _ = _generate_t2v_video(omni_runner_handler)

    assert frames is not None, "Expected video frames output"
    assert hasattr(frames, "shape"), "Expected frames to have a shape attribute"

    assert frames.shape[1] == NUM_FRAMES, f"Expected {NUM_FRAMES} frames, got {frames.shape[1]}"
    assert frames.shape[2] == HEIGHT, f"Expected height {HEIGHT}, got {frames.shape[2]}"
    assert frames.shape[3] == WIDTH, f"Expected width {WIDTH}, got {frames.shape[3]}"

    arr = np.asarray(frames)
    assert arr.std() > 0.01, "Generated video appears blank (std ≈ 0)"


# ------------------------------------------------------------------
# Test: I2V quantized peak memory
# ------------------------------------------------------------------


@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_runner", quant_i2v_params, indirect=True)
def test_wan22_i2v_autoround_w4a16_quant_peak(omni_runner, omni_runner_handler):
    """Measure peak GPU memory of W4A16 quantized I2V model."""
    frames, peak = _generate_i2v_video(omni_runner_handler)

    assert frames is not None, "Expected video frames output"
    _memory_results["quant_i2v"] = peak
    print(f"\nQuantized I2V (W4A16) peak memory: {peak:.0f} MB")


# ------------------------------------------------------------------
# Test: I2V baseline peak memory
# ------------------------------------------------------------------


@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_runner", baseline_i2v_params, indirect=True)
def test_wan22_i2v_autoround_w4a16_baseline_peak(omni_runner, omni_runner_handler):
    """Measure peak GPU memory of BF16 baseline I2V model."""
    frames, peak = _generate_i2v_video(omni_runner_handler)

    assert frames is not None, "Expected video frames output"
    _memory_results["baseline_i2v"] = peak
    print(f"\nBaseline I2V (BF16) peak memory: {peak:.0f} MB")


# ------------------------------------------------------------------
# Test: I2V memory savings
# ------------------------------------------------------------------


@hardware_test(res={"cuda": "H100"})
def test_wan22_i2v_autoround_w4a16_memory_savings():
    """Assert quantized I2V model uses meaningfully less memory than BF16 baseline."""
    quant_peak = _memory_results["quant_i2v"]
    baseline_peak = _memory_results["baseline_i2v"]

    savings = baseline_peak - quant_peak
    print(f"\nQuantized I2V (W4A16) peak memory: {quant_peak:.0f} MB")
    print(f"Baseline I2V (BF16) peak memory:   {baseline_peak:.0f} MB")
    print(f"Savings:                            {savings:.0f} MB")

    # Wan2.2 I2V A14B transformer is ~28 GB in BF16; W4A16 should save ~20 GB.
    # Use a conservative threshold to account for activations and overhead.
    min_savings_mb = 5000
    assert quant_peak + min_savings_mb < baseline_peak, (
        f"Quantized model ({quant_peak:.0f} MB) should use at least "
        f"{min_savings_mb} MB less than baseline ({baseline_peak:.0f} MB)"
    )


# ------------------------------------------------------------------
# Test: T2V quantized peak memory
# ------------------------------------------------------------------


@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_runner", quant_t2v_params, indirect=True)
def test_wan22_t2v_autoround_w4a16_quant_peak(omni_runner, omni_runner_handler):
    """Measure peak GPU memory of W4A16 quantized T2V model."""
    frames, peak = _generate_t2v_video(omni_runner_handler)

    assert frames is not None, "Expected video frames output"
    _memory_results["quant_t2v"] = peak
    print(f"\nQuantized T2V (W4A16) peak memory: {peak:.0f} MB")


# ------------------------------------------------------------------
# Test: T2V baseline peak memory
# ------------------------------------------------------------------


@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_runner", baseline_t2v_params, indirect=True)
def test_wan22_t2v_autoround_w4a16_baseline_peak(omni_runner, omni_runner_handler):
    """Measure peak GPU memory of BF16 baseline T2V model."""
    frames, peak = _generate_t2v_video(omni_runner_handler)

    assert frames is not None, "Expected video frames output"
    _memory_results["baseline_t2v"] = peak
    print(f"\nBaseline T2V (BF16) peak memory: {peak:.0f} MB")


# ------------------------------------------------------------------
# Test: T2V memory savings
# ------------------------------------------------------------------


@hardware_test(res={"cuda": "H100"})
def test_wan22_t2v_autoround_w4a16_memory_savings():
    """Assert quantized T2V model uses meaningfully less memory than BF16 baseline."""
    quant_peak = _memory_results["quant_t2v"]
    baseline_peak = _memory_results["baseline_t2v"]

    savings = baseline_peak - quant_peak
    print(f"\nQuantized T2V (W4A16) peak memory: {quant_peak:.0f} MB")
    print(f"Baseline T2V (BF16) peak memory:   {baseline_peak:.0f} MB")
    print(f"Savings:                            {savings:.0f} MB")

    # Wan2.2 T2V A14B transformer is ~28 GB in BF16; W4A16 should save ~20 GB.
    # Use a conservative threshold to account for activations and overhead.
    min_savings_mb = 5000
    assert quant_peak + min_savings_mb < baseline_peak, (
        f"Quantized model ({quant_peak:.0f} MB) should use at least "
        f"{min_savings_mb} MB less than baseline ({baseline_peak:.0f} MB)"
    )
