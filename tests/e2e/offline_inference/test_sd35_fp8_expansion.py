# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline memory comparison for Stable Diffusion 3.5 tuned FP8 inference.

This covers the reviewer-requested evidence that the validated SD3.5 FP8 setup
still reduces peak memory versus the BF16 baseline after keeping a small set of
quality-sensitive layers in BF16 through ``ignored_layers``.
"""

import gc
import os as _os

import pytest
import torch
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from tests.helpers.env import DeviceMemoryMonitor
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

MODEL = _os.environ.get("SD35_MODEL", "stabilityai/stable-diffusion-3.5-medium")
HEIGHT = 256
WIDTH = 256
NUM_STEPS = 2
PROMPT = "a cozy reading corner with a chair, lamp, and books"
SD35_TUNED_FP8_CONFIG = {
    "method": "fp8",
    "ignored_layers": [
        "proj_out",
        "context_embedder",
        *[f"transformer_blocks.{i}.ff.net.2" for i in range(24)],
        *[f"transformer_blocks.{i}.ff_context.net.2" for i in range(24)],
    ],
}


def _sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=4.5,
        generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
    )


def _first_request_images(outputs) -> list:
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput) and hasattr(req_out, "images")
    return req_out.images


def _generate_image_with_peak_memory(**omni_kwargs) -> tuple[list, float]:
    gc.collect()
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    current_omni_platform.reset_peak_memory_stats()
    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()

    try:
        with OmniRunner(MODEL, enforce_eager=True, **omni_kwargs) as runner:
            current_omni_platform.reset_peak_memory_stats()
            outputs = runner.omni.generate(PROMPT, _sampling_params())
    finally:
        peak_used_mb = monitor.peak_used_mb
        monitor.stop()

    images = _first_request_images(outputs)
    gc.collect()
    current_omni_platform.empty_cache()
    return images, peak_used_mb


@pytest.mark.diffusion
@pytest.mark.slow
@hardware_test(res={"cuda": "L4"})
def test_sd35_tuned_fp8_uses_less_memory_than_baseline():
    """Compare BF16 vs tuned FP8 peak GPU memory for SD3.5 medium."""
    baseline_images, baseline_peak = _generate_image_with_peak_memory()
    cleanup_dist_env_and_memory()
    quant_images, quant_peak = _generate_image_with_peak_memory(
        quantization_config=SD35_TUNED_FP8_CONFIG,
    )

    assert len(baseline_images) >= 1
    assert len(quant_images) >= 1
    assert baseline_images[0].width == WIDTH and baseline_images[0].height == HEIGHT
    assert quant_images[0].width == WIDTH and quant_images[0].height == HEIGHT

    print(f"SD3.5 baseline peak memory:  {baseline_peak:.0f} MB")
    print(f"SD3.5 tuned FP8 peak memory: {quant_peak:.0f} MB")
    print(f"SD3.5 memory savings:        {baseline_peak - quant_peak:.0f} MB")

    assert quant_peak < baseline_peak, (
        f"Expected tuned FP8 peak memory ({quant_peak:.0f} MB) to be lower than "
        f"baseline ({baseline_peak:.0f} MB)"
    )
