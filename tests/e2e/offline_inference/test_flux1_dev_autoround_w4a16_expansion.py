# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for FLUX AutoRound W4A16 quantized inference.

These tests require:
  - A CUDA GPU
  - The quantized model checkpoint (vllm-project-org/FLUX.1-dev-AutoRound-w4a16)

``test_flux_autoround_w4a16_generates_image`` uses the shared ``omni_runner`` fixture
(quantized model). ``test_flux_autoround_w4a16_memory_savings`` must load two different
checkpoints in sequence, so it keeps explicit ``OmniRunner`` context managers.
"""

import gc
import os as _os

import numpy as np
import pytest
import torch
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from tests.helpers.env import DeviceMemoryMonitor
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

pytestmark = pytest.mark.skip(
    reason="Current Hugging Face account is not yet authorized to access this model.",
)

QUANTIZED_MODEL = "vllm-project-org/FLUX.1-dev-AutoRound-w4a16"
BASELINE_MODEL = "black-forest-labs/FLUX.1-dev"

QUANTIZED_MODEL = _os.environ.get("FLUX_AUTOROUND_MODEL", QUANTIZED_MODEL)
BASELINE_MODEL = _os.environ.get("FLUX_BASELINE_MODEL", BASELINE_MODEL)

# Small resolution to keep GPU memory & time manageable
HEIGHT = 256
WIDTH = 256
NUM_STEPS = 2  # minimal for smoke-test

# Indirect param: (model_id, stage_config_path, extra OmniRunner / Omni kwargs)
_OMNI_RUNNER_PARAM = (QUANTIZED_MODEL, None, {"enforce_eager": True})


def _sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=0.0,
        generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
    )


def _first_request_images(outputs) -> list:
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput) and hasattr(req_out, "images")
    return req_out.images


def _generate_image(model_name: str, **extra_kwargs) -> tuple[list, float]:
    """Load a FLUX model, generate one image, return (images, peak_memory_mb)."""
    gc.collect()
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    current_omni_platform.reset_peak_memory_stats()
    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()

    with OmniRunner(model_name, enforce_eager=True, **extra_kwargs) as runner:
        current_omni_platform.reset_peak_memory_stats()
        outputs = runner.omni.generate("a photo of a cat sitting on a laptop keyboard", _sampling_params())

    peak = monitor.peak_used_mb
    monitor.stop()

    images = _first_request_images(outputs)

    gc.collect()
    current_omni_platform.empty_cache()

    return images, peak


@pytest.mark.full_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)
def test_flux_autoround_w4a16_generates_image(omni_runner: OmniRunner):
    """Load the W4A16 quantized FLUX model and verify it produces a valid image."""
    outputs = omni_runner.omni.generate("a photo of a cat sitting on a laptop keyboard", _sampling_params())
    images = _first_request_images(outputs)

    assert len(images) >= 1, "Expected at least one generated image"
    img = images[0]
    assert img.width == WIDTH, f"Expected width {WIDTH}, got {img.width}"
    assert img.height == HEIGHT, f"Expected height {HEIGHT}, got {img.height}"

    arr = np.array(img)
    assert arr.std() > 1.0, "Generated image appears blank (std ≈ 0)"


@pytest.mark.full_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"})
def test_flux_autoround_w4a16_memory_savings():
    """Compare peak GPU memory of quantized vs FP16 baseline.

    The W4A16 model should use meaningfully less memory than the
    BF16/FP16 baseline since weights are 4-bit instead of 16-bit.
    """
    _, quant_peak = _generate_image(QUANTIZED_MODEL)
    cleanup_dist_env_and_memory()
    _, baseline_peak = _generate_image(BASELINE_MODEL)

    print(f"Quantized (W4A16) peak memory: {quant_peak:.0f} MB")
    print(f"Baseline (BF16) peak memory:   {baseline_peak:.0f} MB")
    print(f"Savings:                        {baseline_peak - quant_peak:.0f} MB")

    # W4A16 weights are 4x smaller than BF16/FP16.  FLUX.1-dev transformer
    # is ~12 GB in BF16, so we expect ~9 GB savings on weights alone.
    # Use a conservative threshold to account for activations and overhead.
    min_savings_mb = 2000
    assert quant_peak + min_savings_mb < baseline_peak, (
        f"Quantized model ({quant_peak:.0f} MB) should use at least "
        f"{min_savings_mb} MB less than baseline ({baseline_peak:.0f} MB)"
    )
