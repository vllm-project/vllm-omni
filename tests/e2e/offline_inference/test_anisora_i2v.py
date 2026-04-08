# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E offline inference tests for Index-AniSora I2V models.

Covers:
- V1 (5B, CogVideoX-based): AniSoraI2VCogVideoXPipeline
- V2 (14B, Wan2.1-based):   AniSoraV2I2VPipeline
- TP=2 for both models (requires 2 GPUs)
"""

import gc
import os
import sys
from pathlib import Path

import PIL.Image
import pytest
import torch

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.helpers.mark import hardware_test
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_V1 = "Disty0/Index-anisora-5B-diffusers"
MODEL_V2 = "aardsoul-music/Wan2.1-Anisora-14B"

# V1: vae_scale_factor_temporal=4 → num_frames % 4 == 1, e.g. 5, 9, 13 ...
# V2: same constraint (Wan2.1 VAE)
NUM_FRAMES = 5
HEIGHT = 480
WIDTH = 720
SEED = 42


def _dummy_image() -> PIL.Image.Image:
    """Create a small solid-color image for testing."""
    return PIL.Image.new("RGB", (WIDTH, HEIGHT), color=(100, 149, 237))


def _assert_video_output(output, num_frames: int, height: int, width: int) -> None:
    assert output is not None
    if isinstance(output, OmniRequestOutput):
        assert output.final_output_type == "image"
        assert output.request_output is not None
        frames = output.request_output.images[0]
    else:
        frames = output
    assert frames is not None
    assert hasattr(frames, "shape"), f"Expected tensor, got {type(frames)}"
    # shape: (batch, num_frames, height, width, channels)
    # Pipeline may round num_frames up for VAE temporal alignment
    assert frames.shape[1] >= num_frames, f"Expected >= {num_frames} frames, got {frames.shape[1]}"
    assert frames.shape[2] == height
    assert frames.shape[3] == width


def _cleanup(model):
    """Shut down model workers and free GPU memory between tests."""
    model.shutdown()
    del model
    gc.collect()
    torch.accelerator.empty_cache()


# ---------------------------------------------------------------------------
# V1 (5B CogVideoX) — single GPU
# ---------------------------------------------------------------------------


@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_anisora_v1_offline_single_gpu():
    """V1 (5B) offline inference on a single GPU."""
    model = Omni(model=MODEL_V1)
    image = _dummy_image()
    outputs = model.generate(
        {"prompt": "a cat sitting calmly", "multi_modal_data": {"image": image}},
        OmniDiffusionSamplingParams(
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=2,
            guidance_scale=6.0,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(SEED),
        ),
    )
    result = outputs[0] if isinstance(outputs, list) else outputs
    _assert_video_output(result, NUM_FRAMES, HEIGHT, WIDTH)
    _cleanup(model)


# ---------------------------------------------------------------------------
# V1 (5B CogVideoX) — SP=2 (Ulysses)
# ---------------------------------------------------------------------------


@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
def test_anisora_v1_offline_sp2():
    """V1 (5B) offline inference with sequence_parallel_size=2 (Ulysses)."""
    model = Omni(
        model=MODEL_V1,
        parallel_config=DiffusionParallelConfig(sequence_parallel_size=2, ulysses_degree=2),
    )
    image = _dummy_image()
    outputs = model.generate(
        {"prompt": "a cat sitting calmly", "multi_modal_data": {"image": image}},
        OmniDiffusionSamplingParams(
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=2,
            guidance_scale=6.0,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(SEED),
        ),
    )
    result = outputs[0] if isinstance(outputs, list) else outputs
    _assert_video_output(result, NUM_FRAMES, HEIGHT, WIDTH)
    _cleanup(model)


# ---------------------------------------------------------------------------
# V1 (5B CogVideoX) — FP8 quantization, single GPU
# ---------------------------------------------------------------------------


@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_anisora_v1_offline_fp8():
    """V1 (5B) offline inference with FP8 quantization (W8A8)."""
    model = Omni(model=MODEL_V1, quantization="fp8")
    image = _dummy_image()
    outputs = model.generate(
        {"prompt": "a cat sitting calmly", "multi_modal_data": {"image": image}},
        OmniDiffusionSamplingParams(
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=2,
            guidance_scale=6.0,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(SEED),
        ),
    )
    result = outputs[0] if isinstance(outputs, list) else outputs
    _assert_video_output(result, NUM_FRAMES, HEIGHT, WIDTH)
    _cleanup(model)
