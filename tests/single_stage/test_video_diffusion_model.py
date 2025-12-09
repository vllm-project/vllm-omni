import os
import sys
from pathlib import Path

import pytest
import torch

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

models = ["Wan-AI/Wan2.2-T2V-A14B-Diffusers"]


@pytest.mark.parametrize("model_name", models)
def test_video_diffusion_model(model_name: str):
    m = Omni(
        model=model_name,
        boundary_ratio=0.875,
        flow_shift=5.0,
    )
    # Use minimal settings for testing
    height = 480
    width = 640
    num_frames = 17
    frames = m.generate(
        "A cat sitting on a table",
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=2,
        guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(42),
    )

    assert frames is not None
    assert hasattr(frames, "shape")
    assert frames.shape[1] == height
    assert frames.shape[2] == width
