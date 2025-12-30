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
    # num_frames must satisfy: num_frames % vae_scale_factor_temporal == 1
    # For Wan2.2, vae_scale_factor_temporal=4, so valid values are 5, 9, 13, 17, ...
    height = 480
    width = 640
    num_frames = 5
    outputs = m.generate(
        "A cat sitting on a table",
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=2,
        guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(42),
    )
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output[0]
    if not isinstance(req_out, dict) or "images" not in req_out:
        raise ValueError("Invalid request_output structure or missing 'images' key")

    frames = req_out["images"][0]

    assert frames is not None
    assert hasattr(frames, "shape")
    # frames shape: (batch, num_frames, height, width, channels)
    assert frames.shape[1] == num_frames
    assert frames.shape[2] == height
    assert frames.shape[3] == width
