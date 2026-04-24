import os

import pytest
import torch

from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["Wan-AI/Wan2.2-T2V-A14B-Diffusers"]


@pytest.mark.parametrize("model_name", models)
def test_video_diffusion_model(model_name: str):
    with OmniRunner(
        model_name,
        boundary_ratio=0.875,
        flow_shift=5.0,
    ) as runner:
        # Use minimal settings for testing
        # num_frames must satisfy: num_frames % vae_scale_factor_temporal == 1
        # For Wan2.2, vae_scale_factor_temporal=4, so valid values are 5, 9, 13, 17, ...
        height = 480
        width = 640
        num_frames = 5
        outputs = runner.omni.generate(
            prompts="A cat sitting on a table",
            sampling_params_list=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=2,
                guidance_scale=1.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
            ),
        )
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")

    frames = req_out.images[0]

    assert frames is not None
    assert hasattr(frames, "shape")
    # frames shape: (batch, num_frames, height, width, channels)
    assert frames.shape[1] == num_frames
    assert frames.shape[2] == height
    assert frames.shape[3] == width
