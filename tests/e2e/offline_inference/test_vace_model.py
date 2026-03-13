import os
import sys
from pathlib import Path

import pytest
import torch

from tests.utils import hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Use 1.3B model for faster CI testing
VACE_MODELS = ["Wan-AI/Wan2.1-VACE-1.3B-diffusers"]


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 1, "rocm": 2})
@pytest.mark.parametrize("model_name", VACE_MODELS)
def test_vace_t2v_generation(model_name: str, run_level):
    """Test basic Text-to-Video generation with VACE model."""
    height = 480
    width = 832
    num_frames = 5

    m = Omni(model=model_name, vae_use_tiling=True)
    try:
        outputs = m.generate(
            prompts="A cat sitting on a table",
            sampling_params_list=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=2,
                guidance_scale=5.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
            ),
        )

        first_output = outputs[0]
        assert first_output.final_output_type == "image"

        req_out = first_output.request_output[0]
        assert isinstance(req_out, OmniRequestOutput)

        frames = req_out.images[0]
        assert frames is not None
        assert frames.shape[1] == num_frames
        assert frames.shape[2] == height
        assert frames.shape[3] == width
    finally:
        m.close()


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards={"cuda": 1, "rocm": 2})
@pytest.mark.parametrize("model_name", VACE_MODELS)
def test_vace_r2v_generation(model_name: str, run_level):
    """Test Reference-to-Video (R2V) pipeline with a reference image."""
    import numpy as np
    from PIL import Image

    height = 480
    width = 832
    num_frames = 5

    # Create synthetic reference image
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    arr[:, :, 1] = np.tile(np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1), (1, width))
    arr[:, :, 2] = 128
    reference_image = Image.fromarray(arr)

    m = Omni(model=model_name, vae_use_tiling=True)
    try:
        outputs = m.generate(
            prompts={
                "prompt": "Camera slowly zooms out from the scene",
                "multi_modal_data": {
                    "reference_images": [reference_image],
                },
            },
            sampling_params_list=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=2,
                guidance_scale=5.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
            ),
        )

        first_output = outputs[0]
        assert first_output.final_output_type == "image"

        req_out = first_output.request_output[0]
        assert isinstance(req_out, OmniRequestOutput)

        frames = req_out.images[0]
        assert frames is not None
        assert frames.shape[1] == num_frames
        assert frames.shape[2] == height
        assert frames.shape[3] == width
    finally:
        m.close()
