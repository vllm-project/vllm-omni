import os
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = [os.environ.get("MICRO_WORLD_I2W_MODEL", "amd/Micro-World-I2W")]


@pytest.mark.parametrize("model_name", models)
def test_micro_world_i2w_generation(model_name: str):
    """E2E test: generate action-controlled video from an image via Omni entrypoint."""
    stage_config = str(
        Path(__file__).resolve().parents[3] / "vllm_omni" / "model_executor" / "stage_configs" / "micro_world_i2w.yaml"
    )
    m = Omni(
        model=model_name,
        stage_configs_path=stage_config,
        flow_shift=3.0,
        stage_init_timeout=3000,
        init_timeout=3000,
    )

    height = 352
    width = 640
    num_frames = 17
    num_action_frames = 69
    keyboard_actions = [[1, 0, 0, 0, 0, 0, 0]] * num_action_frames
    mouse_actions = [[0.0, 0.0]] * num_action_frames

    image = Image.new("RGB", (width, height), color=(40, 40, 60))

    outputs = m.generate(
        prompts={
            "prompt": "A first person view walking down a quiet street at night.",
            "multi_modal_data": {"image": image},
        },
        sampling_params_list=OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=2,
            guidance_scale=6.0,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
            extra_args={
                "mouse_actions": mouse_actions,
                "keyboard_actions": keyboard_actions,
            },
        ),
    )
    req_out = outputs[0].request_output
    assert isinstance(req_out, OmniRequestOutput)
    frames = req_out.images[0]
    assert frames.shape[1] == num_frames
    assert frames.shape[2] == height
    assert frames.shape[3] == width
