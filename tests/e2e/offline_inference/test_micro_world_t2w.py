import os
import sys
from pathlib import Path

import pytest
import torch

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = [os.environ.get("MICRO_WORLD_T2W_MODEL", "amd/Micro-World-T2W")]


# ── Unit tests ───────────────────────────────────────────────────────────


def test_micro_world_t2w_action_module():
    """Test ActionModule produces correct output shapes."""
    from vllm_omni.diffusion.models.micro_world.action_module import ActionModule

    # T2W variant (flatten_spatial=True)
    am_t2w = ActionModule(flatten_spatial=True)
    mouse = torch.randn(1, 81, 2)
    keyboard = torch.randn(1, 81, 7)
    grid_sizes = (21, 30, 45)
    out = am_t2w(mouse, keyboard, grid_sizes)
    assert out.shape == (1, 21 * 30 * 45, 1536)

    # I2W variant (flatten_spatial=False)
    am_i2w = ActionModule(flatten_spatial=False)
    out = am_i2w(mouse, keyboard, grid_sizes)
    assert out.shape == (1, 21, 1536)


def test_micro_world_t2w_registry():
    """Test both pipelines are registered in all registries."""
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
    )

    assert "MicroWorldT2WPipeline" in _DIFFUSION_MODELS
    assert "MicroWorldI2WPipeline" in _DIFFUSION_MODELS
    assert "MicroWorldT2WPipeline" in _DIFFUSION_POST_PROCESS_FUNCS
    assert "MicroWorldI2WPipeline" in _DIFFUSION_POST_PROCESS_FUNCS
    assert "MicroWorldT2WPipeline" in _DIFFUSION_PRE_PROCESS_FUNCS
    assert "MicroWorldI2WPipeline" in _DIFFUSION_PRE_PROCESS_FUNCS


# ── Correlation / regression tests ───────────────────────────────────────


def test_action_module_sensitivity():
    """Different actions must produce different outputs (action signal is not ignored)."""
    from vllm_omni.diffusion.models.micro_world.action_module import ActionModule

    torch.manual_seed(0)
    am = ActionModule(flatten_spatial=True)
    am.eval()

    grid_sizes = (5, 22, 40)

    # Walk forward: W=1
    mouse_walk = torch.zeros(1, 69, 2)
    keyboard_walk = torch.zeros(1, 69, 7)
    keyboard_walk[0, :, 0] = 1.0

    # Stand still: all zeros
    mouse_still = torch.zeros(1, 69, 2)
    keyboard_still = torch.zeros(1, 69, 7)

    with torch.no_grad():
        out_walk = am(mouse_walk, keyboard_walk, grid_sizes)
        out_still = am(mouse_still, keyboard_still, grid_sizes)

    mse = ((out_walk - out_still) ** 2).mean().item()
    assert mse > 1e-6, f"Walk and still outputs are too similar (MSE={mse:.8f}). Actions may be silently ignored."


def test_action_module_mouse_sensitivity():
    """Different mouse inputs must produce different outputs."""
    from vllm_omni.diffusion.models.micro_world.action_module import ActionModule

    torch.manual_seed(0)
    am = ActionModule(flatten_spatial=True)
    am.eval()

    grid_sizes = (5, 22, 40)
    keyboard = torch.zeros(1, 69, 7)

    mouse_left = torch.zeros(1, 69, 2)
    mouse_left[0, :, 0] = -5.0  # look left

    mouse_right = torch.zeros(1, 69, 2)
    mouse_right[0, :, 0] = 5.0  # look right

    with torch.no_grad():
        out_left = am(mouse_left, keyboard, grid_sizes)
        out_right = am(mouse_right, keyboard, grid_sizes)

    mse = ((out_left - out_right) ** 2).mean().item()
    assert mse > 1e-6, (
        f"Left and right mouse outputs are too similar (MSE={mse:.8f}). Mouse actions may be silently ignored."
    )


# ── E2E generation tests ─────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", models)
def test_micro_world_t2w_generation(model_name: str):
    """E2E test: generate action-controlled video via Omni entrypoint."""
    stage_config = str(
        Path(__file__).resolve().parents[3] / "vllm_omni" / "model_executor" / "stage_configs" / "micro_world_t2w.yaml"
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
    # 17 latent frames * temporal_ratio(4) + 1 = 69 raw action frames
    num_action_frames = 69
    # Walk forward: W=1, rest=0
    keyboard_actions = [[1, 0, 0, 0, 0, 0, 0]] * num_action_frames
    mouse_actions = [[0.0, 0.0]] * num_action_frames

    outputs = m.generate(
        prompts="A first person view walking through a forest",
        sampling_params_list=OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=2,
            guidance_scale=3.0,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
            extra_args={
                "mouse_actions": mouse_actions,
                "keyboard_actions": keyboard_actions,
            },
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
    assert frames.shape[1] == num_frames
    assert frames.shape[2] == height
    assert frames.shape[3] == width
