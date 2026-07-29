"""Regression tests for Micro-World T2W and I2W pipelines.

Each test regenerates a small fixed-config video and compares its per-frame
RGB mean/std fingerprint against a checked-in reference JSON. Catches:
  - flat-color collapse (std → 0)
  - color drift (means shift)
  - NaN propagation

Regenerate the references with::

    MICRO_WORLD_REGEN=1 pytest tests/e2e/accuracy/micro_world/test_micro_world_regression.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

HEIGHT = 352
WIDTH = 640
NUM_FRAMES = 17
NUM_ACTION_FRAMES = 69
NUM_STEPS = 2
SEED = 42

MEAN_TOL = 0.08
STD_TOL = 0.08

T2W_MODEL = os.environ.get("MICRO_WORLD_T2W_MODEL", "amd/Micro-World-T2W")
I2W_MODEL = os.environ.get("MICRO_WORLD_I2W_MODEL", "amd/Micro-World-I2W")
T2W_STAGE = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "micro_world_t2w.yaml"
I2W_STAGE = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "micro_world_i2w.yaml"
T2W_REF = Path(__file__).with_name("reference_t2w.json")
I2W_REF = Path(__file__).with_name("reference_i2w.json")

T2W_PROMPT = (
    "Running along a cliffside path in a tropical island in first person perspective, "
    "with turquoise waters crashing against the rocks far below, the salty scent of the "
    "ocean carried by the breeze, and the sound of distant waves blending with the calls "
    "of seagulls as the path twists and turns along the jagged cliffs."
)
I2W_PROMPT = (
    "First-person perspective walking down a lively city street at night. "
    "Neon signs and bright billboards glow on both sides, cars drive past with "
    "headlights and taillights streaking slightly. Camera motion directly aligned "
    "with user actions, immersive urban night scene."
)

_KEYBOARD_WALK = [[1, 0, 0, 0, 0, 0, 0]] * NUM_ACTION_FRAMES
_MOUSE_STILL = [[0.0, 0.0]] * NUM_ACTION_FRAMES


def _to_float_frames(frames) -> np.ndarray:
    arr = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else np.asarray(frames)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def _fingerprint(frames) -> dict[str, list[list[float]]]:
    """Per-frame RGB mean and std. Accepts (T,H,W,C), (C,T,H,W) or those with a leading batch dim."""
    arr = _to_float_frames(frames)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D/5D frames, got shape={arr.shape}")
    if arr.shape[-1] == 3:
        per_frame = arr
    elif arr.shape[0] == 3:
        per_frame = np.transpose(arr, (1, 2, 3, 0))
    else:
        raise ValueError(f"Cannot locate RGB channel in shape={arr.shape}")
    means = per_frame.mean(axis=(1, 2)).tolist()
    stds = per_frame.std(axis=(1, 2)).tolist()
    return {"mean": means, "std": stds}


def _compare(actual: dict, reference: dict) -> None:
    a_mean = np.array(actual["mean"])
    a_std = np.array(actual["std"])
    r_mean = np.array(reference["mean"])
    r_std = np.array(reference["std"])

    assert not np.isnan(a_mean).any() and not np.isnan(a_std).any(), "NaN in generated frames"
    assert a_mean.shape == r_mean.shape, f"Frame count mismatch: got {a_mean.shape}, expected {r_mean.shape}"

    mean_mae = float(np.abs(a_mean - r_mean).max())
    std_mae = float(np.abs(a_std - r_std).max())
    print(f"max mean diff={mean_mae:.4f} (tol {MEAN_TOL})  max std diff={std_mae:.4f} (tol {STD_TOL})")
    assert mean_mae < MEAN_TOL, f"per-frame mean drifted: max abs diff={mean_mae:.4f}"
    assert std_mae < STD_TOL, f"per-frame std drifted: max abs diff={std_mae:.4f}"


def _save_or_compare(fp: dict, ref_path: Path) -> None:
    if os.environ.get("MICRO_WORLD_REGEN") == "1":
        ref_path.write_text(json.dumps(fp, indent=2) + "\n")
        pytest.skip(f"Wrote reference fingerprint to {ref_path}")
    if not ref_path.exists():
        pytest.skip(f"Reference fingerprint not found: {ref_path}. Run with MICRO_WORLD_REGEN=1.")
    reference = json.loads(ref_path.read_text())
    _compare(fp, reference)


def _run(
    model: str, stage_config: Path, prompt: str, guidance_scale: float, extra_args: dict, image=None
) -> np.ndarray:
    m = Omni(
        model=model,
        stage_configs_path=str(stage_config),
        flow_shift=3.0,
        stage_init_timeout=3000,
        init_timeout=3000,
    )
    if image is not None:
        prompts = {"prompt": prompt, "multi_modal_data": {"image": image}}
    else:
        prompts = prompt
    outputs = m.generate(
        prompts=prompts,
        sampling_params_list=OmniDiffusionSamplingParams(
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=NUM_STEPS,
            guidance_scale=guidance_scale,
            generator=torch.Generator(current_omni_platform.device_type).manual_seed(SEED),
            extra_args=extra_args,
        ),
    )
    req_out = outputs[0].request_output
    assert isinstance(req_out, OmniRequestOutput)
    return req_out.images[0]


def test_micro_world_t2w_regression():
    frames = _run(
        T2W_MODEL,
        T2W_STAGE,
        prompt=T2W_PROMPT,
        guidance_scale=3.0,
        extra_args={"mouse_actions": _MOUSE_STILL, "keyboard_actions": _KEYBOARD_WALK},
    )
    _save_or_compare(_fingerprint(frames), T2W_REF)


def test_micro_world_i2w_regression():
    image = Image.new("RGB", (WIDTH, HEIGHT), color=(40, 40, 60))
    frames = _run(
        I2W_MODEL,
        I2W_STAGE,
        prompt=I2W_PROMPT,
        guidance_scale=6.0,
        extra_args={"mouse_actions": _MOUSE_STILL, "keyboard_actions": _KEYBOARD_WALK},
        image=image,
    )
    _save_or_compare(_fingerprint(frames), I2W_REF)
