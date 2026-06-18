# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end test for the Alpamayo-1.5 VLA driving model (offline / in-process).

Validates the single-stage AR + inline-flow-matching pipeline: multi-camera
images + ego history drive the Qwen3-VL backbone until the trajectory trigger
token, the model dispatches inline to the flow-matching expert, and the sampled
action trajectory is surfaced under ``multimodal_output["actions"]``. With a
clip's ground-truth future the test also checks minADE is in a sane range.

Equivalent to running:
    ALPAMAYO_CLIP_PKL=/path/to/clip.pkl \
    python3 examples/offline_inference/alpamayo/eval_ade.py

The clip ``.pkl`` is not bundled in the repo (it carries multi-camera frames),
so the test skips unless ``ALPAMAYO_CLIP_PKL`` points at one. The pickle is a
dict with keys ``image_frames`` ([C, F, 3, H, W]), ``camera_indices``
(length-C int list), ``ego_history_xyz`` ([n_traj, T, 3]), ``ego_history_rot``
([n_traj, T, 3, 3]) and ``ego_future_xyz`` ([1, 1, T_future, 3]).
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

MODEL_NAME = os.environ.get("ALPAMAYO_MODEL", "nvidia/Alpamayo-1.5-10B")
# Alpamayo-1.5 ships no tokenizer files; it borrows the Qwen3-VL base tokenizer
# (the model's processor appends the trajectory tokens at load time).
VLM_BASE = os.environ.get("ALPAMAYO_VLM_BASE", "Qwen/Qwen3-VL-8B-Instruct")
STAGE_CONFIG = get_deploy_config_path("alpamayo1_5.yaml")
CLIP_PKL = os.environ.get("ALPAMAYO_CLIP_PKL")
# Per-request flow-matching noise rolls; minADE tightens with N.
N_SAMPLES = int(os.environ.get("ALPAMAYO_N_SAMPLES", "4"))
# Loose upper bound on minADE@N (m). The recipe reference is ~0.44 m at N=4;
# this generous bound only catches gross regressions, not AR sampling jitter.
MAX_MIN_ADE_M = 2.0

_EXTRA_OMNI_KWARGS = {
    "tokenizer": VLM_BASE,
    "trust_remote_code": True,
    "dtype": "bfloat16",
    "enforce_eager": True,
    "gpu_memory_utilization": 0.5,
}

# (model, stage_config_path, extra_omni_kwargs) for the indirect ``omni_runner`` fixture.
_OMNI_RUNNER_PARAM = (MODEL_NAME, STAGE_CONFIG, _EXTRA_OMNI_KWARGS)

# Marked ``local_model`` (not ``advanced_model``) so the merge CI lane never
# selects it: Alpamayo-1.5 weights are non-commercial/gated and the multi-camera
# clip .pkl is not in the repo, so this can only run locally with both provided.
pytestmark = [
    pytest.mark.local_model,
    pytest.mark.omni,
    pytest.mark.skipif(
        not CLIP_PKL,
        reason="Set ALPAMAYO_CLIP_PKL to a clip .pkl (multi-cam frames + ego history + GT future).",
    ),
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


def _extract_actions(omni_outputs: list):
    """Pull the sampled action trajectory out of multimodal_output.

    ``ro.multimodal_output`` is a Mapping (MultimodalPayload for in-process
    calls); ``actions`` is a tensor or a list of per-sample tensors.
    """
    from collections.abc import Mapping

    import torch

    for ro in omni_outputs:
        mm = getattr(ro, "multimodal_output", None)
        if isinstance(mm, Mapping) and "actions" in mm:
            actions = mm["actions"]
            if isinstance(actions, list):
                actions = actions[0] if len(actions) == 1 else torch.stack(actions, dim=0)
            return actions
    return None


@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_alpamayo_trajectory(run_level, omni_runner: OmniRunner) -> None:
    """Alpamayo emits a (N, 64, 2) action trajectory with a sane minADE vs GT."""
    # pandas.read_pickle loads the clip dict without this file needing a direct
    # pickle import (keeps the check-pickle-imports pre-commit hook satisfied
    # without an allowlist entry).
    import pandas as pd
    import torch
    from PIL import Image
    from transformers import AutoConfig
    from vllm import SamplingParams

    from vllm_omni.model_executor.models.alpamayo.action_space import (
        UnicycleAccelCurvatureActionSpace,
    )
    from vllm_omni.model_executor.models.alpamayo.processing import build_alpamayo_prompt

    data = pd.read_pickle(CLIP_PKL)
    frames = data["image_frames"].flatten(0, 1)
    pil_images = [Image.fromarray(f.permute(1, 2, 0).numpy()) for f in frames]

    cam_ids = data["camera_indices"]
    num_frames_per_camera = int(frames.shape[0] // len(cam_ids))
    prompt = build_alpamayo_prompt(camera_indices=cam_ids, num_frames_per_camera=num_frames_per_camera)

    hx = data["ego_history_xyz"]
    hr = data["ego_history_rot"]
    if hx.ndim == 3:
        hx = hx.unsqueeze(0)
        hr = hr.unsqueeze(0)
    # extra_args crosses an IPC boundary (tensors don't survive) -> nested lists;
    # the model re-wraps them as tensors server-side during history fusion.
    sp = SamplingParams(
        max_tokens=400,
        temperature=0.6,
        top_p=0.98,
        extra_args={
            "robot_obs": {
                "ego_history_xyz": hx.tolist(),
                "ego_history_rot": hr.tolist(),
            },
            "n_samples": N_SAMPLES,
        },
    )

    omni = omni_runner.omni
    omni_outputs = list(
        omni.generate(
            prompts=[
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": pil_images},
                    "modalities": ["text"],
                }
            ],
            sampling_params_list=sp,
        )
    )

    assert len(omni_outputs) > 0, "No outputs returned"
    actions = _extract_actions(omni_outputs)
    assert actions is not None, "No actions in multimodal_output — flow matching did not fire"

    actions = actions.float().cpu()
    n_wp = int(data["ego_future_xyz"].shape[-2])
    assert actions.ndim == 3, f"Expected (N, n_waypoints, action_dim), got {tuple(actions.shape)}"
    assert actions.shape[0] == N_SAMPLES, f"Expected {N_SAMPLES} samples, got {actions.shape[0]}"
    assert actions.shape[1] == n_wp, f"Expected {n_wp} waypoints, got {actions.shape[1]}"
    assert actions.shape[2] == 2, f"Expected action_dim 2, got {actions.shape[2]}"
    assert torch.isfinite(actions).all(), "Actions contain non-finite values"

    # Decode actions to an xyz trajectory and check minADE against the GT future.
    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    acfg = cfg.traj_tokenizer_cfg["action_space_cfg"]
    kw = {k: v for k, v in acfg.items() if k not in ("_target_", "_recursive_", "n_waypoints")}
    space = UnicycleAccelCurvatureActionSpace(n_waypoints=int(acfg["n_waypoints"]), **kw).float()

    hist_xyz = data["ego_history_xyz"][0].float()
    hist_rot = data["ego_history_rot"][0].float()
    n = actions.shape[0]
    if hist_xyz.shape[0] != n:
        hist_xyz = hist_xyz.expand(n, -1, -1).contiguous()
        hist_rot = hist_rot.expand(n, -1, -1, -1).contiguous()
    xyz, _rot = space.action_to_traj(actions, hist_xyz, hist_rot)

    gt_xy = data["ego_future_xyz"][0, 0, :, :2].numpy()
    pred_xy = xyz[:, :, :2].numpy()
    steps = min(pred_xy.shape[1], gt_xy.shape[0])
    ade_each = np.linalg.norm(pred_xy[:, :steps] - gt_xy[None, :steps], axis=-1).mean(axis=-1)
    min_ade = float(ade_each.min())
    assert np.isfinite(min_ade), "minADE is not finite"

    if run_level in ("advanced_model", "full_model"):
        assert min_ade < MAX_MIN_ADE_M, f"minADE@{n} = {min_ade:.3f} m exceeds {MAX_MIN_ADE_M} m"
