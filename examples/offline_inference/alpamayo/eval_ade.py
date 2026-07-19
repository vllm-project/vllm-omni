# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo offline ADE eval — drive the engine in-process and decode the
actions produced by the inline flow-matching path against a clip's GT
future trajectory.

Useful as a parity check against the HTTP path: with the same model + seed
the engine-side ADE should match the HTTP client's within ~10 mm
(AR sampling noise). See sibling examples/online_serving/alpamayo/ for
the HTTP path.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

MODEL = os.environ.get("ALPAMAYO_MODEL", "nvidia/Alpamayo-1.5-10B")
VLM_BASE = os.environ.get("ALPAMAYO_VLM_BASE", "Qwen/Qwen3-VL-8B-Instruct")
CLIP_PKL = os.environ.get("ALPAMAYO_CLIP_PKL")
if not CLIP_PKL:
    raise SystemExit(
        "ALPAMAYO_CLIP_PKL must point to a clip .pkl with "
        "image_frames / camera_indices / ego_history_{xyz,rot} / ego_future_xyz."
    )
# Per-request flow-matching noise rolls. The engine reads this from
# ``sampling_params.extra_args["n_samples"]`` to set the FM batch dim
# (triggered.numel() × n_samples). Higher N → tighter minADE@N.
N_SAMPLES = int(os.environ.get("ALPAMAYO_N_SAMPLES", "4"))


def main() -> int:
    from PIL import Image

    import vllm_omni.transformers_utils.configs.alpamayo  # noqa: F401
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(
        model=MODEL,
        tokenizer=VLM_BASE,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        # Cap context so KV-cache sizing stays bounded; the model-config default
        # (262144) needs ~36 GiB KV and can fail engine init when profiling
        # leaves less. Alpamayo prompts are a few thousand tokens.
        max_model_len=32768,
    )

    # pandas.read_pickle loads the clip dict without a direct pickle import here.
    import pandas as pd

    data = pd.read_pickle(CLIP_PKL)
    frames = data["image_frames"].flatten(0, 1)
    pil_images = [Image.fromarray(f.permute(1, 2, 0).numpy()) for f in frames]

    # Server-side fusion now active: the client only constructs the prompt
    # string (camera names + <|traj_history|>×48 placeholders) and ships
    # ego_history via sampling_params.extra_args["robot_obs"]. The server's
    # prepare_runner_inputs swaps placeholders with delta tokens.
    from vllm import SamplingParams

    from vllm_omni.model_executor.models.alpamayo.processing import (
        build_alpamayo_prompt,
    )

    cam_ids = data["camera_indices"]
    num_frames_per_camera = int(frames.shape[0] // len(cam_ids))
    prompt = build_alpamayo_prompt(
        camera_indices=cam_ids,
        num_frames_per_camera=num_frames_per_camera,
    )

    hx = data["ego_history_xyz"]
    hr = data["ego_history_rot"]
    if hx.ndim == 3:
        hx = hx.unsqueeze(0)
        hr = hr.unsqueeze(0)
    # extra_args is serialized via msgpack-style IPC -> tensors don't survive.
    # Convert to nested lists; the server's prepare_runner_inputs re-wraps as tensors.
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

    t = time.time()
    outs = omni.generate(
        prompts=[
            {
                "prompt": prompt,
                "multi_modal_data": {"image": pil_images},
                "modalities": ["text"],
            }
        ],
        sampling_params_list=sp,
    )
    print(f"[gen] {time.time() - t:.1f}s")

    # Print the chain-of-thought reasoning the VLM generated.
    for ro in outs:
        req_out = getattr(ro, "request_output", None)
        text = "".join(o.text or "" for o in getattr(req_out, "outputs", []) or [])
        if text.strip():
            print(f"[reasoning] {text.strip()}")
            break

    # The model's forward() returns OmniOutput; the runner routes its
    # multimodal_outputs payload through the engine output processor and the
    # client sees the sampled trajectory under multimodal_output["actions"].
    # NOTE: ``ro.multimodal_output`` is a MultimodalPayload (a Mapping, not a
    # plain dict) for in-process calls — over HTTP the same data is a dict
    # after JSON deserialization. Use Mapping for both.
    from collections.abc import Mapping

    actions = None
    for ro in outs:
        mm = getattr(ro, "multimodal_output", None)
        if isinstance(mm, Mapping) and "actions" in mm:
            actions = mm["actions"]
            break
    if actions is None:
        print("ERROR: no actions in multimodal_output; flow matching may not have fired", file=sys.stderr)
        return 1
    if isinstance(actions, list):
        actions = actions[0] if len(actions) == 1 else torch.stack(actions, dim=0)
    print(f"[actions] shape={tuple(actions.shape)} dtype={actions.dtype}")

    # Decode with the clip's history (same action_space the model holds).
    from transformers import AutoConfig

    from vllm_omni.model_executor.models.alpamayo.action_space import (
        UnicycleAccelCurvatureActionSpace,
    )

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    acfg = cfg.traj_tokenizer_cfg["action_space_cfg"]
    kw = {k: v for k, v in acfg.items() if k not in ("_target_", "_recursive_", "n_waypoints")}
    space = UnicycleAccelCurvatureActionSpace(n_waypoints=int(acfg["n_waypoints"]), **kw).float()

    hx = data["ego_history_xyz"][0].float()  # (1, T, 3)
    hr = data["ego_history_rot"][0].float()  # (1, T, 3, 3)
    a = actions.float().cpu()  # (n_samples, 64, 2)
    n = a.shape[0]
    if hx.shape[0] != n:
        hx = hx.expand(n, -1, -1).contiguous()
        hr = hr.expand(n, -1, -1, -1).contiguous()
    xyz, _rot = space.action_to_traj(a, hx, hr)  # (n_samples, 64, 3)

    gt_xy = data["ego_future_xyz"][0, 0, :, :2].numpy()
    pred_xy_all = xyz[:, :, :2].numpy()  # (n_samples, 64, 2)
    steps = min(pred_xy_all.shape[1], gt_xy.shape[0])
    ade_each = np.linalg.norm(pred_xy_all[:, :steps] - gt_xy[None, :steps], axis=-1).mean(axis=-1)
    fde_each = np.linalg.norm(pred_xy_all[:, steps - 1] - gt_xy[steps - 1], axis=-1)
    n = pred_xy_all.shape[0]
    print(f"clip={os.path.basename(CLIP_PKL).split('_')[0][:8]}  n_samples={n}")
    print(f"  minADE@{n}  = {ade_each.min():.4f} m")
    print(f"  meanADE@{n} = {ade_each.mean():.4f} m")
    print(f"  minFDE@{n}  = {fde_each.min():.4f} m")
    best = int(ade_each.argmin())
    print(f"  best-sample first 3 GT  xy: {gt_xy[:3].round(3).tolist()}")
    print(f"  best-sample first 3 pred xy: {pred_xy_all[best, :3].round(3).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
