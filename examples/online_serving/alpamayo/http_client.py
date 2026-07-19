# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP client for Alpamayo — no tokenizer, just requests + a thin client-side
``action_space`` decode so we can ADE-compare against the clip's GT future.

The server (started via ``vllm-omni serve``) handles:
  - Tokenization (Qwen3-VL processor + Alpamayo special-token injection)
  - History fusion: ``<|traj_history|>`` placeholders are replaced with
    delta-encoded ego_history tokens read from
    ``sampling_params.extra_args["robot_obs"]`` (see
    ``Alpamayo15ForConditionalGeneration.prepare_runner_inputs``)
  - Flow matching: triggered inline when the AR emits
    ``<|traj_future_start|>``; sampled actions surfaced as
    ``multimodal_output["actions"]``.

The client:
  1. Loads a clip pickle (for the images + ego_history + GT future)
  2. Constructs the Alpamayo prompt string (camera names + frame_idx +
     image placeholders + 48 ``<|traj_history|>`` literals + system /
     instruction). Pure string templating — no tokenizer.
  3. Base64-encodes the images for OpenAI-style chat completion
  4. POSTs to /v1/chat/completions with prompt + multi_modal_data +
     ``vllm_xargs={"robot_obs": ..., "n_samples": N}``
  5. Reads ``multimodal_output["actions"]`` from the response, decodes
     the ``(N_samples, 64, 2)`` actions to ``(N_samples, 64, 3)`` xyz via
     ``UnicycleAccelCurvatureActionSpace`` (instantiated from the
     ``ACTION_SPACE_CFG`` constants below — copied verbatim from the
     Alpamayo-1.5-10B checkpoint's ``config.json``), then prints
     minADE / meanADE against ``ego_future_xyz``.

See ``examples/online_serving/alpamayo/README.md`` for the full
two-terminal recipe (server + client). Env vars on the client:

  - ``ALPAMAYO_SERVER``    (default ``http://localhost:8000``)
  - ``ALPAMAYO_MODEL``     (default ``alpamayo-1.5`` — wire name; must
                            match the server's ``--served-model-name``)
  - ``ALPAMAYO_CLIP_PKL``  (required; clip pickle with images + history
                            + ``ego_future_xyz`` GT)
  - ``ALPAMAYO_N_SAMPLES`` (optional, default 4)
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time

import numpy as np
import requests
import torch
from PIL import Image

SERVER = os.environ.get("ALPAMAYO_SERVER", "http://localhost:8000")
# Must match the server's --served-model-name (or, if that flag was omitted,
# whatever path/id was passed positionally as the model argument). This is
# just a wire name — it has NOTHING to do with where the weights live.
MODEL = os.environ.get("ALPAMAYO_MODEL", "alpamayo-1.5")
CLIP_PKL = os.environ.get("ALPAMAYO_CLIP_PKL")
if not CLIP_PKL:
    raise SystemExit(
        "ALPAMAYO_CLIP_PKL must point to a clip .pkl with "
        "image_frames / camera_indices / ego_history_{xyz,rot} / ego_future_xyz."
    )
N_SAMPLES = int(os.environ.get("ALPAMAYO_N_SAMPLES", "4"))
# Passed to the server as ``extra_args["n_samples"]`` (a per-request knob
# the model reads in ``_run_flow_matching_inline`` to set the FM batch dim
# = triggered.numel() × n_samples). For one HTTP request → one AR rollout
# → N independent FM noise rolls → minADE@N (computed by the offline eval).

# --- Action-space constants -------------------------------------------------
# Copied verbatim from ``traj_tokenizer_cfg["action_space_cfg"]`` in the
# Alpamayo-1.5-10B checkpoint's config.json. These are normalization stats
# baked at training time — fixed per released checkpoint, so embedding them
# here keeps the client free of any config/weights download.
ACTION_SPACE_CFG = {
    "accel_mean": 0.02902694707164455,
    "accel_std": 0.6810426736454882,
    "curvature_mean": 0.0002692167976330542,
    "curvature_std": 0.026148280660833106,
    "accel_bounds": (-9.8, 9.8),
    "curvature_bounds": (-0.33, 0.33),
    "dt": 0.1,
    "n_waypoints": 64,
    "theta_lambda": 1e-06,
    "theta_ridge": 1e-08,
    "v_lambda": 1e-06,
    "v_ridge": 0.0001,
    "a_lambda": 0.0001,
    "a_ridge": 0.0001,
    "kappa_lambda": 0.0001,
    "kappa_ridge": 0.0001,
}

# --- Static template pieces (NO tokenizer needed) ----------------------------
SYSTEM = "You are a driving assistant that generates safe and accurate actions."
INSTRUCTION = "output the chain-of-thought reasoning of the driving process, then output the future trajectory"
CAMERA_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}
VISION_PH = "<|vision_start|><|image_pad|><|vision_end|>"
NUM_HISTORY_TRAJ_TOKENS = 48
HISTORY_BLOCK = "<|traj_history_start|>" + "<|traj_history|>" * NUM_HISTORY_TRAJ_TOKENS + "<|traj_history_end|>"


def build_prompt(camera_indices: list[int], num_frames_per_camera: int) -> str:
    """Replicate ``processing.build_alpamayo_prompt`` — pure string ops."""
    parts: list[str] = []
    for cam_id in camera_indices:
        parts.append(f"{CAMERA_NAMES.get(int(cam_id), f'Camera {cam_id}')}: ")
        for frame_idx in range(num_frames_per_camera):
            parts.append(f"frame {frame_idx} {VISION_PH}")
    cam_block = "".join(parts)
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{cam_block}{HISTORY_BLOCK}{INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>assistant\n<|cot_start|>"
    )


def encode_image(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> int:
    # --- Load clip (for images + ego_history) -------------------------------
    # pandas.read_pickle loads the clip dict without a direct pickle import here.
    import pandas as pd

    data = pd.read_pickle(CLIP_PKL)
    frames = data["image_frames"].flatten(0, 1)
    pil_images = [Image.fromarray(f.permute(1, 2, 0).numpy()) for f in frames]
    cam_ids = data["camera_indices"].tolist()
    num_frames_per_camera = int(frames.shape[0] // len(cam_ids))

    prompt = build_prompt(cam_ids, num_frames_per_camera)

    # --- Build robot_obs dict (the only "domain" data we ship) --------------
    hx = data["ego_history_xyz"]  # (n_traj, T, 3) or similar
    hr = data["ego_history_rot"]
    if hx.ndim == 3:
        hx = hx.unsqueeze(0)
        hr = hr.unsqueeze(0)
    robot_obs = {
        "ego_history_xyz": hx.tolist(),
        "ego_history_rot": hr.tolist(),
    }

    # --- POST to /v1/chat/completions --------------------------------------
    # Use a single "user" message whose content is image_url blocks interleaved
    # with a final text block carrying the FULLY-FORMATTED Alpamayo prompt.
    # The empty chat_template (`{% for m in messages %}{{ m.content }}{% endfor %}`-style)
    # below collapses messages back to the raw string our prompt already contains.
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(i)}"}} for i in pil_images
    ]
    user_content.append({"type": "text", "text": prompt})

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": 400,
        "temperature": 0.6,
        "top_p": 0.98,
        "chat_template": (
            "{% for m in messages %}{% for c in m.content %}"
            "{% if c.type == 'text' %}{{ c.text }}{% endif %}"
            "{% endfor %}{% endfor %}"
        ),
        # vllm_xargs is the official "passthrough to sampling_params.extra_args"
        # channel for the OpenAI-compatible server. Its protocol type permits
        # only flat primitives, so the nested robot_obs dict ships as a JSON
        # string under the ``robot_obs`` key — the model's prepare_runner_inputs
        # accepts either a dict (Python clients) or a JSON string (HTTP).
        # robot_obs as JSON string (vllm_xargs values must be flat primitives);
        # n_samples as a flat int passes natively.
        "vllm_xargs": {
            "robot_obs": json.dumps(robot_obs),
            "n_samples": N_SAMPLES,
        },
    }
    t = time.time()
    resp = requests.post(f"{SERVER}/v1/chat/completions", json=payload, timeout=180)
    print(f"[http] status={resp.status_code} {time.time() - t:.1f}s")
    if resp.status_code != 200:
        print("ERROR body:", resp.text[:500], file=sys.stderr)
        return 1
    body = resp.json()
    choices = body.get("choices", [])

    # --- Print the chain-of-thought reasoning the VLM generated -------------
    msg0 = choices[0].get("message", {}) if choices else {}
    reasoning = (msg0.get("reasoning") or msg0.get("content") or "").strip()
    if reasoning:
        print(f"[reasoning] {reasoning}")

    # --- Extract actions from multimodal_output ----------------------------
    # Chat completion shape: choices[*].message.multimodal_output["actions"]
    # (vs the completion shape with choices[*].multimodal_output)
    actions_raw = None
    for ch in choices:
        for src in (ch.get("message", {}), ch):
            mm = src.get("multimodal_output") if isinstance(src, dict) else None
            if isinstance(mm, dict) and "actions" in mm:
                actions_raw = mm["actions"]
                break
        if actions_raw is not None:
            break
    if actions_raw is None:
        print("ERROR: no actions in response.choices[*].multimodal_output", file=sys.stderr)
        print("response keys:", list(body.keys()), file=sys.stderr)
        print("first choice keys:", list(choices[0].keys()) if choices else "—", file=sys.stderr)
        return 1
    # actions_raw is a nested list of shape (n_samples, n_waypoints=64, action_dim=2).
    actions = torch.tensor(actions_raw, dtype=torch.float32)
    print(f"[actions] shape={tuple(actions.shape)}")

    # --- Decode actions -> xyz trajectory; ADE vs GT ----------------------
    # action_space.py is a self-contained pure-torch module (no engine);
    # the normalization constants come from ACTION_SPACE_CFG above.
    from vllm_omni.model_executor.models.alpamayo.action_space import (
        UnicycleAccelCurvatureActionSpace,
    )

    space = UnicycleAccelCurvatureActionSpace(**ACTION_SPACE_CFG).float()

    n = actions.shape[0]
    hx_f = data["ego_history_xyz"][0].float().expand(n, -1, -1).contiguous()
    hr_f = data["ego_history_rot"][0].float().expand(n, -1, -1, -1).contiguous()
    xyz, _ = space.action_to_traj(actions, hx_f, hr_f)  # (n_samples, 64, 3)
    gt_xy = data["ego_future_xyz"][0, 0, :, :2].numpy()
    pred_xy = xyz[:, :, :2].numpy()
    steps = min(pred_xy.shape[1], gt_xy.shape[0])
    ade_each = np.linalg.norm(pred_xy[:, :steps] - gt_xy[None, :steps], axis=-1).mean(-1)
    print(f"clip={os.path.basename(CLIP_PKL).split('_')[0][:8]}  n={n}")
    print(f"  minADE@{n}  = {ade_each.min():.4f} m")
    print(f"  meanADE@{n} = {ade_each.mean():.4f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
