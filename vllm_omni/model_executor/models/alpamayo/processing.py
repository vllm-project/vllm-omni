# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo tokenizer extension + prompt construction + history-token fusion.

Two responsibilities, exactly as the SGLang reference:

1. **Add the new Alpamayo tokens** to the tokenizer: the ``traj_vocab_size``
   discrete trajectory tokens ``<i0>..<i(N-1)>`` (inserted at the base vocab
   boundary) followed by the Alpamayo special tokens. This reproduces the exact
   ``traj_token_ids`` and final vocab (155697) the model embedding expects.

2. **Prompt construction + history fusion.** The 1.5 prompt interleaves camera
   display names + frame indices with the image placeholders (config
   ``include_camera_ids`` / ``include_frame_nums``), followed by a history block
   ``<|traj_history_start|> <|traj_history|>*N <|traj_history_end|>``. The
   ``<|traj_history|>`` placeholders are replaced with the ego history discretized
   by :class:`DeltaTrajectoryTokenizer`, so the VLM conditions on past motion.

API note: the ego history is delivered on the outer API as
``extra_args["robot_obs"]`` (renamed from the old ``history_traj`` field), but the
in-prompt token mechanism is unchanged — it is still the ``traj_history`` fusion.
"""

from __future__ import annotations

from typing import Any

import einops
import numpy as np
import torch

from vllm_omni.model_executor.models.alpamayo.action_space import (
    DeltaTrajectoryTokenizer,
)

# Trajectory placeholder/marker tokens (value -> literal string).
TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}

# Ordered Alpamayo special tokens. ORDER IS SIGNIFICANT (reproduces the released
# checkpoints' token ids, e.g. traj_future_start == 155681).
SPECIAL_TOKENS_KEYS = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "image_pre_tkn",
    "image_end",
    "traj_history_start",
    "traj_history_pre_tkn",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_future_start",
    "traj_future_pre_tkn",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "vectorized_wm",
    "vectorized_wm_start",
    "vectorized_wm_end",
    "vectorized_wm_pre_tkn",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]

# Camera index -> display name (matches the SGLang reference / training format).
CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}

VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def add_alpamayo_tokens(
    tokenizer: Any,
    traj_vocab_size: int = 4000,
    add_special_tokens: bool = True,
) -> dict[str, Any]:
    """Extend ``tokenizer`` in-place with Alpamayo discrete + special tokens."""
    discrete_tokens = [f"<i{v}>" for v in range(traj_vocab_size)]
    tokenizer.add_tokens(discrete_tokens)
    traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
    traj_token_end_idx = tokenizer.convert_tokens_to_ids(f"<i{traj_vocab_size - 1}>")
    if add_special_tokens:
        tokenizer.add_tokens([f"<|{k}|>" for k in SPECIAL_TOKENS_KEYS], special_tokens=True)
    else:
        tokenizer.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)
    traj_token_ids = {k: tokenizer.convert_tokens_to_ids(v) for k, v in TRAJ_TOKEN.items()}
    return {
        "traj_token_start_idx": traj_token_start_idx,
        "traj_token_end_idx": traj_token_end_idx,
        "traj_token_ids": traj_token_ids,
    }


def build_history_block(num_traj_tokens: int = 48) -> str:
    """``<|traj_history_start|> <|traj_history|>*N <|traj_history_end|>``."""
    return TRAJ_TOKEN["history_start"] + TRAJ_TOKEN["history"] * num_traj_tokens + TRAJ_TOKEN["history_end"]


def build_camera_frame_content(
    camera_indices: list[int] | torch.Tensor,
    num_frames_per_camera: int,
    image_placeholder: str = VISION_PLACEHOLDER,
) -> str:
    """Interleave camera display names + frame indices with image placeholders.

    Produces e.g. ``"Front left camera: frame 0 <img>frame 1 <img>...Front camera:
    frame 0 <img>..."`` (matches the SGLang / 1.5 training prompt format).
    """
    if isinstance(camera_indices, torch.Tensor):
        camera_indices = camera_indices.tolist()
    parts: list[str] = []
    for cam_id in camera_indices:
        cam_name = CAMERA_DISPLAY_NAMES.get(int(cam_id), f"Camera {int(cam_id)}")
        parts.append(f"{cam_name}: ")
        for frame_idx in range(num_frames_per_camera):
            parts.append(f"frame {frame_idx} {image_placeholder}")
    return "".join(parts)


def build_alpamayo_prompt(
    camera_indices: list[int] | torch.Tensor,
    num_frames_per_camera: int,
    *,
    num_traj_tokens: int = 48,
    system: str = "You are a driving assistant that generates safe and accurate actions.",
    instruction: str = (
        "output the chain-of-thought reasoning of the driving process, then output the future trajectory"
    ),
    image_placeholder: str = VISION_PLACEHOLDER,
) -> str:
    """Build the full Alpamayo chat prompt (system + user + assistant<|cot_start|>).

    User content = camera/frame-annotated images + history block + instruction.
    """
    cam_block = build_camera_frame_content(camera_indices, num_frames_per_camera, image_placeholder)
    hist_block = build_history_block(num_traj_tokens)
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{cam_block}{hist_block}{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n<|cot_start|>"
    )


def replace_pad_token(input_ids: torch.Tensor, new_ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Replace ``pad_idx`` placeholders in ``input_ids`` with ``new_ids`` values."""
    mask = input_ids == pad_idx
    return input_ids.masked_scatter(mask, new_ids)


def tokenize_history_trajectory(
    tokenizer: DeltaTrajectoryTokenizer,
    traj_data: dict[str, Any],
    start_idx: int = 0,
) -> torch.Tensor:
    """Discretize the history into delta tokens, offset by ``start_idx``.

    ``traj_data['ego_history_xyz']`` must be 4D ``[B, n_traj, T, 3]``.
    """
    assert traj_data["ego_history_xyz"].ndim == 4, "ego_history_xyz must be [B, n_traj, T, 3]"
    B = traj_data["ego_history_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)
    hist_idx = (
        tokenizer.encode(
            hist_xyz=hist_xyz[:, :1],
            hist_rot=hist_rot[:, :1],
            fut_xyz=hist_xyz,
            fut_rot=hist_rot,
        )
        + start_idx
    )
    return einops.rearrange(hist_idx, "(b n_traj) n -> b (n_traj n)", b=B)


def fuse_traj_tokens(
    input_ids: torch.Tensor,
    robot_obs: dict[str, Any] | None,
    hist_traj_tokenizer: DeltaTrajectoryTokenizer,
    traj_token_start_idx: int,
    history_placeholder_id: int,
) -> torch.Tensor:
    """Replace ``<|traj_history|>`` placeholders with discretized ego-history tokens.

    ``robot_obs`` carries ``ego_history_xyz`` / ``ego_history_rot`` (the renamed
    outer API; the in-prompt mechanism is the same traj_history fusion). The number
    of placeholders must equal the number of produced delta tokens.
    """
    if robot_obs is None or robot_obs.get("ego_history_xyz") is None or robot_obs.get("ego_history_rot") is None:
        return input_ids
    norm: dict[str, Any] = {}
    for k in ("ego_history_xyz", "ego_history_rot"):
        v = robot_obs[k]
        norm[k] = torch.as_tensor(v) if isinstance(v, (list, np.ndarray)) else v
    t = norm["ego_history_xyz"]
    while t.ndim < 4:
        t = t.unsqueeze(0)
    norm["ego_history_xyz"] = t
    r = norm["ego_history_rot"]
    while r.ndim < 5:
        r = r.unsqueeze(0)
    norm["ego_history_rot"] = r
    hist_idx = tokenize_history_trajectory(hist_traj_tokenizer, norm, traj_token_start_idx)
    return replace_pad_token(input_ids, hist_idx.to(input_ids.device), history_placeholder_id)


__all__ = [
    "TRAJ_TOKEN",
    "SPECIAL_TOKENS_KEYS",
    "CAMERA_DISPLAY_NAMES",
    "VISION_PLACEHOLDER",
    "add_alpamayo_tokens",
    "build_history_block",
    "build_camera_frame_content",
    "build_alpamayo_prompt",
    "fuse_traj_tokens",
    "tokenize_history_trajectory",
    "replace_pad_token",
]
