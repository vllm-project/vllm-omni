# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Adapted from NVlabs/alpamayo2 (Apache-2.0), commit 9596749e:
#   src/alpamayo2_super/config.py (build_alpamayo2_super_tokenizer)
#   src/alpamayo2_super/helper.py (create_messages)
#   src/alpamayo2_super/chat_template/conversation.py
#   src/alpamayo2_super/models/utils.py (fuse_traj_tokens)
#   src/alpamayo2_super/models/delta_tokenizer.py (history encoding)
"""Prompt construction for Alpamayo 2 Super trajectory requests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

# Order defines the token ids appended after the trajectory vocabulary.
SPECIAL_TOKENS = {
    key: f"<|{key}|>"
    for key in (
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
    )
}

# Camera index -> name used in the prompt, as in the released data loader.
CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}

_SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
_USER_PROMPT = "output the chain-of-thought reasoning of the driving process, then output the future trajectory."

# History trajectory delta quantization. These are the reference
# DeltaTrajectoryTokenizer's defaults; the checkpoint's hist_traj_tokenizer_cfg
# does not override them.
_HISTORY_XYZ_MIN = (-4.0, -4.0, -10.0)
_HISTORY_XYZ_MAX = (4.0, 4.0, 10.0)


def build_alpamayo2_super_tokenizer(
    name_or_path: str,
    history_vocab_size: int,
    future_vocab_size: int,
) -> PreTrainedTokenizerBase:
    """Extend the checkpoint tokenizer with the trajectory and special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    tokenizer.add_tokens([f"<i{i}>" for i in range(history_vocab_size + future_vocab_size)])
    tokenizer.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)
    return tokenizer


def build_trajectory_messages(
    frames: Sequence[Sequence[Any]],
    camera_indices: Sequence[int],
    config: PretrainedConfig,
) -> list[dict[str, Any]]:
    """Chat messages for chain-of-thought reasoning followed by a trajectory.

    ``frames[c][f]`` is frame ``f`` of camera ``camera_indices[c]``; cameras
    must be sorted by index.
    """
    camera_indices = [int(index) for index in camera_indices]
    if camera_indices != sorted(camera_indices):
        raise ValueError(f"camera_indices must be sorted, got {camera_indices}")
    if len(frames) != len(camera_indices):
        raise ValueError("frames and camera_indices must have the same length")

    content: list[dict[str, Any]] = []
    for camera_index, camera_frames in zip(camera_indices, frames, strict=True):
        if config.include_camera_ids:
            content.append({"type": "text", "text": f"{CAMERA_DISPLAY_NAMES[camera_index]}: "})
        for frame_index, frame in enumerate(camera_frames):
            if config.frame_label == "frame_num":
                content.append({"type": "text", "text": f"frame {frame_index} "})
            content.append({"type": "image", "image": frame})
    history = SPECIAL_TOKENS["traj_history"] * int(config.tokens_per_history_traj)
    content.append(
        {
            "type": "text",
            "text": f"{SPECIAL_TOKENS['traj_history_start']}{history}{SPECIAL_TOKENS['traj_history_end']}",
        }
    )
    content.append({"type": "text", "text": _USER_PROMPT})
    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def encode_history_tokens(history_xyz: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Quantize ``(T, 3)`` history waypoints ending at the ego origin into ``3 * (T - 1)`` bins."""
    deltas = history_xyz[1:] - history_xyz[:-1]
    xyz_min = torch.tensor(_HISTORY_XYZ_MIN, dtype=deltas.dtype, device=deltas.device)
    xyz_max = torch.tensor(_HISTORY_XYZ_MAX, dtype=deltas.dtype, device=deltas.device)
    normalized = (deltas - xyz_min) / (xyz_max - xyz_min)
    return (normalized * (num_bins - 1)).round().long().clamp(0, num_bins - 1).flatten()


def build_trajectory_prompt(
    tokenizer: PreTrainedTokenizerBase,
    config: PretrainedConfig,
    *,
    frames: Sequence[Sequence[Any]],
    camera_indices: Sequence[int],
    ego_history_xyz: torch.Tensor,
) -> list[int]:
    """Token ids of a trajectory prompt with the encoded ego history fused in.

    The caller passes the flattened ``frames`` as the request's images, in
    camera-major order.
    """
    messages = build_trajectory_messages(frames, camera_indices, config)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = torch.tensor(tokenizer.encode(prompt))
    history_ids = encode_history_tokens(
        torch.as_tensor(ego_history_xyz, dtype=torch.float32),
        int(config.history_vocab_size),
    )
    history_ids = history_ids + int(config.traj_ids["history_id0"])
    placeholders = prompt_ids == int(config.traj_ids["history_pad"])
    if int(placeholders.sum()) != history_ids.numel():
        raise ValueError(
            f"ego_history_xyz encodes {history_ids.numel()} tokens but the prompt has "
            f"{int(placeholders.sum())} history placeholders"
        )
    return prompt_ids.masked_scatter(placeholders, history_ids).tolist()
