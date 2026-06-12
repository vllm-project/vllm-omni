# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Alpamayo tokenizer extension + prompt build + history fusion."""

from __future__ import annotations

import os

import pytest
import torch

from vllm_omni.model_executor.models.alpamayo.action_space import (
    DeltaTrajectoryTokenizer,
)
from vllm_omni.model_executor.models.alpamayo.processing import (
    add_alpamayo_tokens,
    build_alpamayo_prompt,
    build_camera_frame_content,
    fuse_traj_tokens,
)

_R1_DIR = "/data/models/Alpamayo-R1-10B"
_EXPECTED_TRAJ_TOKEN_IDS = {
    "history_start": 155674,
    "history_end": 155676,
    "future_start": 155681,
    "future_end": 155683,
    "history": 155684,
    "future": 155685,
}


@pytest.mark.skipif(not os.path.isdir(_R1_DIR), reason="Alpamayo-R1 tokenizer not present locally")
def test_add_alpamayo_tokens_reproduces_config_ids():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_R1_DIR, trust_remote_code=True)
    assert len(tok) == 151669
    info = add_alpamayo_tokens(tok, traj_vocab_size=4000, add_special_tokens=True)
    assert info["traj_token_start_idx"] == 151669
    assert tok.convert_tokens_to_ids("<i0>") == 151669
    for k, expected in _EXPECTED_TRAJ_TOKEN_IDS.items():
        assert info["traj_token_ids"][k] == expected
    assert len(tok) == 155697


# --------------------------------------------------------------------------- #
# prompt construction (camera/frame annotations + history block)
# --------------------------------------------------------------------------- #
def test_camera_frame_content_annotations():
    txt = build_camera_frame_content([0, 1], num_frames_per_camera=4)
    assert "Front left camera: " in txt and "Front camera: " in txt
    # 2 cameras x 4 frames = 8 image placeholders + per-frame "frame N" labels
    assert txt.count("<|image_pad|>") == 8
    assert txt.count("frame 0 ") == 2 and txt.count("frame 3 ") == 2


def test_build_alpamayo_prompt_structure():
    p = build_alpamayo_prompt([0, 1, 2, 6], num_frames_per_camera=4, num_traj_tokens=48)
    assert p.startswith("<|im_start|>system")
    assert p.count("<|image_pad|>") == 16
    assert p.count("<|traj_history|>") == 48
    assert "<|traj_history_start|>" in p and "<|traj_history_end|>" in p
    assert p.rstrip().endswith("<|cot_start|>")
    assert "Front telephoto camera: " in p  # camera id 6


# --------------------------------------------------------------------------- #
# history fusion (robot_obs -> delta tokens replacing <|traj_history|>)
# --------------------------------------------------------------------------- #
def test_fuse_traj_tokens_replaces_placeholders():
    start, ph = 151669, 155684  # traj_token_start_idx, <|traj_history|>
    n_steps = 16
    n_tokens = n_steps * 3
    tok = DeltaTrajectoryTokenizer(num_bins=1000)
    robot_obs = {
        "ego_history_xyz": torch.randn(1, 1, n_steps, 3) * 0.3,
        "ego_history_rot": torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 1, n_steps, 1, 1),
    }
    input_ids = torch.tensor([[10] + [ph] * n_tokens + [20]], dtype=torch.long)
    out = fuse_traj_tokens(input_ids, robot_obs, tok, start, ph)
    assert (out == ph).sum().item() == 0  # all placeholders replaced
    body = out[0, 1:-1]
    assert int(body.min()) >= start and int(body.max()) < start + tok.num_bins
    assert out[0, 0] == 10 and out[0, -1] == 20  # surrounding tokens untouched


def test_fuse_traj_tokens_noop_without_history():
    tok = DeltaTrajectoryTokenizer()
    ids = torch.tensor([[1, 2, 3]])
    assert torch.equal(fuse_traj_tokens(ids, None, tok, 151669, 155684), ids)
