# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for latent-mask serialization helpers."""

import json

import pytest
import torch
from comfyui_vllm_omni.utils.latent_mask import audio_mask_to_json, scalar_mask_to_json, video_mask_to_json

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_scalar_mask_to_json():
    assert scalar_mask_to_json(0.5) == "0.5"
    assert scalar_mask_to_json(1.0) == "1.0"
    with pytest.raises(ValueError):
        scalar_mask_to_json(1.5)
    with pytest.raises(ValueError):
        scalar_mask_to_json(-0.1)


def test_video_mask_to_json_serializes_raw_spatial_mask():
    # A 2D mask is sent as-is; the server resizes it to the latent grid.
    grid = json.loads(video_mask_to_json(torch.full((120, 160), 0.5)))
    assert len(grid) == 120
    assert len(grid[0]) == 160
    assert grid[0][0] == 0.5


def test_video_mask_to_json_serializes_raw_frame_space_mask():
    # A 3D mask is sent as-is (one slice per source frame).
    grid = json.loads(video_mask_to_json(torch.full((22, 120, 160), 0.5)))
    assert len(grid) == 22
    assert len(grid[0]) == 120
    assert len(grid[0][0]) == 160
    assert grid[0][0][0] == 0.5


def test_video_mask_to_json_rejects_wrong_rank():
    with pytest.raises(ValueError):
        video_mask_to_json(torch.zeros(4))
    with pytest.raises(ValueError):
        video_mask_to_json(torch.zeros(1, 1, 1, 1))


def test_audio_mask_to_json_serializes_raw_temporal_mask():
    # A 1D mask is sent as-is (one value per time step).
    grid = json.loads(audio_mask_to_json(torch.full((178,), 0.5)))
    assert len(grid) == 178
    assert grid[0] == 0.5


def test_audio_mask_to_json_serializes_raw_channel_major_mask():
    # A 2D mask is sent as-is ([channel, time]).
    grid = json.loads(audio_mask_to_json(torch.full((2, 178), 0.5)))
    assert len(grid) == 2
    assert len(grid[0]) == 178
    assert grid[0][0] == 0.5


def test_audio_mask_to_json_rejects_wrong_rank():
    with pytest.raises(ValueError):
        audio_mask_to_json(torch.zeros(()))  # scalar
    with pytest.raises(ValueError):
        audio_mask_to_json(torch.zeros(1, 1, 1))  # 3D
