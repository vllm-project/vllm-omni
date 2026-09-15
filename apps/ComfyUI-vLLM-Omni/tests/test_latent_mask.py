import importlib.util
import json
import os

import pytest
import torch

_EXT_ROOT = os.environ.get("VLLM_OMNI_EXT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LM_PATH = os.path.join(_EXT_ROOT, "comfyui_vllm_omni", "utils", "latent_mask.py")
_spec = importlib.util.spec_from_file_location("latent_mask", _LM_PATH)
_lm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lm)
scalar_mask_to_json = _lm.scalar_mask_to_json
video_mask_to_grid_json = _lm.video_mask_to_grid_json


def test_scalar_mask_to_json():
    assert scalar_mask_to_json(0.5) == "0.5"
    assert scalar_mask_to_json(1.0) == "1.0"
    with pytest.raises(ValueError):
        scalar_mask_to_json(1.5)
    with pytest.raises(ValueError):
        scalar_mask_to_json(-0.1)


def test_video_mask_grid_shape():
    mask = torch.zeros(256, 448)
    grid = json.loads(video_mask_to_grid_json(mask, width=448, height=256, num_frames=107))
    assert len(grid) == 2 + 5 * ((107 - 5) // 17)
    assert len(grid[0]) == 256 // 16
    assert len(grid[0][0]) == 448 // 16


def test_video_mask_grid_shape_non_aligned_frames():
    # 96 snaps up to 107 (17n+5), so Tv == 32, matching the server.
    grid = json.loads(video_mask_to_grid_json(torch.zeros(256, 448), width=448, height=256, num_frames=96))
    assert len(grid) == 32


def test_video_mask_grid_shape_short_clip():
    # num_frames <= 5 -> Tv == 2.
    grid = json.loads(video_mask_to_grid_json(torch.zeros(64, 64), width=64, height=64, num_frames=1))
    assert len(grid) == 2


def test_video_mask_grid_shape_unaligned_width():
    # width=500 floors to 480 (multiple of 32) -> gw == 30, not 31.
    grid = json.loads(video_mask_to_grid_json(torch.zeros(64, 64), width=500, height=256, num_frames=22))
    assert len(grid[0][0]) == 30


def test_video_mask_3d_input():
    grid = json.loads(video_mask_to_grid_json(torch.ones(1, 256, 448), width=448, height=256, num_frames=107))
    assert grid[0][0][0] == 1.0


def test_video_mask_temporal():
    # 3D mask: first half 0 (preserve), second half 1 (regenerate).
    mask = torch.zeros(7, 64, 64)
    mask[4:] = 1.0
    grid = json.loads(video_mask_to_grid_json(mask, width=64, height=64, num_frames=22))
    assert len(grid) == 7
    assert all(v == 0.0 for row in grid[0] for v in row)
    assert all(v == 1.0 for row in grid[-1] for v in row)


def test_video_mask_temporal_resize():
    # 2 temporal slices resize to tv == 7.
    mask = torch.zeros(2, 64, 64)
    mask[1] = 1.0
    grid = json.loads(video_mask_to_grid_json(mask, width=64, height=64, num_frames=22))
    assert len(grid) == 7


def test_video_mask_uniform_preserved():
    grid = json.loads(video_mask_to_grid_json(torch.full((64, 64), 0.5), width=64, height=64, num_frames=107))
    assert all(v == 0.5 for row in grid for cell in row for v in cell)


def test_video_mask_values_in_range():
    grid = json.loads(video_mask_to_grid_json(torch.rand(128, 128), width=128, height=128, num_frames=22))
    assert all(0.0 <= v <= 1.0 for row in grid for cell in row for v in cell)
