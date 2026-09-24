# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import torch
import torch.nn.functional as F


def scalar_mask_to_json(value: float) -> str:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"mask value must be in [0, 1], got {value}")
    return str(value)


def _align_frame_count(frame_count: int) -> int:
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    while current % 17 != 5:
        current += 1
    return current


def _video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def video_mask_to_grid(mask: torch.Tensor, *, width: int, height: int, num_frames: int) -> torch.Tensor:
    tv = _video_latent_t(_align_frame_count(num_frames))
    height = int(height) // 32 * 32
    width = int(width) // 32 * 32
    gh, gw = height // 16, width // 16

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim != 3:
        raise ValueError(f"expected a 2D or 3D mask tensor, got {mask.ndim}D")

    grid = F.interpolate(mask.unsqueeze(1).float(), size=(gh, gw), mode="area").squeeze(1)
    if grid.shape[0] == 1:
        grid = grid.expand(tv, gh, gw)
    elif grid.shape[0] != tv:
        grid = F.interpolate(grid.unsqueeze(0).unsqueeze(0), size=(tv, gh, gw), mode="nearest").squeeze(0).squeeze(0)
    return grid


def video_mask_to_grid_json(mask: torch.Tensor, *, width: int, height: int, num_frames: int) -> str:
    return json.dumps(video_mask_to_grid(mask, width=width, height=height, num_frames=num_frames).tolist())
