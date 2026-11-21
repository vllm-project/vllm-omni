# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""3D rotary embedding of the Wan transformer with explicit temporal positions.

AURA attends from the current chunk plus a reference frame to a history whose temporal positions
are not contiguous (anchor chunk, latest chunks, reference frame inside its training segment), so
the frequencies are gathered per frame from explicit position lists. Computed in float64.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

MAX_POSITIONS = 1024


def rope_table(head_dim: int, max_positions: int = MAX_POSITIONS, theta: float = 10000.0) -> torch.Tensor:
    """Complex128 table [max_positions, head_dim // 2]: temporal, height, then width frequencies."""
    sixth = head_dim // 6
    parts = []
    for dim in (head_dim - 4 * sixth, 2 * sixth, 2 * sixth):
        inv_freq = 1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float64) / dim)
        angles = torch.outer(torch.arange(max_positions, dtype=torch.float64), inv_freq)
        parts.append(torch.polar(torch.ones_like(angles), angles))
    return torch.cat(parts, dim=1)


def gather_freqs(table: torch.Tensor, frames: Sequence[int], height: int, width: int) -> torch.Tensor:
    """Frequencies for ``len(frames)`` frames of ``height x width`` tokens, frame i at temporal
    position ``frames[i]``. Returns a complex tensor [1, len(frames) * height * width, 1, head_dim / 2]."""
    if max(frames) >= table.shape[0]:
        raise ValueError(f"temporal position {max(frames)} exceeds the RoPE table ({table.shape[0]} positions)")
    half = table.shape[1]
    third = half // 3
    f_time, f_height, f_width = table.split([half - 2 * third, third, third], dim=1)
    n = len(frames)
    index = torch.tensor(list(frames), device=table.device)
    grid = torch.cat(
        [
            f_time[index].view(n, 1, 1, -1).expand(n, height, width, -1),
            f_height[:height].view(1, height, 1, -1).expand(n, height, width, -1),
            f_width[:width].view(1, 1, width, -1).expand(n, height, width, -1),
        ],
        dim=-1,
    )
    return grid.reshape(1, n * height * width, 1, -1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate x [B, L, heads, head_dim] by complex freqs [1, L, 1, head_dim / 2], in float64
    (autocast off), as pairs of adjacent channels."""
    b, s, h, d = x.shape
    with torch.autocast(device_type=x.device.type, enabled=False):
        real, imag = x.to(torch.float64).reshape(b, s, h, d // 2, 2).unbind(-1)
        cos_sin = torch.view_as_real(freqs)
        cos, sin = cos_sin[..., 0], cos_sin[..., 1]
        rotated = torch.stack([real * cos - imag * sin, real * sin + imag * cos], dim=-1)
    return rotated.reshape(b, s, h, d).to(x.dtype)
