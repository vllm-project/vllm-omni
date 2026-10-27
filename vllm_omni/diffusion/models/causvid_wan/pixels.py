# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LingBot-aligned pixel conversion for CausVid outputs.

Streaming and offline consumers share the same host layout LingBot uses for
realtime muxing: contiguous ``(B, T, H, W, 3)`` uint8 RGB. Converting before
worker→engine SHM packing cuts payload size ~4× versus float BCTHW and is what
keeps long R>1 rollouts inside a typical ``/dev/shm`` budget.
"""

from __future__ import annotations

import torch


def float_bcthw_to_uint8_bthwc(video: torch.Tensor) -> torch.Tensor:
    """``(B, 3, T, H, W)`` in ``[-1, 1]`` → contiguous ``(B, T, H, W, 3)`` uint8."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"expected float BCTHW RGB video, got shape {tuple(video.shape)}")
    if not video.is_floating_point():
        raise ValueError(f"expected floating video, got dtype {video.dtype}")
    frames = (video / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 4, 1)
    return frames.float().mul_(255).round_().to(torch.uint8).contiguous()
