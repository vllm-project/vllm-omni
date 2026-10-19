# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exact planar conversion of one decoded chunk to host RGB bytes."""

import numpy as np
import torch


def _interleave_planar(planar: np.ndarray) -> np.ndarray:
    _, frame_count, height, width = planar.shape
    output = np.empty((frame_count, height, width, 3), dtype=np.uint8)
    for channel in range(3):
        output[..., channel] = planar[channel]
    return output


def planar_uint8_tensor(video: torch.Tensor) -> torch.Tensor:
    """Prepare exact [3,F,H,W] uint8 pixels without a host transfer."""
    if video.ndim != 5 or video.shape[:2] != (1, 3) or not video.is_floating_point():
        raise ValueError("LingBot pixel conversion expects floating-point [1,3,F,H,W] video")
    frames = (video[0] / 2 + 0.5).clamp(0, 1)
    return frames.float().mul_(255).round_().to(torch.uint8)
