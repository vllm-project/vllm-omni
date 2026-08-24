# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
from diffusers.video_processor import VideoProcessor


def prepare_video_for_transport(
    video: torch.Tensor,
    video_processor: VideoProcessor,
) -> torch.Tensor:
    """Consume a decoded BCTHW video and prepare BTHWC float32 on-device."""
    if video.ndim != 5:
        raise ValueError(f"Expected a decoded BCTHW video, got shape {tuple(video.shape)}.")

    if video_processor.config.do_normalize:
        video.mul_(0.5).add_(0.5).clamp_(0, 1)
    # Diffusers NumPy/PIL output is BTHWC float32. Materialize that layout on
    # the producing device; the generic Worker IPC owns the subsequent D2H.
    return video.permute(0, 2, 3, 4, 1).to(dtype=torch.float32, copy=True)
