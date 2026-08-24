# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion post-processing helpers."""

from vllm_omni.diffusion.postprocess.rife_interpolator import (
    FrameInterpolator,
    interpolate_video_tensor,
)
from vllm_omni.diffusion.postprocess.video import prepare_video_for_transport

__all__ = [
    "FrameInterpolator",
    "interpolate_video_tensor",
    "prepare_video_for_transport",
]
