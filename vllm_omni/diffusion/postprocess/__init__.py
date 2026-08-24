# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion post-processing helpers."""

from vllm_omni.diffusion.postprocess.rife_interpolator import (
    FrameInterpolator,
    interpolate_video_tensor,
)

__all__ = ["FrameInterpolator", "interpolate_video_tensor"]
