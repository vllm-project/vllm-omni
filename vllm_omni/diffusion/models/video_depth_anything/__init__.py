# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Video Depth Anything video depth estimation."""

from __future__ import annotations

from typing import Any

__all__ = ["VideoDepthAnythingPipeline"]


def __getattr__(name: str) -> Any:
    # Lazy, so the config resolver can import ``.checkpoints`` without the pipeline's dependencies.
    if name == "VideoDepthAnythingPipeline":
        from .pipeline_video_depth_anything import VideoDepthAnythingPipeline

        return VideoDepthAnythingPipeline
    raise AttributeError(name)
