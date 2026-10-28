# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Example and serving defaults for the PAN 2 text-to-video / image-to-video pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults


def get_pan2_video_generation_defaults(
    extra_body: Mapping[str, Any] | None = None,
) -> VideoGenerationDefaults:
    """PAN 2's default geometry: 704x1248, 313 frames (13 s) at 24 fps, 50 steps, CFG 2.5.

    Any size that is a multiple of 32 and any ``4k+1`` frame count can be requested explicitly.
    """
    del extra_body
    return VideoGenerationDefaults(
        width=1248,
        height=704,
        num_frames=313,
        num_inference_steps=50,
        fps=24.0,
        guidance_scale=2.5,
        output="pan2_output.mp4",
        dimension_multiple=32,
        # ``None`` lets the pipeline apply its own negative prompt default.
        default_negative_prompt=None,
    )
