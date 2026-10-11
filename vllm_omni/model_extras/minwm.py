# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request defaults for the minWM Action2V causal DMD pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults

# A minWM camera trajectory string such as "a*4,w*8,s*7"; see vllm_omni/diffusion/models/minwm/camera.py.
MINWM_EXTRA_BODY_PARAMS: frozenset[str] = frozenset({"trajectory"})
MINWM_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()


def get_minwm_video_generation_defaults(
    extra_body: Mapping[str, Any] | None = None,
) -> VideoGenerationDefaults:
    """The 77-frame 480x832 clip the Action2V generator was distilled on, at four DMD steps without CFG."""
    del extra_body
    return VideoGenerationDefaults(
        width=832,
        height=480,
        num_frames=77,
        num_inference_steps=4,
        fps=16,
        output="minwm_action2v_output.mp4",
        guidance_scale=1.0,
        default_negative_prompt=None,
    )
