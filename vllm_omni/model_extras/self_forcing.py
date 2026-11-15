# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request defaults for the Self-Forcing causal DMD pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults

SELF_FORCING_EXTRA_BODY_PARAMS: frozenset[str] = frozenset()
SELF_FORCING_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()


def get_self_forcing_video_generation_defaults(
    extra_body: Mapping[str, Any] | None = None,
) -> VideoGenerationDefaults:
    """The 81-frame 480x832 clip the checkpoint was distilled on, at four DMD steps without CFG."""
    del extra_body
    return VideoGenerationDefaults(
        width=832,
        height=480,
        num_frames=81,
        num_inference_steps=4,
        fps=16,
        output="self_forcing_output.mp4",
        guidance_scale=1.0,
        default_negative_prompt=None,
    )
