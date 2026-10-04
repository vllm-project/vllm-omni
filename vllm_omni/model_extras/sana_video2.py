# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request defaults for the official SANA-Video 2.0 5B release."""

from collections.abc import Mapping
from typing import Any

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults

SANA_VIDEO2_EXTRA_BODY_PARAMS = frozenset({"motion_score", "high_motion", "flow_shift"})


def get_sana_video2_video_generation_defaults(
    extra_body: Mapping[str, Any] | None = None,
) -> VideoGenerationDefaults:
    return VideoGenerationDefaults(
        width=1280,
        height=736,
        num_frames=193,
        fixed_num_frames=False,
        num_inference_steps=50,
        fps=24,
        guidance_scale=8.0,
        flow_shift=12.0,
        dimension_multiple=32,
        default_negative_prompt=None,
        output="sana_video2_output.mp4",
    )
