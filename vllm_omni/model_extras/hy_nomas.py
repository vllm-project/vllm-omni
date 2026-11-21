# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
from collections.abc import Mapping

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults

HY_NOMAS_EXTRA_BODY_PARAMS = frozenset(
    {
        "segment_frame_boundaries",
        "camera_trajectory",
        "segment_has_camera",
        "flow_shift",
        "duration",
        "generate_sound",
        "session_id",
        "ar_diffusion_tick",
        "reset",
        "close_session",
    }
)


def get_hy_nomas_video_generation_defaults(
    extra_body: Mapping[str, object] | None = None,
) -> VideoGenerationDefaults:
    num_frames = 121
    duration = None if extra_body is None else extra_body.get("duration")
    if duration is not None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float, str)):
            raise ValueError("Hy-Nomas duration must be a positive finite number of seconds.")
        try:
            seconds = float(duration)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Hy-Nomas duration must be a positive finite number of seconds.") from exc
        if not math.isfinite(seconds) or not 0 < seconds < (2**63 - 1) / 24:
            raise ValueError("Hy-Nomas duration must be a positive finite number of seconds.")
        intervals = seconds * 24
        num_frames = round(intervals) + 1
        if (
            num_frames < 5
            or (num_frames - 1) % 4
            or not math.isclose(intervals, num_frames - 1, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise ValueError("Hy-Nomas duration must span a positive multiple of four frame intervals at 24 fps.")
    return VideoGenerationDefaults(
        width=1280,
        height=704,
        num_frames=num_frames,
        num_inference_steps=8,
        guidance_scale=1.0,
        flow_shift=5.0,
        fps=24.0,
        dimension_multiple=32,
        output="hy_nomas_output.mp4",
    )
