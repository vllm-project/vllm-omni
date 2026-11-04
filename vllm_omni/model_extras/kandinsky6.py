# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request fields and example defaults for the Kandinsky 6 TI2VA pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults

# Forwarded to ``OmniDiffusionSamplingParams.extra_args`` by the shared examples
# and the OpenAI video API's ``extra_body``.
KANDINSKY6_EXTRA_BODY_PARAMS = frozenset(
    {
        # bool, default True: also sample the audio track (TI2VA). False -> video only.
        "sample_audio",
        # "pretrain" (text-only) | "tail_cond_first_frame" (reference image).
        "visual_cond_scheme",
    }
)
KANDINSKY6_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()


def get_kandinsky6_video_generation_defaults(
    extra_body: Mapping[str, Any] | None = None,
) -> VideoGenerationDefaults:
    """Kandinsky 6 Pro production geometry: 480x864, 125 frames @ 24 fps, 50 steps, CFG 5.

    These are defaults, not a fixed contract: any ``4k+1`` frame count and
    16-divisible resolution can be requested explicitly, so ``duration_seconds``
    is left unset (the serving layer only pins ``num_frames`` for
    fixed-duration pipelines such as MAGI-2).
    """
    del extra_body
    return VideoGenerationDefaults(
        width=864,
        height=480,
        num_frames=125,
        num_inference_steps=50,
        fps=24.0,
        guidance_scale=5.0,
        output="kandinsky6_output.mp4",
        # ``None`` lets the pipeline apply its own tuned negative prompt.
        default_negative_prompt=None,
        duration_seconds=None,
    )
