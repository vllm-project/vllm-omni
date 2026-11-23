# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request fields supported by the MAGI-2 Preview pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults

MAGI2_EXTRA_BODY_PARAMS = frozenset(
    {
        "seconds",
        "resolution",
        "image_path",
        "output_width",
        "output_height",
        "deterministic",
    }
)
MAGI2_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()


def magi2_preserves_reference_image_size(
    *,
    model: str | None,
    revision: str | None = None,
) -> bool:
    """MAGI-2 owns checkpoint-aligned reference resize and padding."""
    del model, revision
    return True


def get_magi2_video_generation_defaults(
    extra_body: Mapping[str, Any] | None = None,
) -> VideoGenerationDefaults:
    resolution = str((extra_body or {}).get("resolution", "540p")).lower()
    if resolution == "272p":
        width, height = 448, 256
    elif resolution == "540p":
        width, height = 896, 512
    else:
        raise ValueError("MAGI-2 Preview resolution must be '272p' or '540p'.")
    return VideoGenerationDefaults(
        width=width,
        height=height,
        num_frames=125,
        num_inference_steps=100,
        fps=12.5,
        duration_seconds=10.0,
        output="magi2_output.mp4",
        default_negative_prompt=None,
    )
