# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request fields supported by the MAGI-2 pipeline."""

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
        "refiner_num_inference_steps",
    }
)
MAGI2_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()

# Output (width, height, num_frames, fps) per resolution. 1080p refines a 540p
# Preview and doubles its frame rate over the same 10 seconds.
_MAGI2_OUTPUTS: dict[str, tuple[int, int, int, float]] = {
    "272p": (448, 256, 125, 12.5),
    "540p": (896, 512, 125, 12.5),
    "1080p": (1920, 1088, 249, 25.0),
}
_MAGI2_RESOLUTION_BY_SIZE = {(width, height): name for name, (width, height, _, _) in _MAGI2_OUTPUTS.items()}


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
    *,
    requested_size: tuple[int, int] | None = None,
) -> VideoGenerationDefaults:
    """Return the defaults of the requested resolution.

    A request without ``resolution`` takes the preset of its requested
    ``(width, height)``; any other size keeps the 540p defaults, and the
    pipeline rejects it.
    """
    resolution_value = (extra_body or {}).get("resolution")
    if resolution_value is not None:
        resolution = str(resolution_value).lower()
    elif requested_size is not None:
        resolution = _MAGI2_RESOLUTION_BY_SIZE.get(requested_size, "540p")
    else:
        resolution = "540p"
    try:
        width, height, num_frames, fps = _MAGI2_OUTPUTS[resolution]
    except KeyError:
        raise ValueError("MAGI-2 resolution must be '272p', '540p' or '1080p'.") from None
    return VideoGenerationDefaults(
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=100,
        fps=fps,
        duration_seconds=10.0,
        output="magi2_output.mp4",
        default_negative_prompt=None,
    )
