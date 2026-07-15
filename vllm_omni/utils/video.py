# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight video frame normalization helpers."""

import math
from collections.abc import Sequence

import numpy as np
from PIL import Image


def normalize_decoded_video_frames(
    video_input: Sequence[Image.Image],
    *,
    default_fps: float,
) -> tuple[np.ndarray, float]:
    """Convert decoded image frames to a contiguous THWC uint8 video array."""
    if not video_input:
        raise ValueError("video_edit received an empty decoded video frame sequence.")

    fps = getattr(video_input, "fps", None)
    if fps is None:
        fps = getattr(video_input, "frame_rate", None)
    try:
        frame_rate = float(fps) if fps is not None else default_fps
    except (TypeError, ValueError):
        frame_rate = default_fps
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        frame_rate = default_fps

    frames = []
    frame_size = None
    for index, frame in enumerate(video_input):
        if not isinstance(frame, Image.Image):
            raise ValueError(
                f"video_edit decoded video frame at index {index} must be a PIL.Image.Image, got {type(frame)}."
            )
        if frame_size is None:
            frame_size = frame.size
        elif frame.size != frame_size:
            raise ValueError(
                "video_edit decoded video frames must have identical dimensions; "
                f"frame 0 is {frame_size}, frame {index} is {frame.size}."
            )
        frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))

    return np.ascontiguousarray(np.stack(frames)), frame_rate
