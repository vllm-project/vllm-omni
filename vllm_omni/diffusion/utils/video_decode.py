# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decode video file paths to RGB frames.

Path suffix checks and bounded first/last-N imageio decoding. Model-specific
conditioning and decode-budget stay in each pipeline's preprocess layer.

Currently used by Cosmos3 V2V preprocess and transfer ``control_path``.
Unifying with ``OmniVideoBackend`` / ``_decode_video_bytes`` is a follow-up.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any, Literal

import numpy as np
import PIL.Image

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def is_video_file_path(value: Any) -> bool:
    """True when ``value`` is a filesystem path with a known video suffix.

    Serving may leave uploaded references as temp paths such as
    ``/tmp/vllm_omni_video_reference_*.mp4`` instead of decoded frames.
    """
    if not isinstance(value, str | os.PathLike):
        return False
    suffix = os.path.splitext(os.fspath(value))[1].lower()
    return suffix in VIDEO_EXTENSIONS


def _frame_to_uint8_rgb(value: Any) -> np.ndarray:
    if isinstance(value, PIL.Image.Image):
        return np.array(value.convert("RGB"), dtype=np.uint8, copy=True)
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.transpose(array[:3], (1, 2, 0))
        if np.issubdtype(array.dtype, np.floating):
            if array.size and (array.min() < 0.0 or array.max() > 1.0):
                array = np.clip(array, -1.0, 1.0) * 0.5 + 0.5
            array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return array[..., :3].astype(np.uint8)
    raise TypeError(f"Video decode expected an RGB frame, got {type(value)!r}.")


def decode_path_video_frames(
    path: str | Path,
    *,
    max_frames: int | None = None,
    keep: Literal["first", "last"] = "first",
) -> list[np.ndarray]:
    """Decode a video file to RGB uint8 frames (H, W, C).

    ``keep='first'`` stops after ``max_frames``. ``keep='last'`` still scans the
    file and retains only the tail window.
    """
    media_path = Path(path)
    if not media_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {media_path}")
    if keep not in {"first", "last"}:
        raise ValueError("Video keep must be either 'first' or 'last'.")
    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError("Video max_frames must be positive.")
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "Video path decoding requires imageio. Install imageio[ffmpeg] or provide decoded frames."
        ) from exc

    limit = None if max_frames is None else int(max_frames)
    if keep == "last" and limit is not None:
        window: deque[np.ndarray] = deque(maxlen=limit)
        for frame in iio.imiter(media_path):
            window.append(_frame_to_uint8_rgb(frame))
        frames = list(window)
    else:
        frames = []
        for frame in iio.imiter(media_path):
            frames.append(_frame_to_uint8_rgb(frame))
            if limit is not None and len(frames) >= limit:
                break
    if not frames:
        raise ValueError(f"Video path produced no frames: {media_path}")
    return frames
