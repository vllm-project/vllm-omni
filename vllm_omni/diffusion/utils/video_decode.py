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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import PIL.Image

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_REQUEST_VIDEO_DECODE_CACHE_BYTES = 256 * 1024 * 1024


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


@dataclass(frozen=True)
class _RequestVideoDecodeKey:
    path: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    max_frames: int | None
    keep: Literal["first", "last"]
    decoder: str = "imageio.v3"
    output_format: str = "uint8-rgb-v1"


class RequestVideoDecodeContext:
    """Request-local bounded reuse for identical path-video decode operations."""

    def __init__(
        self,
        *,
        max_bytes: int = DEFAULT_REQUEST_VIDEO_DECODE_CACHE_BYTES,
        before_access: Callable[[], None] | None = None,
        after_access: Callable[[], None] | None = None,
    ) -> None:
        if max_bytes < 0:
            raise ValueError(f"Video decode cache max_bytes must be non-negative, got {max_bytes}.")
        self.max_bytes = int(max_bytes)
        self.before_access = before_access
        self.after_access = after_access
        self._frames: dict[_RequestVideoDecodeKey, list[np.ndarray]] = {}
        self._retained_bytes = 0
        self._closed = False
        self.decode_count = 0
        self.cache_hit_count = 0
        self.decoded_bytes = 0

    @property
    def retained_bytes(self) -> int:
        return self._retained_bytes

    def decode_path_video_frames(
        self,
        path: str | Path,
        *,
        max_frames: int | None = None,
        keep: Literal["first", "last"] = "first",
    ) -> list[np.ndarray]:
        if self._closed:
            raise RuntimeError("Cannot decode video with a closed request context.")

        media_path = Path(path)
        try:
            resolved_path = media_path.resolve(strict=True)
            stat = resolved_path.stat()
        except FileNotFoundError:
            return decode_path_video_frames(media_path, max_frames=max_frames, keep=keep)

        limit = None if max_frames is None else int(max_frames)
        key = _RequestVideoDecodeKey(
            path=os.fspath(resolved_path),
            device=stat.st_dev,
            inode=stat.st_ino,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            max_frames=limit,
            keep=keep,
        )
        if self.before_access is not None:
            self.before_access()
        cached = self._frames.get(key)
        if cached is not None:
            self.cache_hit_count += 1
            if self.after_access is not None:
                self.after_access()
            return cached

        frames = decode_path_video_frames(media_path, max_frames=limit, keep=keep)
        decoded_bytes = sum(frame.nbytes for frame in frames)
        self.decode_count += 1
        self.decoded_bytes += decoded_bytes
        if decoded_bytes <= self.max_bytes - self._retained_bytes:
            self._frames[key] = frames
            self._retained_bytes += decoded_bytes
        if self.after_access is not None:
            self.after_access()
        return frames

    def close(self) -> None:
        self._frames.clear()
        self._retained_bytes = 0
        self._closed = True

    def __enter__(self) -> RequestVideoDecodeContext:
        if self._closed:
            raise RuntimeError("Cannot re-enter a closed request video decode context.")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
