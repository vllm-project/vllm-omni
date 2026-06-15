# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frame sampling strategies for streaming video input.

Three strategies are supported:

- ``uniform``: Sample frames evenly across the entire buffer (first/last
  anchored). Best for understanding the full timeline.
- ``latest_frames``: Take only the most recent *N* frames. Best for
  reacting to "what just happened".
- ``latest_seconds``: Take all frames within the last *T* seconds.
  Best for time-bounded context windows.
"""

from __future__ import annotations

import time as _time
from typing import Any, Literal

from vllm.logger import init_logger

logger = init_logger(__name__)

SamplingStrategy = Literal["uniform", "latest_frames", "latest_seconds"]

_DEFAULT_NUM_FRAMES = 8
_DEFAULT_WINDOW_SECONDS = 5.0


def sample_frames(
    frame_buffer: list[Any],
    *,
    strategy: SamplingStrategy = "uniform",
    num_frames: int = _DEFAULT_NUM_FRAMES,
    window_seconds: float = _DEFAULT_WINDOW_SECONDS,
    frame_timestamps: list[float] | None = None,
) -> list[Any]:
    """Sample frames from *frame_buffer* using the given strategy.

    Args:
        frame_buffer: Ordered list of frame objects (newest at the end).
        strategy: One of ``"uniform"``, ``"latest_frames"``,
            ``"latest_seconds"``.
        num_frames: Target number of frames for ``uniform`` and
            ``latest_frames``; cap for ``latest_seconds`` when too many
            frames fall within the window.
        window_seconds: Time window in seconds (only for
            ``latest_seconds``).
        frame_timestamps: Monotonic timestamps aligned with
            *frame_buffer* (required for ``latest_seconds``).

    Returns:
        A list of sampled frame objects (subset of *frame_buffer*).
    """
    if not frame_buffer:
        return []

    if strategy == "uniform":
        return _sample_uniform(frame_buffer, num_frames)
    if strategy == "latest_frames":
        return _sample_latest_frames(frame_buffer, num_frames)
    if strategy == "latest_seconds":
        return _sample_latest_seconds(
            frame_buffer,
            num_frames,
            window_seconds,
            frame_timestamps,
        )

    # Should never reach here if Pydantic validates the Literal,
    # but be defensive.
    logger.warning("Unknown sampling strategy %r, falling back to uniform", strategy)
    return _sample_uniform(frame_buffer, num_frames)


def _sample_uniform(frames: list[Any], num_frames: int) -> list[Any]:
    """First/last anchored uniform stride sampling."""
    n = len(frames)
    if n <= num_frames:
        return list(frames)
    indices = [round(i * (n - 1) / (num_frames - 1)) for i in range(num_frames)]
    return [frames[i] for i in indices]


def _sample_latest_frames(frames: list[Any], num_frames: int) -> list[Any]:
    """Take the most recent *num_frames* frames."""
    return list(frames[-num_frames:])


def _sample_latest_seconds(
    frames: list[Any],
    num_frames: int,
    window_seconds: float,
    frame_timestamps: list[float] | None,
) -> list[Any]:
    """Take frames within the last *window_seconds*.

    Falls back to the single most recent frame if timestamps are
    unavailable or the window is empty.  If the window contains more
    than *num_frames*, applies uniform sub-sampling.
    """
    if frame_timestamps is None or len(frame_timestamps) != len(frames):
        logger.warning("latest_seconds requires frame_timestamps; falling back to latest single frame")
        return [frames[-1]]

    now = _time.monotonic()
    cutoff = now - window_seconds

    recent = [f for f, ts in zip(frames, frame_timestamps) if ts >= cutoff]

    if not recent:
        # Window is empty (all frames are older than cutoff).
        return [frames[-1]]

    if len(recent) > num_frames:
        return _sample_uniform(recent, num_frames)

    return recent
