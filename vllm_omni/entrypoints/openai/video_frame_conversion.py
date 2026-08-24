# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Planar video frame conversion helpers for MP4 response encoding."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import DTypeLike

if TYPE_CHECKING:
    import av


_planar_scratch_local = threading.local()


def _validate_frame_conversion_workers(value: object) -> int:
    """Validate a positive CPU frame-conversion worker count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("frame_conversion_workers must be a positive integer.")
    return value


def _planar_scratch_dtype(common_dtype: np.dtype) -> np.dtype:
    common_dtype = np.dtype(common_dtype)
    if np.issubdtype(common_dtype, np.floating):
        return common_dtype
    return np.dtype(np.float64)


def _planar_channel_scratch(height: int, width: int, common_dtype: np.dtype) -> np.ndarray:
    """Return reusable per-thread scratch storage for one planar channel."""
    scratch_dtype = _planar_scratch_dtype(common_dtype)
    scratch = getattr(_planar_scratch_local, "channel", None)
    if scratch is None or scratch.shape != (height, width) or scratch.dtype != scratch_dtype:
        scratch = np.empty((height, width), dtype=scratch_dtype)
        _planar_scratch_local.channel = scratch
    return cast(np.ndarray, scratch)


def _clear_planar_channel_scratch() -> None:
    if hasattr(_planar_scratch_local, "channel"):
        delattr(_planar_scratch_local, "channel")


def _build_planar_video_frame(frame: np.ndarray, common_dtype: np.dtype) -> av.VideoFrame:
    """Build one quantized GBR PyAV frame using thread-local scratch storage."""
    import av

    height, width = frame.shape[:2]
    scratch = None if frame.dtype == np.uint8 else _planar_channel_scratch(height, width, common_dtype)
    av_frame = av.VideoFrame(width, height, format="gbrp")
    for plane, channel in zip(av_frame.planes, (1, 2, 0)):
        if plane.height < height or plane.line_size < width:
            raise ValueError("PyAV video plane is smaller than the requested frame dimensions.")
        plane_view = np.frombuffer(
            memoryview(plane),
            dtype=np.uint8,
            count=plane.height * plane.line_size,
        ).reshape(plane.height, plane.line_size)
        plane_view.fill(0)
        if frame.dtype == np.uint8:
            plane_view[:height, :width] = frame[..., channel]
        else:
            scratch_buffer = cast(np.ndarray, scratch)
            np.copyto(scratch_buffer, frame[..., channel], casting="unsafe")
            np.clip(scratch_buffer, 0.0, 1.0, out=scratch_buffer)
            scratch_buffer *= 255.0
            np.rint(scratch_buffer, out=scratch_buffer)
            plane_view[:height, :width] = scratch_buffer
    return av_frame


def _iter_baseline_planar_video_frames(
    frames: list[np.ndarray],
    common_dtype: np.dtype,
) -> Iterator[av.VideoFrame]:
    """Yield planar PyAV frames while retaining only one channel scratch buffer."""
    import av

    height, width = frames[0].shape[:2]
    scratch_dtype: DTypeLike = np.float64 if np.issubdtype(common_dtype, np.bool_) else common_dtype
    scratch = None if common_dtype == np.uint8 else np.empty((height, width), dtype=scratch_dtype)

    for frame in frames:
        av_frame = av.VideoFrame(width, height, format="gbrp")
        for plane, channel in zip(av_frame.planes, (1, 2, 0)):
            if plane.height < height or plane.line_size < width:
                raise ValueError("PyAV video plane is smaller than the requested frame dimensions.")
            plane_view = np.frombuffer(  # type: ignore[call-overload]  # VideoPlane exposes the buffer protocol at runtime.
                plane,
                dtype=np.uint8,
                count=plane.height * plane.line_size,
            ).reshape(plane.height, plane.line_size)
            plane_view.fill(0)
            if frame.dtype == np.uint8:
                plane_view[:height, :width] = frame[..., channel]
            else:
                scratch_buffer = cast(np.ndarray, scratch)
                np.copyto(scratch_buffer, frame[..., channel], casting="unsafe")
                np.clip(scratch_buffer, 0.0, 1.0, out=scratch_buffer)
                scratch_buffer *= 255.0
                np.rint(scratch_buffer, out=scratch_buffer)
                plane_view[:height, :width] = scratch_buffer
        yield av_frame


def _iter_planar_video_frames(
    frames: list[np.ndarray],
    common_dtype: np.dtype,
    worker_count: int = 1,
) -> Generator[av.VideoFrame, None, None]:
    """Yield planar PyAV frames in order with bounded conversion parallelism."""
    worker_count = min(len(frames), _validate_frame_conversion_workers(worker_count))
    if worker_count <= 1:
        try:
            for frame in frames:
                yield _build_planar_video_frame(frame, common_dtype)
        finally:
            _clear_planar_channel_scratch()
        return

    executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="video-planar")
    pending: deque[Future[av.VideoFrame]] = deque()
    max_pending = 2 * worker_count
    frame_iter = iter(frames)
    try:
        for frame in frame_iter:
            pending.append(executor.submit(_build_planar_video_frame, frame, common_dtype))
            if len(pending) == max_pending:
                break

        while pending:
            converted_frame = pending.popleft().result()
            try:
                next_input = next(frame_iter)
            except StopIteration:
                pass
            else:
                pending.append(executor.submit(_build_planar_video_frame, next_input, common_dtype))
            yield converted_frame
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
