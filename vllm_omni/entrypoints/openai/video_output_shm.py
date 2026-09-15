# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import atexit
import contextlib
import os
import threading
import time
from collections.abc import Iterator, Mapping
from multiprocessing import resource_tracker, shared_memory

import numpy as np
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.videos import VideoSharedMemoryHandle

logger = init_logger(__name__)

_LEASES: dict[str, float] = {}
_TRACKED_NAMES: dict[str, str] = {}
_LEASES_CONDITION = threading.Condition()
_LEASE_SWEEPER: threading.Thread | None = None


def _unlink_segment(name: str) -> None:
    segment = shared_memory.SharedMemory(name=name)
    try:
        segment.unlink()
    finally:
        segment.close()


def _unregister_missing_segment(name: str) -> None:
    with _LEASES_CONDITION:
        _LEASES.pop(name, None)
        tracked_name = _TRACKED_NAMES.pop(name, None)
        _LEASES_CONDITION.notify_all()
    if tracked_name is not None:
        resource_tracker.unregister(tracked_name, "shared_memory")


def _reap_expired_segments(*, now: float | None = None) -> int:
    current = time.monotonic() if now is None else now
    with _LEASES_CONDITION:
        expired = [name for name, deadline in _LEASES.items() if deadline <= current]
        for name in expired:
            del _LEASES[name]

    for name in expired:
        try:
            _unlink_segment(name)
        except FileNotFoundError:
            _unregister_missing_segment(name)
        except OSError:
            logger.warning("Failed to unlink expired video shared memory %s", name, exc_info=True)
            with _LEASES_CONDITION:
                if name in _TRACKED_NAMES:
                    _LEASES[name] = time.monotonic() + 1.0
                    _LEASES_CONDITION.notify_all()
        else:
            with _LEASES_CONDITION:
                _TRACKED_NAMES.pop(name, None)
    return len(expired)


def _lease_sweeper() -> None:
    while True:
        with _LEASES_CONDITION:
            while not _LEASES:
                _LEASES_CONDITION.wait()
            delay = min(_LEASES.values()) - time.monotonic()
            if delay > 0:
                _LEASES_CONDITION.wait(timeout=delay)
                continue
        _reap_expired_segments()


def _register_lease(name: str, tracked_name: str, ttl_seconds: int) -> None:
    global _LEASE_SWEEPER

    if type(ttl_seconds) is not int or ttl_seconds <= 0:
        raise ValueError(f"ttl_seconds must be a positive integer, got {ttl_seconds!r}")
    with _LEASES_CONDITION:
        _LEASES[name] = time.monotonic() + ttl_seconds
        _TRACKED_NAMES[name] = tracked_name
        if _LEASE_SWEEPER is None or not _LEASE_SWEEPER.is_alive():
            _LEASE_SWEEPER = threading.Thread(
                target=_lease_sweeper,
                name="video-output-shm-sweeper",
                daemon=True,
            )
            _LEASE_SWEEPER.start()
        _LEASES_CONDITION.notify_all()


def _forget_lease(name: str) -> None:
    with _LEASES_CONDITION:
        _LEASES.pop(name, None)
        _TRACKED_NAMES.pop(name, None)
        _LEASES_CONDITION.notify_all()


def export_video_frames_to_shm(frames: np.ndarray, *, ttl_seconds: int) -> VideoSharedMemoryHandle:
    if type(ttl_seconds) is not int or ttl_seconds <= 0:
        raise ValueError(f"ttl_seconds must be a positive integer, got {ttl_seconds!r}")
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must be uint8 with shape (frames, height, width, 3)")

    contiguous = np.ascontiguousarray(frames)
    if contiguous.nbytes <= 0:
        raise ValueError("frames must not be empty")

    segment = shared_memory.SharedMemory(create=True, size=contiguous.nbytes)
    committed = False
    try:
        target = np.ndarray(contiguous.shape, dtype=np.uint8, buffer=segment.buf)
        np.copyto(target, contiguous)
        handle = VideoSharedMemoryHandle(
            name=segment.name,
            shape=contiguous.shape,
            nbytes=contiguous.nbytes,
            expires_at=time.time() + ttl_seconds,
        )
        tracked_name = f"/{segment.name}" if os.name == "posix" else segment.name
        _register_lease(segment.name, tracked_name, ttl_seconds)
        committed = True
        return handle
    finally:
        if not committed:
            with contextlib.suppress(FileNotFoundError):
                segment.unlink()
            _forget_lease(segment.name)
        segment.close()


def _coerce_handle(handle: VideoSharedMemoryHandle | Mapping[str, object]) -> VideoSharedMemoryHandle:
    if isinstance(handle, VideoSharedMemoryHandle):
        return handle
    return VideoSharedMemoryHandle.model_validate(handle)


def release_video_frames(handle: VideoSharedMemoryHandle | Mapping[str, object]) -> None:
    """Idempotently unlink a same-host video handle without mapping it."""
    parsed = _coerce_handle(handle)
    try:
        _unlink_segment(parsed.name)
    except FileNotFoundError:
        _unregister_missing_segment(parsed.name)
    else:
        _forget_lease(parsed.name)


@contextlib.contextmanager
def borrowed_video_frames(
    handle: VideoSharedMemoryHandle | Mapping[str, object],
    *,
    borrow: bool = True,
) -> Iterator[np.ndarray]:
    """Map one same-host video handle; views are invalid after context exit."""
    parsed = _coerce_handle(handle)
    if parsed.expires_at <= time.time():
        raise ValueError("shared-memory video handle has expired")
    segment = shared_memory.SharedMemory(name=parsed.name)
    try:
        view = np.ndarray(parsed.shape, dtype=np.uint8, buffer=segment.buf[: parsed.nbytes])
        yield view if borrow else view.copy()
    finally:
        release_video_frames(parsed)
        with contextlib.suppress(BufferError):
            segment.close()


def _cleanup_leased_segments() -> None:
    with _LEASES_CONDITION:
        names = list(_LEASES)
        _LEASES.clear()
    for name in names:
        try:
            _unlink_segment(name)
        except FileNotFoundError:
            _unregister_missing_segment(name)
        except OSError:
            pass
        else:
            with _LEASES_CONDITION:
                _TRACKED_NAMES.pop(name, None)


atexit.register(_cleanup_leased_segments)
