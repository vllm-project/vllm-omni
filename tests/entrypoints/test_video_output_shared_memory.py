# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import subprocess
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai import video_output_shm as shm_module
from vllm_omni.entrypoints.openai.protocol.videos import VideoData, VideoSharedMemoryHandle
from vllm_omni.entrypoints.openai.serving_video import _is_loopback_host
from vllm_omni.entrypoints.openai.video_output_shm import (
    _reap_expired_segments,
    borrowed_video_frames,
    export_video_frames_to_shm,
    release_video_frames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _frames() -> np.ndarray:
    return np.arange(2 * 4 * 6 * 3, dtype=np.uint8).reshape(2, 4, 6, 3)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("10.0.0.2", False),
        (None, False),
    ],
)
def test_shared_memory_host_gate(host: str | None, expected: bool) -> None:
    assert _is_loopback_host(host) is expected


def test_shared_memory_handle_validates_shape_and_size() -> None:
    with pytest.raises(ValidationError, match="nbytes must match"):
        VideoSharedMemoryHandle(
            name="segment",
            shape=(2, 4, 6, 3),
            nbytes=1,
            expires_at=1,
        )


def test_expired_handle_is_rejected_before_mapping() -> None:
    handle = VideoSharedMemoryHandle(
        name="expired-segment",
        shape=(2, 4, 6, 3),
        nbytes=2 * 4 * 6 * 3,
        expires_at=time.time() - 1,
    )

    with pytest.raises(ValueError, match="has expired"):
        with borrowed_video_frames(handle):
            pass


def test_video_data_rejects_multiple_transport_payloads() -> None:
    with pytest.raises(ValidationError, match="only one video transport payload"):
        VideoData(b64_json="payload", url="https://example.com/video.mp4")


def test_export_rejects_invalid_ttl_before_allocating(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_allocation(*_args: object, **_kwargs: object) -> None:
        pytest.fail("shared memory was allocated")

    monkeypatch.setattr(shared_memory, "SharedMemory", fail_allocation)

    with pytest.raises(ValueError, match="ttl_seconds must be a positive integer"):
        export_video_frames_to_shm(_frames(), ttl_seconds=0)


def test_export_rejects_non_uint8_video() -> None:
    with pytest.raises(ValueError, match="uint8"):
        export_video_frames_to_shm(_frames().astype(np.float32), ttl_seconds=10)


def test_export_failure_unlinks_the_allocated_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    original_shared_memory = shared_memory.SharedMemory
    created_names: list[str] = []

    def tracked_shared_memory(*args, **kwargs):
        segment = original_shared_memory(*args, **kwargs)
        if kwargs.get("create"):
            created_names.append(segment.name)
        return segment

    def fail_copy(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("copy failed")

    monkeypatch.setattr(shared_memory, "SharedMemory", tracked_shared_memory)
    monkeypatch.setattr(np, "copyto", fail_copy)

    with pytest.raises(RuntimeError, match="copy failed"):
        export_video_frames_to_shm(_frames(), ttl_seconds=10)

    assert len(created_names) == 1
    with pytest.raises(FileNotFoundError):
        original_shared_memory(name=created_names[0])


def test_release_is_idempotent() -> None:
    handle = export_video_frames_to_shm(_frames(), ttl_seconds=10)

    release_video_frames(handle)
    release_video_frames(handle)

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.name)


def test_borrowed_video_is_a_shared_view_and_releases_the_segment() -> None:
    handle = export_video_frames_to_shm(_frames(), ttl_seconds=10)

    with borrowed_video_frames(handle) as borrowed:
        other = shared_memory.SharedMemory(name=handle.name)
        try:
            mapped = np.ndarray(handle.shape, dtype=np.uint8, buffer=other.buf)
            mapped[0, 0, 0, 0] = 231
            assert borrowed[0, 0, 0, 0] == 231
            assert borrowed.base is not None
        finally:
            other.close()

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.name)


def test_copy_mode_survives_context_exit() -> None:
    expected = _frames()
    handle = export_video_frames_to_shm(expected, ttl_seconds=10)

    with borrowed_video_frames(handle, borrow=False) as copied:
        survivor = copied

    np.testing.assert_array_equal(survivor, expected)
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.name)


def test_expired_unclaimed_segment_is_reaped() -> None:
    handle = export_video_frames_to_shm(_frames(), ttl_seconds=10)

    assert _reap_expired_segments(now=float("inf")) >= 1
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.name)


def test_failed_expiry_cleanup_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = export_video_frames_to_shm(_frames(), ttl_seconds=10)
    original_unlink = shm_module._unlink_segment

    def fail_unlink(name: str) -> None:
        raise OSError(f"cannot unlink {name}")

    monkeypatch.setattr(shm_module, "_unlink_segment", fail_unlink)
    assert _reap_expired_segments(now=float("inf")) >= 1
    assert handle.name in shm_module._LEASES

    monkeypatch.setattr(shm_module, "_unlink_segment", original_unlink)
    assert _reap_expired_segments(now=float("inf")) >= 1
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.name)


@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="requires POSIX shared memory")
def test_creator_crash_does_not_leave_named_segment() -> None:
    script = """
import time
import numpy as np
from vllm_omni.entrypoints.openai.video_output_shm import export_video_frames_to_shm
handle = export_video_frames_to_shm(np.zeros((2, 4, 6, 3), dtype=np.uint8), ttl_seconds=3600)
print(f'HANDLE {handle.name}', flush=True)
time.sleep(3600)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    name = ""
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            line = process.stdout.readline()
            if line.startswith("HANDLE "):
                name = line.split(maxsplit=1)[1].strip()
                break
        assert name
        segment_path = Path("/dev/shm") / name
        assert segment_path.exists()

        process.kill()
        process.wait(timeout=10)
        deadline = time.time() + 10
        while segment_path.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert not segment_path.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        if name and (Path("/dev/shm") / name).exists():
            segment = shared_memory.SharedMemory(name=name)
            segment.unlink()
            segment.close()
