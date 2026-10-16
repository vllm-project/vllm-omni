# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from typing import BinaryIO
from uuid import UUID

import pytest

from vllm_omni.model_executor.models.step_audio2 import step_audio2_dit_trt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _write_artifact(name: str, temporary: Path, stream: BinaryIO, payload: bytes) -> None:
    if name.endswith(".onnx"):
        temporary.write_bytes(payload)
    else:
        stream.write(payload)


@pytest.mark.parametrize("name", ["dit_chunk.onnx", "dit_chunk.plan"])
def test_atomic_publication_supports_same_process_concurrency(tmp_path: Path, name: str) -> None:
    destination = tmp_path / name
    destination.write_bytes(b"existing")
    payloads = [b"first" * 1024, b"second" * 1024]
    ready = Barrier(3)
    release = Event()
    temporary_paths: list[Path] = []
    paths_lock = Lock()

    def publish(payload: bytes) -> None:
        def write(temporary: Path, stream: BinaryIO) -> None:
            _write_artifact(name, temporary, stream, payload)
            with paths_lock:
                temporary_paths.append(temporary)
            ready.wait(timeout=10)
            assert release.wait(timeout=10)

        step_audio2_dit_trt._publish_atomically(
            destination,
            write,
            requires_path_write=name.endswith(".onnx"),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish, payload) for payload in payloads]
        ready.wait(timeout=10)
        try:
            assert destination.read_bytes() == b"existing"
            assert len(set(temporary_paths)) == 2
            assert all(path.parent == destination.parent for path in temporary_paths)
        finally:
            release.set()
        for future in futures:
            future.result(timeout=10)

    assert destination.read_bytes() in payloads
    assert not list(tmp_path.glob(f"{destination.name}.tmp.*"))


def test_atomic_publication_does_not_remove_unowned_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "dit_chunk.onnx"
    destination.write_bytes(b"existing")
    collision_id = UUID(int=1)
    monkeypatch.setattr(step_audio2_dit_trt.uuid, "uuid4", lambda: collision_id)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.{collision_id.hex}")
    temporary.write_bytes(b"foreign")
    writer_called = False

    def write(path: Path, stream: BinaryIO) -> None:
        nonlocal writer_called
        del path, stream
        writer_called = True

    with pytest.raises(FileExistsError):
        step_audio2_dit_trt._publish_atomically(destination, write)

    assert not writer_called
    assert destination.read_bytes() == b"existing"
    assert temporary.read_bytes() == b"foreign"


def test_atomic_publication_cleans_up_after_writer_failure(tmp_path: Path) -> None:
    destination = tmp_path / "dit_chunk.onnx"
    destination.write_bytes(b"existing")
    write_error = RuntimeError("injected writer failure")
    temporary: Path | None = None

    def fail_write(path: Path, stream: BinaryIO) -> None:
        nonlocal temporary
        temporary = path
        stream.write(b"partial")
        raise write_error

    with pytest.raises(RuntimeError) as error:
        step_audio2_dit_trt._publish_atomically(destination, fail_write)

    assert error.value is write_error
    assert destination.read_bytes() == b"existing"
    assert temporary is not None
    assert not temporary.exists()


def test_atomic_publication_cleans_up_after_fstat_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "dit_chunk.onnx"
    fstat_error = OSError("injected fstat failure")
    owned_id = UUID(int=2)
    monkeypatch.setattr(step_audio2_dit_trt.uuid, "uuid4", lambda: owned_id)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.{owned_id.hex}")
    writer_called = False

    def fail_fstat(fd: int) -> os.stat_result:
        raise fstat_error

    def write(path: Path, stream: BinaryIO) -> None:
        nonlocal writer_called
        del path, stream
        writer_called = True

    monkeypatch.setattr(os, "fstat", fail_fstat)

    with pytest.raises(OSError) as error:
        step_audio2_dit_trt._publish_atomically(destination, write, requires_path_write=True)

    assert error.value is fstat_error
    assert not writer_called
    assert not destination.exists()
    assert not temporary.exists()


def test_atomic_publication_restores_permissions_when_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "dit_chunk.onnx"
    writer_error = RuntimeError("injected writer failure")
    temporary_id = UUID(int=3)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.{temporary_id.hex}")
    original_unlink = Path.unlink
    cleanup_attempted = False

    def fail_write(path: Path, stream: BinaryIO) -> None:
        del stream
        path.write_bytes(b"partial")
        raise writer_error

    def fail_unlink(path: Path, *, missing_ok: bool = False) -> None:
        nonlocal cleanup_attempted
        del missing_ok
        assert path == temporary
        cleanup_attempted = True
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(step_audio2_dit_trt.uuid, "uuid4", lambda: temporary_id)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    previous_umask = os.umask(0o222)
    try:
        with pytest.raises(RuntimeError) as error:
            step_audio2_dit_trt._publish_atomically(destination, fail_write, requires_path_write=True)
    finally:
        os.umask(previous_umask)

    assert error.value is writer_error
    assert cleanup_attempted
    assert stat.S_IMODE(temporary.stat().st_mode) == 0o444
    original_unlink(temporary)


@pytest.mark.parametrize(
    ("name", "requires_path_write", "expected_fstat_calls", "expected_fchmod_calls"),
    [("dit_chunk.onnx", True, 1, 2), ("dit_chunk.plan", False, 0, 0)],
)
def test_atomic_publication_preserves_permissions_with_restrictive_umask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    requires_path_write: bool,
    expected_fstat_calls: int,
    expected_fchmod_calls: int,
) -> None:
    destination = tmp_path / name
    original_fstat = os.fstat
    original_fchmod = os.fchmod
    fstat_calls = 0
    fchmod_calls = 0

    def track_fstat(fd: int) -> os.stat_result:
        nonlocal fstat_calls
        fstat_calls += 1
        if not requires_path_write:
            raise OSError("plan publication must not require fstat")
        return original_fstat(fd)

    def track_fchmod(fd: int, mode: int) -> None:
        nonlocal fchmod_calls
        fchmod_calls += 1
        if not requires_path_write:
            raise OSError("plan publication must not require fchmod")
        original_fchmod(fd, mode)

    monkeypatch.setattr(os, "fstat", track_fstat)
    monkeypatch.setattr(os, "fchmod", track_fchmod)
    previous_umask = os.umask(0o222)
    try:
        step_audio2_dit_trt._publish_atomically(
            destination,
            lambda temporary, stream: _write_artifact(name, temporary, stream, b"complete"),
            requires_path_write=requires_path_write,
        )
    finally:
        os.umask(previous_umask)

    assert destination.read_bytes() == b"complete"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o444
    assert fstat_calls == expected_fstat_calls
    assert fchmod_calls == expected_fchmod_calls


def test_atomic_publication_cleans_up_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "dit_chunk.plan"
    destination.write_bytes(b"existing")
    replace_error = OSError("injected replace failure")
    temporary: Path | None = None

    def write(path: Path, stream: BinaryIO) -> None:
        nonlocal temporary
        temporary = path
        stream.write(b"complete")

    def fail_replace(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        raise replace_error

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError) as error:
        step_audio2_dit_trt._publish_atomically(destination, write)

    assert error.value is replace_error
    assert destination.read_bytes() == b"existing"
    assert temporary is not None
    assert not temporary.exists()
