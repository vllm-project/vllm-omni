# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 unit tests for vllm_omni.entrypoints.stage_utils IPC and utility functions."""

import json
import os
import tempfile
from multiprocessing import shared_memory as _shm

import pytest

from vllm_omni.entrypoints.stage_utils import (
    OmniStageTaskType,
    SHUTDOWN_TASK,
    _ensure_parent_dir,
    _to_dict,
    append_jsonl,
    is_profiler_task,
    shm_read_bytes,
    shm_write_bytes,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _force_unlink_shm(meta: dict) -> None:
    """Best-effort cleanup if a SHM segment was created but not consumed by shm_read_bytes."""
    name = meta.get("name")
    if not name:
        return
    try:
        shm = _shm.SharedMemory(name=name)
    except FileNotFoundError:
        return
    try:
        shm.unlink()
    finally:
        try:
            shm.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OmniStageTaskType & helpers
# ---------------------------------------------------------------------------
class TestOmniStageTaskType:
    """Tests for OmniStageTaskType enum and related helpers."""

    def test_enum_values(self) -> None:
        """All expected task types exist with correct string values."""
        assert OmniStageTaskType.GENERATE.value == "generate"
        assert OmniStageTaskType.ABORT.value == "abort"
        assert OmniStageTaskType.SHUTDOWN.value == "shutdown"
        assert OmniStageTaskType.PROFILER_START.value == "profiler_start"
        assert OmniStageTaskType.PROFILER_STOP.value == "profiler_stop"

    def test_shutdown_task_constant(self) -> None:
        """SHUTDOWN_TASK dict contains the correct type."""
        assert SHUTDOWN_TASK["type"] is OmniStageTaskType.SHUTDOWN

    def test_is_profiler_task_true(self) -> None:
        """is_profiler_task returns True for profiler task types."""
        assert is_profiler_task(OmniStageTaskType.PROFILER_START) is True
        assert is_profiler_task(OmniStageTaskType.PROFILER_STOP) is True

    def test_is_profiler_task_false(self) -> None:
        """is_profiler_task returns False for non-profiler task types."""
        assert is_profiler_task(OmniStageTaskType.GENERATE) is False
        assert is_profiler_task(OmniStageTaskType.ABORT) is False
        assert is_profiler_task(OmniStageTaskType.SHUTDOWN) is False


# ---------------------------------------------------------------------------
# Shared-memory round-trip
# ---------------------------------------------------------------------------
class TestSharedMemory:
    """Tests for shm_write_bytes / shm_read_bytes round-trip."""

    def test_roundtrip(self) -> None:
        """Data written to SHM can be read back identically."""
        payload = b"hello shared memory"
        meta = shm_write_bytes(payload)
        try:
            assert "name" in meta
            assert meta["size"] == len(payload)
            recovered = shm_read_bytes(meta)
            assert recovered == payload
        finally:
            _force_unlink_shm(meta)

    def test_roundtrip_large(self) -> None:
        """Larger payloads survive the round-trip."""
        payload = os.urandom(1024 * 64)
        meta = shm_write_bytes(payload)
        try:
            recovered = shm_read_bytes(meta)
            assert recovered == payload
        finally:
            _force_unlink_shm(meta)

    def test_empty_payload_raises(self) -> None:
        """SharedMemory(create=True, size=0) is invalid in CPython — expect ValueError."""
        with pytest.raises(ValueError):
            shm_write_bytes(b"")


# ---------------------------------------------------------------------------
# append_jsonl
# ---------------------------------------------------------------------------
class TestAppendJsonl:
    """Tests for the append_jsonl helper."""

    def test_creates_file_and_writes_record(self) -> None:
        """A new JSONL file is created with the correct record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.jsonl")
            append_jsonl(path, {"event": "test", "value": 42})

            with open(path, encoding="utf-8") as f:
                line = f.readline().strip()
            record = json.loads(line)
            assert record["event"] == "test"
            assert record["value"] == 42

    def test_appends_multiple_records(self) -> None:
        """Multiple calls append multiple lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.jsonl")
            append_jsonl(path, {"a": 1})
            append_jsonl(path, {"a": 2})
            append_jsonl(path, {"a": 3})

            with open(path, encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            assert len(lines) == 3
            assert json.loads(lines[2])["a"] == 3

    def test_creates_parent_dirs(self) -> None:
        """Parent directories are created when they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "log.jsonl")
            append_jsonl(path, {"ok": True})
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# _ensure_parent_dir
# ---------------------------------------------------------------------------
class TestEnsureParentDir:
    """Tests for _ensure_parent_dir helper."""

    def test_creates_nested_dirs(self) -> None:
        """Nested parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "c", "file.txt")
            _ensure_parent_dir(path)
            assert os.path.isdir(os.path.join(tmpdir, "a", "b", "c"))

    def test_no_error_on_existing(self) -> None:
        """No error when parent directory already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "file.txt")
            _ensure_parent_dir(path)


# ---------------------------------------------------------------------------
# _to_dict
# ---------------------------------------------------------------------------
class TestToDict:
    """Tests for the _to_dict conversion helper."""

    def test_dict_passthrough(self) -> None:
        """A plain dict is returned as-is."""
        d = {"key": "value", "num": 42}
        result = _to_dict(d)
        assert result == d

    def test_non_dict_fallback(self) -> None:
        """Scalars fall through OmegaConf conversion and dict() coercion to empty dict."""
        # _to_dict tries _omega_to_dict then dict(x); int is not dict-convertible → {}.
        result = _to_dict(42)
        assert result == {}

    def test_list_of_tuples(self) -> None:
        """A list of 2-tuples is convertible to dict."""
        result = _to_dict([("a", 1), ("b", 2)])
        assert result == {"a": 1, "b": 2}
