# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SharedMemoryConnector focusing on TP / CFG / metadata fallback."""

import fcntl
import os
import subprocess
import sys
import time
from multiprocessing import shared_memory as shm_pkg
from uuid import uuid4

import pytest
import torch

from vllm_omni.distributed.omni_connectors.connectors import shm_connector as shm_conn_module
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
    SharedMemoryConnector,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture()
def connector():
    c = SharedMemoryConnector({})
    yield c
    c.close()


# ── Key-based read (the fundamental SHM path) ────────────────────────


class TestKeyBasedReadWrite:
    def test_put_then_get_by_key(self, connector):
        data = {"hello": "world", "n": 42}
        ok, size, meta = connector.put("s0", "s1", "test_key_1", data)
        assert ok
        assert size > 0
        assert "shm" in meta
        assert "test_key_1" in connector._pending_keys

        result = connector.get("s0", "s1", "test_key_1", metadata=None)
        assert result is not None
        obj, rsize = result
        assert obj == data
        assert rsize == size
        assert "test_key_1" not in connector._pending_keys
        assert connector._metrics["gets"] == 1

    def test_tensor_payload_removes_lock_file(self, connector):
        key = "tensor_payload"
        payload = torch.ones(2, 2)
        ok, _, metadata = connector.put("s0", "s1", key, payload)
        assert ok

        result = connector.get("s0", "s1", key, metadata=metadata)

        assert result is not None
        assert torch.equal(result[0], payload)
        assert not os.path.exists(f"/dev/shm/shm_{key}_lockfile.lock")

    def test_falsey_payload_removes_lock_file(self, connector):
        key = "falsey_payload"
        ok, _, metadata = connector.put("s0", "s1", key, 0)
        assert ok

        result = connector.get("s0", "s1", key, metadata=metadata)

        assert result is not None
        assert result[0] == 0
        assert not os.path.exists(f"/dev/shm/shm_{key}_lockfile.lock")

    def test_get_nonexistent_key_returns_none(self, connector):
        result = connector.get("s0", "s1", "no_such_key_xyz", metadata=None)
        assert result is None

    def test_get_empty_shm_race_returns_none(self, connector, monkeypatch):
        def raise_empty_file(*args, **kwargs):
            raise ValueError("cannot mmap an empty file")

        monkeypatch.setattr(
            "vllm_omni.distributed.omni_connectors.connectors.shm_connector.shm_pkg.SharedMemory",
            raise_empty_file,
        )

        result = connector.get("s0", "s1", "not_ready_yet", metadata=None)

        assert result is None

    def test_rank_aware_keys_independent(self, connector):
        """Each TP rank writes/reads its own key — simulates homogeneous TP."""
        payloads = {}
        for rank in range(4):
            key = f"req1_s0_0_{rank}_{rank}"
            data = {"rank": rank, "values": list(range(rank, rank + 3))}
            ok, _, _ = connector.put("s0", "s1", key, data)
            assert ok
            payloads[rank] = data

        for rank in range(4):
            key = f"req1_s0_0_{rank}_{rank}"
            result = connector.get("s0", "s1", key, metadata=None)
            assert result is not None
            obj, _ = result
            assert obj == payloads[rank]


# ── Metadata fallback behaviour ──────────────────────────────────────


class TestMetadataFallback:
    def test_rdma_style_metadata_falls_back_to_key(self, connector):
        """source_host/source_port metadata should be ignored; key read used."""
        data = {"payload": True}
        connector.put("s0", "s1", "fb_key_1", data)

        rdma_meta = {"source_host": "10.0.0.1", "source_port": 12345}
        result = connector.get("s0", "s1", "fb_key_1", metadata=rdma_meta)
        assert result is not None
        obj, _ = result
        assert obj == data

    def test_non_dict_metadata_falls_back_to_key(self, connector):
        data = {"val": 99}
        connector.put("s0", "s1", "fb_key_2", data)

        result = connector.get("s0", "s1", "fb_key_2", metadata="not_a_dict")
        assert result is not None
        obj, _ = result
        assert obj == data

    def test_empty_dict_metadata_falls_back_to_key(self, connector):
        data = {"x": 1}
        connector.put("s0", "s1", "fb_key_3", data)

        result = connector.get("s0", "s1", "fb_key_3", metadata={})
        assert result is not None
        obj, _ = result
        assert obj == data

    def test_shm_handle_metadata_still_works(self, connector):
        """When metadata contains a proper 'shm' handle, use it directly."""
        data = {"direct": True}
        ok, size, meta = connector.put("s0", "s1", "shm_direct_1", data)
        assert ok
        result = connector.get("s0", "s1", "shm_direct_1", metadata=meta)
        assert result is not None
        obj, _ = result
        assert obj == data

    def test_metadata_keyed_by_request_id(self, connector):
        """Metadata wrapped as {get_key: actual_meta} should be unwrapped."""
        data = {"wrapped": True}
        ok, size, meta = connector.put("s0", "s1", "wrap_key", data)
        assert ok
        wrapped = {"wrap_key": meta}
        result = connector.get("s0", "s1", "wrap_key", metadata=wrapped)
        assert result is not None
        obj, _ = result
        assert obj == data


# ── Heterogeneous TP multi-key read ──────────────────────────────────


class TestHeteroTPMultiKey:
    def test_receiver_reads_multiple_sender_keys(self, connector):
        """Simulates from_tp=2 -> to_tp=1: receiver reads 2 keys and merges."""
        for sender_rank in range(2):
            key = f"req1_s0_0_{sender_rank}_0"
            data = {"sender": sender_rank, "shard": [sender_rank * 10]}
            connector.put("s0", "s1", key, data)

        shards = []
        for sender_rank in range(2):
            key = f"req1_s0_0_{sender_rank}_0"
            result = connector.get("s0", "s1", key, metadata=None)
            assert result is not None
            obj, _ = result
            shards.append(obj)

        assert len(shards) == 2
        assert shards[0]["sender"] == 0
        assert shards[1]["sender"] == 1

    def test_sender_writes_multiple_receiver_keys(self, connector):
        """Simulates from_tp=1 -> to_tp=2: sender writes 2 sliced keys."""
        for recv_rank in range(2):
            key = f"req1_s0_0_0_{recv_rank}"
            data = {"target": recv_rank, "slice": list(range(recv_rank, recv_rank + 2))}
            connector.put("s0", "s1", key, data)

        for recv_rank in range(2):
            key = f"req1_s0_0_0_{recv_rank}"
            result = connector.get("s0", "s1", key, metadata=None)
            assert result is not None
            obj, _ = result
            assert obj["target"] == recv_rank


# ── Cleanup ──────────────────────────────────────────────────────────


class TestCleanup:
    def test_cleanup_removes_unconsumed_segment(self, connector):
        data = {"leak": True}
        connector.put("s0", "s1", "cleanup_req_42", data)
        assert "cleanup_req_42" in connector._pending_keys

        connector.cleanup("req_42")
        assert "cleanup_req_42" not in connector._pending_keys

        result = connector.get("s0", "s1", "cleanup_req_42", metadata=None)
        assert result is None

    def test_cleanup_noop_for_consumed_segment(self, connector):
        data = {"consumed": True}
        connector.put("s0", "s1", "consumed_req_99", data)
        connector.get("s0", "s1", "consumed_req_99", metadata=None)

        connector.cleanup("req_99")
        assert "consumed_req_99" not in connector._pending_keys

    def test_close_cleans_all_pending(self, connector):
        for i in range(3):
            connector.put("s0", "s1", f"close_test_{i}", {"i": i})

        assert len(connector._pending_keys) == 3
        connector.close()
        assert len(connector._pending_keys) == 0


# ── Lock-file lifecycle (issue 24 of #7636) ──────────────────────────


class TestLockFileLifecycle:
    """A lock file lives exactly as long as its segment — no longer.

    Covers the removal paths of the lifecycle contract: successful read,
    failed read with a dead segment, failed ``put()`` (segment never came to
    life), and the once-per-process startup sweep for orphans left behind by
    abnormally terminated processes.
    """

    @staticmethod
    def _fresh_key() -> str:
        return f"lc_{uuid4().hex[:8]}"

    @staticmethod
    def _lock_path(key: str) -> str:
        return f"/dev/shm/shm_{key}_lockfile.lock"

    @staticmethod
    def _seg_path(key: str) -> str:
        return f"/dev/shm/{key}"

    @staticmethod
    def _backdate(path: str, seconds: float = 120.0) -> None:
        """Pretend *path* was created *seconds* ago (grace is 60 s)."""
        t = time.time() - seconds
        os.utime(path, (t, t))

    @staticmethod
    def _force_unlink_segment(key: str) -> None:
        seg = shm_pkg.SharedMemory(name=key)
        seg.close()
        seg.unlink()

    def test_roundtrip_removes_lock_and_segment(self, connector):
        key = self._fresh_key()
        ok, _, meta = connector.put("s0", "s1", key, {"hello": "lc"})
        assert ok
        assert os.path.exists(self._lock_path(key))
        assert os.path.exists(self._seg_path(key))

        result = connector.get("s0", "s1", key, metadata=meta)

        assert result is not None and result[0] == {"hello": "lc"}
        assert not os.path.exists(self._lock_path(key))
        assert not os.path.exists(self._seg_path(key))

    def test_put_write_failure_removes_lock_file(self, connector, monkeypatch):
        def _boom(*args, **kwargs):
            raise OSError("simulated segment write failure")

        monkeypatch.setattr(shm_conn_module, "shm_write_bytes", _boom)
        key = self._fresh_key()
        ok, size, meta = connector.put("s0", "s1", key, {"x": 1})

        assert (ok, size, meta) == (False, 0, None)
        assert not os.path.exists(self._lock_path(key))
        assert not os.path.exists(self._seg_path(key))

    def test_put_failure_then_retry_succeeds(self, connector):
        key = self._fresh_key()
        orig = shm_conn_module.shm_write_bytes

        def _boom(*args, **kwargs):
            raise OSError("simulated segment write failure")

        shm_conn_module.shm_write_bytes = _boom
        try:
            ok, _, _ = connector.put("s0", "s1", key, {"x": 1})
        finally:
            shm_conn_module.shm_write_bytes = orig
        assert ok is False
        assert not os.path.exists(self._lock_path(key))

        ok, _, meta = connector.put("s0", "s1", key, {"x": 2})
        assert ok
        result = connector.get("s0", "s1", key, metadata=meta)
        assert result is not None and result[0] == {"x": 2}
        assert not os.path.exists(self._lock_path(key))

    def test_get_failure_dead_segment_removes_lock_file(self, connector):
        key = self._fresh_key()
        _, _, meta = connector.put("s0", "s1", key, {"data": "w" * 128})

        self._force_unlink_segment(key)  # segment reaped under us

        result = connector.get("s0", "s1", key, metadata=meta)
        assert result is None
        assert not os.path.exists(self._lock_path(key))

    def test_get_deserialize_failure_after_read_removes_lock_file(self, connector, monkeypatch):
        """Read succeeds (segment already unlinked), then deserialize fails."""
        key = self._fresh_key()
        _, _, meta = connector.put("s0", "s1", key, {"data": "y" * 128})

        def _boom(raw):
            raise ValueError("corrupted payload")

        monkeypatch.setattr(SharedMemoryConnector, "deserialize_obj", staticmethod(_boom))
        result = connector.get("s0", "s1", key, metadata=meta)

        assert result is None
        assert not os.path.exists(self._seg_path(key))  # unlinked by the read
        assert not os.path.exists(self._lock_path(key))  # lock must not outlive it

    def test_get_failure_live_segment_keeps_lock_file(self, connector, monkeypatch):
        key = self._fresh_key()
        _, _, meta = connector.put("s0", "s1", key, {"data": "z" * 128})
        try:

            def _boom(handle):
                raise OSError("simulated read failure")

            monkeypatch.setattr(shm_conn_module, "shm_read_bytes", _boom)
            result = connector.get("s0", "s1", key, metadata=meta)

            assert result is None
            assert os.path.exists(self._seg_path(key))
            assert os.path.exists(self._lock_path(key))  # retry is still possible
        finally:
            self._force_unlink_segment(key)
            try:
                os.remove(self._lock_path(key))
            except OSError:
                pass

    def test_sweep_removes_stale_lock_without_segment(self):
        key = self._fresh_key()
        lock = self._lock_path(key)
        open(lock, "wb").close()
        self._backdate(lock)

        removed = shm_conn_module._sweep_stale_lock_files(grace=60.0)

        assert removed >= 1
        assert not os.path.exists(lock)

    def test_sweep_spares_lock_with_live_segment(self):
        key = self._fresh_key()
        seg = shm_pkg.SharedMemory(create=True, size=8, name=key)
        try:
            lock = self._lock_path(key)
            open(lock, "wb").close()
            self._backdate(lock)

            shm_conn_module._sweep_stale_lock_files(grace=60.0)

            assert os.path.exists(lock)
            assert os.path.exists(self._seg_path(key))
        finally:
            seg.close()
            seg.unlink()
            try:
                os.remove(self._lock_path(key))
            except OSError:
                pass

    def test_sweep_spares_fresh_lock(self):
        key = self._fresh_key()
        lock = self._lock_path(key)
        open(lock, "wb").close()
        try:
            shm_conn_module._sweep_stale_lock_files(grace=60.0)
            assert os.path.exists(lock)
        finally:
            try:
                os.remove(lock)
            except OSError:
                pass

    def test_sweep_spares_held_lock(self):
        key = self._fresh_key()
        lock = self._lock_path(key)
        open(lock, "wb").close()
        self._backdate(lock)

        fd = os.open(lock, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)  # a critical section holds the lock
            shm_conn_module._sweep_stale_lock_files(grace=60.0)
            assert os.path.exists(lock)
        finally:
            os.close(fd)

        shm_conn_module._sweep_stale_lock_files(grace=60.0)
        assert not os.path.exists(lock)

    def test_sweep_ignores_unrelated_files(self):
        extras = [
            "/dev/shm/lc_dummy.bin",
            "/dev/shm/shm_x.lock",  # shm_ prefix but not a transfer lock file
            "/dev/shm/plain_lockfile.lock",
        ]
        try:
            for path in extras:
                open(path, "wb").close()
                self._backdate(path)

            shm_conn_module._sweep_stale_lock_files(grace=60.0)

            for path in extras:
                assert os.path.exists(path)
        finally:
            for path in extras:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def test_sweep_runs_once_per_process(self, monkeypatch):
        calls = []

        def _counting_sweep(*args, **kwargs):
            calls.append(1)
            return 0

        monkeypatch.setattr(shm_conn_module, "_swept_this_process", False)
        monkeypatch.setattr(shm_conn_module, "_sweep_stale_lock_files", _counting_sweep)

        SharedMemoryConnector({})
        SharedMemoryConnector({})

        assert len(calls) == 1
        assert shm_conn_module._swept_this_process is True

    def test_crashed_subprocess_lock_swept_by_new_process(self):
        key = self._fresh_key()
        lock = self._lock_path(key)
        # The subprocess only touches the lock file (no vllm_omni import) and
        # dies via os._exit, skipping every cleanup path — the leak from a
        # crashed duplex process, reproduced.
        script = "import os, sys\nopen(sys.argv[1], 'wb').close()\nos._exit(1)\n"
        proc = subprocess.run([sys.executable, "-c", script, lock])
        assert proc.returncode == 1
        assert os.path.exists(lock)

        self._backdate(lock)  # the restart happens some time after the crash
        shm_conn_module._sweep_stale_lock_files(grace=60.0)

        assert not os.path.exists(lock)
