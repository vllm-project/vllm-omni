# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for SharedMemoryConnector focusing on TP / CFG / metadata fallback."""

import fcntl
import os
import time
import uuid

import pytest
import torch

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
    def test_deadline_receive_does_not_wait_for_writer_lock(self, connector):
        key = "deadline_locked_payload"
        connector.put("0", "1", key, {"value": 7})
        with open(f"/dev/shm/shm_{key}_lockfile.lock", "rb+") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            assert connector.get_with_deadline("0", "1", key, deadline=time.monotonic() + 1) is None
        assert connector.get_with_deadline("0", "1", key, deadline=time.monotonic() - 1) is None
        assert connector.get_with_deadline("0", "1", key, deadline=time.monotonic() + 1)[0] == {"value": 7}

    @pytest.mark.parametrize("with_handle", [False, True], ids=["key", "handle"])
    def test_deadline_receive_leaves_due_sweep_for_ordinary_get(self, connector, mocker, monkeypatch, with_handle):
        from vllm_omni.distributed.omni_connectors.connectors import shm_connector as module

        key = f"deadline_sweep_{uuid.uuid4().hex}"
        ok, size, metadata = connector.put("0", "1", key, "ready")
        assert ok
        monkeypatch.delitem(module._next_sweep, connector.shm_directory, raising=False)
        sweep = mocker.patch.object(module, "_sweep_orphan_locks")
        monkeypatch.setattr(time, "monotonic", lambda: 1.0)
        source = metadata if with_handle else {"source_host": "localhost"}

        assert connector.get_with_deadline("0", "1", key, source, deadline=1.001) == ("ready", size)
        sweep.assert_not_called()
        assert connector.get("0", "1", key) is None
        sweep.assert_called_once_with(connector.shm_directory)

    def test_deadline_expiring_before_read_preserves_payload(self, connector, monkeypatch):
        from vllm_omni.distributed.omni_connectors.connectors import shm_connector as module

        key = f"deadline_unread_{uuid.uuid4().hex}"
        ok, size, _ = connector.put("0", "1", key, "still here")
        assert ok
        now = 1.0
        current_lock = module._current_lock

        def check_lock_after_deadline(fd, path):
            nonlocal now
            now = 2.0
            return current_lock(fd, path)

        with monkeypatch.context() as patch:
            patch.setattr(time, "monotonic", lambda: now)
            patch.setattr(module, "_current_lock", check_lock_after_deadline)
            assert connector.get_with_deadline("0", "1", key, deadline=2.0) is None
        assert connector.get("0", "1", key) == ("still here", size)

    def test_reading_other_handle_preserves_tracked_publication(self, connector):
        prefix = f"handle_alias_{uuid.uuid4().hex}_"
        key = f"{prefix}0"
        ok, size, handle = connector.put("0", "1", f"{prefix}source", "source payload")
        assert ok
        assert connector.put("0", "1", key, "keep tracked")[0]

        assert connector.get("0", "1", key, metadata=handle) == ("source payload", size)
        assert connector.cleanup_prefix(prefix) == 1

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

    def test_get_does_not_open_shm_before_writer_sizes_it(self, connector, tmp_path, monkeypatch, mocker):
        import threading

        from vllm_omni.distributed.omni_connectors.connectors import shm_connector as module

        # Isolate the publication lock from the host's coordination directory.
        monkeypatch.setattr(connector, "_lock_file", lambda name: str(tmp_path / "writer.lock"))
        entered, release = threading.Event(), threading.Event()
        original = os.ftruncate
        opens = mocker.spy(module.shm_pkg, "SharedMemory")

        def pause_size(fd, size):
            entered.set()
            assert release.wait(timeout=5)
            return original(fd, size)

        key = "sizing_" + uuid.uuid4().hex[:16]
        result = []
        with monkeypatch.context() as patch:
            patch.setattr(os, "ftruncate", pause_size)
            writer = threading.Thread(target=lambda: result.append(connector.put("0", "1", key, {"ready": True})))
            writer.start()
            try:
                assert entered.wait(timeout=5)
                before_read = opens.call_count
                assert connector.get("0", "1", key) is None
                assert opens.call_count == before_read
            finally:
                release.set()
                writer.join(timeout=5)
        assert not writer.is_alive()
        assert result[0][0]
        assert connector.get("0", "1", key, metadata=result[0][2])[0] == {"ready": True}

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
    def test_read_can_retry_after_segment_unlink_fails(self, connector, monkeypatch):
        from multiprocessing.shared_memory import SharedMemory

        key = f"read_retry_{uuid.uuid4().hex}"
        ok, size, metadata = connector.put("0", "1", key, "unread")
        assert ok

        def fail_unlink(segment):
            raise PermissionError("temporary unlink failure")

        with monkeypatch.context() as patch:
            patch.setattr(SharedMemory, "unlink", fail_unlink)
            assert connector.get("0", "1", key, metadata) is None
            assert key in connector._pending_keys
            assert connector._shm_exists(key)
            assert os.path.exists(connector._lock_file(key))

        assert connector.get("0", "1", key, metadata) == ("unread", size)
        assert key not in connector._pending_keys
        assert not connector._shm_exists(key)
        assert not os.path.exists(connector._lock_file(key))

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

        assert connector.cleanup("cleanup_req_42") is True
        assert "cleanup_req_42" not in connector._pending_keys

        result = connector.get("s0", "s1", "cleanup_req_42", metadata=None)
        assert result is None

    def test_cleanup_noop_for_consumed_segment(self, connector):
        data = {"consumed": True}
        connector.put("s0", "s1", "consumed_req_99", data)
        connector.get("s0", "s1", "consumed_req_99", metadata=None)

        assert connector.cleanup("consumed_req_99") is False
        assert "consumed_req_99" not in connector._pending_keys

    def test_deferred_cleanup_keeps_lock_through_segment_gap(self, connector, monkeypatch):
        key = f"release_gap_{uuid.uuid4().hex}"
        assert connector.put("0", "1", key, "original")[0]
        lock_path = connector._lock_file(key)
        now = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: now)
        with open(lock_path, "rb") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            assert not connector.cleanup_tracked(key, defer_if_busy=True)
            connector._unlink_segment(key)
            now += 60
            connector.reap_consumed()
            assert key in connector._pending_keys
            assert os.path.exists(lock_path)

        now += 60
        connector.reap_consumed()
        assert key not in connector._pending_keys
        assert not os.path.exists(lock_path)

    def test_busy_release_without_storage_identity_preserves_replacement(self, connector, monkeypatch):
        monkeypatch.setattr(connector, "_segment_identity", lambda name: None)
        key = f"release_no_identity_{uuid.uuid4().hex}"
        assert connector.put("0", "1", key, "original")[0]
        with open(connector._lock_file(key), "rb") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            assert not connector.cleanup_tracked(key, defer_if_busy=True)

        writer = SharedMemoryConnector({})
        try:
            assert writer.put("0", "1", key, "replacement")[0]
            with monkeypatch.context() as patch:
                later = time.monotonic() + 60
                patch.setattr(time, "monotonic", lambda: later)
                connector.reap_consumed()
            assert writer.get("0", "1", key)[0] == "replacement"
        finally:
            writer.close()

    def test_unlink_error_preserves_lock_and_tracking_for_retry(self, connector, monkeypatch):
        prefix = f"cleanup_error_{uuid.uuid4().hex}_"
        key = f"{prefix}0"
        assert connector.put("0", "1", key, "payload")[0]

        def fail_unlink(name):
            raise OSError("temporary failure unlinking the segment")

        with monkeypatch.context() as patch:
            patch.setattr(connector, "_unlink_segment", fail_unlink)
            assert connector.cleanup(key) is False
        assert key in connector._pending_keys
        assert connector._shm_exists(key)
        assert os.path.exists(connector._lock_file(key))

        assert connector.cleanup_prefix(prefix) == 1
        assert key not in connector._pending_keys
        assert not connector._shm_exists(key)
        assert not os.path.exists(connector._lock_file(key))

    def test_close_cleans_all_pending(self, connector):
        for i in range(3):
            connector.put("s0", "s1", f"close_test_{i}", {"i": i})

        assert len(connector._pending_keys) == 3
        connector.close()
        assert len(connector._pending_keys) == 0


@pytest.mark.parametrize("suffix", ["_1_2", "_1", "_suffix"])
def test_cleanup_preserves_other_request_keys(connector, suffix):
    request_id = f"ownership_{uuid.uuid4().hex}"
    own_key = f"{request_id}_0_0"
    sibling_key = f"{request_id}{suffix}_0_0"
    assert connector.put("0", "1", own_key, "own")[0]
    assert connector.put("0", "1", sibling_key, "sibling")[0]

    # A raw request id is not an exact key and must not match any chunks.
    connector.cleanup(request_id)
    assert connector.get("0", "1", own_key)[0] == "own"
    connector.cleanup(own_key)
    assert connector.get("0", "1", sibling_key)[0] == "sibling"


class TestLockFileLifecycle:
    @pytest.mark.parametrize("has_segment", [False, True])
    def test_failed_put_keeps_lock_only_for_live_segment(self, connector, monkeypatch, has_segment):
        key = f"failed_put_{uuid.uuid4().hex}"
        lock = connector._lock_file(key)
        if has_segment:
            assert connector.put("0", "1", key, "existing")[0]

        def fail_write(payload, name):
            raise OSError("injected write failure")

        monkeypatch.setattr(connector, "_write_bytes", fail_write)
        try:
            assert not connector.put("0", "1", key, "replacement")[0]
            assert os.path.exists(lock) == has_segment
            if has_segment:
                assert connector.get("0", "1", key)[0] == "existing"
        finally:
            connector.cleanup(key)

    def test_poll_keeps_fresh_or_busy_lock_then_reclaims_orphan(self, monkeypatch, tmp_path):
        monkeypatch.setattr(SharedMemoryConnector, "shm_directory", str(tmp_path))
        receiver = SharedMemoryConnector({})
        key = f"poll_{uuid.uuid4().hex}"
        lock = tmp_path / f"shm_{key}_lockfile.lock"
        source = {"source_host": "localhost", "source_port": 1234}
        try:
            lock.touch()
            assert receiver.get("0", "1", key, metadata=source) is None
            assert lock.exists()
            os.utime(lock, (1, 1))
            with lock.open("rb") as active:
                fcntl.flock(active, fcntl.LOCK_EX)
                assert receiver.get("0", "1", key, metadata=source) is None
                assert lock.exists()
            assert receiver.get("0", "1", key, metadata=source) is None
            assert not lock.exists()
        finally:
            receiver.close()

    @pytest.mark.parametrize("segment_gone", [False, True], ids=["live", "reaped"])
    def test_handle_read_failure_preserves_only_live_segment_lock(self, connector, monkeypatch, segment_gone):
        key = f"handle_{uuid.uuid4().hex}"
        ok, _, metadata = connector.put("0", "1", key, "payload")
        assert ok

        def fail_read(handle):
            raise OSError("temporary read failure")

        with monkeypatch.context() as patch:
            if segment_gone:
                connector._unlink_segment(metadata["shm"]["name"])
            else:
                patch.setattr(connector, "_read_bytes", fail_read)
            assert connector.get("0", "1", key, metadata) is None
        assert os.path.exists(connector._lock_file(key)) == (not segment_gone)
        if not segment_gone:
            assert connector.get("0", "1", key, metadata)[0] == "payload"

    def test_sweep_revisits_young_orphans_and_preserves_live_segments(self, monkeypatch, tmp_path):
        monkeypatch.setattr(SharedMemoryConnector, "shm_directory", str(tmp_path))
        now = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: now)
        orphan = tmp_path / "shm_orphan_lockfile.lock"
        young = tmp_path / "shm_young_lockfile.lock"
        live = tmp_path / "shm_live_lockfile.lock"
        unrelated = tmp_path / "unrelated.lock"
        nonempty = tmp_path / "shm_nonempty_lockfile.lock"
        symlink = tmp_path / "shm_symlink_lockfile.lock"
        for path in (orphan, live, unrelated):
            path.touch()
            os.utime(path, (1, 1))
        nonempty.write_bytes(b"preserve")
        os.utime(nonempty, (1, 1))
        symlink.symlink_to(unrelated)
        os.utime(symlink, (1, 1), follow_symlinks=False)
        (tmp_path / "live").touch()
        young.touch()

        receiver = SharedMemoryConnector({})
        try:
            assert not orphan.exists()
            assert live.exists() and young.exists() and unrelated.exists()
            os.utime(young, (1, 1))
            receiver.get("0", "1", "unrelated_key")
            assert young.exists()
            now += 61
            receiver.get("0", "1", "unrelated_key")
            assert not young.exists()
            assert live.exists() and unrelated.exists()
            assert nonempty.read_bytes() == b"preserve"
            assert symlink.is_symlink()
        finally:
            receiver.close()

    def test_separate_segment_storage_cannot_reclaim_by_lock_directory(self, tmp_path):
        class SeparateStorage(SharedMemoryConnector):
            shm_directory = str(tmp_path)

            def _shm_exists(self, key):
                return key == "external"

            def _read_bytes(self, metadata):
                raise OSError("temporarily unavailable")

        lock = tmp_path / "shm_external_lockfile.lock"
        lock.touch()
        os.utime(lock, (1, 1))
        receiver = SeparateStorage({})
        try:
            assert lock.exists()
            assert receiver.get("0", "1", "external") is None
            assert lock.exists()
        finally:
            receiver.close()

    def test_cleanup_preserves_busy_tracked_key_for_retry(self, connector):
        key = f"busy_cleanup_{uuid.uuid4().hex}"
        assert connector.put("0", "1", key, "payload")[0]
        lock = connector._lock_file(key)
        with open(lock, "rb") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            assert connector.cleanup(key) is False
            assert key in connector._pending_keys and connector._shm_exists(key)
            assert os.path.exists(lock)
        assert connector.cleanup(key) is True
        assert key not in connector._pending_keys
        assert not os.path.exists(lock) and not connector._shm_exists(key)

    @pytest.mark.parametrize("retire", ["get", "cleanup"])
    def test_waiting_same_key_writer_publishes_readable_replacement(self, connector, monkeypatch, retire):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        key = f"same_key_{uuid.uuid4().hex}"
        receiver = SharedMemoryConnector({})
        retiring, writer_opened = threading.Event(), threading.Event()
        hook = "_read_bytes" if retire == "get" else "_unlink_segment"
        backend = getattr(receiver, hook)
        flock = fcntl.flock

        def wait_for_writer(arg):
            try:
                with open(receiver._lock_file(key), "rb") as probe, pytest.raises(BlockingIOError):
                    flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                retiring.set()
            assert writer_opened.wait(timeout=5)
            return backend(arg)

        def observe_write_lock(file, operation):
            if operation == fcntl.LOCK_EX and retiring.is_set():
                writer_opened.set()
            return flock(file, operation)

        try:
            assert connector.put("0", "1", key, "first")[0]
            with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=1) as executor:
                patch.setattr(receiver, hook, wait_for_writer)
                patch.setattr(fcntl, "flock", observe_write_lock)
                args = ("0", "1", key) if retire == "get" else (key,)
                first = executor.submit(getattr(receiver, retire), *args)
                assert retiring.wait(timeout=5)
                assert connector.put("0", "1", key, "second")[0]
                previous = first.result(timeout=5)
                if retire == "get":
                    assert previous[0] == "first"
                else:
                    assert previous is True
            result = receiver.get("0", "1", key)
            assert result is not None and result[0] == "second"
            assert not os.path.exists(connector._lock_file(key))
        finally:
            receiver.close()


def test_sender_reaps_keys_consumed_by_another_connector(connector):
    receiver = SharedMemoryConnector({})
    prefix = f"reap_{uuid.uuid4().hex}"
    try:
        for index in range(256):
            key = f"{prefix}_{index}"
            assert connector.put("0", "1", key, index)[0]
            assert receiver.get("0", "1", key)[0] == index
            assert len(connector._pending_keys) <= 1
        connector.reap_consumed()
        assert not connector._pending_keys
    finally:
        receiver.close()


def test_reap_rotates_past_unread_keys(connector):
    receiver = SharedMemoryConnector({})
    prefix = f"rotate_{uuid.uuid4().hex}"
    try:
        for index in range(130):
            assert connector.put("0", "1", f"{prefix}_{index}", index)[0]
        for index in range(65, 130):
            assert receiver.get("0", "1", f"{prefix}_{index}")[0] == index
        for _ in range(3):
            connector.reap_consumed()
        assert len(connector._pending_keys) == 65
        assert connector.get("0", "1", f"{prefix}_0")[0] == 0
    finally:
        receiver.close()


def test_consumed_bad_payload_removes_lock_file(connector, monkeypatch):
    key = f"bad_payload_{uuid.uuid4().hex}"
    assert connector.put("0", "1", key, "payload")[0]

    def fail_deserialize(_data):
        raise ValueError("invalid payload")

    monkeypatch.setattr(connector._serde, "deserialize", fail_deserialize)
    assert connector.get("0", "1", key) is None
    assert not os.path.exists(f"/dev/shm/shm_{key}_lockfile.lock")
    connector.reap_consumed()
    assert key not in connector._pending_keys


@pytest.mark.parametrize("suffix", ["_1_2", "_1", "_0", "_suffix"])
def test_cleanup_prefix_only_matches_own_chunk_keys(connector, suffix):
    request_id = f"prefix_{uuid.uuid4().hex}"
    own_keys = [f"{request_id}_0_{i}" for i in range(3)]
    sibling_key = f"{request_id}{suffix}_0_0"
    for key in own_keys:
        assert connector.put("0", "1", key, key)[0]
    assert connector.put("0", "1", sibling_key, "sibling")[0]
    # A consumed chunk is no longer counted as reclaimed.
    assert connector.get("0", "1", own_keys[0])[0] == own_keys[0]

    assert connector.cleanup_prefix(f"{request_id}_0_") == 2
    for key in own_keys:
        assert key not in connector._pending_keys
        assert connector.get("0", "1", key) is None
    assert connector.get("0", "1", sibling_key)[0] == "sibling"


# ── Arrival wakeups ───────────────────────────────────────────────────


def _stage_connector(stage_id):
    return SharedMemoryConnector({"stage_id": stage_id})


def test_wakeup_discovery_only_scans_deployment_directory(monkeypatch, tmp_path):
    from vllm_omni.distributed.omni_connectors.connectors import shm_connector as module

    directory = tmp_path / "deployment"
    monkeypatch.setattr(module.SharedMemoryConnector, "_wakeup_directory", lambda self: str(directory))
    original_scandir = os.scandir
    scanned = []

    def scoped_scandir(path):
        # A parent-directory scan would recreate the performance regression.
        assert os.fspath(path) == str(directory)
        scanned.append(path)
        return original_scandir(path)

    sender, receiver = _stage_connector(86), _stage_connector(87)
    try:
        generation = receiver.get_wakeup_generation()
        assert generation is not None
        assert os.path.dirname(receiver._wake_path) == str(directory)
        monkeypatch.setattr(module.glob.os, "scandir", scoped_scandir)
        sender._wake_receiver(87)
        assert receiver.wait_for_change(generation, timeout=0.5)
        assert scanned
    finally:
        sender.close()
        receiver.close()


def test_missing_wakeup_directory_keeps_payload_transport_working(monkeypatch, tmp_path):
    from vllm_omni.distributed.omni_connectors.connectors import shm_connector as module

    directory = tmp_path / "missing"
    monkeypatch.setattr(module.SharedMemoryConnector, "_wakeup_directory", lambda self: str(directory))
    sender, receiver = _stage_connector(88), _stage_connector(89)
    key = f"wake_{uuid.uuid4().hex}"
    try:
        assert sender.put("88", "89", key, {"value": 1})[0]
        assert not directory.exists()
        assert receiver.get("88", "89", key)[0] == {"value": 1}
    finally:
        sender.close()
        receiver.close()


def test_put_wakes_waiting_receiver():
    sender, receiver = _stage_connector(70), _stage_connector(71)
    try:
        generation = receiver.get_wakeup_generation()
        assert generation is not None
        start = time.monotonic()
        assert receiver.wait_for_change(generation, timeout=0.05) is False
        assert time.monotonic() - start >= 0.04  # no arrival: blocks for the timeout

        key = f"wake_{uuid.uuid4().hex}"
        sender.put("70", "71", key, {"value": 1})
        start = time.monotonic()
        assert receiver.wait_for_change(generation, timeout=5) is True
        assert time.monotonic() - start < 0.5
        assert receiver.get("70", "71", key)[0] == {"value": 1}
        # Drained: the next wait blocks again.
        assert receiver.wait_for_change(receiver.get_wakeup_generation(), timeout=0.02) is False
    finally:
        sender.close()
        receiver.close()


def test_receiver_does_not_spin_after_sender_closes():
    sender, receiver = _stage_connector(72), _stage_connector(73)
    try:
        receiver.get_wakeup_generation()
        sender.put("72", "73", f"wake_{uuid.uuid4().hex}", {"value": 1})
        sender.close()  # the only external writer goes away
        generation = receiver.get_wakeup_generation()
        receiver.wait_for_change(generation, timeout=1)
        generation = receiver.get_wakeup_generation()
        start = time.monotonic()
        assert receiver.wait_for_change(generation, timeout=0.05) is False
        assert time.monotonic() - start >= 0.04
    finally:
        receiver.close()


def test_wakeups_can_be_disabled_and_close_unlinks_fifo(monkeypatch):
    receiver = _stage_connector(74)
    assert receiver.get_wakeup_generation() is not None
    path = receiver._wake_path
    assert os.path.exists(path)
    receiver.close()
    assert not os.path.exists(path)

    monkeypatch.setenv("VLLM_OMNI_SHM_WAKEUP", "0")
    sender, receiver = _stage_connector(75), _stage_connector(76)
    try:
        assert receiver.get_wakeup_generation() is None  # caller keeps its timed poll
        key = f"wake_{uuid.uuid4().hex}"
        assert sender.put("75", "76", key, {"value": 2})[0]
        assert receiver.get("75", "76", key)[0] == {"value": 2}
    finally:
        sender.close()
        receiver.close()


def _sibling_receiver(stage_id, ready, result):
    receiver = _stage_connector(stage_id)
    generation = receiver.get_wakeup_generation()
    ready.set()
    start = time.monotonic()
    woke = receiver.wait_for_change(generation, timeout=5)
    result.put((woke, time.monotonic() - start))
    receiver.close()


def _sibling_sender(stage_id, key):
    sender = _stage_connector(stage_id - 1)
    sender.put(str(stage_id - 1), str(stage_id), key, {"value": 3})


def test_sibling_stage_processes_share_wakeups():
    # Stage engine processes are siblings under the launching process.
    import multiprocessing

    ctx = multiprocessing.get_context("fork")
    ready, result = ctx.Event(), ctx.Queue()
    key = f"wake_{uuid.uuid4().hex}"
    receiver = ctx.Process(target=_sibling_receiver, args=(78, ready, result))
    receiver.start()
    assert ready.wait(5)
    time.sleep(0.05)
    sender = ctx.Process(target=_sibling_sender, args=(78, key))
    sender.start()
    sender.join(5)
    woke, waited = result.get(timeout=5)
    receiver.join(5)
    assert woke and waited < 1.0
    cleanup = SharedMemoryConnector({})
    cleanup.cleanup(key)


def test_same_stage_receivers_are_independent_across_close_and_restart():
    sender = _stage_connector(80)
    first, second = _stage_connector(81), _stage_connector(81)
    replacement = None
    try:
        first_generation = first.get_wakeup_generation()
        second_generation = second.get_wakeup_generation()
        assert first._wake_path != second._wake_path
        sender._wake_receiver(81)
        assert first.wait_for_change(first_generation, timeout=0.5)
        assert second.wait_for_change(second_generation, timeout=0.5)
        second_generation = second.get_wakeup_generation()
        first.close()
        assert first.get_wakeup_generation() is None
        assert os.path.exists(second._wake_path)
        replacement = _stage_connector(81)
        replacement_generation = replacement.get_wakeup_generation()
        sender._wake_receiver(81)
        assert second.wait_for_change(second_generation, timeout=0.5)
        assert replacement.wait_for_change(replacement_generation, timeout=0.5)
    finally:
        sender.close()
        first.close()
        second.close()
        if replacement is not None:
            replacement.close()


def test_partial_receiver_open_closes_fd_and_removes_only_own_fifo(monkeypatch, tmp_path):
    import errno

    from vllm_omni.distributed.omni_connectors.connectors import shm_connector as module

    monkeypatch.setattr(module.SharedMemoryConnector, "_wakeup_directory", lambda self: str(tmp_path))
    receiver = _stage_connector(82)
    peer_path = tmp_path / "stage_82_peer"
    os.mkfifo(peer_path, 0o600)
    original_open = os.open
    opened = []

    def fail_hold_open(path, flags, *args, **kwargs):
        if flags & os.O_WRONLY:
            raise OSError(errno.EMFILE, "injected fd exhaustion")
        fd = original_open(path, flags, *args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(module.os, "open", fail_hold_open)
    try:
        assert receiver.get_wakeup_generation() is None
        assert list(tmp_path.iterdir()) == [peer_path]
        assert len(opened) == 1
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(opened[0])
    finally:
        receiver.close()


def test_wakeup_ignores_stale_fifo_and_non_fifo_paths(monkeypatch, tmp_path):
    from vllm_omni.distributed.omni_connectors.connectors import shm_connector as module

    monkeypatch.setattr(module.SharedMemoryConnector, "_wakeup_directory", lambda self: str(tmp_path))
    sender, receiver = _stage_connector(83), _stage_connector(84)
    plain = tmp_path / "84_plain"
    plain.write_text("unchanged")
    (tmp_path / "84_link").symlink_to(plain)
    os.mkfifo(tmp_path / "84_stale", 0o600)
    try:
        generation = receiver.get_wakeup_generation()
        sender._wake_receiver(84)
        assert receiver.wait_for_change(generation, timeout=0.5)
        assert plain.read_text() == "unchanged"
        assert (tmp_path / "84_stale").exists()
    finally:
        sender.close()
        receiver.close()


def test_waiter_does_not_read_fd_reused_after_close(monkeypatch, tmp_path):
    from vllm_omni.distributed.omni_connectors.connectors import shm_connector as module

    receiver = _stage_connector(85)
    generation = receiver.get_wakeup_generation()
    read_fd = receiver._wake_read_fd
    path = tmp_path / "unrelated_ipc"
    path.write_bytes(b"untouched")
    reused = []

    def close_during_select(*args):
        receiver.close()
        fd = os.open(path, os.O_RDONLY)
        if fd != read_fd:
            os.dup2(fd, read_fd)
            os.close(fd)
        reused.append(read_fd)
        return [read_fd], [], []

    monkeypatch.setattr(module.select, "select", close_during_select)
    try:
        assert not receiver.wait_for_change(generation, timeout=0.1)
        assert os.lseek(read_fd, 0, os.SEEK_CUR) == 0
    finally:
        for fd in reused:
            os.close(fd)
        receiver.close()


def test_deployment_scope_survives_different_worker_parents(monkeypatch):
    from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        load_omni_transfer_config,
        resolve_connector_spec,
    )

    # Build connectors the way stage workers do: the launcher stores the scope
    # in the edge's extra, and the factory passes that extra as the config.
    transfer_config = load_omni_transfer_config(
        config_dict={"stage_args": [{"stage_id": 0}, {"stage_id": 1, "engine_input_source": [0]}]}
    )
    edge = transfer_config.connectors[("0", "1")]
    monkeypatch.setattr(os, "getppid", lambda: 111)
    receiver = OmniConnectorFactory.create_connector(resolve_connector_spec(edge, stage_id=1, role="receiver"))
    monkeypatch.setattr(os, "getppid", lambda: 222)
    sender = OmniConnectorFactory.create_connector(resolve_connector_spec(edge, stage_id=0, role="sender"))
    try:
        assert receiver._wake_directory == sender._wake_directory
        generation = receiver.get_wakeup_generation()
        sender._wake_receiver(1)
        assert receiver.wait_for_change(generation, timeout=0.1)
        directory = receiver._wake_directory
    finally:
        receiver.close()
        sender.close()
    assert not os.path.exists(directory)


@pytest.mark.parametrize("value,expected", [("bad", 0.005), ("nan", 0.005), ("-1", 0.005), ("20", 0.02)])
def test_receive_poll_interval_validation(monkeypatch, value, expected):
    from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport import _recv_poll_seconds

    monkeypatch.setenv("VLLM_OMNI_CONNECTOR_RECV_POLL_MS", value)
    assert _recv_poll_seconds() == expected


def test_put_limits_reaping_work_with_unread_backlog(connector, monkeypatch):
    """A backlog must not turn every data send into a large filesystem sweep."""
    prefix = f"reap_budget_{uuid.uuid4().hex}"
    for index in range(128):
        assert connector.put("0", "1", f"{prefix}_{index}", index)[0]

    checked_paths = []
    exists = os.path.exists

    def track_exists(path):
        if str(path).startswith(f"/dev/shm/{prefix}"):
            checked_paths.append(path)
        return exists(path)

    monkeypatch.setattr(os.path, "exists", track_exists)
    assert connector.put("0", "1", f"{prefix}_next", "next")[0]
    # Keep per-send housekeeping small, independently of the backlog length.
    assert len(checked_paths) <= 8


def test_reaping_keeps_consumed_records_bounded_behind_unread_keys(connector):
    """Small send-time sweeps must still catch up with ongoing consumption."""
    receiver = SharedMemoryConnector({})
    prefix = f"reap_mixed_{uuid.uuid4().hex}"
    unread_count = 128
    try:
        for index in range(unread_count):
            assert connector.put("0", "1", f"{prefix}_held_{index}", index)[0]
        for index in range(2048):
            key = f"{prefix}_consumed_{index}"
            assert connector.put("0", "1", key, index)[0]
            assert receiver.get("0", "1", key)[0] == index
            assert len(connector._pending_keys) <= 2 * unread_count
        for index in range(unread_count):
            assert receiver.get("0", "1", f"{prefix}_held_{index}")[0] == index
        for _ in range(4):
            connector.reap_consumed()
        assert not connector._pending_keys
    finally:
        receiver.close()


def test_put_refuses_when_peer_shared_memory_is_not_visible():
    """A sender that cannot see the receiver's IPC namespace must not report success."""
    base = 50000 + (os.getpid() % 10000)
    sender = SharedMemoryConnector({"stage_id": base})
    key = f"peer_{uuid.uuid4().hex}"
    try:
        ok, size, meta = sender.put(str(base), str(base + 1), key, {"a": 1})
        assert ok is False
        assert size == 0
        assert meta is None
        receiver = SharedMemoryConnector({"stage_id": base + 1})
        try:
            assert sender.put(str(base), str(base + 1), key, {"a": 1})[0] is True
            assert receiver.get(str(base), str(base + 1), key)[0] == {"a": 1}
        finally:
            receiver.close()
        assert sender.put(str(base), str(base + 1), f"{key}-gone", {"a": 1})[0] is False
    finally:
        sender.close()


def test_visibility_is_scoped_to_deployment():
    """A same-numbered stage from another deployment in this namespace is not our peer."""
    base = 50000 + (os.getpid() % 10000)
    sender = SharedMemoryConnector({"stage_id": base, "wakeup_scope": f"b-{uuid.uuid4().hex}"})
    other_deployment = SharedMemoryConnector({"stage_id": base + 1, "wakeup_scope": f"a-{uuid.uuid4().hex}"})
    key = f"scoped_{uuid.uuid4().hex}"
    try:
        assert sender.put(str(base), str(base + 1), key, {"a": 1})[0] is False
        receiver = SharedMemoryConnector({"stage_id": base + 1, "wakeup_scope": sender.config["wakeup_scope"]})
        try:
            assert sender.put(str(base), str(base + 1), key, {"a": 1})[0] is True
            assert receiver.get(str(base), str(base + 1), key)[0] == {"a": 1}
        finally:
            receiver.close()
    finally:
        other_deployment.close()
        sender.close()


def test_separate_config_loads_share_one_deployment_scope():
    """Head and --headless each load the config. The same master endpoint must match."""
    from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        load_omni_transfer_config,
        resolve_connector_spec,
    )

    config = {"stage_args": [{"stage_id": 0}, {"stage_id": 1, "engine_input_source": [0]}]}
    scope = f"10.0.0.8:{18000 + (os.getpid() % 1000)}"
    head = load_omni_transfer_config(config_dict=config, deployment_scope=scope)
    headless = load_omni_transfer_config(config_dict=config, deployment_scope=scope)
    other = load_omni_transfer_config(config_dict=config, deployment_scope=scope + "-other")
    sender = OmniConnectorFactory.create_connector(
        resolve_connector_spec(head.connectors[("0", "1")], stage_id=0, role="sender")
    )
    stranger = OmniConnectorFactory.create_connector(
        resolve_connector_spec(other.connectors[("0", "1")], stage_id=1, role="receiver")
    )
    key = f"deploy_{uuid.uuid4().hex}"
    receiver = None
    try:
        assert sender.put("0", "1", key, {"a": 1})[0] is False
        receiver = OmniConnectorFactory.create_connector(
            resolve_connector_spec(headless.connectors[("0", "1")], stage_id=1, role="receiver")
        )
        assert sender.put("0", "1", key, {"a": 1})[0] is True
        assert receiver.get("0", "1", key)[0] == {"a": 1}
    finally:
        if receiver is not None:
            receiver.close()
        stranger.close()
        sender.close()


def test_closing_one_receiver_keeps_the_other_visible():
    base = 60000 + (os.getpid() % 1000)
    scope = f"live-{uuid.uuid4().hex}"
    sender = SharedMemoryConnector({"stage_id": base, "wakeup_scope": scope})
    first = SharedMemoryConnector({"stage_id": base + 1, "wakeup_scope": scope})
    second = SharedMemoryConnector({"stage_id": base + 1, "wakeup_scope": scope})
    key = f"both_{uuid.uuid4().hex}"
    try:
        first.close()
        assert sender.put(str(base), str(base + 1), key, {"a": 1})[0] is True
        assert second.get(str(base), str(base + 1), key)[0] == {"a": 1}
    finally:
        second.close()
        sender.close()


def test_head_deployment_id_matches_across_master_address_spellings():
    """127.0.0.1 and localhost are one master. Connectors must share its id."""
    import msgspec
    import zmq
    from vllm.utils.network_utils import get_open_port

    from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        load_omni_transfer_config,
        resolve_connector_spec,
    )
    from vllm_omni.engine.stage_engine_startup import OmniMasterServer, fetch_deployment_id

    port = get_open_port()
    server = OmniMasterServer(master_address="127.0.0.1", master_port=port, stage_ids=[0, 1])
    server.start()
    config = {"stage_args": [{"stage_id": 0}, {"stage_id": 1, "engine_input_source": [0]}]}
    sender = receiver = literal_sender = literal_receiver = None
    ctx = zmq.Context()
    try:
        from vllm_omni.engine.stage_engine_startup import ensure_matching_deployment_id

        via_ip = fetch_deployment_id(omni_master_address="127.0.0.1", omni_master_port=port)
        via_name = fetch_deployment_id(omni_master_address="localhost", omni_master_port=port)
        assert via_ip == via_name == server.deployment_id
        sock = ctx.socket(zmq.DEALER)
        sock.connect(f"tcp://localhost:{port}")
        sock.send(msgspec.msgpack.encode({"stage_id": 0}))
        assert sock.poll(timeout=5_000)
        registered = msgspec.msgpack.decode(sock.recv())["deployment_id"]
        assert registered == server.deployment_id
        ensure_matching_deployment_id(expected=via_ip, actual=registered)
        sock.close(linger=0)

        head = load_omni_transfer_config(config_dict=config, deployment_scope=via_ip)
        headless = load_omni_transfer_config(config_dict=config, deployment_scope=via_name)
        sender = OmniConnectorFactory.create_connector(
            resolve_connector_spec(head.connectors[("0", "1")], stage_id=0, role="sender")
        )
        receiver = OmniConnectorFactory.create_connector(
            resolve_connector_spec(headless.connectors[("0", "1")], stage_id=1, role="receiver")
        )
        key = f"spell_{uuid.uuid4().hex}"
        assert sender.put("0", "1", key, {"a": 1})[0] is True
        assert receiver.get("0", "1", key)[0] == {"a": 1}

        literal_sender = OmniConnectorFactory.create_connector(
            resolve_connector_spec(
                load_omni_transfer_config(config_dict=config, deployment_scope=f"127.0.0.1:{port}").connectors[
                    ("0", "1")
                ],
                stage_id=0,
                role="sender",
            )
        )
        literal_receiver = OmniConnectorFactory.create_connector(
            resolve_connector_spec(
                load_omni_transfer_config(config_dict=config, deployment_scope=f"localhost:{port}").connectors[
                    ("0", "1")
                ],
                stage_id=1,
                role="receiver",
            )
        )
        assert literal_sender.put("0", "1", key + "-x", {"a": 1})[0] is False
    finally:
        for connector in (literal_receiver, literal_sender, receiver, sender):
            if connector is not None:
                connector.close()
        ctx.term()
        server.stop()


def test_registration_rejects_stale_prefetched_deployment_id():
    """Headless must not spawn after the master id drifts from the prefetch."""
    from vllm_omni.engine.stage_engine_startup import ensure_matching_deployment_id

    ensure_matching_deployment_id(expected="same-id", actual="same-id")
    with pytest.raises(RuntimeError, match="deployment id changed"):
        ensure_matching_deployment_id(expected="prefetch-old", actual="register-new")
