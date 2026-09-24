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

    monkeypatch.setattr(connector, "deserialize_obj", fail_deserialize)
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
