# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SharedMemoryConnector focusing on TP / CFG / metadata fallback."""

import pytest

from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
    SharedMemoryConnector,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture()
def connector():
    c = SharedMemoryConnector({"shm_threshold_bytes": 64})
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

    def test_get_nonexistent_key_returns_none(self, connector):
        result = connector.get("s0", "s1", "no_such_key_xyz", metadata=None)
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
