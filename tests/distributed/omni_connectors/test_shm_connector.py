# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SharedMemoryConnector focusing on TP / CFG / metadata fallback."""

import os

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
        """KV key shape ``omni_{from}_to_{to}_kv_cache_{request_id}``."""
        key = "omni_s0_to_s1_kv_cache_req_42"
        connector.put("s0", "s1", key, {"leak": True})
        assert key in connector._pending_keys

        connector.cleanup("req_42")
        assert key not in connector._pending_keys

        result = connector.get("s0", "s1", key, metadata=None)
        assert result is None

    def test_cleanup_noop_for_consumed_segment(self, connector):
        key = "omni_s0_to_s1_kv_cache_req_99"
        connector.put("s0", "s1", key, {"consumed": True})
        connector.get("s0", "s1", key, metadata=None)

        connector.cleanup("req_99")
        assert key not in connector._pending_keys

    def test_close_cleans_all_pending(self, connector):
        for i in range(3):
            connector.put("s0", "s1", f"close_test_{i}", {"i": i})

        assert len(connector._pending_keys) == 3
        connector.close()
        assert len(connector._pending_keys) == 0

    def test_cleanup_sweeps_chunk_keys_and_lock_files(self, connector):
        """Abort sweep: chunk-style keys ``{ext_req_id}_{stage}_{chunk}`` are
        matched by external request id; both the SHM segments and their
        lock files must be removed, leaving other requests untouched.
        """
        for chunk in range(2):
            ok, _, _ = connector.put("s0", "s1", f"req-c1_0_{chunk}", {"chunk": chunk})
            assert ok
        ok, _, _ = connector.put("s0", "s1", "other-req_0_0", {"keep": True})
        assert ok

        connector.cleanup("req-c1")

        for chunk in range(2):
            assert connector.get("s0", "s1", f"req-c1_0_{chunk}", metadata=None) is None
            assert not os.path.exists(f"/dev/shm/shm_req-c1_0_{chunk}_lockfile.lock")
        assert not any(k.startswith("req-c1_") for k in connector._pending_keys)

        result = connector.get("s0", "s1", "other-req_0_0", metadata=None)
        assert result is not None
        assert result[0] == {"keep": True}

    def test_cleanup_does_not_touch_id_prefixed_sibling(self, connector):
        """``abc`` must not sweep ``abc_def``. A bare ``request_id + "_"``
        prefix match would unlink a live sibling's chunks, stranding its
        consumer on a chunk that no longer exists — strictly worse than
        the leak this sweep fixes.
        """
        connector.put("s0", "s1", "abc_0_0", {"mine": True})
        connector.put("s0", "s1", "abc_def_0_0", {"sibling": True})

        connector.cleanup("abc")

        assert "abc_0_0" not in connector._pending_keys
        assert "abc_def_0_0" in connector._pending_keys
        result = connector.get("s0", "s1", "abc_def_0_0", metadata=None)
        assert result is not None
        assert result[0] == {"sibling": True}

    def test_cleanup_does_not_match_bare_id_suffix(self, connector):
        """Numeric request ids are common, and a bare ``"_" + request_id``
        suffix match would make ``cleanup("0")`` unlink chunk 0 of every
        request in flight.
        """
        connector.put("s0", "s1", "other-req_0_0", {"keep": True})
        connector.put("s0", "s1", "0_0_0", {"mine": True})

        connector.cleanup("0")

        assert "0_0_0" not in connector._pending_keys
        assert "other-req_0_0" in connector._pending_keys
        result = connector.get("s0", "s1", "other-req_0_0", metadata=None)
        assert result is not None
        assert result[0] == {"keep": True}

    def test_cleanup_sweeps_rank_aware_kv_keys(self, connector):
        """Rank-aware KV shape ``{req}_{from_stage}_{chunk}_{from}_{to}``
        stays covered; stage names need not be numeric.
        """
        connector.put("s0", "s1", "req-tp_s0_0_0_1", {"rank": 1})
        connector.put("s0", "s1", "req-tp_s0_0_1_0", {"rank": 0})

        connector.cleanup("req-tp")

        assert not any(k.startswith("req-tp") for k in connector._pending_keys)

    def test_get_removes_lock_file_even_for_falsy_payload(self, connector):
        """The lock file must be tied to segment consumption, not payload
        truthiness — an empty-dict payload previously leaked its lock file.
        """
        ok, _, _ = connector.put("s0", "s1", "falsy_key_1", {})
        assert ok

        result = connector.get("s0", "s1", "falsy_key_1", metadata=None)
        assert result is not None
        assert result[0] == {}
        assert not os.path.exists("/dev/shm/shm_falsy_key_1_lockfile.lock")
