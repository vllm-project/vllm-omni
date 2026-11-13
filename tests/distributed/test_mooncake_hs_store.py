# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in native Store checks for HS publication and validation.

Set MOONCAKE_CONFIG_PATH to a single-host Store JSON configuration and start
its master before running. Tests own UUID-scoped keys and remove only those
keys. They exercise real Store operations, but no GPU KV transfer or model.
"""

import hashlib
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import MooncakeStoreConnectorStats

from vllm_omni.distributed.kv_transfer.mooncake import connector as connector_module
from vllm_omni.distributed.kv_transfer.mooncake.config import MooncakeConfig
from vllm_omni.distributed.kv_transfer.mooncake.connector import Attempt, OmniMooncakeStoreConnector, PendingSave
from vllm_omni.distributed.kv_transfer.mooncake.hs import decode_hs, hs_key

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class NativeStore(Protocol):
    def get(self, key: str) -> bytes | None: ...

    def get_batch(self, keys: list[str]) -> list[bytes]: ...

    def put(self, key: str, payload: bytes) -> int: ...

    def is_exist(self, key: str) -> int: ...

    def batch_is_exist(self, keys: list[str]) -> list[int]: ...

    def remove(self, key: str, force: bool = False) -> int: ...


@dataclass
class NativeTokenDatabase:
    prefix: str

    def key_for(self, block_hash: bytes) -> str:
        return self.prefix + block_hash.hex()


@dataclass
class NativeWorkerSeam:
    store: NativeStore
    token_dbs: list[NativeTokenDatabase]
    kv_connector_stats: MooncakeStoreConnectorStats = field(default_factory=MooncakeStoreConnectorStats)

    def _record_kv_connector_operation(self, operation, duration_seconds, num_keys, **fields):
        self.kv_connector_stats.record_operation(operation, duration_seconds, num_keys, **fields)


@pytest.fixture
def native_store():
    if not os.environ.get("MOONCAKE_CONFIG_PATH"):
        pytest.skip("Set MOONCAKE_CONFIG_PATH to opt in to a running single-host Mooncake Store")
    mooncake = pytest.importorskip("mooncake.store", reason="Mooncake Store is not installed")
    from vllm.distributed.mooncake_store import MooncakeStoreConfig, setup_mooncake_store

    store = mooncake.MooncakeDistributedStore()
    namespace = "omni-m1-hs-test-" + uuid.uuid4().hex
    keys: list[str] = []
    try:
        setup_mooncake_store(store, MooncakeStoreConfig.load_from_config(), "127.0.0.1")
        yield store, namespace, keys
    finally:
        for key in keys:
            store.remove(key, force=True)
        store.close()


@pytest.fixture
def native_connector(native_store):
    store, namespace, keys = native_store
    connector = OmniMooncakeStoreConnector.__new__(OmniMooncakeStoreConnector)
    connector._closed = True
    connector._lock = threading.RLock()
    connector._reserved_bytes = 0
    connector.namespace = namespace
    connector.block_size = 4
    connector.hidden_size = 8
    connector.hidden_layer = 24
    connector.hs_dtype = torch.bfloat16
    connector._row_bytes = 2 * 8 * 2
    connector.settings = MooncakeConfig(
        config_file=os.environ["MOONCAKE_CONFIG_PATH"],
        namespace=namespace,
        model_revision="native-test-model-revision",
        load_timeout_seconds=2,
    )
    connector.connector_worker = NativeWorkerSeam(
        store=store,
        token_dbs=[NativeTokenDatabase(namespace + ":kv:")],
    )
    attempt = Attempt("request", uuid.uuid4().hex, 5, [b"prefix-0"])
    hs = hs_key(namespace, attempt.hashes[0])
    kv = connector.connector_worker.token_dbs[0].key_for(attempt.hashes[0])
    keys.extend([hs, hs + ":ready", kv])
    return connector, attempt, hs, kv


def tensors(offset=0):
    embedding = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8) + offset
    return {"embedding": embedding, "hidden": embedding + 100}


def test_native_hs_save_read_is_exact(native_connector):
    connector, attempt, key, kv_key = native_connector
    store = connector.connector_worker.store
    # This sentinel represents a committed KV object; the test deliberately
    # does not claim to exercise attention-KV copying.
    assert store.put(kv_key, b"test-kv-commit") == 0
    expected = tensors()
    connector._save_hs(PendingSave(attempt, [(0, 4, expected)], 0))
    payload = store.get(key)
    assert store.get(key + ":ready") == hashlib.sha256(payload).digest()
    actual = connector._read_hs(attempt, 4)
    for name in expected:
        assert actual[name].shape == expected[name].shape
        assert actual[name].dtype == expected[name].dtype
        assert torch.equal(actual[name], expected[name])


def test_native_missing_and_evicted_hs_fail_before_assembly(native_connector):
    connector, attempt, key, kv_key = native_connector
    store = connector.connector_worker.store
    assert store.get(key) in (None, b"")
    with pytest.raises(ValueError, match="missing HS"):
        connector._read_hs(attempt, 4)
    assert store.put(kv_key, b"test-kv-commit") == 0
    connector._save_hs(PendingSave(attempt, [(0, 4, tensors())], 0))
    assert store.batch_is_exist([key, key + ":ready"]) == [1, 1]
    assert store.remove(key, force=True) == 0
    with pytest.raises(ValueError, match="missing HS"):
        connector._read_hs(attempt, 4)


def test_native_short_object_rejected_even_with_matching_digest(native_connector):
    connector, attempt, key, _ = native_connector
    store = connector.connector_worker.store
    short = b"incomplete-tensor"
    assert store.put(key, short) == 0
    assert store.put(key + ":ready", hashlib.sha256(short).digest()) == 0
    with pytest.raises(ValueError):
        connector._read_hs(attempt, 4)


def test_native_hs_without_completion_is_not_readable(native_connector):
    connector, attempt, key, kv_key = native_connector
    store = connector.connector_worker.store
    assert store.put(kv_key, b"test-kv-commit") == 0
    connector._save_hs(PendingSave(attempt, [(0, 4, tensors())], 0))
    assert store.remove(key + ":ready", force=True) == 0
    with pytest.raises(ValueError, match="completion record"):
        connector._read_hs(attempt, 4)


def test_native_refill_replaces_stale_completion_after_eviction(native_connector):
    connector, attempt, key, kv_key = native_connector
    store = connector.connector_worker.store
    assert store.put(kv_key, b"test-kv-commit") == 0
    connector._save_hs(PendingSave(attempt, [(0, 4, tensors())], 0))
    assert store.remove(key, force=True) == 0
    replacement = tensors(10)
    connector._save_hs(PendingSave(attempt, [(0, 4, replacement)], 0))
    restored = connector._read_hs(attempt, 4)
    assert all(torch.equal(restored[name], replacement[name]) for name in replacement)


def test_native_concurrent_duplicate_writes_publish_one_complete_winner(native_connector):
    connector, attempt, key, kv_key = native_connector
    store = connector.connector_worker.store
    assert store.put(kv_key, b"test-kv-commit") == 0
    alternatives = [tensors(0), tensors(10)]
    barrier = threading.Barrier(2)

    def write(index):
        barrier.wait(timeout=10)
        connector._save_hs(PendingSave(attempt, [(0, 4, alternatives[index])], 0))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(write, index) for index in range(2)]
        for future in futures:
            future.result(timeout=30)
    payload = store.get(key)
    assert store.get(key + ":ready") == hashlib.sha256(payload).digest()
    actual = decode_hs(payload, connector._spec(attempt.hashes[0], 0))
    assert any(all(torch.equal(actual[name], candidate[name]) for name in actual) for candidate in alternatives)
    # A later duplicate must leave both the existing object and its commit
    # record usable. M1 relies on the pinned Store's immutable-key behavior.
    connector._save_hs(PendingSave(attempt, [(0, 4, tensors(20))], 0))
    assert store.get(key) == payload
    restored = connector._read_hs(attempt, 4)
    assert all(torch.equal(restored[name], actual[name]) for name in actual)


def test_native_batched_read_spans_batches_and_rejects_a_gap(native_store, native_connector, monkeypatch):
    store, namespace, cleanup = native_store
    connector, attempt, _, _ = native_connector
    attempt.hashes = [b"prefix-0", b"prefix-1", b"prefix-2"]
    expected = [tensors(10 * index) for index in range(3)]
    for block_hash in attempt.hashes:
        key = hs_key(namespace, block_hash)
        kv_key = connector.connector_worker.token_dbs[0].key_for(block_hash)
        cleanup.extend([key, key + ":ready", kv_key])
        assert store.put(kv_key, b"test-kv-commit") == 0
    connector._save_hs(PendingSave(attempt, [(4 * i, 4 * i + 4, expected[i]) for i in range(3)], 0))
    # Two blocks per get_batch call, so this prefix needs two Store calls.
    monkeypatch.setattr(connector_module, "_READ_BATCH_BYTES", 2 * connector.block_size * connector._row_bytes)
    restored = connector._read_hs(attempt, 12)
    for name in ("embedding", "hidden"):
        assert torch.equal(restored[name], torch.cat([block[name] for block in expected]))
    assert store.remove(hs_key(namespace, attempt.hashes[1]), force=True) == 0
    with pytest.raises(ValueError, match="missing HS"):
        connector._read_hs(attempt, 12)
