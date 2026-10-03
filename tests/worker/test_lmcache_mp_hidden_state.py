# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the LMCache MP hidden-state backend."""

import pickle
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.lmcache_mp_hidden_state import (
    MPHiddenStateBackend,
    layer_order,
    stack_layers,
    unstack_layers,
)

CHUNK_SIZE = 4
HIDDEN = 3
MM_KEYS = ("0", "7")


class _Future:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value


class _FakeReqClient:
    """Records calls and serves whatever the test staged."""

    def __init__(self):
        self.calls = []
        self.registered = True
        self.stored = True
        self.retrieve_result = (b"", 0)
        self.lookup_result = 0

    def register_hidden_state(self, *args):
        self.calls.append(("register", args))
        return _Future(self.registered)

    def store_hidden_state(self, key, instance_id, data):
        self.calls.append(("store", key, instance_id, data))
        return _Future(self.stored)

    def retrieve_hidden_state(self, key, instance_id):
        self.calls.append(("retrieve", key, instance_id))
        return _Future(self.retrieve_result)

    def lookup_hidden_state(self, key, instance_id):
        self.calls.append(("lookup", key, instance_id))
        return _Future(self.lookup_result)


class _FakeAdapter:
    def __init__(self):
        self.req_client = _FakeReqClient()
        self.instance_id = 11
        self.model_name = "test-model"
        self.world_size = 1
        self.lmcache_tokens_per_chunk = CHUNK_SIZE

    def _create_key(self, token_ids, start, end, request_id):
        return SimpleNamespace(token_ids=tuple(token_ids), start=start, end=end, request_id=request_id)


def _layers(rows: int) -> dict[str, torch.Tensor]:
    return {key: torch.full((rows, HIDDEN), float(i)) for i, key in enumerate(layer_order(MM_KEYS))}


def test_stack_then_unstack_round_trips():
    layers = _layers(8)

    stacked = stack_layers(layers, MM_KEYS)

    assert stacked is not None
    assert stacked.shape == (8, len(layer_order(MM_KEYS)), HIDDEN)
    for key, restored in unstack_layers(stacked, MM_KEYS).items():
        assert torch.equal(restored, layers[key])


def test_stack_refuses_layers_of_different_width():
    layers = _layers(8)
    layers["hidden"] = torch.zeros(8, HIDDEN + 1)

    # One object per chunk covers every layer, so a single width is required.
    assert stack_layers(layers, MM_KEYS) is None


def test_stack_refuses_a_missing_layer():
    layers = _layers(8)
    layers.pop("hidden")

    assert stack_layers(layers, MM_KEYS) is None


def test_store_registers_the_layout_once():
    adapter = _FakeAdapter()
    backend = MPHiddenStateBackend(adapter)
    stacked = stack_layers(_layers(2 * CHUNK_SIZE), MM_KEYS)

    assert backend.store(list(range(2 * CHUNK_SIZE)), stacked, 0, "req")
    assert backend.store(list(range(4 * CHUNK_SIZE)), stacked, 2 * CHUNK_SIZE, "req")

    registers = [c for c in adapter.req_client.calls if c[0] == "register"]
    assert len(registers) == 1
    assert registers[0][1] == (11, "test-model", 1, len(layer_order(MM_KEYS)), HIDDEN, "float32")


def test_store_splits_the_range_into_chunks():
    adapter = _FakeAdapter()
    backend = MPHiddenStateBackend(adapter)
    stacked = stack_layers(_layers(3 * CHUNK_SIZE), MM_KEYS)

    assert backend.store(list(range(3 * CHUNK_SIZE)), stacked, CHUNK_SIZE, "req")

    _, key, _, data = next(c for c in adapter.req_client.calls if c[0] == "store")
    chunks = pickle.loads(data)
    assert len(chunks) == 3
    assert all(c.shape[0] == CHUNK_SIZE for c in chunks)
    # The range the server is told must match the rows actually sent.
    assert (key.start, key.end) == (CHUNK_SIZE, 4 * CHUNK_SIZE)


def test_store_reports_failure_when_the_server_refuses_the_layout():
    adapter = _FakeAdapter()
    adapter.req_client.registered = False
    backend = MPHiddenStateBackend(adapter)
    stacked = stack_layers(_layers(CHUNK_SIZE), MM_KEYS)

    assert not backend.store(list(range(CHUNK_SIZE)), stacked, 0, "req")
    assert not any(c[0] == "store" for c in adapter.req_client.calls)


def test_retrieve_reassembles_the_chunks():
    adapter = _FakeAdapter()
    backend = MPHiddenStateBackend(adapter)
    stacked = stack_layers(_layers(2 * CHUNK_SIZE), MM_KEYS)
    chunks = list(stacked.split(CHUNK_SIZE, dim=0))
    adapter.req_client.retrieve_result = (pickle.dumps(chunks), 2 * CHUNK_SIZE)

    restored = backend.retrieve(list(range(2 * CHUNK_SIZE)), "req")

    assert restored is not None
    assert torch.equal(restored, stacked)


def test_retrieve_returns_none_on_a_miss():
    adapter = _FakeAdapter()
    backend = MPHiddenStateBackend(adapter)

    assert backend.retrieve(list(range(CHUNK_SIZE)), "req") is None


def test_retrieve_survives_an_rpc_failure():
    adapter = _FakeAdapter()
    backend = MPHiddenStateBackend(adapter)

    def boom(key, instance_id):
        raise RuntimeError("server gone")

    adapter.req_client.retrieve_hidden_state = boom

    # A cache that cannot be reached is a miss, not a failed step.
    assert backend.retrieve(list(range(CHUNK_SIZE)), "req") is None


def test_lookup_returns_the_covered_token_count():
    adapter = _FakeAdapter()
    adapter.req_client.lookup_result = 2 * CHUNK_SIZE
    backend = MPHiddenStateBackend(adapter)

    assert backend.lookup(list(range(4 * CHUNK_SIZE)), "req") == 2 * CHUNK_SIZE


def test_a_changed_hidden_shape_is_refused_rather_than_reregistered():
    adapter = _FakeAdapter()
    backend = MPHiddenStateBackend(adapter)
    assert backend.store(list(range(CHUNK_SIZE)), stack_layers(_layers(CHUNK_SIZE), MM_KEYS), 0, "req")

    wider = {k: torch.zeros(CHUNK_SIZE, HIDDEN + 2) for k in layer_order(MM_KEYS)}

    assert not backend.store(list(range(CHUNK_SIZE)), stack_layers(wider, MM_KEYS), 0, "req")
    assert len([c for c in adapter.req_client.calls if c[0] == "register"]) == 1


@pytest.mark.parametrize("mm_keys", [(), ("0",), ("0", "7")])
def test_layer_order_always_ends_with_the_final_tap(mm_keys):
    assert layer_order(mm_keys)[-1] == "hidden"
    assert len(layer_order(mm_keys)) == len(mm_keys) + 1
