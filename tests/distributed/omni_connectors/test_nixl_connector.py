# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for the NixlConnector metadata handshake.

NIXL itself is stubbed out, so these tests exercise the ZMQ control plane and
the payload normalisation rather than any actual RDMA transfer.
"""

import sys
import types

import msgspec
import pytest
import torch

pytestmark = [pytest.mark.cpu, pytest.mark.parallel, pytest.mark.core_model]

PORT = 47431


class _FakeNixlAgent:
    def __init__(self, name, config=None):
        self.name = name
        self.registered = []

    def get_reg_descs(self, regions, memory_type):
        return ("reg", tuple(regions), memory_type)

    def register_memory(self, descs, backends=None):
        self.registered.append(descs)

    def deregister_memory(self, descs):
        if descs in self.registered:
            self.registered.remove(descs)

    def get_agent_metadata(self):
        return f"agent-metadata-{self.name}".encode()


@pytest.fixture
def nixl_connector_cls(monkeypatch):
    """Import NixlConnector with vLLM's optional NIXL dependency stubbed out."""
    stub = types.ModuleType("vllm.distributed.nixl_utils")
    stub.NixlWrapper = _FakeNixlAgent
    stub.nixl_agent_config = None
    monkeypatch.setitem(sys.modules, "vllm.distributed.nixl_utils", stub)

    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import NixlConnector

    return NixlConnector


@pytest.fixture
def producer(nixl_connector_cls):
    connector = nixl_connector_cls({"role": "sender", "host": "127.0.0.1", "zmq_port": PORT})
    yield connector
    connector.close()


@pytest.fixture
def consumer(nixl_connector_cls):
    connector = nixl_connector_cls({"role": "receiver", "sender_host": "127.0.0.1", "sender_zmq_port": PORT})
    yield connector
    connector.close()


def test_put_publishes_its_handshake_endpoint(producer):
    ok, size, metadata = producer.put("0", "1", "req-0", torch.arange(8, dtype=torch.float32))

    assert ok is True
    assert size == 32
    assert metadata["sender_host"] == "127.0.0.1"
    assert metadata["sender_zmq_port"] == PORT


def test_handshake_serves_metadata_when_caller_has_none(producer, consumer):
    _, _, published = producer.put("0", "1", "req-1", torch.arange(4, dtype=torch.float32))

    resolved = consumer._resolve_metadata("req-1", None)

    # msgpack has no tuple type, so region descriptors arrive as lists; get()
    # re-tuples them before handing them to NIXL.
    assert resolved == msgspec.msgpack.decode(msgspec.msgpack.encode(published))
    assert [tuple(region) for region in resolved["regions"]] == published["regions"]


def test_metadata_passed_by_caller_skips_the_handshake(consumer):
    """The diffusion path forwards put()'s metadata and must not need a socket."""
    direct = {"schema_version": 1, "kind": "tensors", "tensor_specs": []}

    assert consumer._resolve_metadata("req-2", direct) is direct


def test_unknown_key_is_retried_then_reported(producer, consumer):
    consumer._handshake_max_wait_s = 0.2

    assert consumer._resolve_metadata("never-published", None) is None


def test_transfer_done_releases_the_producer_buffer(producer, consumer):
    _, _, metadata = producer.put("0", "1", "req-3", torch.arange(4, dtype=torch.float32))
    assert producer._pending and producer._agent.registered

    consumer._notify_transfer_done("req-3", metadata)

    assert producer._pending == {}
    assert producer._published == {}
    assert producer._agent.registered == []


def test_handshake_stays_off_without_configuration(nixl_connector_cls):
    connector = nixl_connector_cls({})
    try:
        assert connector._zmq_ctx is None
        assert connector._listener_thread is None
        _, _, metadata = connector.put("0", "1", "req-4", torch.zeros(2))
        assert "sender_host" not in metadata
    finally:
        connector.close()


@pytest.mark.parametrize(
    "payload,expected_kind",
    [
        (torch.zeros(4), "tensors"),
        ([torch.zeros(2), torch.ones(3)], "tensors"),
        ({"hidden": torch.zeros(4), "meta": {"token_role_ids": [1, 2]}}, "structured"),
        ({"prompt": "a cat", "steps": 8}, "object"),
    ],
)
def test_payload_kinds_round_trip_through_the_handshake(producer, consumer, payload, expected_kind):
    _, _, published = producer.put("0", "1", "req-kind", payload)
    assert published["kind"] == expected_kind

    resolved = consumer._resolve_metadata("req-kind", None)

    assert resolved["kind"] == expected_kind
    assert resolved["tensor_specs"] == published["tensor_specs"]


def test_receive_device_ignores_the_producer_index(nixl_connector_cls):
    """A producer on cuda:3 must not dictate the consumer's card."""
    connector = nixl_connector_cls({})
    try:
        assert connector._resolve_receive_device("cpu") == torch.device("cpu")
        assert connector._resolve_receive_device(None) == torch.device("cpu")
    finally:
        connector.close()


def test_receive_device_config_wins(nixl_connector_cls):
    connector = nixl_connector_cls({"receive_device": "cpu"})
    try:
        assert connector._resolve_receive_device("cuda:3") == torch.device("cpu")
    finally:
        connector.close()
