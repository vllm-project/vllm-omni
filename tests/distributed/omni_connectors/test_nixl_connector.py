# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Unit tests for the NixlConnector metadata handshake.

NIXL itself is stubbed out, so these tests exercise the ZMQ control plane and
the payload normalisation rather than any actual RDMA transfer.
"""

import sys
import time
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
    stub.NixlWrapper = _FakeNixlAgent  # type: ignore[attr-defined]
    stub.nixl_agent_config = None  # type: ignore[attr-defined]
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
    assert [tuple(region) for group in resolved["descriptor_groups"] for region in group["regions"]] == [
        region for group in published["descriptor_groups"] for region in group["regions"]
    ]


def test_metadata_passed_by_caller_skips_the_handshake(consumer):
    """The diffusion path forwards put()'s metadata and must not need a socket."""
    direct = {"schema_version": 1, "kind": "tensors", "tensor_specs": []}

    assert consumer._resolve_metadata("req-2", direct) is direct


def test_source_endpoint_metadata_overrides_configured_sender(consumer, monkeypatch):
    queried = []
    expected = {"schema_version": 1, "kind": "tensors", "tensor_specs": []}

    def query(get_key, host, port):
        queried.append((get_key, host, port))
        return expected

    monkeypatch.setattr(consumer, "_query_metadata_at", query)

    resolved = consumer._resolve_metadata(
        "req-tp4-rank3",
        {"source_host": "10.0.0.3", "source_port": PORT + 3 * 16},
    )

    assert resolved is expected
    assert queried == [("req-tp4-rank3", "10.0.0.3", PORT + 3 * 16)]


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


def test_idle_producer_lease_expires_without_another_put(nixl_connector_cls):
    connector = nixl_connector_cls({"role": "sender", "lease_seconds": 0.01})
    try:
        connector.put("0", "1", "req-expire", torch.zeros(2))

        deadline = time.monotonic() + 1.0
        while connector._pending and time.monotonic() < deadline:
            time.sleep(0.01)

        assert connector._pending == {}
        assert connector._published == {}
        assert connector._agent.registered == []
        assert connector._registered_descs == []
    finally:
        connector.close()


def test_close_stops_lease_reaper(nixl_connector_cls):
    connector = nixl_connector_cls({"role": "sender"})
    lease_thread = connector._lease_thread

    connector.close()

    assert lease_thread is not None
    assert not lease_thread.is_alive()
    assert connector._lease_thread is None


def test_deferred_transfer_is_retained_while_active(nixl_connector_cls):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _DeferredTransfer

    connector = nixl_connector_cls({"role": "receiver"})
    released = []
    connector._agent.check_xfer_state = lambda handle: "PROC"
    connector._agent.release_xfer_handle = lambda handle: released.append(("handle", handle))
    connector._agent.release_dlist_handle = lambda handle: released.append(("dlist", handle))
    connector._agent.remove_remote_agent = lambda agent: released.append(("agent", agent))
    transfer = _DeferredTransfer(
        tensors=[torch.zeros(1)],
        registrations=["registration"],
        dlists=["local", "remote"],
        handles=["transfer"],
        remote_agent="producer",
    )
    connector._defer_transfer(transfer)
    try:
        connector._reap_deferred_transfers()

        assert released == []
        assert transfer in connector._deferred_transfers
        assert transfer.tensors
    finally:
        connector._agent.check_xfer_state = lambda handle: "DONE"
        connector.close()


def test_deferred_transfer_releases_exactly_once_after_done(nixl_connector_cls):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _DeferredTransfer

    connector = nixl_connector_cls({"role": "receiver"})
    released = []
    connector._agent.check_xfer_state = lambda handle: "DONE"
    connector._agent.release_xfer_handle = lambda handle: released.append(("handle", handle))
    connector._agent.release_dlist_handle = lambda handle: released.append(("dlist", handle))
    connector._agent.remove_remote_agent = lambda agent: released.append(("agent", agent))
    connector._agent.deregister_memory = lambda descs: released.append(("registration", descs))
    transfer = _DeferredTransfer(
        tensors=[torch.zeros(1)],
        registrations=["registration"],
        dlists=["local", "remote"],
        handles=["transfer"],
        remote_agent="producer",
    )
    connector._defer_transfer(transfer)
    try:
        connector._reap_deferred_transfers()
        connector._reap_deferred_transfers()

        assert released == [
            ("handle", "transfer"),
            ("dlist", "local"),
            ("dlist", "remote"),
            ("agent", "producer"),
            ("registration", "registration"),
        ]
        assert connector._deferred_transfers == []
        assert transfer.tensors == []
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


def test_structured_payload_groups_descriptors_by_memory_type(producer, monkeypatch):
    monkeypatch.setattr(
        producer,
        "_resolve_memory_type",
        lambda tensor: "DRAM" if tensor.dtype == torch.uint8 else "VRAM",
    )

    _, _, metadata = producer.put(
        "0",
        "1",
        "req-mixed",
        {"meta": "value", "hidden": torch.ones(4, dtype=torch.float32)},
    )

    assert metadata["schema_version"] == 1
    assert metadata["descriptor_groups"] == [
        {
            "memory_type": "DRAM",
            "tensor_indices": [0],
            "regions": metadata["descriptor_groups"][0]["regions"],
        },
        {
            "memory_type": "VRAM",
            "tensor_indices": [1],
            "regions": metadata["descriptor_groups"][1]["regions"],
        },
    ]
    assert len(producer._agent.registered) == 2


def test_partial_group_registration_failure_rolls_back(producer, monkeypatch):
    monkeypatch.setattr(
        producer,
        "_resolve_memory_type",
        lambda tensor: "DRAM" if tensor.dtype == torch.uint8 else "VRAM",
    )
    original_register = producer._agent.register_memory

    def fail_second_group(descs, backends=None):
        if descs[2] == "VRAM":
            raise RuntimeError("registration failed")
        original_register(descs, backends=backends)

    monkeypatch.setattr(producer._agent, "register_memory", fail_second_group)

    success, _, metadata = producer.put(
        "0",
        "1",
        "req-partial",
        {"meta": "value", "hidden": torch.ones(4, dtype=torch.float32)},
    )

    assert success is False
    assert metadata is None
    assert producer._agent.registered == []
    assert producer._registered_descs == []
    assert "req-partial" not in producer._pending


def test_reusing_put_key_releases_previous_registration(producer):
    producer.put("0", "1", "req-reuse", torch.zeros(2))
    previous_registration = producer._agent.registered[0]

    producer.put("0", "1", "req-reuse", torch.ones(2))

    assert previous_registration not in producer._agent.registered
    assert len(producer._agent.registered) == 1
    assert len(producer._registered_descs) == 1


def test_expired_snapshot_cannot_claim_replacement(producer):
    from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import _PendingPayload

    old = _PendingPayload([torch.zeros(1)], ["old"], 0.0)
    replacement = _PendingPayload([torch.ones(1)], ["new"], time.monotonic() + 60)
    producer._pending["req-race"] = replacement

    assert producer._take_pending("req-race", expected=old) is None
    assert producer._pending["req-race"] is replacement


@pytest.mark.parametrize(
    "indices",
    [
        [0, 0],
        [0],
        [0, 2],
        [-1, 0],
    ],
)
def test_descriptor_groups_require_exact_tensor_index_partition(nixl_connector_cls, indices):
    metadata = {
        "schema_version": 1,
        "descriptor_groups": [
            {
                "memory_type": "DRAM",
                "tensor_indices": indices,
                "regions": [(1, 4, 0, "")] * len(indices),
            }
        ],
    }

    with pytest.raises(RuntimeError, match="exact partition"):
        nixl_connector_cls._validated_descriptor_groups(metadata, 2)


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
