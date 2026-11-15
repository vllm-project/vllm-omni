# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import gc
import struct
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import msgspec
import pytest
import torch

from vllm_omni.distributed.omni_connectors.utils.tensor_payload import (
    UnsupportedTensorPayloadError,
    decode_tensor_payload,
    plan_tensor_payload,
)

pytestmark = [pytest.mark.cpu, pytest.mark.parallel, pytest.mark.core_model]


def pack(value):
    plan = plan_tensor_payload(value, torch.device("cpu"))
    frame = torch.empty(plan.size, dtype=torch.uint8)
    plan.pack_into(frame)
    return frame


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
def test_values_shapes_and_storage_ownership(dtype):
    tensor = torch.arange(24).to(dtype).reshape(4, 6).t()
    payload = {
        "tensor": tensor,
        "nested": [tensor, (torch.tensor(3).to(dtype), torch.empty((0, 3), dtype=dtype))],
        "meta": {"text": "你好", "bytes": b"hello", "flag": True, "none": None},
    }
    frame = pack(payload)
    result = decode_tensor_payload(frame)
    assert result is not None
    # Pool reuse cannot mutate returned values, including non-contiguous inputs.
    frame.zero_()
    assert torch.equal(result["tensor"], tensor)
    assert result["tensor"].dtype == dtype
    assert result["tensor"].shape == tensor.shape
    assert result["tensor"] is result["nested"][0]
    assert type(result["nested"][1]) is tuple
    assert result["nested"][1][1].shape == (0, 3)
    assert result["nested"][1][0].item() == torch.tensor(3).to(dtype).item()
    assert result["meta"] == payload["meta"]


def test_sender_does_not_copy_tensor_to_host(monkeypatch):
    tensor = torch.arange(256, dtype=torch.float32)

    def fail_cpu(*args, **kwargs):
        raise AssertionError("payload copied to CPU")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", fail_cpu)
        frame = pack({"x": tensor})
    result = decode_tensor_payload(frame)
    assert result is not None
    assert torch.equal(result["x"], tensor)


def test_queued_snapshot_budget_and_cancellation(connector):
    payload = {"x": torch.ones(512)}
    prepared = connector.prepare_payload(payload)
    assert 0 < connector._snapshot_bytes <= connector.pool_size
    with pytest.raises(MemoryError, match="queued tensor snapshots"):
        connector.prepare_payload(payload)
    del prepared
    gc.collect()
    assert connector._snapshot_bytes == 0
    connector.prepare_payload(payload)
    gc.collect()
    assert connector._snapshot_bytes == 0


def test_failed_snapshot_releases_budget(connector, monkeypatch):
    from vllm_omni.distributed.omni_connectors.utils.tensor_payload import TensorPayloadPlan

    def fail_snapshot(*args, **kwargs):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(TensorPayloadPlan, "snapshot", fail_snapshot)
    with pytest.raises(RuntimeError, match="copy failed"):
        connector.prepare_payload({"x": torch.ones(4)})
    assert connector._snapshot_bytes == 0


def test_no_tensor_and_unsupported_values_fall_back():
    for value in (
        {"x": 1},
        {"x": torch.ones(2), "y": object()},
        {1: torch.ones(2)},
        {"x": torch.empty(2, device="meta")},
        {"x": torch.ones(2).to_sparse()},
    ):
        with pytest.raises(UnsupportedTensorPayloadError):
            plan_tensor_payload(value, torch.device("cpu"))


def test_cycles_and_metadata_limit():
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(UnsupportedTensorPayloadError, match="Cyclic"):
        pack({"x": torch.ones(1), "cycle": cycle})
    with pytest.raises(UnsupportedTensorPayloadError, match="1 MiB"):
        pack({"x": torch.ones(1), "text": "x" * (1024 * 1024)})


def test_small_and_legacy_frames():
    for value in (b"", b"hello", msgspec.msgpack.encode({"legacy": "long enough for a header"})):
        assert decode_tensor_payload(torch.tensor(list(value), dtype=torch.uint8)) is None


def rewrite_metadata(frame, mutate):
    raw = bytes(frame.tolist())
    magic, size = struct.unpack("<8sQ", raw[:16])
    metadata = msgspec.msgpack.decode(raw[16 : 16 + size])
    old_offset = (16 + size + 63) // 64 * 64
    mutate(metadata)
    encoded = msgspec.msgpack.encode(metadata)
    new_offset = (16 + len(encoded) + 63) // 64 * 64
    data = struct.pack("<8sQ", magic, len(encoded)) + encoded
    return torch.tensor(list(data + bytes(new_offset - len(data)) + raw[old_offset:]), dtype=torch.uint8)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(version=2),
        lambda m: m.update(data_size=m["data_size"] + 1),
        lambda m: m["tensors"][0].update(offset=1),
        lambda m: m["tensors"][0].update(nbytes=1),
        lambda m: m["tensors"][0].update(dtype="invalid"),
        lambda m: m["tensors"][0].update(shape=[-1]),
        lambda m: m.update(tree=["tensor", 999]),
        lambda m: m.update(tree=["dict", [["x", ["tensor", 0]], ["x", ["tensor", 0]]]]),
    ],
)
def test_invalid_metadata_fails_before_destination_clone(mutation, monkeypatch):
    frame = rewrite_metadata(pack({"x": torch.ones(4)}), mutation)

    def fail_clone(*args, **kwargs):
        raise AssertionError("clone before validation")

    monkeypatch.setattr(torch.Tensor, "clone", fail_clone)
    with pytest.raises(ValueError):
        decode_tensor_payload(frame)


def test_truncated_frame_and_oversized_header():
    frame = pack({"x": torch.ones(4)})
    with pytest.raises(ValueError):
        decode_tensor_payload(frame[:-1])
    bad = torch.tensor(list(struct.pack("<8sQ", b"OMNITNSR", 2**63)), dtype=torch.uint8)
    with pytest.raises(ValueError):
        decode_tensor_payload(bad)


@pytest.fixture
def connector(monkeypatch):
    from vllm_omni.distributed.omni_connectors.connectors import mooncake_transfer_engine_connector as module

    def init_pool(self):
        self.pool = torch.empty(self.pool_size, dtype=torch.uint8)
        self.base_ptr = self.pool.data_ptr()
        self.allocator = module.BufferAllocator(self.pool_size)

    with monkeypatch.context() as patch:
        patch.setattr(module, "TransferEngine", object)
        patch.setattr(module.MooncakeTransferEngineConnector, "_init_engine", lambda self: None)
        patch.setattr(module.MooncakeTransferEngineConnector, "_init_pool", init_pool)
        patch.setattr(module.MooncakeTransferEngineConnector, "_init_listener", lambda self, role: None)
        instance = module.MooncakeTransferEngineConnector(
            {"host": "127.0.0.1", "memory_pool_size": 4096, "tensor_payload": True}
        )
    yield instance
    instance.close()


def decode_buffer(connector, buffer, size):
    return connector._decode_result("key", buffer, size, False, 0.0, 0.0, 0.0, 0.0, 0.0)[0]


def test_connector_roundtrip_and_pool_reuse(connector):
    from vllm_omni.distributed.omni_connectors.utils.memory_pool import ManagedBuffer

    payload = {"x": torch.arange(32, dtype=torch.float32)}
    success, size, metadata = connector.put("A", "B", "key", payload)
    assert success and metadata["is_fast_path"] is False
    src = connector._local_buffers["key@A_B"][2]
    frame = src.tensor.clone()
    connector.cleanup("key", "A", "B")
    recv = ManagedBuffer(connector.allocator, connector.allocator.alloc(size), size, connector.pool)
    recv.tensor.copy_(frame)
    result = decode_buffer(connector, recv, size)
    assert connector.allocator.free_blocks == [(0, 4096)]
    connector.pool.zero_()
    assert torch.equal(result["x"], payload["x"])


@pytest.mark.parametrize("enabled", [False, True])
def test_connector_legacy_object_fallback(connector, enabled):
    import numpy as np

    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

    connector.tensor_payload = enabled
    value = {"array": np.arange(8), "tensor": torch.ones(2)}
    success, size, _ = connector.put("A", "B", "legacy", value)
    assert success
    stored = connector._local_buffers["legacy@A_B"][2]
    result = OmniSerializer.deserialize(stored.to_bytes())
    assert np.array_equal(result["array"], value["array"])
    assert torch.equal(result["tensor"], value["tensor"])


@pytest.mark.parametrize(
    "wrap",
    [lambda tensor: {"x": tensor}, lambda tensor: tensor, lambda tensor: tensor.numpy().tobytes()],
    ids=["packed", "tensor", "bytes"],
)
def test_connector_copy_failure_and_pool_exhaustion(connector, monkeypatch, wrap):
    def fail_copy(*args, **kwargs):
        raise RuntimeError("copy failed")

    value = wrap(torch.ones(4))
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "copy_", fail_copy)
        assert connector.put("A", "B", "bad-copy", value) == (False, 0, None)
    assert connector.allocator.free_blocks == [(0, 4096)]
    assert connector._metrics["errors"] == 1
    assert connector.put("A", "B", "too-big", wrap(torch.ones(4096))) == (False, 0, None)
    assert connector.allocator.free_blocks == [(0, 4096)]
    assert connector._metrics["errors"] == 2
    assert not connector._local_buffers


def test_connector_decode_failure_releases_pool(connector):
    from vllm_omni.distributed.omni_connectors.utils.memory_pool import ManagedBuffer

    invalid = rewrite_metadata(pack({"x": torch.ones(2)}), lambda m: m.update(version=99))
    size = invalid.numel()
    recv = ManagedBuffer(connector.allocator, connector.allocator.alloc(size), size, connector.pool)
    recv.tensor.copy_(invalid)
    with pytest.raises(ValueError, match="version"):
        decode_buffer(connector, recv, size)
    assert connector.allocator.free_blocks == [(0, 4096)]


@pytest.mark.parametrize(
    "host, expected", [("127.0.0.1", "127.0.0.1:123"), ("::1", "[::1]:123"), ("[::1]", "[::1]:123")]
)
def test_ipv6_endpoints(host, expected):
    from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import _endpoint

    assert _endpoint(host, 123) == expected


def test_prepared_frame_owns_values_before_background_put(connector):
    tensor = torch.arange(8, dtype=torch.float32)
    metadata = {"step": 1}
    payload = {"nested": [tensor], "meta": metadata}
    prepared = connector.prepare_payload(payload)
    tensor.fill_(-1)
    metadata["step"] = 2
    ok, size, metadata = connector.put("A", "B", "snapshot", prepared)
    assert ok and not metadata["is_fast_path"]
    stored = connector._local_buffers["snapshot@A_B"][2]
    result = decode_tensor_payload(stored.tensor[:size])
    assert torch.equal(result["nested"][0], torch.arange(8, dtype=torch.float32))
    assert result["meta"]["step"] == 1


def test_prepare_legacy_payload_is_unchanged(connector):
    connector.tensor_payload = False
    payload = {"x": torch.ones(4)}
    assert connector.prepare_payload(payload) is payload


def test_pool_created_under_inference_mode_accepts_background_writes(connector, monkeypatch):
    from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
        MooncakeTransferEngineConnector,
    )

    connector.engine = SimpleNamespace(register_memory=lambda *_: 0)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    with torch.inference_mode():
        MooncakeTransferEngineConnector._init_pool(connector)
        prepared = connector.prepare_payload({"x": torch.arange(8)})
    assert not torch.is_inference(connector.pool)
    with ThreadPoolExecutor(max_workers=1) as executor:
        ok, _, _ = executor.submit(connector.put, "A", "B", "inference", prepared).result()
    assert ok


def test_prepare_raw_tensor_preserves_fast_path(connector):
    tensor = torch.arange(8)
    assert connector.prepare_payload(tensor) is tensor
    ok, _, metadata = connector.put("A", "B", "raw", tensor)
    assert ok and metadata["is_fast_path"]
