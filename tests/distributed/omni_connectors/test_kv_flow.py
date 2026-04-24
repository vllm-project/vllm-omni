import json
import struct

import numpy as np
import pytest
import torch

import vllm_omni.distributed.omni_connectors.kv_transfer_manager as kv_transfer_manager_module
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    KVCacheTransferData,
    OmniKVCacheConfig,
    OmniKVTransferManager,
)
from vllm_omni.distributed.omni_connectors.utils.kv_utils import normalize_layer_kv
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.cache]


class MockConnector:
    def __init__(self):
        self.store = {}

    def put(self, from_stage, to_stage, put_key, data):
        # The manager now passes full key as put_key
        key = f"{from_stage}->{to_stage}:{put_key}"
        self.store[key] = data
        return True, len(str(data)), None  # (success, size, metadata)

    def get(self, from_stage, to_stage, get_key, metadata=None):
        # The manager now passes full key as get_key
        key = f"{from_stage}->{to_stage}:{get_key}"
        if key in self.store:
            return self.store[key], len(str(self.store[key]))
        return None


@pytest.fixture
def mock_connector():
    return MockConnector()


@pytest.fixture
def kv_config():
    return OmniKVCacheConfig(
        connector_config={"type": "mock"},
        from_stage="stage1",
        to_stage="stage2",
        stage_id="stage2",  # Acting as receiver for some tests
        need_recv_cache=True,
        need_send_cache=True,
        recv_timeout=1.0,  # Short timeout for tests
    )


@pytest.fixture
def common_constants():
    return {
        "num_layers": 2,
        "num_heads": 4,
        "head_dim": 16,
        "block_size": 8,
        "seq_len": 20,
        "req_id": "req_test_1",
    }


def _decode_stored_payload(data):
    if isinstance(data, torch.Tensor) and data.dtype == torch.uint8 and data.dim() == 1:
        return KVCacheTransferData.from_bytes(data.cpu().numpy().tobytes())

    if isinstance(data, (bytes, bytearray, memoryview)):
        return KVCacheTransferData.from_bytes(data)

    return data


def _make_serialized_payload() -> tuple[bytes, torch.Tensor]:
    key_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    payload = KVCacheTransferData(
        request_id="req-payload",
        layer_blocks={"key_cache": [key_tensor], "value_cache": [None]},
        block_ids=[1],
        metadata={"seq_len": 3},
    ).to_bytes()
    return payload, key_tensor


def _rewrite_serialized_header(payload: bytes, mutate_header) -> bytes:
    header_len = struct.unpack(">I", payload[:4])[0]
    header = json.loads(payload[4 : 4 + header_len])
    mutate_header(header)
    new_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(new_header)) + new_header + payload[4 + header_len :]


def test_manager_extraction(kv_config, mock_connector, common_constants):
    """Test extraction and sending logic in OmniKVTransferManager."""
    num_layers = common_constants["num_layers"]
    block_size = common_constants["block_size"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    seq_len = common_constants["seq_len"]
    req_id = common_constants["req_id"]

    num_blocks = 10
    kv_caches = []
    for _ in range(num_layers):
        k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        # Stack K and V to create [2, num_blocks, block_size, n_heads, head_dim]
        layer_cache = torch.stack([k_cache, v_cache], dim=0)
        kv_caches.append(layer_cache)

    block_ids = [1, 3, 5]
    finished_reqs = {req_id: {"block_ids": block_ids, "seq_len": seq_len}}

    manager = OmniKVTransferManager(kv_config)
    # Mock the connector factory or injection
    manager._connector = mock_connector

    processed = manager.handle_finished_requests_kv_transfer(finished_reqs, kv_caches, block_size, "float32")

    assert req_id in processed

    # Check if data was put into connector
    # Manager builds full key: omni_{from}_to_{to}_kv_cache_{req_id}
    full_request_id = f"omni_stage1_to_stage2_kv_cache_{req_id}"
    expected_key = f"stage1->stage2:{full_request_id}"
    assert expected_key in mock_connector.store

    data = _decode_stored_payload(mock_connector.store[expected_key])
    assert data["request_id"] == req_id
    assert "layer_blocks" in data
    assert len(data["layer_blocks"]["key_cache"]) == num_layers

    # Verify shape of extracted tensor: [seq_len, heads, dim]
    # Note: Manager detaches and moves to CPU
    expected_shape = (seq_len, num_heads, head_dim)
    assert data["layer_blocks"]["key_cache"][0].shape == expected_shape


def test_from_bytes_rejects_out_of_bounds_header_len():
    payload, _ = _make_serialized_payload()
    bad_payload = struct.pack(">I", len(payload)) + payload[4:]

    with pytest.raises(ValueError, match="header_len"):
        KVCacheTransferData.from_bytes(bad_payload)

    with pytest.raises(ValueError, match="header_len"):
        KVCacheTransferData.from_bytes_gpu(torch.tensor(list(bad_payload), dtype=torch.uint8))


def test_from_bytes_rejects_out_of_bounds_tensor_span():
    payload, _ = _make_serialized_payload()
    bad_payload = _rewrite_serialized_header(payload, lambda header: header["td"][0].update({"o": 4096}))

    with pytest.raises(ValueError, match="tensor span"):
        KVCacheTransferData.from_bytes(bad_payload)

    with pytest.raises(ValueError, match="tensor span"):
        KVCacheTransferData.from_bytes_gpu(torch.tensor(list(bad_payload), dtype=torch.uint8))


def test_from_bytes_rejects_unsupported_dtype():
    payload, _ = _make_serialized_payload()
    bad_payload = _rewrite_serialized_header(payload, lambda header: header["td"][0].update({"d": "cuda"}))

    with pytest.raises(ValueError, match="Unsupported dtype"):
        KVCacheTransferData.from_bytes(bad_payload)

    with pytest.raises(ValueError, match="Unsupported dtype"):
        KVCacheTransferData.from_bytes_gpu(torch.tensor(list(bad_payload), dtype=torch.uint8))


def test_from_bytes_uses_explicit_layer_index_descriptor():
    payload, key_tensor = _make_serialized_payload()
    payload_with_explicit_index = _rewrite_serialized_header(
        payload,
        lambda header: header["td"][0].update({"n": "key_cache_extra_suffix", "i": 0}),
    )

    data = KVCacheTransferData.from_bytes(payload_with_explicit_index)

    assert torch.equal(data["layer_blocks"]["key_cache"][0], key_tensor)


def test_update_sender_info_uses_configured_source_stage():
    config = OmniKVCacheConfig(
        connector_config={"type": "mock"},
        stage_id=2,
        engine_input_source=[1],
        need_recv_cache=True,
    )
    manager = OmniKVTransferManager(config)

    manager.update_sender_info(
        {
            0: {"host": "10.0.0.1", "zmq_port": 50151},
            1: {"host": "10.0.0.2", "zmq_port": 50152},
        }
    )

    assert manager.config.connector_config["sender_host"] == "10.0.0.2"
    assert manager.config.connector_config["sender_zmq_port"] == 50152


def test_clone_received_payload_tensors_breaks_buffer_alias():
    payload, key_tensor = _make_serialized_payload()
    raw = np.frombuffer(bytearray(payload), dtype=np.uint8)
    data = KVCacheTransferData.from_bytes(memoryview(raw))

    OmniKVTransferManager._clone_received_payload_tensors(data)
    raw[:] = 0

    assert torch.equal(data["layer_blocks"]["key_cache"][0], key_tensor)


def test_receive_kv_cache_uses_exponential_backoff(monkeypatch):
    config = OmniKVCacheConfig(
        connector_config={"type": "mock"},
        from_stage="sender",
        stage_id="receiver",
        need_recv_cache=True,
        recv_timeout=0.3,
    )
    manager = OmniKVTransferManager(config)

    class _NeverReadyConnector:
        def get(self, **kwargs):
            del kwargs
            return None

    manager._connector = _NeverReadyConnector()

    now = {"value": 0.0}
    sleep_intervals = []

    monkeypatch.setattr(kv_transfer_manager_module.time, "time", lambda: now["value"])

    def _fake_sleep(interval: float) -> None:
        sleep_intervals.append(interval)
        now["value"] += interval

    monkeypatch.setattr(kv_transfer_manager_module.time, "sleep", _fake_sleep)

    data, size = manager.receive_kv_cache_for_request("req-backoff")

    assert (data, size) == (None, 0)
    assert sleep_intervals == pytest.approx([0.01, 0.02, 0.04, 0.08, 0.16])


def test_manager_extraction_tuple_layout(kv_config, mock_connector, common_constants):
    """Test extraction with tuple layout."""
    num_layers = common_constants["num_layers"]
    block_size = common_constants["block_size"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    seq_len = common_constants["seq_len"]
    req_id = common_constants["req_id"]

    num_blocks = 10
    kv_caches = []
    for _ in range(num_layers):
        k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        kv_caches.append((k_cache, v_cache))

    block_ids = [1, 3, 5]
    finished_reqs = {req_id: {"block_ids": block_ids, "seq_len": seq_len}}

    manager = OmniKVTransferManager(kv_config)
    manager._connector = mock_connector

    processed = manager.handle_finished_requests_kv_transfer(finished_reqs, kv_caches, block_size, "float32")
    assert req_id in processed

    full_request_id = f"omni_stage1_to_stage2_kv_cache_{req_id}"
    expected_key = f"stage1->stage2:{full_request_id}"
    assert expected_key in mock_connector.store

    data = _decode_stored_payload(mock_connector.store[expected_key])
    expected_shape = (seq_len, num_heads, head_dim)
    for idx in range(len(kv_caches)):
        assert data["layer_blocks"]["key_cache"][idx].shape == expected_shape
        assert data["layer_blocks"]["value_cache"][idx].shape == expected_shape


def test_manager_extraction_mismatched_kv_block_counts(kv_config, mock_connector, common_constants):
    """Mismatched key/value block counts should not crash extraction."""
    block_size = common_constants["block_size"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    req_id = common_constants["req_id"]

    key_blocks = torch.randn(3, block_size, num_heads, head_dim)
    value_blocks = torch.randn(2, block_size, num_heads, head_dim)
    kv_caches = [(key_blocks, value_blocks)]

    finished_reqs = {req_id: {"block_ids": [0, 1, 2], "seq_len": 32}}

    manager = OmniKVTransferManager(kv_config)
    manager._connector = mock_connector

    processed = manager.handle_finished_requests_kv_transfer(finished_reqs, kv_caches, block_size, "float32")
    assert req_id in processed

    full_request_id = f"omni_stage1_to_stage2_kv_cache_{req_id}"
    expected_key = f"stage1->stage2:{full_request_id}"
    assert expected_key in mock_connector.store

    data = _decode_stored_payload(mock_connector.store[expected_key])
    expected_shape = (2 * block_size, num_heads, head_dim)
    assert data["layer_blocks"]["key_cache"][0].shape == expected_shape
    assert data["layer_blocks"]["value_cache"][0].shape == expected_shape


@pytest.mark.parametrize(
    "invalid_case",
    ["invalid_stacked_shape", "invalid_tuple_length", "non_tensor_entries"],
)
def test_normalize_layer_kv_rejects_invalid_inputs(kv_config, common_constants, invalid_case):
    """_normalize_layer_kv should reject malformed KV representations."""
    block_size = common_constants["block_size"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    req_id = common_constants["req_id"]

    if invalid_case == "invalid_stacked_shape":
        layer_kv = torch.randn(3, block_size, num_heads, head_dim)
    elif invalid_case == "invalid_tuple_length":
        layer_kv = (
            torch.randn(2, block_size, num_heads, head_dim),
            torch.randn(2, block_size, num_heads, head_dim),
            torch.randn(2, block_size, num_heads, head_dim),
        )
    else:
        layer_kv = (torch.randn(2, block_size, num_heads, head_dim), "not-a-tensor")

    normalized = normalize_layer_kv(layer_kv, req_id=req_id, layer_idx=0)
    assert normalized is None


def test_manager_reception(kv_config, mock_connector, common_constants):
    """Test reception and injection logic in OmniKVTransferManager."""
    num_layers = common_constants["num_layers"]
    block_size = common_constants["block_size"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    seq_len = common_constants["seq_len"]
    req_id = common_constants["req_id"]

    expected_shape = (seq_len, num_heads, head_dim)
    key_cache = [torch.randn(expected_shape) for _ in range(num_layers)]
    value_cache = [torch.randn(expected_shape) for _ in range(num_layers)]

    layer_blocks = {"key_cache": key_cache, "value_cache": value_cache}
    metadata = {
        "block_size": block_size,
        "num_layers": num_layers,
        "dtype": "float32",
        "seq_len": seq_len,
    }

    data_to_receive = {
        "request_id": req_id,
        "layer_blocks": layer_blocks,
        "metadata": metadata,
        "block_ids": [],
    }

    # In setUp, from_stage="stage1", stage_id="stage2". recv_stages=("stage1", "stage2")

    manager = OmniKVTransferManager(kv_config)
    manager._connector = mock_connector

    # Pre-populate connector with data
    # Manager builds full key: omni_{from}_to_{to}_kv_cache_{req_id}
    full_request_id = f"omni_stage1_to_stage2_kv_cache_{req_id}"
    store_key = f"stage1->stage2:{full_request_id}"
    mock_connector.store[store_key] = data_to_receive

    req = OmniDiffusionRequest(
        prompts=["test_recv"],
        sampling_params=OmniDiffusionSamplingParams(),
        request_ids=[req_id],
    )
    # req.need_kv_receive = True # Implicitly handled by receive_kv_cache check? No, manager doesn't check it, runner does.
    # But receive_kv_cache in manager checks request_id. Which we need to fix in manager next.
    success = manager.receive_kv_cache(req, target_device=torch.device("cpu"))

    assert success
    assert hasattr(req, "past_key_values")
    assert hasattr(req, "kv_metadata")

    assert len(req.past_key_values.key_cache) == num_layers
    assert torch.allclose(req.past_key_values.key_cache[0], key_cache[0])
    assert req.kv_metadata["seq_len"] == seq_len


def test_manager_reception_prefers_parent_request_id_for_batched_request(kv_config, mock_connector, common_constants):
    """Batched diffusion requests must fetch KV using the parent/global request ID."""
    num_layers = common_constants["num_layers"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    seq_len = common_constants["seq_len"]
    parent_req_id = common_constants["req_id"]

    expected_shape = (seq_len, num_heads, head_dim)
    key_cache = [torch.randn(expected_shape) for _ in range(num_layers)]
    value_cache = [torch.randn(expected_shape) for _ in range(num_layers)]

    data_to_receive = {
        "request_id": parent_req_id,
        "layer_blocks": {"key_cache": key_cache, "value_cache": value_cache},
        "metadata": {"seq_len": seq_len},
        "block_ids": [],
    }

    manager = OmniKVTransferManager(kv_config)
    manager._connector = mock_connector

    full_request_id = f"omni_stage1_to_stage2_kv_cache_{parent_req_id}"
    store_key = f"stage1->stage2:{full_request_id}"
    mock_connector.store[store_key] = data_to_receive

    req = OmniDiffusionRequest(
        prompts=["prompt-a", "prompt-b"],
        sampling_params=OmniDiffusionSamplingParams(),
        request_ids=[f"{parent_req_id}-0", f"{parent_req_id}-1"],
        request_id=parent_req_id,
    )

    success = manager.receive_kv_cache(req, target_device=torch.device("cpu"))

    assert success
    assert req.kv_metadata["seq_len"] == seq_len
    assert torch.allclose(req.past_key_values.key_cache[0], key_cache[0])


def test_receive_multi_kv_cache_uses_parent_request_id_for_cfg_collection(kv_config):
    manager = OmniKVTransferManager(kv_config)

    seen = {}

    def collect_cfg(request_id, cfg_request_ids, kv_transfer_manager, target_device):
        seen["request_id"] = request_id
        seen["cfg_request_ids"] = cfg_request_ids
        seen["kv_transfer_manager"] = kv_transfer_manager
        seen["target_device"] = target_device
        return {"cfg_text_kv_metadata": {"ok": True}}

    req = OmniDiffusionRequest(
        prompts=["prompt-a", "prompt-b"],
        sampling_params=OmniDiffusionSamplingParams(),
        request_ids=["req-parent-0", "req-parent-1"],
        request_id="req-parent",
    )
    req.sampling_params.cfg_kv_request_ids = {"cfg_text": "req-parent__cfg_text"}

    manager.receive_kv_cache = lambda request, target_device=None: request is req

    success = manager.receive_multi_kv_cache(
        req,
        cfg_kv_collect_func=collect_cfg,
        target_device=torch.device("cpu"),
    )

    assert success
    assert seen["request_id"] == "req-parent"
    assert seen["cfg_request_ids"] == {"cfg_text": "req-parent__cfg_text"}
    assert seen["kv_transfer_manager"] is manager
    assert seen["target_device"] == torch.device("cpu")
    assert req.sampling_params.cfg_text_kv_metadata == {"ok": True}


def test_integration_flow(common_constants):
    """Simulate extraction -> connector -> reception."""
    num_layers = common_constants["num_layers"]
    block_size = common_constants["block_size"]
    num_heads = common_constants["num_heads"]
    head_dim = common_constants["head_dim"]
    req_id = common_constants["req_id"]

    sender_config = OmniKVCacheConfig(
        connector_config={"type": "mock"}, from_stage="sender", to_stage="receiver", need_send_cache=True
    )
    sender_manager = OmniKVTransferManager(sender_config)
    connector = MockConnector()
    sender_manager._connector = connector  # Shared connector

    # Create Data
    num_blocks = 5
    kv_caches = []
    for _ in range(num_layers):
        layer = torch.randn(2, num_blocks, block_size, num_heads, head_dim)
        kv_caches.append(layer)

    finished_reqs = {req_id: {"block_ids": [0, 1], "seq_len": 10}}

    # Send
    sender_manager.handle_finished_requests_kv_transfer(finished_reqs, kv_caches, block_size, "float32")

    receiver_config = OmniKVCacheConfig(
        connector_config={"type": "mock"},
        from_stage="sender",
        stage_id="receiver",
        need_recv_cache=True,
        recv_timeout=1.0,
    )
    receiver_manager = OmniKVTransferManager(receiver_config)
    # Share the same mock connector instance
    receiver_manager._connector = connector

    req = OmniDiffusionRequest(
        prompts=["test_integ"],
        sampling_params=OmniDiffusionSamplingParams(),
        request_ids=[req_id],
    )

    # Receive
    success = receiver_manager.receive_kv_cache(req)

    # Verify
    assert success
    assert req.past_key_values is not None
    assert req.kv_metadata["seq_len"] == 10


def test_manager_extraction_no_connector(kv_config, common_constants):
    """Test extraction when connector is unavailable (should still return IDs)."""
    block_size = common_constants["block_size"]
    req_id = common_constants["req_id"]

    manager = OmniKVTransferManager(kv_config)
    # Force connector to be None
    manager._connector = None
    manager.config.connector_config = None
    finished_reqs = {req_id: {"block_ids": [1, 2], "seq_len": 10}}

    processed = manager.handle_finished_requests_kv_transfer(
        finished_reqs, kv_caches=[], block_size=block_size, cache_dtype="float32"
    )

    assert req_id in processed
