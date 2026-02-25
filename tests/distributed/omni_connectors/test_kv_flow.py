import pytest
import torch

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVCacheConfig,
    OmniKVTransferManager,
)
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

    data = mock_connector.store[expected_key]
    assert data["request_id"] == req_id
    assert "layer_blocks" in data
    assert len(data["layer_blocks"]["key_cache"]) == num_layers

    # Verify shape of extracted tensor: [seq_len, heads, dim]
    # Note: Manager detaches and moves to CPU
    expected_shape = (seq_len, num_heads, head_dim)
    assert data["layer_blocks"]["key_cache"][0].shape == expected_shape


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
