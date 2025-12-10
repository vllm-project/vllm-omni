# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest

from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer


def test_pickle_serialization():
    """Test basic pickle serialization."""
    data = {"key": "value", "list": [1, 2, 3]}
    serialized = OmniSerializer.serialize(data, method="cloudpickle")
    assert isinstance(serialized, bytes)

    deserialized = OmniSerializer.deserialize(serialized, method="cloudpickle")
    assert data == deserialized


def test_create_shm_connector():
    """Test creating SharedMemoryConnector via Factory."""
    spec = ConnectorSpec(name="SharedMemoryConnector", extra={"shm_threshold_bytes": 1024})
    connector = OmniConnectorFactory.create_connector(spec)
    assert isinstance(connector, SharedMemoryConnector)
    assert connector.threshold == 1024


def test_create_unknown_connector():
    """Test error when creating unknown connector."""
    spec = ConnectorSpec(name="UnknownConnector")
    with pytest.raises(ValueError):
        OmniConnectorFactory.create_connector(spec)


@pytest.fixture
def shm_connector():
    config = {"shm_threshold_bytes": 100}  # Small threshold for testing
    return SharedMemoryConnector(config)


def test_put_get_inline(shm_connector):
    """Test inline transfer for small data."""
    data = {"small": "data"}
    # Ensure data is smaller than threshold (100 bytes)

    success, size, metadata = shm_connector.put("stage_0", "stage_1", "req_1", data)
    assert success is True
    assert "inline_bytes" in metadata
    assert "shm" not in metadata

    # Retrieve
    retrieved_data, ret_size = shm_connector.get("stage_0", "stage_1", "req_1", metadata)
    assert data == retrieved_data
    assert size == ret_size


def test_put_get_shm(shm_connector, monkeypatch):
    """Test SHM transfer logic for large data (Mocked)."""
    # Create data larger than 100 bytes
    data = {"large": "x" * 200}

    # Mock SHM return values
    mock_handle = {"name": "test_shm", "size": 200}
    mock_write = MagicMock(return_value=mock_handle)
    monkeypatch.setattr("vllm_omni.distributed.omni_connectors.connectors.shm_connector.shm_write_bytes", mock_write)

    # When reading, return the serialized bytes of the data
    serialized_data = shm_connector.serialize_obj(data)
    mock_read = MagicMock(return_value=serialized_data)
    monkeypatch.setattr("vllm_omni.distributed.omni_connectors.connectors.shm_connector.shm_read_bytes", mock_read)

    # Put
    success, size, metadata = shm_connector.put("stage_0", "stage_1", "req_2", data)

    assert success is True
    # Should use SHM because data > threshold
    assert "shm" in metadata
    assert metadata["shm"] == mock_handle
    assert "inline_bytes" not in metadata

    mock_write.assert_called_once()

    # Get
    retrieved_data, ret_size = shm_connector.get("stage_0", "stage_1", "req_2", metadata)

    assert data == retrieved_data
    mock_read.assert_called_once_with(mock_handle)


def test_get_invalid_metadata(shm_connector):
    """Test get with invalid metadata."""
    result = shm_connector.get("stage_0", "stage_1", "req_3", {})
    assert result is None

    result = shm_connector.get("stage_0", "stage_1", "req_3", {"unknown": "format"})
    assert result is None
