"""
Pytest configuration and fixtures for vLLM-omni tests.
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock

import torch
import numpy as np

from vllm_omni.configs import load_config


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_async_llm() -> AsyncMock:
    """Mock AsyncLLM for testing."""
    mock = AsyncMock()
    mock.process.return_value = {"output": "test_output", "hidden_states": None}
    return mock


@pytest.fixture
async def mock_omni_engine() -> AsyncMock:
    """Mock OmniEngine for testing."""
    mock = AsyncMock()
    mock.execute.return_value = {"result": "test_result"}
    return mock


@pytest.fixture
def sample_config() -> dict:
    """Sample configuration for testing."""
    return {
        "general": {
            "name": "vllm-omni-test",
            "version": "0.1.0",
            "debug": True,
            "log_level": "DEBUG"
        },
        "model": {
            "device": "cpu",
            "dtype": "float32",
            "max_model_len": 1024
        },
        "engines": {
            "ar_engine": {"enabled": True, "max_batch_size": 32},
            "diffusion_engine": {"enabled": True, "max_batch_size": 8}
        }
    }


@pytest.fixture
def sample_text_input() -> str:
    """Sample text input for testing."""
    return "Hello, world! This is a test input for vLLM-omni."


@pytest.fixture
def sample_image_input() -> np.ndarray:
    """Sample image input for testing."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_audio_input() -> np.ndarray:
    """Sample audio input for testing."""
    return np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz


@pytest.fixture
def sample_multimodal_input() -> dict:
    """Sample multimodal input for testing."""
    return {
        "text": "Describe this image",
        "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "audio": np.random.randn(16000).astype(np.float32)
    }


@pytest.fixture
def mock_torch_device() -> str:
    """Mock torch device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Pytest markers
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
