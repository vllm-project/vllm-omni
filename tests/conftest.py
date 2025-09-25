"""
Shared fixtures and configuration for vLLM-omni tests.
"""

import pytest
import torch
from unittest.mock import Mock
from vllm_omni.config import OmniStageConfig, DiTConfig, DiTCacheConfig, create_ar_stage_config, create_dit_stage_config


@pytest.fixture(scope="session")
def device():
    """Get available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_ar_stage_config():
    """Sample AR stage configuration for testing."""
    return create_ar_stage_config(
        stage_id=0,
        model_path="test-ar-model",
        input_modalities=["text"],
        output_modalities=["text"]
    )


@pytest.fixture
def sample_dit_stage_config():
    """Sample DiT stage configuration for testing."""
    dit_config = DiTConfig(
        model_type="dit",
        scheduler_type="ddpm",
        num_inference_steps=10,
        guidance_scale=7.5
    )
    
    return create_dit_stage_config(
        stage_id=1,
        model_path="test-dit-model",
        input_modalities=["text"],
        output_modalities=["image"],
        dit_config=dit_config
    )


@pytest.fixture
def sample_stage_configs(sample_ar_stage_config, sample_dit_stage_config):
    """Sample stage configurations for testing."""
    return [sample_ar_stage_config, sample_dit_stage_config]


@pytest.fixture
def mock_vllm_config():
    """Mock vLLM configuration."""
    config = Mock()
    config.model = "test-model"
    config.tensor_parallel_size = 1
    config.pipeline_parallel_size = 1
    return config


@pytest.fixture
def mock_dit_cache_config():
    """Mock DiT cache configuration."""
    from vllm_omni.config import DiTCacheTensor
    
    cache_tensors = [
        DiTCacheTensor(
            name="test_tensor",
            shape=[1, 512, 512],
            dtype="float32",
            persistent=True
        )
    ]
    
    return DiTCacheConfig(
        cache_tensors=cache_tensors,
        max_cache_size=1024 * 1024 * 1024,  # 1GB
        cache_strategy="fifo",
        enable_optimization=True
    )