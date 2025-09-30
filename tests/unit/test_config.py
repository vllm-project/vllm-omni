"""
Unit tests for configuration modules.
"""

import pytest
from vllm_omni.config import (
    OmniStageConfig,
    DiTConfig,
    DiTCacheConfig,
    DiTCacheTensor,
    create_ar_stage_config,
    create_dit_stage_config,
)


class TestOmniStageConfig:
    def test_ar_stage_config_creation(self):
        """Test AR stage configuration creation."""
        config = OmniStageConfig(
            stage_id=0,
            engine_type="AR",
            model_path="test-model",
            input_modalities=["text"],
            output_modalities=["text"]
        )
        assert config.engine_type == "AR"
        assert config.stage_id == 0
        assert config.model_path == "test-model"
        assert config.input_modalities == ["text"]
        assert config.output_modalities == ["text"]
    
    def test_dit_stage_config_creation(self):
        """Test DiT stage configuration creation."""
        dit_config = DiTConfig(
            model_type="dit",
            scheduler_type="ddpm",
            num_inference_steps=50
        )
        config = OmniStageConfig(
            stage_id=1,
            engine_type="DiT",
            model_path="test-dit-model",
            input_modalities=["text"],
            output_modalities=["image"],
            dit_config=dit_config
        )
        assert config.engine_type == "DiT"
        assert config.dit_config is not None
        assert config.dit_config.model_type == "dit"
    
    def test_invalid_engine_type(self):
        """Test validation of engine type."""
        with pytest.raises(ValueError):
            OmniStageConfig(
                stage_id=0,
                engine_type="INVALID",
                model_path="test-model",
                input_modalities=["text"],
                output_modalities=["text"]
            )
    
    def test_empty_modalities(self):
        """Test validation of empty modalities."""
        with pytest.raises(ValueError):
            OmniStageConfig(
                stage_id=0,
                engine_type="AR",
                model_path="test-model",
                input_modalities=[],
                output_modalities=["text"]
            )
        
        with pytest.raises(ValueError):
            OmniStageConfig(
                stage_id=0,
                engine_type="AR",
                model_path="test-model",
                input_modalities=["text"],
                output_modalities=[]
            )
    
    def test_dit_requires_config(self):
        """Test that DiT engine requires dit_config."""
        with pytest.raises(ValueError):
            OmniStageConfig(
                stage_id=1,
                engine_type="DiT",
                model_path="test-dit-model",
                input_modalities=["text"],
                output_modalities=["image"]
            )


class TestDiTConfig:
    def test_dit_config_defaults(self):
        """Test DiT configuration with defaults."""
        config = DiTConfig(
            model_type="dit",
            scheduler_type="ddpm",
            num_inference_steps=50
        )
        assert config.use_diffusers is False
        assert config.diffusers_pipeline is None
        assert config.guidance_scale == 7.5
        assert config.height == 512
        assert config.width == 512
    
    def test_diffusers_config(self):
        """Test DiT configuration with diffusers."""
        config = DiTConfig(
            model_type="dit",
            scheduler_type="ddpm",
            num_inference_steps=50,
            use_diffusers=True,
            diffusers_pipeline="stable-diffusion"
        )
        assert config.use_diffusers is True
        assert config.diffusers_pipeline == "stable-diffusion"


class TestDiTCacheConfig:
    def test_cache_config_creation(self):
        """Test cache configuration creation."""
        cache_tensors = [
            DiTCacheTensor(
                name="test_tensor",
                shape=[1, 512, 512],
                dtype="float32",
                persistent=True
            )
        ]
        
        config = DiTCacheConfig(
            cache_tensors=cache_tensors,
            max_cache_size=1024 * 1024 * 1024,
            cache_strategy="fifo"
        )
        
        assert len(config.cache_tensors) == 1
        assert config.max_cache_size == 1024 * 1024 * 1024
        assert config.cache_strategy == "fifo"
        assert config.enable_optimization is True


class TestHelperFunctions:
    def test_create_ar_stage_config(self):
        """Test create_ar_stage_config helper function."""
        config = create_ar_stage_config(
            stage_id=0,
            model_path="test-model"
        )
        
        assert config.stage_id == 0
        assert config.engine_type == "AR"
        assert config.model_path == "test-model"
        assert config.input_modalities == ["text"]
        assert config.output_modalities == ["text"]
    
    def test_create_dit_stage_config(self):
        """Test create_dit_stage_config helper function."""
        dit_config = DiTConfig(
            model_type="dit",
            scheduler_type="ddpm",
            num_inference_steps=50
        )
        
        config = create_dit_stage_config(
            stage_id=1,
            model_path="test-dit-model",
            dit_config=dit_config
        )
        
        assert config.stage_id == 1
        assert config.engine_type == "DiT"
        assert config.model_path == "test-dit-model"
        assert config.input_modalities == ["text"]
        assert config.output_modalities == ["image"]
        assert config.dit_config is not None

