# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for TeaCache extractor functions.

This module provides a generic testing framework for model-specific extractor functions
used by TeaCache. Each model's extractor can be tested by:
1. Creating a fixture that returns model module
2. Creating a fixture that returns sample inputs for that model
3. Creating a test class that inherits from BaseExtractorTest
4. Implementing any model-specific test methods

Currently implemented:
- TestFlux2KleinExtractor: Flux2Klein model extractor
"""

from abc import ABC, abstractmethod
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.extractors import extract_flux2_klein_context
from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
    Flux2Transformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="function", autouse=True)
def setup_tp_group():
    """Set up TP group for each test function"""
    with patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1):
        with patch("vllm.distributed.parallel_state.get_tp_group") as mock_get_tp_group:
            mock_tp_group = MagicMock()
            mock_tp_group.world_size = 1
            mock_get_tp_group.return_value = mock_tp_group
            yield


class BaseExtractorTest(ABC):
    """Base class for testing TeaCache extractors.

    Subclasses should implement:
    - get_extractor(): Return extractor function
    - get_module(): Return model module
    - get_sample_inputs(): Return sample inputs for model
    """

    @abstractmethod
    def get_extractor(self):
        """Return extractor function to test."""
        pass

    @abstractmethod
    def get_module(self):
        """Return model module instance."""
        pass

    @abstractmethod
    def get_sample_inputs(self):
        """Return sample inputs for model."""
        pass


class TestFlux2KleinExtractor(BaseExtractorTest):
    """Test extract_flux2_klein_context function."""

    def get_extractor(self):
        return extract_flux2_klein_context

    @pytest.fixture
    def flux2_klein_module(self):
        """Create a minimal Flux2Transformer2DModel for testing."""
        model = Flux2Transformer2DModel(
            num_layers=2,
            num_single_layers=2,
            num_attention_heads=48,
            attention_head_dim=128,
            joint_attention_dim=15360,
        )
        return model

    def get_module(self, flux2_klein_module):
        return flux2_klein_module

    @pytest.fixture
    def sample_inputs(self):
        """Create sample input tensors for Flux2Klein.

        Note: hidden_states uses in_channels=128 (default for Flux2Klein),
        not inner_dim=6144. The x_embedder projects from 128 -> 6144.
        encoder_hidden_states uses joint_attention_dim=15360 (model default),
        which then gets projected to inner_dim=6144 by context_embedder.
        """
        batch_size = 1
        img_seq_len = 1024
        txt_seq_len = 512
        in_channels = 128  # Model default in_channels
        txt_dim = 15360  # Model default joint_attention_dim

        return {
            "hidden_states": torch.randn(batch_size, img_seq_len, in_channels),
            "encoder_hidden_states": torch.randn(batch_size, txt_seq_len, txt_dim),
            "timestep": torch.tensor([500]),
            "img_ids": torch.randint(0, 64, (batch_size, img_seq_len, 4)),
            "txt_ids": torch.randint(0, 64, (batch_size, txt_seq_len, 4)),
            "guidance": torch.tensor([3.5]),
        }

    def get_sample_inputs(self, sample_inputs):
        return sample_inputs

    def test_modulated_input_shape(self, flux2_klein_module, sample_inputs):
        """Test that modulated_input has correct shape matching the model's inner_dim.

        Note: After x_embedder projection, hidden_states are projected from
        in_channels (128) to inner_dim (6144), so modulated_input should match
        the projected shape, not the input shape.
        """
        context = extract_flux2_klein_context(flux2_klein_module, **sample_inputs)

        batch_size, img_seq_len, _ = sample_inputs["hidden_states"].shape
        inner_dim = flux2_klein_module.inner_dim
        assert context.modulated_input.shape == (batch_size, img_seq_len, inner_dim)

    def test_run_transformer_blocks_callable(self, flux2_klein_module, sample_inputs):
        """Test that run_transformer_blocks is callable."""
        context = extract_flux2_klein_context(flux2_klein_module, **sample_inputs)
        assert callable(context.run_transformer_blocks)

    def test_postprocess_callable(self, flux2_klein_module, sample_inputs):
        """Test that postprocess is callable."""
        context = extract_flux2_klein_context(flux2_klein_module, **sample_inputs)
        assert callable(context.postprocess)

    def test_extra_states_contains_full_transformer(self, flux2_klein_module, sample_inputs):
        """Test that extra_states contains run_flux2_full_transformer_with_single."""
        context = extract_flux2_klein_context(flux2_klein_module, **sample_inputs)

        assert context.extra_states is not None
        assert "run_flux2_full_transformer_with_single" in context.extra_states
        assert callable(context.extra_states["run_flux2_full_transformer_with_single"])

    def test_without_guidance(self, flux2_klein_module, sample_inputs):
        """Test context extraction works without guidance (no CFG)."""
        inputs = sample_inputs.copy()
        inputs["guidance"] = None

        context = extract_flux2_klein_context(flux2_klein_module, **inputs)

        assert context is not None
        assert context.temb is not None

    def test_invalid_module_raises_error(self):
        """Test that invalid module without transformer_blocks raises ValueError."""
        invalid_module = Mock()
        invalid_module.transformer_blocks = []

        with pytest.raises(ValueError, match="Module must have transformer_blocks"):
            extract_flux2_klein_context(
                invalid_module,
                hidden_states=torch.randn(1, 1024, 6144),
                encoder_hidden_states=torch.randn(1, 512, 15360),
                timestep=torch.tensor([500]),
                img_ids=torch.randint(0, 64, (1, 1024, 4)),
                txt_ids=torch.randint(0, 64, (1, 512, 4)),
            )
