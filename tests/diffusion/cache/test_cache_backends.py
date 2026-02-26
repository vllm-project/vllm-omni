# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for cache backends (cache-dit and teacache).

This module tests the cache backend implementations:
- CacheDiTBackend: cache-dit acceleration backend
- TeaCacheBackend: TeaCache hook-based backend
- Cache selector function: get_cache_backend
- DiffusionCacheConfig: configuration dataclass
"""

from unittest.mock import Mock, patch

import pytest

from vllm_omni.diffusion.cache.cache_dit_backend import (
    CacheDiTBackend,
)
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestCacheDiTBackend:
    """Test CacheDiTBackend implementation."""

    def test_init_with_dict(self):
        """Test initialization with dictionary config."""
        config_dict = {"Fn_compute_blocks": 4, "max_warmup_steps": 8}
        backend = CacheDiTBackend(config_dict)
        assert backend.config.Fn_compute_blocks == 4
        assert backend.config.max_warmup_steps == 8
        assert backend.enabled is False

    def test_init_with_config_object(self):
        """Test initialization with DiffusionCacheConfig object."""
        config = DiffusionCacheConfig(Fn_compute_blocks=4)
        backend = CacheDiTBackend(config)
        assert backend.config.Fn_compute_blocks == 4
        assert backend.enabled is False

    @patch("vllm_omni.diffusion.cache.cache_dit_backend.cache_dit")
    def test_enable_single_transformer(self, mock_cache_dit):
        """Test enabling cache-dit on single-transformer pipeline."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.__class__.__name__ = "DiTPipeline"
        mock_transformer = Mock()
        mock_pipeline.transformer = mock_transformer

        # Mock cache_dit functions
        mock_cache_dit.enable_cache = Mock()
        mock_cache_dit.refresh_context = Mock()

        backend = CacheDiTBackend({"Fn_compute_blocks": 2})
        backend.enable(mock_pipeline)

        # Verify cache-dit was enabled
        assert backend.enabled is True
        assert backend._refresh_func is not None
        mock_cache_dit.enable_cache.assert_called_once()

    @patch("vllm_omni.diffusion.cache.cache_dit_backend.cache_dit")
    def test_refresh(self, mock_cache_dit):
        """Test refreshing cache context with SCM mask policy updates when num_inference_steps changes."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.__class__.__name__ = "DiTPipeline"
        mock_transformer = Mock()
        mock_pipeline.transformer = mock_transformer

        # Mock cache_dit functions
        mock_cache_dit.enable_cache = Mock()
        mock_cache_dit.refresh_context = Mock()
        mock_steps_mask_50 = [1, 0, 1, 0, 1] * 10  # Mock mask for 50 steps
        mock_steps_mask_100 = [1, 0, 1, 0, 1] * 20  # Mock mask for 100 steps
        mock_cache_dit.steps_mask = Mock(side_effect=[mock_steps_mask_50, mock_steps_mask_100])

        # Enable cache-dit with SCM enabled (using mask policy)
        config = DiffusionCacheConfig(
            scm_steps_mask_policy="fast",
            scm_steps_policy="dynamic",
        )
        backend = CacheDiTBackend(config)
        backend.enable(mock_pipeline)

        # First refresh with 50 steps
        backend.refresh(mock_pipeline, num_inference_steps=50)
        assert backend._last_num_inference_steps == 50

        # Verify steps_mask was called with mask policy (not direct steps mask)
        mock_cache_dit.steps_mask.assert_called_with(mask_policy="fast", total_steps=50)
        assert mock_cache_dit.steps_mask.call_count == 1

        # Verify refresh_context was called with cache_config (SCM path)
        mock_cache_dit.refresh_context.assert_called_once()
        call_args = mock_cache_dit.refresh_context.call_args
        assert call_args[0][0] == mock_transformer
        # Check that cache_config was passed (not num_inference_steps directly when SCM is enabled)
        assert "cache_config" in call_args[1]
        cache_config_arg = call_args[1]["cache_config"]
        assert cache_config_arg is not None

        # Change num_inference_steps and refresh again
        mock_cache_dit.refresh_context.reset_mock()
        backend.refresh(mock_pipeline, num_inference_steps=100)

        # Verify steps_mask was called again with new num_inference_steps (using mask policy)
        assert mock_cache_dit.steps_mask.call_count == 2
        # Check the last call was with 100 steps and mask policy
        assert mock_cache_dit.steps_mask.call_args_list[-1].kwargs["total_steps"] == 100
        assert mock_cache_dit.steps_mask.call_args_list[-1].kwargs["mask_policy"] == "fast"

        # Verify refresh_context was called again with updated mask
        mock_cache_dit.refresh_context.assert_called_once()
        call_args = mock_cache_dit.refresh_context.call_args
        assert call_args[0][0] == mock_transformer
        assert "cache_config" in call_args[1]
        assert backend._last_num_inference_steps == 100


class TestTeaCacheBackend:
    """Test TeaCacheBackend implementation."""

    def test_init(self):
        """Test initialization."""
        config = DiffusionCacheConfig(rel_l1_thresh=0.3)
        backend = TeaCacheBackend(config)
        assert backend.config.rel_l1_thresh == 0.3
        assert backend.enabled is False

    @patch("vllm_omni.diffusion.cache.teacache.backend.apply_teacache_hook")
    def test_enable(self, mock_apply_hook):
        """Test enabling TeaCache on pipeline."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.__class__.__name__ = "QwenImagePipeline"
        mock_transformer = Mock()
        mock_transformer.__class__.__name__ = "QwenImageTransformer2DModel"
        mock_pipeline.transformer = mock_transformer

        config = DiffusionCacheConfig(rel_l1_thresh=0.3)
        backend = TeaCacheBackend(config)
        backend.enable(mock_pipeline)

        # Verify hook was applied
        assert backend.enabled is True
        mock_apply_hook.assert_called_once()

    @patch("vllm_omni.diffusion.cache.teacache.backend.apply_teacache_hook")
    def test_enable_with_coefficients(self, mock_apply_hook):
        """Test enabling TeaCache with custom coefficients."""
        mock_pipeline = Mock()
        mock_pipeline.__class__.__name__ = "QwenImagePipeline"
        mock_transformer = Mock()
        mock_transformer.__class__.__name__ = "QwenImageTransformer2DModel"
        mock_pipeline.transformer = mock_transformer

        config = DiffusionCacheConfig(rel_l1_thresh=0.3, coefficients=[1.0, 0.5, 0.2, 0.1, 0.05])
        backend = TeaCacheBackend(config)
        backend.enable(mock_pipeline)

        assert backend.enabled is True
        mock_apply_hook.assert_called_once()

    @patch("vllm_omni.diffusion.cache.teacache.backend.apply_teacache_hook")
    def test_refresh(self, mock_apply_hook):
        """Test refreshing TeaCache state."""
        mock_pipeline = Mock()
        mock_pipeline.__class__.__name__ = "QwenImagePipeline"
        mock_transformer = Mock()
        mock_transformer.__class__.__name__ = "QwenImageTransformer2DModel"
        mock_pipeline.transformer = mock_transformer

        # Mock hook registry
        mock_hook = Mock()
        mock_registry = Mock()
        mock_registry.get_hook = Mock(return_value=mock_hook)
        mock_registry.reset_hook = Mock()
        mock_transformer._hook_registry = mock_registry

        config = DiffusionCacheConfig()
        backend = TeaCacheBackend(config)
        backend.enable(mock_pipeline)

        # Test refresh
        backend.refresh(mock_pipeline, num_inference_steps=50)
        mock_registry.reset_hook.assert_called_once()


class TestCacheSelector:
    """Test cache backend selector function."""

    def test_get_cache_backend_none(self):
        """Test getting None backend."""
        backend = get_cache_backend(None, None)
        assert backend is None

        backend = get_cache_backend("none", None)
        assert backend is None

    def test_get_cache_backend_cache_dit(self):
        """Test getting cache-dit backend."""
        config_dict = {"Fn_compute_blocks": 4}
        backend = get_cache_backend("cache_dit", config_dict)
        assert isinstance(backend, CacheDiTBackend)
        assert backend.config.Fn_compute_blocks == 4

    def test_get_cache_backend_tea_cache(self):
        """Test getting teacache backend."""
        config_dict = {"rel_l1_thresh": 0.3}
        backend = get_cache_backend("tea_cache", config_dict)
        assert isinstance(backend, TeaCacheBackend)
        assert backend.config.rel_l1_thresh == 0.3

    def test_get_cache_backend_invalid(self):
        """Test getting invalid backend raises error."""
        with pytest.raises(ValueError, match="Unsupported cache backend"):
            get_cache_backend("invalid_backend", {})
