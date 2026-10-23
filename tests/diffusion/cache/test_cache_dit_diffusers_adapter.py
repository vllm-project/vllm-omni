# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Tests for Cache DiT integration with the DiffusersAdapter backend.

Validates that DiffusersAdapterPipeline no longer blocks cache backends
and that the component discovery + attrgetter path resolution works
end-to-end with CacheDiTBackend.
"""

from operator import attrgetter
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend
from vllm_omni.diffusion.cache.cachedit.backend import _dit_module_names
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_adapter_mock(*, dit_modules=None, transformer=None, unet=None):
    """Build a mock that mimics DiffusersAdapterPipeline's structure."""
    pipeline = Mock()
    pipeline.__class__.__name__ = "DiffusersAdapterPipeline"

    hf_pipe = SimpleNamespace()
    if transformer is not None:
        hf_pipe.transformer = transformer
    if unet is not None:
        hf_pipe.unet = unet

    pipeline._pipeline = hf_pipe

    if dit_modules is not None:
        pipeline._dit_modules = dit_modules
    else:
        pipeline._dit_modules = []

    pipeline.default_num_inference_steps = 50
    return pipeline


class TestDiffusersAdapterCacheDiTIntegration:

    def test_attrgetter_resolves_pipeline_transformer(self):
        """_pipeline.transformer paths resolve through attrgetter."""
        mock_transformer = Mock(spec=torch.nn.Module)
        adapter = _make_adapter_mock(
            dit_modules=["_pipeline.transformer"],
            transformer=mock_transformer,
        )

        resolved = attrgetter("_pipeline.transformer")(adapter)
        assert resolved is mock_transformer

    def test_dit_module_names_resolves_adapter_paths(self):
        """_dit_module_names() correctly resolves _pipeline.X attrgetter paths."""
        mock_transformer = Mock(spec=torch.nn.Module)
        adapter = _make_adapter_mock(
            dit_modules=["_pipeline.transformer"],
            transformer=mock_transformer,
        )

        names = _dit_module_names(adapter)
        assert names == ("_pipeline.transformer",)

    def test_dit_module_names_skips_missing_components(self):
        """_dit_module_names() skips paths that don't resolve to a module."""
        adapter = _make_adapter_mock(dit_modules=["_pipeline.transformer"])
        # _pipeline has no .transformer attribute
        adapter._pipeline = SimpleNamespace()

        names = _dit_module_names(adapter)
        assert names == ()

    @patch("vllm_omni.diffusion.cache.cachedit.backend.BlockAdapter")
    @patch("vllm_omni.diffusion.cache.cachedit.backend.cache_dit")
    def test_cache_dit_enable_with_attrgetter_paths(
        self, mock_cache_dit, mock_block_adapter
    ):
        """CacheDiTBackend.enable() works with _pipeline.X attrgetter paths."""
        mock_transformer = Mock(spec=torch.nn.Module)
        mock_transformer._cache_dit_adapter_config = None
        adapter = _make_adapter_mock(
            dit_modules=["_pipeline.transformer"],
            transformer=mock_transformer,
        )

        mock_cache_dit.enable_cache = Mock()

        backend = CacheDiTBackend(DiffusionCacheConfig())
        backend.enable(adapter)

        assert backend.enabled is True
        mock_cache_dit.enable_cache.assert_called_once()

    @patch("vllm_omni.diffusion.cache.cachedit.backend.BlockAdapter")
    @patch("vllm_omni.diffusion.cache.cachedit.backend.cache_dit")
    def test_cache_dit_refresh_after_enable(
        self, mock_cache_dit, mock_block_adapter
    ):
        """refresh() works with adapter pipeline after enable()."""
        mock_transformer = Mock(spec=torch.nn.Module)
        mock_transformer._cache_dit_adapter_config = None
        adapter = _make_adapter_mock(
            dit_modules=["_pipeline.transformer"],
            transformer=mock_transformer,
        )

        mock_cache_dit.enable_cache = Mock()
        mock_cache_dit.refresh_context = Mock()

        backend = CacheDiTBackend(DiffusionCacheConfig())
        backend.enable(adapter)
        backend.refresh(adapter, num_inference_steps=50)

        mock_cache_dit.refresh_context.assert_called_once()

    def test_empty_dit_modules_raises_on_enable(self):
        """enable() raises ValueError when no DiT modules are discovered."""
        adapter = _make_adapter_mock(dit_modules=[])

        backend = CacheDiTBackend(DiffusionCacheConfig())
        with pytest.raises(ValueError, match="no declared DiT modules"):
            backend.enable(adapter)

    @patch("vllm_omni.diffusion.cache.cachedit.backend.BlockAdapter")
    @patch("vllm_omni.diffusion.cache.cachedit.backend.cache_dit")
    def test_unet_path_resolves_but_cache_dit_may_reject(
        self, mock_cache_dit, mock_block_adapter
    ):
        """UNet-based HF pipelines resolve via attrgetter; cache_dit decides support."""
        mock_unet = Mock(spec=torch.nn.Module)
        mock_unet._cache_dit_adapter_config = None
        adapter = _make_adapter_mock(
            dit_modules=["_pipeline.unet"],
            unet=mock_unet,
        )

        mock_cache_dit.enable_cache = Mock(
            side_effect=ValueError("no compatible adapter")
        )

        backend = CacheDiTBackend(DiffusionCacheConfig())
        with pytest.raises(ValueError, match="Failed to enable Cache-DiT"):
            backend.enable(adapter)

    def test_resolve_default_num_inference_steps_on_adapter(self):
        """Adapter's default_num_inference_steps resolves via base class method."""
        adapter = _make_adapter_mock()
        assert adapter.default_num_inference_steps == 50


class TestDiscoverComponents:
    """Test the _discover_components logic from DiffusersAdapterPipeline."""

    def test_transformer_discovered(self):
        """Transformer-based HF pipeline produces correct _dit_modules."""
        pipe = SimpleNamespace(
            transformer=Mock(spec=torch.nn.Module),
            text_encoder=Mock(spec=torch.nn.Module),
            text_encoder_2=None,
            text_encoder_3=None,
            vae=Mock(spec=torch.nn.Module),
        )

        dit = [
            f"_pipeline.{attr}"
            for attr in ("transformer", "unet")
            if getattr(pipe, attr, None) is not None
        ]
        encoders = [
            f"_pipeline.{attr}"
            for attr in ("text_encoder", "text_encoder_2", "text_encoder_3")
            if getattr(pipe, attr, None) is not None
        ]
        vaes = [
            f"_pipeline.{attr}"
            for attr in ("vae",)
            if getattr(pipe, attr, None) is not None
        ]

        assert dit == ["_pipeline.transformer"]
        assert encoders == ["_pipeline.text_encoder"]
        assert vaes == ["_pipeline.vae"]

    def test_unet_discovered(self):
        """UNet-based HF pipeline (SDXL) produces correct _dit_modules."""
        pipe = SimpleNamespace(
            unet=Mock(spec=torch.nn.Module),
            text_encoder=Mock(spec=torch.nn.Module),
            text_encoder_2=Mock(spec=torch.nn.Module),
            vae=Mock(spec=torch.nn.Module),
        )

        dit = [
            f"_pipeline.{attr}"
            for attr in ("transformer", "unet")
            if getattr(pipe, attr, None) is not None
        ]

        assert dit == ["_pipeline.unet"]

    def test_no_dit_module(self):
        """HF pipeline with neither transformer nor unet produces empty list."""
        pipe = SimpleNamespace(
            text_encoder=Mock(spec=torch.nn.Module),
            vae=Mock(spec=torch.nn.Module),
        )

        dit = [
            f"_pipeline.{attr}"
            for attr in ("transformer", "unet")
            if getattr(pipe, attr, None) is not None
        ]

        assert dit == []
