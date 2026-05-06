# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for diffusion engine plugin extensibility hooks.

This module tests:
- Platform hooks: get_diffusion_worker_cls, get_diffusion_model_runner_cls
- Registry API: register_diffusion_model
- Worker integration: model runner resolved via platform hook
"""

from unittest.mock import patch

import pytest

from vllm_omni.diffusion.registry import (
    _DIFFUSION_MODELS,
    _DIFFUSION_POST_PROCESS_FUNCS,
    _DIFFUSION_PRE_PROCESS_FUNCS,
    register_diffusion_model,
)
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestPlatformDiffusionHooks:
    """Test OmniPlatform diffusion hook defaults."""

    def test_get_diffusion_worker_cls_default(self):
        """Test default diffusion worker class path."""
        result = OmniPlatform.get_diffusion_worker_cls()
        assert result == "vllm_omni.diffusion.worker.diffusion_worker.DiffusionWorker"

    def test_get_diffusion_model_runner_cls_default(self):
        """Test default diffusion model runner class path."""
        result = OmniPlatform.get_diffusion_model_runner_cls()
        assert result == "vllm_omni.diffusion.worker.diffusion_model_runner.DiffusionModelRunner"

    def test_oot_enum_exists(self):
        """Test OOT is a valid platform enum value."""
        assert OmniPlatformEnum.OOT.value == "oot"

    def test_is_out_of_tree(self):
        """Test is_out_of_tree() returns True for OOT platform."""
        platform = OmniPlatform.__new__(OmniPlatform)
        platform._omni_enum = OmniPlatformEnum.OOT
        assert platform.is_out_of_tree() is True
        assert platform.is_cuda() is False


class TestRegisterDiffusionModel:
    """Test register_diffusion_model public API."""

    @pytest.fixture(autouse=True)
    def cleanup_registry(self):
        """Restore global registry dicts after each test."""
        original_models = _DIFFUSION_MODELS.copy()
        original_pre = _DIFFUSION_PRE_PROCESS_FUNCS.copy()
        original_post = _DIFFUSION_POST_PROCESS_FUNCS.copy()
        yield
        _DIFFUSION_MODELS.clear()
        _DIFFUSION_MODELS.update(original_models)
        _DIFFUSION_PRE_PROCESS_FUNCS.clear()
        _DIFFUSION_PRE_PROCESS_FUNCS.update(original_pre)
        _DIFFUSION_POST_PROCESS_FUNCS.clear()
        _DIFFUSION_POST_PROCESS_FUNCS.update(original_post)

    def test_register_new_model(self):
        """Test registering a new diffusion model with pre/post process functions."""
        register_diffusion_model(
            model_arch="TestPipeline",
            module_name="test_plugin.diffusion.pipeline",
            class_name="TestPipeline",
            pre_process_func_name="test_pre_process",
            post_process_func_name="test_post_process",
        )
        assert "TestPipeline" in _DIFFUSION_MODELS
        assert _DIFFUSION_MODELS["TestPipeline"] == (
            "test_plugin.diffusion.pipeline",
            "",
            "TestPipeline",
        )
        assert _DIFFUSION_PRE_PROCESS_FUNCS["TestPipeline"] == "test_pre_process"
        assert _DIFFUSION_POST_PROCESS_FUNCS["TestPipeline"] == "test_post_process"


class TestWorkerUsesHook:
    """Test that DiffusionWorker resolves model runner via platform hook."""

    @patch("vllm_omni.diffusion.worker.diffusion_worker.resolve_obj_by_qualname")
    @patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform")
    def test_model_runner_resolved_via_platform(self, mock_platform, mock_resolve):
        """Test model runner class is resolved from platform hook return value."""
        from unittest.mock import Mock

        from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

        mock_runner_instance = Mock()
        mock_runner_cls = Mock(return_value=mock_runner_instance)
        mock_platform.get_diffusion_model_runner_cls.return_value = "custom.path"
        mock_resolve.return_value = mock_runner_cls

        with patch.object(DiffusionWorker, "init_device"):
            worker = DiffusionWorker(local_rank=0, rank=0, od_config=Mock(), skip_load_model=True)

        assert worker.model_runner is mock_runner_instance
        mock_platform.get_diffusion_model_runner_cls.assert_called_once()
        mock_resolve.assert_called_once_with("custom.path")
