# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for diffusion engine plugin extensibility hooks.

This module tests:
- Platform hooks: get_diffusion_worker_cls, get_diffusion_model_runner_cls
- Registry API: register_diffusion_model
- Worker integration: model runner resolved via platform hook
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm_omni.diffusion.registry import (
    _DIFFUSION_IR_OP_PRIORITY_FUNCS,
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
        original_ir_op_priority = _DIFFUSION_IR_OP_PRIORITY_FUNCS.copy()
        yield
        _DIFFUSION_MODELS.clear()
        _DIFFUSION_MODELS.update(original_models)
        _DIFFUSION_PRE_PROCESS_FUNCS.clear()
        _DIFFUSION_PRE_PROCESS_FUNCS.update(original_pre)
        _DIFFUSION_POST_PROCESS_FUNCS.clear()
        _DIFFUSION_POST_PROCESS_FUNCS.update(original_post)
        _DIFFUSION_IR_OP_PRIORITY_FUNCS.clear()
        _DIFFUSION_IR_OP_PRIORITY_FUNCS.update(original_ir_op_priority)

    def test_register_new_model(self):
        """Test registering a new diffusion model with pre/post process functions."""
        register_diffusion_model(
            model_arch="TestPipeline",
            module_name="test_plugin.diffusion.pipeline",
            class_name="TestPipeline",
            pre_process_func_name="test_pre_process",
            post_process_func_name="test_post_process",
            ir_op_priority_func_name="test_ir_op_priority",
        )
        assert "TestPipeline" in _DIFFUSION_MODELS
        assert _DIFFUSION_MODELS["TestPipeline"] == (
            "test_plugin.diffusion.pipeline",
            "",
            "TestPipeline",
        )
        assert _DIFFUSION_PRE_PROCESS_FUNCS["TestPipeline"] == "test_pre_process"
        assert _DIFFUSION_POST_PROCESS_FUNCS["TestPipeline"] == "test_post_process"
        assert _DIFFUSION_IR_OP_PRIORITY_FUNCS["TestPipeline"] == "test_ir_op_priority"

    def test_register_model_accepts_deprecated_action_postprocess_keyword(self):
        """Deprecated action hook keyword is accepted but not registered."""
        register_diffusion_model(
            model_arch="LegacyActionPipeline",
            module_name="test_plugin.diffusion.pipeline",
            class_name="LegacyActionPipeline",
            post_process_func_name="test_post_process",
            action_post_process_func_name="test_action_post_process",
        )

        assert "LegacyActionPipeline" in _DIFFUSION_MODELS
        assert _DIFFUSION_POST_PROCESS_FUNCS["LegacyActionPipeline"] == "test_post_process"


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

    @patch("vllm_omni.diffusion.worker.diffusion_worker.get_diffusion_ir_op_priority_func")
    @patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform")
    def test_ir_op_priority_hook_receives_platform_default(self, mock_platform, mock_get_hook):
        """Test model IR priority hook merges from the platform default."""
        from vllm_omni.diffusion.worker.diffusion_worker import _resolve_ir_op_priority

        od_config = SimpleNamespace(model_class_name="TestPipeline")
        vllm_config = SimpleNamespace()
        default_priority = object()
        merged_priority = object()
        hook = Mock(return_value=merged_priority)
        mock_platform.get_default_ir_op_priority.return_value = default_priority
        mock_get_hook.return_value = hook

        assert _resolve_ir_op_priority(od_config, vllm_config) is merged_priority
        mock_platform.get_default_ir_op_priority.assert_called_once_with(vllm_config)
        mock_get_hook.assert_called_once_with(od_config)
        hook.assert_called_once_with(default_priority, vllm_config=vllm_config)
