# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for WorkerWrapperBase class.

This module tests the WorkerWrapperBase implementation:
- Initialization with and without worker extensions
- Custom pipeline initialization
- Method delegation via execute_method
- Attribute delegation via __getattr__
- Dynamic worker class extension
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from vllm_omni.diffusion.worker.diffusion_worker import (
    CustomPipelineWorkerExtension,
    DiffusionWorker,
    WorkerWrapperBase,
)

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def mock_od_config():
    """Create a mock OmniDiffusionConfig for use in tests."""
    config = Mock()
    config.num_gpus = 1
    config.master_port = 12345
    config.enable_sleep_mode = False
    config.cache_backend = None
    config.cache_config = None
    config.model = "test-model"
    config.diffusion_load_format = None
    config.dtype = "float32"
    config.max_cpu_loras = 0
    config.lora_path = None
    config.lora_scale = 1.0
    return config


class TestExtension:
    """Simple test extension adding one custom method."""

    def custom_method(self):
        return "extension_method"


class MockCustomPipeline:
    """Mock custom pipeline for testing."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return "pipeline_output"


# -------------------------------------------------------------------------
# Tests: Initialization
# -------------------------------------------------------------------------


class TestWorkerWrapperBaseInitialization:
    """Test WorkerWrapperBase initialization behavior."""

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_basic_initialization(self, mock_worker_init, mock_od_config):
        """Test basic initialization without extensions."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )

        assert wrapper.gpu_id == 0
        assert wrapper.od_config == mock_od_config
        assert wrapper.base_worker_class == DiffusionWorker
        assert wrapper.worker_extension_cls is None
        assert wrapper.custom_pipeline_args is None
        assert wrapper.worker is not None

        mock_worker_init.assert_called_once_with(
            local_rank=0,
            rank=0,
            od_config=mock_od_config,
        )


# -------------------------------------------------------------------------
# Tests: Worker Extension Functionality
# -------------------------------------------------------------------------


class TestWorkerWrapperBaseExtension:
    """Test WorkerWrapperBase worker extension functionality."""

    def test_prepare_worker_class_without_extension(self, mock_od_config):
        """Test _prepare_worker_class without a worker extension."""
        with patch.object(DiffusionWorker, "__init__", return_value=None):
            wrapper = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
            )
            worker_class = wrapper._prepare_worker_class()
            assert worker_class == DiffusionWorker

    def test_prepare_worker_class_with_extension_class(self, mock_od_config):
        """Test _prepare_worker_class with an explicit extension class."""

        class TestExtension:
            def custom_method(self):
                return "extension_method"

        with patch.object(DiffusionWorker, "__init__", return_value=None):
            wrapper = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
                worker_extension_cls=TestExtension,
            )

            assert hasattr(wrapper.worker.__class__, "custom_method")
            assert TestExtension in wrapper.worker.__class__.__bases__

    @patch("vllm.utils.import_utils.resolve_obj_by_qualname")
    def test_prepare_worker_class_with_extension_string(self, mock_resolve, mock_od_config):
        """Test _prepare_worker_class with worker extension as string."""
        mock_resolve.return_value = TestExtension

        with patch.object(DiffusionWorker, "__init__", return_value=None):
            wrapper = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
                worker_extension_cls="tests.diffusion.test_worker_wrapper_base.TestExtension",
            )

            assert hasattr(wrapper.worker.__class__, "custom_method")


# -------------------------------------------------------------------------
# Tests: Method Delegation
# -------------------------------------------------------------------------


class TestWorkerWrapperBaseDelegation:
    """Test WorkerWrapperBase delegation to wrapped worker."""

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_generate_delegation(self, mock_worker_init, mock_od_config):
        """Test that generate() delegates to worker.generate()."""
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        mock_output = Mock()
        wrapper.worker.generate = Mock(return_value=mock_output)

        mock_requests = [Mock()]
        result = wrapper.generate(mock_requests)

        wrapper.worker.generate.assert_called_once_with(mock_requests)
        assert result == mock_output

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_execute_model_delegation(self, mock_worker_init, mock_od_config):
        """Test that execute_model() delegates to worker.execute_model()."""
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        mock_output = Mock()
        wrapper.worker.execute_model = Mock(return_value=mock_output)

        mock_reqs = [Mock()]
        result = wrapper.execute_model(mock_reqs, mock_od_config)

        wrapper.worker.execute_model.assert_called_once_with(mock_reqs, mock_od_config)
        assert result == mock_output

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_load_weights_delegation(self, mock_worker_init, mock_od_config):
        """Test that load_weights() delegates to worker.load_weights()."""
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        expected_result = {"weight1", "weight2"}
        wrapper.worker.load_weights = Mock(return_value=expected_result)

        mock_weights = [("weight1", Mock()), ("weight2", Mock())]
        result = wrapper.load_weights(mock_weights)

        wrapper.worker.load_weights.assert_called_once_with(mock_weights)
        assert result == expected_result

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_sleep_delegation(self, mock_worker_init, mock_od_config):
        """Test that sleep() delegates to worker.sleep()."""
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.sleep = Mock(return_value=True)
        result = wrapper.sleep(level=1)

        wrapper.worker.sleep.assert_called_once_with(1)
        assert result is True

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_wake_up_delegation(self, mock_worker_init, mock_od_config):
        """Test that wake_up() delegates to worker.wake_up()."""
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.wake_up = Mock(return_value=True)

        result = wrapper.wake_up(tags=["weights"])
        wrapper.worker.wake_up.assert_called_once_with(["weights"])
        assert result is True

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_shutdown_delegation(self, mock_worker_init, mock_od_config):
        """Test that shutdown() delegates to worker.shutdown()."""
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.shutdown = Mock(return_value=None)

        result = wrapper.shutdown()
        wrapper.worker.shutdown.assert_called_once()
        assert result is None


# -------------------------------------------------------------------------
# Tests: execute_method
# -------------------------------------------------------------------------


class TestWorkerWrapperBaseExecuteMethod:
    """Test WorkerWrapperBase.execute_method functionality."""

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_execute_method_success(self, mock_worker_init, mock_od_config):
        """Test execute_method successfully calls worker method."""
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.test_method = Mock(return_value="method_result")

        result = wrapper.execute_method("test_method", "arg1", kwarg1="value1")

        wrapper.worker.test_method.assert_called_once_with("arg1", kwarg1="value1")
        assert result == "method_result"

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_execute_method_with_no_args(self, mock_worker_init, mock_od_config):
        """Test execute_method with no arguments."""
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.no_args_method = Mock(return_value="no_args_result")

        result = wrapper.execute_method("no_args_method")
        wrapper.worker.no_args_method.assert_called_once_with()
        assert result == "no_args_result"

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_execute_method_error(self, mock_worker_init, mock_od_config):
        """Test execute_method raises exception on error."""
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.error_method = Mock(side_effect=RuntimeError("Test error"))

        with pytest.raises(RuntimeError, match="Test error"):
            wrapper.execute_method("error_method")

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_execute_method_invalid_type(self, mock_worker_init, mock_od_config):
        """Test execute_method with invalid method type."""
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)

        with pytest.raises(AssertionError, match="Method must be str"):
            wrapper.execute_method(b"bytes_method")


# -------------------------------------------------------------------------
# Tests: __getattr__ delegation
# -------------------------------------------------------------------------


class TestWorkerWrapperBaseGetAttr:
    """Test WorkerWrapperBase.__getattr__ delegation."""

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_getattr_delegation(self, mock_worker_init, mock_od_config):
        """Test __getattr__ delegates to worker attributes."""
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.custom_attribute = "test_value"
        assert wrapper.custom_attribute == "test_value"

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_getattr_method_access(self, mock_worker_init, mock_od_config):
        """Test __getattr__ delegates to worker methods."""
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.custom_method = Mock(return_value="method_result")

        result = wrapper.custom_method()
        wrapper.worker.custom_method.assert_called_once()
        assert result == "method_result"

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_getattr_missing_attribute(self, mock_worker_init, mock_od_config):
        """Test __getattr__ raises AttributeError for missing attributes."""
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        with pytest.raises(AttributeError):
            _ = wrapper.nonexistent_attribute


# -------------------------------------------------------------------------
# Tests: Edge Cases
# -------------------------------------------------------------------------


class TestWorkerWrapperBaseEdgeCases:
    """Test WorkerWrapperBase edge cases and special scenarios."""

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_extension_conflict_warning(self, mock_worker_init, mock_od_config, caplog):
        """Test a warning is logged when an extension conflicts with worker."""

        class ConflictExtension:
            def load_model(self):
                return "extension_load_model"

        with patch.object(DiffusionWorker, "load_model"):
            wrapper = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
                worker_extension_cls=ConflictExtension,
            )
            assert wrapper.worker is not None

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_multiple_extensions_same_class(self, mock_worker_init, mock_od_config):
        """Test that applying same extension twice doesn't duplicate it."""

        class TestExtension:
            def custom_method(self):
                return "extension"

        with patch.object(DiffusionWorker, "__init__", return_value=None):
            wrapper1 = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
                worker_extension_cls=TestExtension,
            )
            wrapper2 = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
                worker_extension_cls=TestExtension,
            )

            assert hasattr(wrapper1.worker, "custom_method")
            assert hasattr(wrapper2.worker, "custom_method")


# -------------------------------------------------------------------------
# Tests: CustomPipelineWorkerExtension
# -------------------------------------------------------------------------


class TestCustomPipelineWorkerExtension:
    """Test CustomPipelineWorkerExtension functionality."""

    @patch("torch.cuda.empty_cache")
    @patch("gc.collect")
    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_re_init_pipeline_basic(self, mock_worker_init, mock_gc_collect, mock_empty_cache, mock_od_config):
        """Test basic re_init_pipeline functionality."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
            worker_extension_cls=CustomPipelineWorkerExtension,
        )

        # Setup mock model_runner and pipeline
        mock_model_runner = Mock()
        mock_pipeline = Mock()
        mock_model_runner.pipeline = mock_pipeline
        wrapper.worker.model_runner = mock_model_runner
        wrapper.worker.init_lora_manager = Mock()
        wrapper.worker.load_model = Mock()

        custom_args = {"pipeline_class": "tests.diffusion.test_worker_wrapper_base.MockCustomPipeline"}

        # Call re_init_pipeline
        wrapper.worker.re_init_pipeline(custom_args)

        # Verify load_model was called with correct arguments
        wrapper.worker.load_model.assert_called_once_with(
            load_format="custom_pipeline",
            custom_pipeline_name="tests.diffusion.test_worker_wrapper_base.MockCustomPipeline",
        )
        wrapper.worker.init_lora_manager.assert_called_once()

    @patch("torch.cuda.empty_cache")
    @patch("gc.collect")
    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_re_init_pipeline_cleanup(self, mock_worker_init, mock_gc_collect, mock_empty_cache, mock_od_config):
        """Test that re_init_pipeline properly cleans up old pipeline."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
            worker_extension_cls=CustomPipelineWorkerExtension,
        )

        # Setup mock model_runner with pipeline
        mock_model_runner = Mock()
        mock_pipeline = Mock()
        mock_model_runner.pipeline = mock_pipeline
        wrapper.worker.model_runner = mock_model_runner
        wrapper.worker.init_lora_manager = Mock()
        wrapper.worker.load_model = Mock()

        custom_args = {"pipeline_class": "tests.diffusion.test_worker_wrapper_base.MockCustomPipeline"}

        # Call re_init_pipeline
        wrapper.worker.re_init_pipeline(custom_args)

        # Verify cleanup was performed
        mock_gc_collect.assert_called_once()
        mock_empty_cache.assert_called_once()

    @patch("torch.cuda.empty_cache")
    @patch("gc.collect")
    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_re_init_pipeline_none_pipeline(self, mock_worker_init, mock_gc_collect, mock_empty_cache, mock_od_config):
        """Test re_init_pipeline when pipeline is None."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
            worker_extension_cls=CustomPipelineWorkerExtension,
        )

        # Setup mock model_runner with None pipeline
        mock_model_runner = Mock()
        mock_model_runner.pipeline = None
        wrapper.worker.model_runner = mock_model_runner
        wrapper.worker.init_lora_manager = Mock()
        wrapper.worker.load_model = Mock()

        custom_args = {"pipeline_class": "tests.diffusion.test_worker_wrapper_base.MockCustomPipeline"}

        # Should not raise an error
        wrapper.worker.re_init_pipeline(custom_args)

        # Verify load_model was still called
        wrapper.worker.load_model.assert_called_once()

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_custom_pipeline_args_initialization(self, mock_worker_init, mock_od_config):
        """Test initialization with custom_pipeline_args calls re_init_pipeline."""
        custom_args = {"pipeline_class": "tests.diffusion.test_worker_wrapper_base.MockCustomPipeline"}

        with patch.object(DiffusionWorker, "__init__", return_value=None):
            with patch.object(WorkerWrapperBase, "_prepare_worker_class") as mock_prepare:
                # Create a mock worker class with re_init_pipeline
                mock_worker_class = Mock()
                mock_worker_instance = Mock()
                mock_worker_instance.re_init_pipeline = Mock()
                mock_worker_class.return_value = mock_worker_instance
                mock_prepare.return_value = mock_worker_class

                _ = WorkerWrapperBase(
                    gpu_id=0,
                    od_config=mock_od_config,
                    base_worker_class=DiffusionWorker,
                    custom_pipeline_args=custom_args,
                )

                # Verify re_init_pipeline was called with custom_pipeline_args
                mock_worker_instance.re_init_pipeline.assert_called_once_with(custom_args)

    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_custom_pipeline_with_explicit_extension(self, mock_worker_init, mock_od_config):
        """Test that explicit worker_extension_cls is preserved when custom_pipeline_args is provided."""

        class CustomExtension:
            def re_init_pipeline(self, custom_pipeline_args: dict[str, Any]):
                return "custom_re_init_pipeline"

            def custom_extension_method(self):
                return "custom_extension_method"

        custom_args = {"pipeline_class": "tests.diffusion.test_worker_wrapper_base.MockCustomPipeline"}

        with patch.object(DiffusionWorker, "__init__", return_value=None):
            wrapper = WorkerWrapperBase(
                gpu_id=0,
                od_config=mock_od_config,
                base_worker_class=DiffusionWorker,
                worker_extension_cls=CustomExtension,
                custom_pipeline_args=custom_args,
            )

            # Should still have the explicitly provided extension
            assert CustomExtension in wrapper.worker.__class__.__bases__
            assert hasattr(wrapper.worker, "custom_extension_method")

    @patch("torch.cuda.empty_cache")
    @patch("gc.collect")
    @patch.object(DiffusionWorker, "__init__", return_value=None)
    def test_re_init_pipeline_multiple_calls(self, mock_worker_init, mock_gc_collect, mock_empty_cache, mock_od_config):
        """Test calling re_init_pipeline multiple times."""
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
            worker_extension_cls=CustomPipelineWorkerExtension,
        )

        # Setup mock model_runner
        mock_model_runner = Mock()
        mock_pipeline1 = Mock()
        mock_pipeline2 = Mock()
        mock_model_runner.pipeline = mock_pipeline1
        wrapper.worker.model_runner = mock_model_runner
        wrapper.worker.init_lora_manager = Mock()
        wrapper.worker.load_model = Mock()

        # First call
        custom_args1 = {"pipeline_class": "tests.diffusion.test_worker_wrapper_base.MockCustomPipeline"}
        wrapper.worker.re_init_pipeline(custom_args1)

        # Update pipeline for second call
        mock_model_runner.pipeline = mock_pipeline2

        # Second call
        custom_args2 = {"pipeline_class": "tests.diffusion.test_worker_wrapper_base.MockCustomPipeline"}
        wrapper.worker.re_init_pipeline(custom_args2)

        # Verify load_model was called twice with different pipelines
        assert wrapper.worker.load_model.call_count == 2
        assert wrapper.worker.init_lora_manager.call_count == 2
