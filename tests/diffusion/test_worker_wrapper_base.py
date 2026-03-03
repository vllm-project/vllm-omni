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

import pytest
from pytest_mock import MockerFixture

from vllm_omni.diffusion.worker.diffusion_worker import (
    CustomPipelineWorkerExtension,
    DiffusionWorker,
    WorkerWrapperBase,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def mock_od_config(mocker: MockerFixture):
    """Create a mock OmniDiffusionConfig for use in tests."""
    config = mocker.Mock()
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

    def test_basic_initialization(self, mocker: MockerFixture, mock_od_config):
        """Test basic initialization without extensions."""
        mock_worker_init = mocker.patch.object(DiffusionWorker, "__init__", return_value=None)

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

    def test_prepare_worker_class_without_extension(self, mocker: MockerFixture, mock_od_config):
        """Test _prepare_worker_class without a worker extension."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
        )
        worker_class = wrapper._prepare_worker_class()
        assert worker_class == DiffusionWorker

    def test_prepare_worker_class_with_extension_class(self, mocker: MockerFixture, mock_od_config):
        """Test _prepare_worker_class with an explicit extension class."""

        class TestExtension:
            def custom_method(self):
                return "extension_method"

        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
            worker_extension_cls=TestExtension,
        )

        assert hasattr(wrapper.worker.__class__, "custom_method")
        assert TestExtension in wrapper.worker.__class__.__bases__

    def test_prepare_worker_class_with_extension_string(self, mocker: MockerFixture, mock_od_config):
        """Test _prepare_worker_class with worker extension as string."""
        mock_resolve = mocker.patch("vllm.utils.import_utils.resolve_obj_by_qualname")
        mock_resolve.return_value = TestExtension

        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
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

    def test_generate_delegation(self, mocker: MockerFixture, mock_od_config):
        """Test that generate() delegates to worker.generate()."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        mock_output = mocker.Mock()
        wrapper.worker.generate = mocker.Mock(return_value=mock_output)

        mock_requests = [mocker.Mock()]
        result = wrapper.generate(mock_requests)

        wrapper.worker.generate.assert_called_once_with(mock_requests)
        assert result == mock_output

    def test_execute_model_delegation(self, mocker: MockerFixture, mock_od_config):
        """Test that execute_model() delegates to worker.execute_model()."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        mock_output = mocker.Mock()
        wrapper.worker.execute_model = mocker.Mock(return_value=mock_output)

        mock_reqs = [mocker.Mock()]
        result = wrapper.execute_model(mock_reqs, mock_od_config)

        wrapper.worker.execute_model.assert_called_once_with(mock_reqs, mock_od_config)
        assert result == mock_output

    def test_load_weights_delegation(self, mocker: MockerFixture, mock_od_config):
        """Test that load_weights() delegates to worker.load_weights()."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        expected_result = {"weight1", "weight2"}
        wrapper.worker.load_weights = mocker.Mock(return_value=expected_result)

        mock_weights = [("weight1", mocker.Mock()), ("weight2", mocker.Mock())]
        result = wrapper.load_weights(mock_weights)

        wrapper.worker.load_weights.assert_called_once_with(mock_weights)
        assert result == expected_result

    def test_sleep_delegation(self, mocker: MockerFixture, mock_od_config):
        """Test that sleep() delegates to worker.sleep()."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.sleep = mocker.Mock(return_value=True)
        result = wrapper.sleep(level=1)

        wrapper.worker.sleep.assert_called_once_with(1)
        assert result is True

    def test_wake_up_delegation(self, mocker: MockerFixture, mock_od_config):
        """Test that wake_up() delegates to worker.wake_up()."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.wake_up = mocker.Mock(return_value=True)

        result = wrapper.wake_up(tags=["weights"])
        wrapper.worker.wake_up.assert_called_once_with(["weights"])
        assert result is True

    def test_shutdown_delegation(self, mocker: MockerFixture, mock_od_config):
        """Test that shutdown() delegates to worker.shutdown()."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.shutdown = mocker.Mock(return_value=None)

        result = wrapper.shutdown()
        wrapper.worker.shutdown.assert_called_once()
        assert result is None


# -------------------------------------------------------------------------
# Tests: execute_method
# -------------------------------------------------------------------------


class TestWorkerWrapperBaseExecuteMethod:
    """Test WorkerWrapperBase.execute_method functionality."""

    def test_execute_method_success(self, mocker: MockerFixture, mock_od_config):
        """Test execute_method successfully calls worker method."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.test_method = mocker.Mock(return_value="method_result")

        result = wrapper.execute_method("test_method", "arg1", kwarg1="value1")

        wrapper.worker.test_method.assert_called_once_with("arg1", kwarg1="value1")
        assert result == "method_result"

    def test_execute_method_with_no_args(self, mocker: MockerFixture, mock_od_config):
        """Test execute_method with no arguments."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.no_args_method = mocker.Mock(return_value="no_args_result")

        result = wrapper.execute_method("no_args_method")
        wrapper.worker.no_args_method.assert_called_once_with()
        assert result == "no_args_result"

    def test_execute_method_error(self, mocker: MockerFixture, mock_od_config):
        """Test execute_method raises exception on error."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.error_method = mocker.Mock(side_effect=RuntimeError("Test error"))

        with pytest.raises(RuntimeError, match="Test error"):
            wrapper.execute_method("error_method")

    def test_execute_method_invalid_type(self, mocker: MockerFixture, mock_od_config):
        """Test execute_method with invalid method type."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)

        with pytest.raises(AssertionError, match="Method must be str"):
            wrapper.execute_method(b"bytes_method")


# -------------------------------------------------------------------------
# Tests: __getattr__ delegation
# -------------------------------------------------------------------------


class TestWorkerWrapperBaseGetAttr:
    """Test WorkerWrapperBase.__getattr__ delegation."""

    def test_getattr_delegation(self, mocker: MockerFixture, mock_od_config):
        """Test __getattr__ delegates to worker attributes."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.custom_attribute = "test_value"
        assert wrapper.custom_attribute == "test_value"

    def test_getattr_method_access(self, mocker: MockerFixture, mock_od_config):
        """Test __getattr__ delegates to worker methods."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        wrapper.worker.custom_method = mocker.Mock(return_value="method_result")

        result = wrapper.custom_method()
        wrapper.worker.custom_method.assert_called_once()
        assert result == "method_result"

    def test_getattr_missing_attribute(self, mocker: MockerFixture, mock_od_config):
        """Test __getattr__ raises AttributeError for missing attributes."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        wrapper = WorkerWrapperBase(gpu_id=0, od_config=mock_od_config, base_worker_class=DiffusionWorker)
        with pytest.raises(AttributeError):
            _ = wrapper.nonexistent_attribute


# -------------------------------------------------------------------------
# Tests: Edge Cases
# -------------------------------------------------------------------------


class TestWorkerWrapperBaseEdgeCases:
    """Test WorkerWrapperBase edge cases and special scenarios."""

    def test_extension_conflict_warning(self, mocker: MockerFixture, mock_od_config, caplog):
        """Test a warning is logged when an extension conflicts with worker."""
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)

        class ConflictExtension:
            def load_model(self):
                return "extension_load_model"

        mocker.patch.object(DiffusionWorker, "load_model")
        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
            worker_extension_cls=ConflictExtension,
        )
        assert wrapper.worker is not None

    def test_multiple_extensions_same_class(self, mocker: MockerFixture, mock_od_config):
        """Test that applying same extension twice doesn't duplicate it."""

        class TestExtension:
            def custom_method(self):
                return "extension"

        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
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

    def test_re_init_pipeline_basic(self, mocker: MockerFixture, mock_od_config):
        """Test basic re_init_pipeline functionality."""
        mocker.patch("torch.cuda.empty_cache")
        mocker.patch("gc.collect")
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)

        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
            worker_extension_cls=CustomPipelineWorkerExtension,
        )

        # Setup mock model_runner and pipeline
        mock_model_runner = mocker.Mock()
        mock_pipeline = mocker.Mock()
        mock_model_runner.pipeline = mock_pipeline
        wrapper.worker.model_runner = mock_model_runner
        wrapper.worker.init_lora_manager = mocker.Mock()
        wrapper.worker.load_model = mocker.Mock()

        custom_args = {"pipeline_class": "tests.diffusion.test_worker_wrapper_base.MockCustomPipeline"}

        # Call re_init_pipeline
        wrapper.worker.re_init_pipeline(custom_args)

        # Verify load_model was called with correct arguments
        wrapper.worker.load_model.assert_called_once_with(
            load_format="custom_pipeline",
            custom_pipeline_name="tests.diffusion.test_worker_wrapper_base.MockCustomPipeline",
        )
        wrapper.worker.init_lora_manager.assert_called_once()

    def test_re_init_pipeline_cleanup(self, mocker: MockerFixture, mock_od_config):
        """Test that re_init_pipeline properly cleans up old pipeline."""
        mock_gc_collect = mocker.patch("gc.collect")
        mock_empty_cache = mocker.patch("torch.cuda.empty_cache")
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)

        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
            worker_extension_cls=CustomPipelineWorkerExtension,
        )

        # Setup mock model_runner with pipeline
        mock_model_runner = mocker.Mock()
        mock_pipeline = mocker.Mock()
        mock_model_runner.pipeline = mock_pipeline
        wrapper.worker.model_runner = mock_model_runner
        wrapper.worker.init_lora_manager = mocker.Mock()
        wrapper.worker.load_model = mocker.Mock()

        custom_args = {"pipeline_class": "tests.diffusion.test_worker_wrapper_base.MockCustomPipeline"}

        # Call re_init_pipeline
        wrapper.worker.re_init_pipeline(custom_args)

        # Verify cleanup was performed
        mock_gc_collect.assert_called_once()
        mock_empty_cache.assert_called_once()

    def test_re_init_pipeline_none_pipeline(self, mocker: MockerFixture, mock_od_config):
        """Test re_init_pipeline when pipeline is None."""
        mocker.patch("torch.cuda.empty_cache")
        mocker.patch("gc.collect")
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)

        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
            worker_extension_cls=CustomPipelineWorkerExtension,
        )

        # Setup mock model_runner with None pipeline
        mock_model_runner = mocker.Mock()
        mock_model_runner.pipeline = None
        wrapper.worker.model_runner = mock_model_runner
        wrapper.worker.init_lora_manager = mocker.Mock()
        wrapper.worker.load_model = mocker.Mock()

        custom_args = {"pipeline_class": "tests.diffusion.test_worker_wrapper_base.MockCustomPipeline"}

        # Should not raise an error
        wrapper.worker.re_init_pipeline(custom_args)

        # Verify load_model was still called
        wrapper.worker.load_model.assert_called_once()

    def test_custom_pipeline_args_initialization(self, mocker: MockerFixture, mock_od_config):
        """Test initialization with custom_pipeline_args calls re_init_pipeline."""
        custom_args = {"pipeline_class": "tests.diffusion.test_worker_wrapper_base.MockCustomPipeline"}

        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
        mock_prepare = mocker.patch.object(WorkerWrapperBase, "_prepare_worker_class")
        # Create a mock worker class with re_init_pipeline
        mock_worker_class = mocker.Mock()
        mock_worker_instance = mocker.Mock()
        mock_worker_instance.re_init_pipeline = mocker.Mock()
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

    def test_custom_pipeline_with_explicit_extension(self, mocker: MockerFixture, mock_od_config):
        """Test that explicit worker_extension_cls is preserved when custom_pipeline_args is provided."""

        class CustomExtension:
            def re_init_pipeline(self, custom_pipeline_args: dict[str, Any]):
                return "custom_re_init_pipeline"

            def custom_extension_method(self):
                return "custom_extension_method"

        custom_args = {"pipeline_class": "tests.diffusion.test_worker_wrapper_base.MockCustomPipeline"}

        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)
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

    def test_re_init_pipeline_multiple_calls(self, mocker: MockerFixture, mock_od_config):
        """Test calling re_init_pipeline multiple times."""
        mocker.patch("torch.cuda.empty_cache")
        mocker.patch("gc.collect")
        mocker.patch.object(DiffusionWorker, "__init__", return_value=None)

        wrapper = WorkerWrapperBase(
            gpu_id=0,
            od_config=mock_od_config,
            base_worker_class=DiffusionWorker,
            worker_extension_cls=CustomPipelineWorkerExtension,
        )

        # Setup mock model_runner
        mock_model_runner = mocker.Mock()
        mock_pipeline1 = mocker.Mock()
        mock_pipeline2 = mocker.Mock()
        mock_model_runner.pipeline = mock_pipeline1
        wrapper.worker.model_runner = mock_model_runner
        wrapper.worker.init_lora_manager = mocker.Mock()
        wrapper.worker.load_model = mocker.Mock()

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
