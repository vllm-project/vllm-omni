# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for DiffusionWorker class.

This module tests the DiffusionWorker implementation:
- load_weights: Loading model weights
- sleep: Putting worker into sleep mode (levels 1 and 2)
- wake_up: Waking worker from sleep mode
"""

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.diffusion.worker.diffusion_worker import (
    DiffusionWorker,
    _create_diffusion_worker_vllm_config,
    _make_diffusion_vllm_model_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]


@pytest.fixture
def mock_od_config(mocker: MockerFixture):
    """Create a mock OmniDiffusionConfig."""
    config = mocker.Mock()
    config.num_gpus = 1
    config.master_port = 12345
    config.enable_sleep_mode = False
    config.cache_backend = None
    config.cache_config = None
    config.model = "test-model"
    return config


@pytest.fixture
def mock_gpu_worker(mocker: MockerFixture, mock_od_config):
    """Create a DiffusionWorker with mocked initialization."""
    mocker.patch.object(DiffusionWorker, "init_device")
    mocker.patch.object(DiffusionWorker, "load_model")
    worker = DiffusionWorker(local_rank=0, rank=0, od_config=mock_od_config)
    # Mock the model_runner with pipeline
    worker.model_runner = mocker.Mock()
    worker.model_runner.pipeline = mocker.Mock()
    worker.device = torch.device("cuda", 0)
    worker._sleep_saved_buffers = {}
    return worker


class TestDiffusionWorkerLoadWeights:
    """Test DiffusionWorker.load_weights method."""

    def test_load_weights_calls_pipeline(self, mocker: MockerFixture, mock_gpu_worker):
        """Test that load_weights delegates to model_runner.load_weights."""
        # Setup mock weights
        mock_weights = [
            ("layer1.weight", torch.randn(10, 10)),
            ("layer2.weight", torch.randn(20, 20)),
        ]
        expected_loaded = {"layer1.weight", "layer2.weight"}

        # Configure model_runner mock
        mock_gpu_worker.model_runner.load_weights = mocker.Mock(return_value=expected_loaded)

        # Call load_weights
        result = mock_gpu_worker.load_weights(mock_weights)

        # Verify model_runner.load_weights was called with the weights
        mock_gpu_worker.model_runner.load_weights.assert_called_once_with(mock_weights)
        assert result == expected_loaded

    def test_load_weights_empty_iterable(self, mocker: MockerFixture, mock_gpu_worker):
        """Test load_weights with empty weights iterable."""
        mock_gpu_worker.model_runner.load_weights = mocker.Mock(return_value=set())

        result = mock_gpu_worker.load_weights([])

        mock_gpu_worker.model_runner.load_weights.assert_called_once_with([])
        assert result == set()


def test_diffusion_vllm_model_config_supplies_dtype_for_quant_methods():
    from types import SimpleNamespace

    from vllm_omni.quantization import build_quant_config

    od_config = SimpleNamespace(
        model="dummy",
        dtype=torch.bfloat16,
        quantization_config=build_quant_config(
            {
                "quant_method": "modelopt",
                "quant_algo": "FP8",
                "ignore": [],
            }
        ),
        tf_model_config=SimpleNamespace(),
        enforce_eager=True,
        is_moe=False,
    )

    model_config = _make_diffusion_vllm_model_config(od_config)

    assert model_config.dtype is torch.bfloat16
    assert model_config.quantization == "modelopt"
    assert model_config.quantization_config is od_config.quantization_config
    assert model_config.is_quantized()


class TestDiffusionWorkerSleep:
    """Test DiffusionWorker.sleep method."""

    @pytest.fixture(autouse=True)
    def setup_allocator(self, mocker: MockerFixture):
        """
        Unified interception of Allocators, and provision of default security values.
        """
        self.mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")
        self.mock_allocator = mocker.Mock()
        self.mock_allocator_class.get_instance.return_value = self.mock_allocator
        self.mock_allocator.get_current_usage.return_value = 4 * 1024**3
        self.mock_allocator.sleep = mocker.Mock()

    def test_sleep_level_1(self, mocker: MockerFixture, mock_gpu_worker):
        """Test sleep mode level 1 (offload weights only)."""
        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")
        mock_platform = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform")
        mock_platform.get_free_memory.side_effect = [10 * 1024**3, 12 * 1024**3]
        mock_platform.get_device_total_memory.return_value = 80 * 1024**3
        mock_get_process_memory = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_process_gpu_memory")

        # Setup process-scoped memory mocks
        # Before sleep: 3GB used
        # After sleep: 1GB used (freed 2GB)
        initial_usage = 3 * 1024**3
        mock_get_process_memory.side_effect = [
            initial_usage,
            1 * 1024**3,
        ]

        # Setup allocator mock
        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.sleep = mocker.Mock()
        mock_allocator.get_current_usage.return_value = initial_usage

        # Call sleep with level 1
        result = mock_gpu_worker.sleep(level=1)

        # Verify sleep was called with correct tags
        mock_allocator.sleep.assert_called_once_with(offload_tags=("weights",))
        assert bool(result) is True
        # Verify buffers were NOT saved (level 1 doesn't save buffers)
        assert len(mock_gpu_worker._sleep_saved_buffers) == 0

    def test_sleep_level_2(self, mocker: MockerFixture, mock_gpu_worker):
        """Test sleep mode level 2 (offload all, save buffers)."""
        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")
        mock_platform = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform")
        mock_platform.get_free_memory.side_effect = [5 * 1024**3, 10 * 1024**3]
        mock_platform.get_device_total_memory.return_value = 80 * 1024**3
        mock_get_process_memory = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_process_gpu_memory")

        # Setup process-scoped memory mocks
        initial_usage = 5 * 1024**3
        mock_get_process_memory.side_effect = [
            initial_usage,  # Before sleep
            1 * 1024**3,  # After sleep (freed 4GB)
        ]

        # Setup allocator mock
        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.sleep = mocker.Mock()
        mock_allocator.get_current_usage.return_value = initial_usage

        # Mock pipeline buffers
        mock_buffer1 = torch.randn(10, 10)
        mock_buffer2 = torch.randn(20, 20)
        mock_gpu_worker.model_runner.pipeline.named_buffers = mocker.Mock(
            return_value=[
                ("buffer1", mock_buffer1),
                ("buffer2", mock_buffer2),
            ]
        )

        # Call sleep with level 2
        result = mock_gpu_worker.sleep(level=2)

        # Verify sleep was called with empty tags (offload all)
        mock_allocator.sleep.assert_called_once_with(offload_tags=tuple())
        assert bool(result) is True

        # Verify buffers were saved
        assert len(mock_gpu_worker._sleep_saved_buffers) == 2
        assert "buffer1" in mock_gpu_worker._sleep_saved_buffers
        assert "buffer2" in mock_gpu_worker._sleep_saved_buffers

    def test_sleep_memory_freed_validation(self, mocker: MockerFixture, mock_gpu_worker):
        """Test that sleep validates memory was actually freed."""
        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")
        mock_platform = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform")
        mock_platform.get_free_memory.return_value = 10 * 1024**3
        mock_platform.get_device_total_memory.return_value = 80 * 1024**3
        mock_get_process_memory = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_process_gpu_memory")

        # Simulate process memory increase (should trigger assertion error)
        initial_usage = 1 * 1024**3
        mock_get_process_memory.side_effect = [
            initial_usage,  # Before sleep: 1GB used
            3 * 1024**3,  # After sleep: 3GB used (negative freed)
        ]

        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.sleep = mocker.Mock()
        mock_allocator.get_current_usage.return_value = initial_usage

        # This should raise an assertion error
        result = mock_gpu_worker.sleep(level=1)
        assert result == initial_usage

    def test_sleep_falls_back_to_device_memory_when_nvml_unavailable(self, mocker: MockerFixture, mock_gpu_worker):
        """Test sleep uses device-scoped fallback when NVML is unavailable."""

        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")
        mock_platform = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform")
        mock_get_process_memory = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.get_process_gpu_memory")
        mock_get_process_memory.side_effect = [None, None]
        mock_platform.get_free_memory.side_effect = [
            1 * 1024**3,  # Before sleep
            3 * 1024**3,  # After sleep
        ]
        mock_platform.get_device_total_memory.return_value = 8 * 1024**3

        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.sleep = mocker.Mock()
        mock_allocator.get_current_usage.return_value = 2 * 1024**3

        result = mock_gpu_worker.sleep(level=1)

        mock_allocator.sleep.assert_called_once_with(offload_tags=("weights",))
        assert bool(result) is True


class TestDiffusionWorkerWakeUp:
    """Test DiffusionWorker.wake_up method."""

    def test_wake_up_without_buffers(self, mocker: MockerFixture, mock_gpu_worker):
        """Test wake_up without saved buffers (level 1 sleep)."""
        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")

        # Setup allocator mock
        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.wake_up = mocker.Mock()
        mock_allocator.get_current_usage.return_value = 10 * 1024**3

        # Ensure no saved buffers
        mock_gpu_worker._sleep_saved_buffers = {}

        # Call wake_up
        result = mock_gpu_worker.wake_up(tags=["weights"])

        # Verify allocator.wake_up was called
        mock_allocator.wake_up.assert_called_once_with(["weights"])
        assert bool(result) is True

    def test_wake_up_with_buffers(self, mocker: MockerFixture, mock_gpu_worker):
        """Test wake_up with saved buffers (level 2 sleep)."""
        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")

        # Setup allocator mock
        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.wake_up = mocker.Mock()
        mock_allocator.get_current_usage.return_value = 10 * 1024**3

        # Create saved buffers
        saved_buffer1 = torch.randn(10, 10)
        saved_buffer2 = torch.randn(20, 20)
        mock_gpu_worker._sleep_saved_buffers = {
            "buffer1": saved_buffer1,
            "buffer2": saved_buffer2,
        }

        # Mock pipeline buffers (these will be restored)
        mock_buffer1 = mocker.Mock()
        mock_buffer1.data = mocker.Mock()
        mock_buffer2 = mocker.Mock()
        mock_buffer2.data = mocker.Mock()

        mock_gpu_worker.model_runner.pipeline.named_buffers = mocker.Mock(
            return_value=[
                ("buffer1", mock_buffer1),
                ("buffer2", mock_buffer2),
            ]
        )

        # Call wake_up
        result = mock_gpu_worker.wake_up(tags=None)

        # Verify allocator.wake_up was called
        mock_allocator.wake_up.assert_called_once_with(None)

        # Verify buffers were restored
        mock_buffer1.data.copy_.assert_called_once()
        mock_buffer2.data.copy_.assert_called_once()

        # Verify saved buffers were cleared
        assert len(mock_gpu_worker._sleep_saved_buffers) == 0
        assert bool(result) is True

    def test_wake_up_partial_buffer_restore(self, mocker: MockerFixture, mock_gpu_worker):
        """Test wake_up only restores buffers that were saved."""
        mock_allocator_class = mocker.patch("vllm.device_allocator.cumem.CuMemAllocator")

        # Setup allocator mock
        mock_allocator = mocker.Mock()
        mock_allocator_class.get_instance = mocker.Mock(return_value=mock_allocator)
        mock_allocator.wake_up = mocker.Mock()
        mock_allocator.get_current_usage.return_value = 10 * 1024**3

        # Only save buffer1, not buffer2
        saved_buffer1 = torch.randn(10, 10)
        mock_gpu_worker._sleep_saved_buffers = {
            "buffer1": saved_buffer1,
        }

        # Mock pipeline has both buffers
        mock_buffer1 = mocker.Mock()
        mock_buffer1.data = mocker.Mock()
        mock_buffer2 = mocker.Mock()
        mock_buffer2.data = mocker.Mock()

        mock_gpu_worker.model_runner.pipeline.named_buffers = mocker.Mock(
            return_value=[
                ("buffer1", mock_buffer1),
                ("buffer2", mock_buffer2),
            ]
        )

        # Call wake_up
        result = mock_gpu_worker.wake_up()

        # Verify only buffer1 was restored
        mock_buffer1.data.copy_.assert_called_once()
        # buffer2 should NOT be restored since it wasn't saved
        mock_buffer2.data.copy_.assert_not_called()

        assert result is True


class TestWorkerVllmConfigAdditionalConfig:
    """Test worker-side VllmConfig construction with additional_config."""

    def test_create_diffusion_worker_vllm_config_passes_additional_config(self, mocker: MockerFixture):
        mock_vllm_config = mocker.Mock()
        mock_vllm_cls = mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.VllmConfig",
            return_value=mock_vllm_config,
        )
        od_config = mocker.Mock(additional_config={"torchair_graph_config": {"enabled": True}})

        result = _create_diffusion_worker_vllm_config(torch.device("cpu"), od_config)

        assert result is mock_vllm_config
        assert mock_vllm_cls.call_args.kwargs["additional_config"] == od_config.additional_config

    def test_create_diffusion_worker_vllm_config_falls_back_when_constructor_rejects_additional_config(
        self, mocker: MockerFixture
    ):
        mock_vllm_config = mocker.Mock()
        mock_vllm_cls = mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.VllmConfig",
            side_effect=[
                TypeError("VllmConfig.__init__() got an unexpected keyword argument 'additional_config'"),
                mock_vllm_config,
            ],
        )
        od_config = mocker.Mock(additional_config={"ascend_scheduler_config": {"foo": "bar"}})

        result = _create_diffusion_worker_vllm_config(torch.device("cpu"), od_config)

        assert result is mock_vllm_config
        assert mock_vllm_cls.call_count == 2
        assert getattr(mock_vllm_config, "additional_config") == od_config.additional_config

    def test_create_diffusion_worker_vllm_config_reraises_other_type_errors(self, mocker: MockerFixture):
        mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.VllmConfig",
            side_effect=TypeError("additional_config values must be hashable"),
        )
        od_config = mocker.Mock(additional_config={"ascend_scheduler_config": {"foo": "bar"}})

        with pytest.raises(TypeError, match="hashable"):
            _create_diffusion_worker_vllm_config(torch.device("cpu"), od_config)
