# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for GPUDiffusionWorker class.

This module tests the GPUDiffusionWorker implementation:
- load_weights: Loading model weights
- sleep: Putting worker into sleep mode (levels 1 and 2)
- wake_up: Waking worker from sleep mode
"""

from unittest.mock import Mock, patch

import pytest
import torch

from vllm_omni.diffusion.worker.gpu_diffusion_worker import GPUDiffusionWorker


@pytest.fixture
def mock_od_config():
    """Create a mock OmniDiffusionConfig."""
    config = Mock()
    config.num_gpus = 1
    config.master_port = 12345
    config.enable_sleep_mode = False
    config.cache_backend = None
    config.cache_config = None
    config.model = "test-model"
    return config


@pytest.fixture
def mock_gpu_worker(mock_od_config):
    """Create a GPUDiffusionWorker with mocked initialization."""
    with patch.object(GPUDiffusionWorker, "init_device"):
        worker = GPUDiffusionWorker(local_rank=0, rank=0, od_config=mock_od_config)
        # Mock the model_runner with pipeline
        worker.model_runner = Mock()
        worker.model_runner.pipeline = Mock()
        worker._sleep_saved_buffers = {}
        return worker


class TestGPUDiffusionWorkerLoadWeights:
    """Test GPUDiffusionWorker.load_weights method."""

    def test_load_weights_calls_pipeline(self, mock_gpu_worker):
        """Test that load_weights delegates to model_runner.load_weights."""
        # Setup mock weights
        mock_weights = [
            ("layer1.weight", torch.randn(10, 10)),
            ("layer2.weight", torch.randn(20, 20)),
        ]
        expected_loaded = {"layer1.weight", "layer2.weight"}

        # Configure model_runner mock
        mock_gpu_worker.model_runner.load_weights = Mock(return_value=expected_loaded)

        # Call load_weights
        result = mock_gpu_worker.load_weights(mock_weights)

        # Verify model_runner.load_weights was called with the weights
        mock_gpu_worker.model_runner.load_weights.assert_called_once_with(mock_weights)
        assert result == expected_loaded

    def test_load_weights_empty_iterable(self, mock_gpu_worker):
        """Test load_weights with empty weights iterable."""
        mock_gpu_worker.model_runner.load_weights = Mock(return_value=set())

        result = mock_gpu_worker.load_weights([])

        mock_gpu_worker.model_runner.load_weights.assert_called_once_with([])
        assert result == set()


class TestGPUDiffusionWorkerSleep:
    """Test GPUDiffusionWorker.sleep method."""

    @patch("vllm_omni.diffusion.worker.gpu_diffusion_worker.torch.cuda.mem_get_info")
    @patch("vllm.device_allocator.cumem.CuMemAllocator")
    def test_sleep_level_1(self, mock_allocator_class, mock_mem_info, mock_gpu_worker):
        """Test sleep mode level 1 (offload weights only)."""
        # Setup memory info mocks
        # Before sleep: 1GB free, 8GB total
        # After sleep: 3GB free, 8GB total (freed 2GB)
        mock_mem_info.side_effect = [
            (1 * 1024**3, 8 * 1024**3),  # Before sleep
            (3 * 1024**3, 8 * 1024**3),  # After sleep
        ]

        # Setup allocator mock
        mock_allocator = Mock()
        mock_allocator_class.get_instance = Mock(return_value=mock_allocator)
        mock_allocator.sleep = Mock()

        # Call sleep with level 1
        result = mock_gpu_worker.sleep(level=1)

        # Verify sleep was called with correct tags
        mock_allocator.sleep.assert_called_once_with(offload_tags=("weights",))
        assert result is True
        # Verify buffers were NOT saved (level 1 doesn't save buffers)
        assert len(mock_gpu_worker._sleep_saved_buffers) == 0

    @patch("vllm_omni.diffusion.worker.gpu_diffusion_worker.torch.cuda.mem_get_info")
    @patch("vllm.device_allocator.cumem.CuMemAllocator")
    def test_sleep_level_2(self, mock_allocator_class, mock_mem_info, mock_gpu_worker):
        """Test sleep mode level 2 (offload all, save buffers)."""
        # Setup memory info mocks
        mock_mem_info.side_effect = [
            (1 * 1024**3, 8 * 1024**3),  # Before sleep
            (5 * 1024**3, 8 * 1024**3),  # After sleep (freed 4GB)
        ]

        # Setup allocator mock
        mock_allocator = Mock()
        mock_allocator_class.get_instance = Mock(return_value=mock_allocator)
        mock_allocator.sleep = Mock()

        # Mock pipeline buffers
        mock_buffer1 = torch.randn(10, 10)
        mock_buffer2 = torch.randn(20, 20)
        mock_gpu_worker.model_runner.pipeline.named_buffers = Mock(
            return_value=[
                ("buffer1", mock_buffer1),
                ("buffer2", mock_buffer2),
            ]
        )

        # Call sleep with level 2
        result = mock_gpu_worker.sleep(level=2)

        # Verify sleep was called with empty tags (offload all)
        mock_allocator.sleep.assert_called_once_with(offload_tags=tuple())
        assert result is True

        # Verify buffers were saved
        assert len(mock_gpu_worker._sleep_saved_buffers) == 2
        assert "buffer1" in mock_gpu_worker._sleep_saved_buffers
        assert "buffer2" in mock_gpu_worker._sleep_saved_buffers

    @patch("vllm_omni.diffusion.worker.gpu_diffusion_worker.torch.cuda.mem_get_info")
    @patch("vllm.device_allocator.cumem.CuMemAllocator")
    def test_sleep_memory_freed_validation(self, mock_allocator_class, mock_mem_info, mock_gpu_worker):
        """Test that sleep validates memory was actually freed."""
        # Simulate memory increase (should trigger assertion error)
        mock_mem_info.side_effect = [
            (3 * 1024**3, 8 * 1024**3),  # Before sleep: 3GB free
            (1 * 1024**3, 8 * 1024**3),  # After sleep: 1GB free (negative freed!)
        ]

        mock_allocator = Mock()
        mock_allocator_class.get_instance = Mock(return_value=mock_allocator)
        mock_allocator.sleep = Mock()

        # This should raise an assertion error
        with pytest.raises(AssertionError, match="Memory usage increased after sleeping"):
            mock_gpu_worker.sleep(level=1)


class TestGPUDiffusionWorkerWakeUp:
    """Test GPUDiffusionWorker.wake_up method."""

    @patch("vllm.device_allocator.cumem.CuMemAllocator")
    def test_wake_up_without_buffers(self, mock_allocator_class, mock_gpu_worker):
        """Test wake_up without saved buffers (level 1 sleep)."""
        # Setup allocator mock
        mock_allocator = Mock()
        mock_allocator_class.get_instance = Mock(return_value=mock_allocator)
        mock_allocator.wake_up = Mock()

        # Ensure no saved buffers
        mock_gpu_worker._sleep_saved_buffers = {}

        # Call wake_up
        result = mock_gpu_worker.wake_up(tags=["weights"])

        # Verify allocator.wake_up was called
        mock_allocator.wake_up.assert_called_once_with(["weights"])
        assert result is True

    @patch("vllm.device_allocator.cumem.CuMemAllocator")
    def test_wake_up_with_buffers(self, mock_allocator_class, mock_gpu_worker):
        """Test wake_up with saved buffers (level 2 sleep)."""
        # Setup allocator mock
        mock_allocator = Mock()
        mock_allocator_class.get_instance = Mock(return_value=mock_allocator)
        mock_allocator.wake_up = Mock()

        # Create saved buffers
        saved_buffer1 = torch.randn(10, 10)
        saved_buffer2 = torch.randn(20, 20)
        mock_gpu_worker._sleep_saved_buffers = {
            "buffer1": saved_buffer1,
            "buffer2": saved_buffer2,
        }

        # Mock pipeline buffers (these will be restored)
        mock_buffer1 = Mock()
        mock_buffer1.data = Mock()
        mock_buffer2 = Mock()
        mock_buffer2.data = Mock()

        mock_gpu_worker.model_runner.pipeline.named_buffers = Mock(
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
        assert result is True

    @patch("vllm.device_allocator.cumem.CuMemAllocator")
    def test_wake_up_partial_buffer_restore(self, mock_allocator_class, mock_gpu_worker):
        """Test wake_up only restores buffers that were saved."""
        # Setup allocator mock
        mock_allocator = Mock()
        mock_allocator_class.get_instance = Mock(return_value=mock_allocator)
        mock_allocator.wake_up = Mock()

        # Only save buffer1, not buffer2
        saved_buffer1 = torch.randn(10, 10)
        mock_gpu_worker._sleep_saved_buffers = {
            "buffer1": saved_buffer1,
        }

        # Mock pipeline has both buffers
        mock_buffer1 = Mock()
        mock_buffer1.data = Mock()
        mock_buffer2 = Mock()
        mock_buffer2.data = Mock()

        mock_gpu_worker.model_runner.pipeline.named_buffers = Mock(
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
