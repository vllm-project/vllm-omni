"""Tests for process-scoped GPU memory accounting."""

import os
from unittest import mock

import pytest


class TestParseCudaVisibleDevices:
    def test_empty(self):
        from vllm_omni.worker.gpu_memory_utils import parse_cuda_visible_devices

        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            assert parse_cuda_visible_devices() == []

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            assert parse_cuda_visible_devices() == []

    def test_integer_indices(self):
        from vllm_omni.worker.gpu_memory_utils import parse_cuda_visible_devices

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3,5"}):
            assert parse_cuda_visible_devices() == [2, 3, 5]

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            assert parse_cuda_visible_devices() == [0]

    def test_uuids(self):
        from vllm_omni.worker.gpu_memory_utils import parse_cuda_visible_devices

        uuid1 = "GPU-12345678-1234-1234-1234-123456789abc"
        uuid2 = "GPU-87654321-4321-4321-4321-cba987654321"
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": f"{uuid1},{uuid2}"}):
            assert parse_cuda_visible_devices() == [uuid1, uuid2]

    def test_mig_ids(self):
        from vllm_omni.worker.gpu_memory_utils import parse_cuda_visible_devices

        mig1 = "MIG-GPU-12345678-1234-1234-1234-123456789abc/0/0"
        mig2 = "MIG-GPU-12345678-1234-1234-1234-123456789abc/1/0"
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": f"{mig1},{mig2}"}):
            assert parse_cuda_visible_devices() == [mig1, mig2]

    def test_spaces(self):
        from vllm_omni.worker.gpu_memory_utils import parse_cuda_visible_devices

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": " 2 , 3 , 5 "}):
            assert parse_cuda_visible_devices() == [2, 3, 5]


class TestGetProcessGpuMemory:
    @pytest.mark.skipif(not os.path.exists("/dev/nvidia0"), reason="No GPU")
    def test_returns_memory_for_current_process(self):
        import torch

        from vllm_omni.worker.gpu_memory_utils import get_process_gpu_memory

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda:0")
        tensor = torch.zeros(1000, 1000, device=device)

        memory = get_process_gpu_memory(0)
        assert memory >= 0

        del tensor
        torch.cuda.empty_cache()

    def test_raises_on_invalid_device(self):
        from vllm_omni.worker.gpu_memory_utils import get_process_gpu_memory

        with (
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlInit"),
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlShutdown"),
            mock.patch("vllm.third_party.pynvml.nvmlDeviceGetCount", return_value=1),
        ):
            with pytest.raises(RuntimeError, match="Invalid GPU device"):
                get_process_gpu_memory(5)

    def test_returns_zero_when_process_not_found(self):
        from vllm_omni.worker.gpu_memory_utils import get_process_gpu_memory

        with (
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlInit"),
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlShutdown"),
            mock.patch("vllm.third_party.pynvml.nvmlDeviceGetCount", return_value=8),
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlDeviceGetHandleByIndex"),
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlDeviceGetComputeRunningProcesses", return_value=[]),
        ):
            memory = get_process_gpu_memory(0)
            assert memory == 0

    def test_uses_uuid_when_provided(self):
        from vllm_omni.worker.gpu_memory_utils import get_process_gpu_memory

        uuid = "GPU-12345678-1234-1234-1234-123456789abc"
        mock_handle = mock.MagicMock()

        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": uuid}),
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlInit"),
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlShutdown"),
            mock.patch("vllm.third_party.pynvml.nvmlDeviceGetHandleByUUID", return_value=mock_handle) as mock_by_uuid,
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlDeviceGetComputeRunningProcesses", return_value=[]),
        ):
            memory = get_process_gpu_memory(0)
            assert memory == 0
            mock_by_uuid.assert_called_once_with(uuid)

    def test_raises_on_invalid_uuid(self):
        from vllm_omni.worker.gpu_memory_utils import get_process_gpu_memory

        uuid = "GPU-invalid-uuid"

        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": uuid}),
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlInit"),
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlShutdown"),
            mock.patch("vllm.third_party.pynvml.nvmlDeviceGetHandleByUUID", side_effect=Exception("Invalid UUID")),
        ):
            with pytest.raises(RuntimeError, match="Failed to get NVML handle"):
                get_process_gpu_memory(0)

    def test_returns_none_on_nvml_init_failure(self):
        from vllm_omni.worker.gpu_memory_utils import get_process_gpu_memory

        with (
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlInit", side_effect=Exception("NVML unavailable")),
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlShutdown"),
        ):
            result = get_process_gpu_memory(0)
            assert result is None


class TestIsProcessScopedMemoryAvailable:
    def test_returns_true_when_nvml_works(self):
        from vllm_omni.worker.gpu_memory_utils import is_process_scoped_memory_available

        with (
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlInit"),
            mock.patch("vllm_omni.worker.gpu_memory_utils.nvmlShutdown"),
        ):
            assert is_process_scoped_memory_available() is True

    def test_returns_false_when_nvml_fails(self):
        from vllm_omni.worker.gpu_memory_utils import is_process_scoped_memory_available

        with mock.patch(
            "vllm_omni.worker.gpu_memory_utils.nvmlInit",
            side_effect=Exception("NVML unavailable"),
        ):
            assert is_process_scoped_memory_available() is False
