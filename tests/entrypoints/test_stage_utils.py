import os
import sys
from unittest.mock import MagicMock

import pytest

from vllm_omni.entrypoints.stage_utils import set_stage_devices


def _make_dummy_torch(call_log):
    class _Props:
        def __init__(self, total):
            self.total_memory = total

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def set_device(idx):
            call_log.append(idx)

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_properties(idx):
            return _Props(total=16000)

        @staticmethod
        def mem_get_info(idx):
            return (8000, 16000)

        @staticmethod
        def get_device_name(idx):
            return f"gpu-{idx}"

    class _Torch:
        cuda = _Cuda

    return _Torch


def _make_mock_platform(device_type: str = "cuda", env_var: str = "CUDA_VISIBLE_DEVICES"):
    """Create a mock platform for testing."""
    mock_platform = MagicMock()
    mock_platform.device_type = device_type
    mock_platform.device_control_env_var = env_var
    return mock_platform


@pytest.mark.usefixtures("clean_gpu_memory_between_tests")
def test_set_stage_devices_respects_logical_ids(monkeypatch):
    # Preserve an existing logical mapping and ensure devices "0,1" map through it.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    call_log: list[int] = []
    dummy_torch = _make_dummy_torch(call_log)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    # Mock the platform at the source module where it's defined
    mock_platform = _make_mock_platform(device_type="cuda", env_var="CUDA_VISIBLE_DEVICES")
    monkeypatch.setattr(
        "vllm_omni.platforms.current_omni_platform",
        mock_platform,
    )

    set_stage_devices(stage_id=0, devices="0,1")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6,7"


@pytest.mark.usefixtures("clean_gpu_memory_between_tests")
def test_set_stage_devices_npu_platform(monkeypatch):
    """Test that set_stage_devices works correctly for NPU platform."""
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5")
    call_log: list[int] = []

    # Create NPU mock torch
    class _Npu:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def set_device(idx):
            call_log.append(idx)

        @staticmethod
        def device_count():
            return 2

    class _NpuTorch:
        npu = _Npu

    monkeypatch.setitem(sys.modules, "torch", _NpuTorch)

    # Mock NPU platform at the source module where it's defined
    mock_platform = _make_mock_platform(device_type="npu", env_var="ASCEND_RT_VISIBLE_DEVICES")
    monkeypatch.setattr(
        "vllm_omni.platforms.current_omni_platform",
        mock_platform,
    )

    set_stage_devices(stage_id=0, devices="0,1")

    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"
