import os
import sys
from multiprocessing import shared_memory as shm

import pytest
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.stage_utils import cleanup_shm_from_ipc_meta, set_stage_devices

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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


def _make_mock_platform(mocker, device_type: str = "cuda", env_var: str = "CUDA_VISIBLE_DEVICES"):
    """Create a mock platform for testing.
    mocker object has to be passed in to utilize this helper function.
    """
    mock_platform = mocker.MagicMock()
    mock_platform.device_type = device_type
    mock_platform.device_control_env_var = env_var
    return mock_platform


@pytest.mark.usefixtures("clean_gpu_memory_between_tests")
def test_set_stage_devices_respects_logical_ids(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    # Preserve an existing logical mapping and ensure devices "0,1" map through it.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    call_log: list[int] = []
    dummy_torch = _make_dummy_torch(call_log)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    # Mock the platform at the source module where it's defined
    mock_platform = _make_mock_platform(mocker, device_type="cuda", env_var="CUDA_VISIBLE_DEVICES")
    monkeypatch.setattr(
        "vllm_omni.platforms.current_omni_platform",
        mock_platform,
    )

    set_stage_devices(stage_id=0, devices="0,1")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6,7"


@pytest.mark.usefixtures("clean_gpu_memory_between_tests")
def test_set_stage_devices_npu_platform(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
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
    mock_platform = _make_mock_platform(mocker, device_type="npu", env_var="ASCEND_RT_VISIBLE_DEVICES")
    monkeypatch.setattr(
        "vllm_omni.platforms.current_omni_platform",
        mock_platform,
    )

    set_stage_devices(stage_id=0, devices="0,1")

    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"


def test_cleanup_shm_from_ipc_meta_unlinks_segment():
    seg = shm.SharedMemory(create=True, size=8)
    seg.buf[:4] = b"test"
    name = seg.name
    seg.close()

    cleaned = cleanup_shm_from_ipc_meta({"engine_outputs_shm": {"name": name, "size": 8}})
    assert cleaned is True

    with pytest.raises(FileNotFoundError):
        shm.SharedMemory(name=name)


def test_cleanup_shm_from_ipc_meta_returns_false_for_invalid_meta():
    assert cleanup_shm_from_ipc_meta({}) is False
    assert cleanup_shm_from_ipc_meta({"engine_outputs_shm": {"size": 8}}) is False
