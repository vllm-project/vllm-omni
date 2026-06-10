import os
import sys

import pytest
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.stage_utils import _map_device_list, set_stage_devices


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


@pytest.mark.core_model
@pytest.mark.cpu
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


@pytest.mark.core_model
@pytest.mark.cpu
def test_set_stage_devices_handles_not_enough_devices(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
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

    # Keep the logical mapping and resolve to the visible subset.
    set_stage_devices(stage_id=0, devices="0,1,2,3")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6,7"


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


# ---- _map_device_list unit tests ----


@pytest.mark.core_model
@pytest.mark.cpu
def test_map_device_list_index_mapping():
    """Device IDs < num_visible not in set are treated as indices."""
    result = _map_device_list(0, ["0", "1"], ["6", "7", "8", "9"])
    assert result == ["6", "7"]


@pytest.mark.core_model
@pytest.mark.cpu
def test_map_device_list_physical_fallback():
    """Device IDs >= num_visible are treated as physical IDs and passed through."""
    result = _map_device_list(1, ["2"], ["0", "1"])
    assert result == ["2"]


@pytest.mark.core_model
@pytest.mark.cpu
def test_map_device_list_physical_fallback_multi():
    """Multiple device IDs >= num_visible all pass through as physical IDs."""
    result = _map_device_list(1, ["2", "3"], ["0", "1"])
    assert result == ["2", "3"]


@pytest.mark.core_model
@pytest.mark.cpu
def test_map_device_list_physical_fallback_mixed():
    """When some devices are >= num_visible but some < num_visible, index mapping applies for the subset."""
    # "1" < num_visible(2) so index mapping applies: visible[1] = "5"
    # "2" >= num_visible(2) so it's dropped via the partial mapping path
    result = _map_device_list(1, ["1", "2"], ["0", "5"])
    assert result == ["5"]


@pytest.mark.core_model
@pytest.mark.cpu
def test_map_device_list_index_with_gaps():
    """Index-based mapping works with non-contiguous visible device lists."""
    result = _map_device_list(0, ["0", "1"], ["4", "5", "7"])
    assert result == ["4", "5"]


@pytest.mark.core_model
@pytest.mark.cpu
def test_map_device_list_index_mapping_no_idempotency():
    """Device IDs < num_visible not in visible set map via index even when no ID matches literally."""
    # "0","1" with visible ["2","3"]: idempotency check fails (none in set),
    # but index mapping succeeds: visible[0]="2", visible[1]="3"
    result = _map_device_list(0, ["0", "1"], ["2", "3"])
    assert result == ["2", "3"]


@pytest.mark.core_model
@pytest.mark.cpu
def test_map_device_list_partial_mapping():
    """When only a subset can map, returns the mapped subset (no error)."""
    result = _map_device_list(0, ["0", "1", "2"], ["5", "6"])
    assert result == ["5", "6"]


@pytest.mark.core_model
@pytest.mark.cpu
def test_map_device_list_multi_replica_offset_cvd():
    """Logical indices from split_devices_for_replicas must map through an offset
    CUDA_VISIBLE_DEVICES, not be treated as physical IDs.
    """
    visible = ["2", "3", "4", "5"]
    assert _map_device_list(1, ["1"], visible) == ["3"]
    assert _map_device_list(1, ["2"], visible) == ["4"]
    assert _map_device_list(1, ["2", "3"], visible) == ["4", "5"]


@pytest.mark.core_model
@pytest.mark.cpu
def test_map_device_list_raises_on_non_numeric():
    """Non-numeric device IDs raise ValueError."""
    with pytest.raises(ValueError, match="must be non-negative integers"):
        _map_device_list(0, ["a"], ["0", "1"])
