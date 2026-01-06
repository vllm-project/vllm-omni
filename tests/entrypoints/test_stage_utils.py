import os
import sys

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


@pytest.mark.usefixtures("clean_gpu_memory_between_tests")
def test_set_stage_devices_respects_logical_ids(monkeypatch):
    # Preserve an existing logical mapping and ensure devices "0,1" map through it.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    call_log: list[int] = []
    dummy_torch = _make_dummy_torch(call_log)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    monkeypatch.setattr("vllm_omni.utils.detect_device_type", lambda: "cuda")
    monkeypatch.setattr("vllm_omni.utils.get_device_control_env_var", lambda: "CUDA_VISIBLE_DEVICES")

    set_stage_devices(stage_id=0, devices="0,1")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6,7"
