# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Sidecar allocation must stop when the device allocator stalls."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from tests.dfx.reliability.helpers import (
    _build_sidecar_cmd,
    allocate_device_chunk,
    reliability_scenarios_for_eager_oom,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_allocate_device_chunk_returns_chunk_when_free_memory_drops() -> None:
    free = {"bytes": 100}

    def allocate(req_elems: int) -> str:
        free["bytes"] -= req_elems
        return f"chunk-{req_elems}"

    chunk = allocate_device_chunk(
        10,
        allocate=allocate,
        free_bytes=lambda: free["bytes"],
        timeout_sec=1.0,
    )

    assert chunk == "chunk-10"


def test_allocate_device_chunk_rejects_runtime_error() -> None:
    def allocate(req_elems: int) -> str:
        raise RuntimeError(f"oom {req_elems}")

    assert (
        allocate_device_chunk(
            4,
            allocate=allocate,
            free_bytes=lambda: 0,
            timeout_sec=1.0,
        )
        is None
    )


def test_allocate_device_chunk_rejects_when_free_memory_does_not_fall() -> None:
    def allocate(req_elems: int) -> str:
        return f"chunk-{req_elems}"

    assert (
        allocate_device_chunk(
            4,
            allocate=allocate,
            free_bytes=lambda: 50,
            timeout_sec=1.0,
        )
        is None
    )


def test_allocate_device_chunk_rejects_stalled_allocator() -> None:
    def allocate(req_elems: int) -> str:
        time.sleep(2.0)
        return f"chunk-{req_elems}"

    started = time.monotonic()
    chunk = allocate_device_chunk(
        4,
        allocate=allocate,
        free_bytes=lambda: 50,
        timeout_sec=0.05,
    )

    assert chunk is None
    assert time.monotonic() - started < 1.0


def test_sidecar_program_uses_rejection_helper_and_compiles() -> None:
    command = _build_sidecar_cmd(0, 0.95, 0, True, 256)
    program = command[3]

    assert "allocate_device_chunk" in program
    compile(program, "<oom-sidecar>", "exec")


def test_sidecar_program_runs_in_fresh_subprocess(tmp_path: Path) -> None:
    (tmp_path / "torch.py").write_text(
        """
class _DeviceProperties:
    total_memory = 2048


class _Cuda:
    @staticmethod
    def init():
        pass

    @staticmethod
    def set_device(device):
        pass

    @staticmethod
    def get_device_properties(device):
        return _DeviceProperties()

    @staticmethod
    def mem_get_info(device):
        return 1024, 2048

    @staticmethod
    def empty(*args, **kwargs):
        raise AssertionError("target ratio zero must not allocate")


cuda = _Cuda()
float16 = object()
""",
        encoding="utf-8",
    )
    command = _build_sidecar_cmd(0, 0.0, 1, False, None)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(path for path in (str(tmp_path), env.get("PYTHONPATH", "")) if path)

    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
    assert "READY:" in completed.stdout
    assert completed.stdout.rstrip().endswith("DONE")


def test_qwen_hip_oom_scenarios_force_eager_request_allocations() -> None:
    base = [{"test_name": "qwen", "server_params": {"model": "Qwen/Qwen3-Omni"}}]
    scenarios = reliability_scenarios_for_eager_oom(base, is_hip=True, stage_ids=(0, 1))

    assert scenarios is not base
    assert "update" not in base[0]["server_params"]
    for scenario in scenarios:
        assert scenario["server_params"]["update"] == {
            "stages": {
                0: {"enforce_eager": True},
                1: {"enforce_eager": True},
            }
        }


def test_qwen_cuda_oom_scenarios_preserve_normal_runtime_config() -> None:
    base = [{"test_name": "qwen", "server_params": {"model": "Qwen/Qwen3-Omni"}}]
    scenarios = reliability_scenarios_for_eager_oom(base, is_hip=False, stage_ids=(0, 1))

    assert scenarios == base
    assert scenarios is not base
