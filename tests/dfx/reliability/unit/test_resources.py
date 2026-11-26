# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU reliability coverage: resources."""

from __future__ import annotations

import ast
import os
import select
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from tests.dfx.reliability import helpers
from tests.dfx.reliability.unit._loader import RELIABILITY_ROOT, exec_ast_nodes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_THINKER_STAGE_PATTERN = "StageEngineCoreProc_stage0"


def test_parse_nvidia_compute_pid_used_memory_mb() -> None:
    output = "42, 512\nnot-a-pid, 128\n43, 2 MiB\n"

    assert helpers._parse_nvidia_compute_pid_used_memory_mb(output) == {42: 512, 43: 2}


def test_parse_rocm_compute_pid_used_memory_mb() -> None:
    output = """{
        "system": {
            "PID123": "python, 1, 1048576, 0, 0",
            "pid456": "process,with,comma, 2, 2097152, 0, 12",
            "PID789": "python, 1, UNKNOWN, UNKNOWN, UNKNOWN",
            "not-a-pid": "python, 1, 4194304, 0, 0"
        }
    }"""

    assert helpers._parse_rocm_compute_pid_used_memory_mb(output) == {123: 1, 456: 2}


def test_gpu_process_query_spec_selects_rocm() -> None:
    platform = SimpleNamespace(is_rocm=lambda: True, is_cuda=lambda: False)

    command, parser = helpers._gpu_process_query_spec(platform) or (None, None)

    assert command == ["rocm-smi", "--showpids", "--json"]
    assert parser is helpers._parse_rocm_compute_pid_used_memory_mb


def test_query_gpu_compute_pid_used_memory_mb_uses_bounded_rocm_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        helpers,
        "_gpu_process_query_spec",
        lambda: (
            ["rocm-smi", "--showpids", "--json"],
            helpers._parse_rocm_compute_pid_used_memory_mb,
        ),
    )
    calls: list[tuple[list[str], int]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append((command, kwargs["timeout"]))
        return subprocess.CompletedProcess(
            command,
            0,
            '{"system": {"PID42": "python, 1, 3145728, 0, 0"}}',
            "",
        )

    monkeypatch.setattr(helpers.subprocess, "run", fake_run)

    assert helpers.query_gpu_compute_pid_used_memory_mb() == {42: 3}
    assert calls == [(["rocm-smi", "--showpids", "--json"], 5)]


def test_query_gpu_compute_pid_used_memory_mb_handles_missing_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        helpers,
        "_gpu_process_query_spec",
        lambda: (
            ["rocm-smi", "--showpids", "--json"],
            helpers._parse_rocm_compute_pid_used_memory_mb,
        ),
    )

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("rocm-smi")

    monkeypatch.setattr(helpers.subprocess, "run", fake_run)

    assert helpers.query_gpu_compute_pid_used_memory_mb() is None


def test_process_cleanup_remains_mandatory_without_gpu_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = SimpleNamespace(reliability_fault_snapshot={"tree_pids": [42]})
    monotonic_values = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(helpers.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(helpers.time, "sleep", lambda _: None)
    monkeypatch.setattr(helpers, "list_alive_pids", lambda _: [42])
    monkeypatch.setattr(helpers, "query_gpu_compute_pid_used_memory_mb", lambda: None)
    monkeypatch.setattr(helpers, "_linux_fd_counts", lambda _: {})

    with pytest.raises(AssertionError, match="residual server-tree processes remain alive"):
        helpers.assert_no_server_tree_process_residual_and_gpu_release(
            server,
            scenario="rocm-no-telemetry",
            timeout_sec=1.0,
        )


def test_process_cleanup_can_pass_without_gpu_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    server = SimpleNamespace(reliability_fault_snapshot={"tree_pids": [42]})
    monotonic_values = iter([0.0, 0.0])
    monkeypatch.setattr(helpers.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(helpers, "list_alive_pids", lambda _: [])
    monkeypatch.setattr(helpers, "query_gpu_compute_pid_used_memory_mb", lambda: None)
    monkeypatch.setattr(helpers, "_linux_fd_counts", lambda _: {})

    helpers.assert_no_server_tree_process_residual_and_gpu_release(
        server,
        scenario="rocm-no-telemetry",
        timeout_sec=1.0,
    )


def test_qwen_worker_pattern_is_the_thinker_stage() -> None:
    source = (RELIABILITY_ROOT / "test_qwen3_omni.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RUNTIME_WORKER_PATTERN" for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Constant)
    assert assignment.value.value == _THINKER_STAGE_PATTERN


def test_stage0_pattern_does_not_select_stage1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpers, "_list_server_process_tree", lambda _server: [1, 2, 3])
    names = {
        1: ("python", "serve"),
        2: ("VLLM::StageEngineCoreProc_stage1_replica0_DP0", "stage1"),
        3: ("VLLM::StageEngineCoreProc_stage0_replica0_DP0", "stage0"),
    }
    monkeypatch.setattr(helpers, "_safe_proc_info", lambda pid: names[pid])

    hits = helpers._pids_in_server_tree_substring_match(object(), _THINKER_STAGE_PATTERN, limit=1)

    assert hits == [3]


@pytest.mark.skipif(os.name == "nt", reason="SIGKILL is not a Windows signal")
def test_sigkill_records_dead_pid(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class _Proc:
        pid = 10

    class _Server:
        proc = _Proc()

    monkeypatch.setattr(helpers, "_list_server_process_tree", lambda _server: [10, 20])
    monkeypatch.setattr(helpers, "_log_server_process_tree", lambda _server: None)
    names = {
        10: ("python", "serve"),
        20: ("VLLM::StageEngineCoreProc_stage0_replica0_DP0", "stage0"),
    }
    monkeypatch.setattr(helpers, "_safe_proc_info", lambda pid: names[pid])
    killed: list[tuple[int, int]] = []
    state = {"dead": False}

    def _kill(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        state["dead"] = True

    monkeypatch.setattr(helpers.os, "kill", _kill)
    monkeypatch.setattr(helpers.psutil, "pid_exists", lambda pid: pid == 20 and not state["dead"])

    def _process(pid: int) -> object:
        class _Status:
            def status(self) -> str:
                return helpers.psutil.STATUS_RUNNING

        return _Status()

    monkeypatch.setattr(helpers.psutil, "Process", _process)

    inject = helpers.make_process_kill_fault_injector(
        grep_patterns=_THINKER_STAGE_PATTERN,
        signal_name="SIGKILL",
        limit=1,
        post_kill_wait_seconds=0.0,
    )
    inject(_Server())

    assert killed == [(20, getattr(signal, "SIGKILL"))]
    output = capsys.readouterr().out
    assert "pid=20" in output
    assert "dead=1" in output


def test_allocate_device_chunk_returns_chunk_when_free_memory_drops() -> None:
    free = {"bytes": 100}

    def allocate(req_elems: int) -> str:
        free["bytes"] -= req_elems
        return f"chunk-{req_elems}"

    chunk = helpers.allocate_device_chunk(
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
        helpers.allocate_device_chunk(
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
        helpers.allocate_device_chunk(
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
    chunk = helpers.allocate_device_chunk(
        4,
        allocate=allocate,
        free_bytes=lambda: 50,
        timeout_sec=0.05,
    )

    assert chunk is None
    assert time.monotonic() - started < 1.0


def test_sidecar_program_uses_rejection_helper_and_compiles() -> None:
    command = helpers._build_sidecar_cmd(0, 0.95, 0, True, 256)
    program = command[3]

    assert "allocate_device_chunk" in program
    compile(program, "<oom-sidecar>", "exec")


@pytest.mark.skipif(os.name == "nt", reason="OOM sidecar pipe polling requires Linux")
@pytest.mark.parametrize("outcome", ["ready", "error", "eof", "timeout", "partial_line"])
def test_sidecar_startup_drains_progress_and_reaps_failed_process(monkeypatch, outcome):
    payloads = {
        "ready": b"STARTING:import_torch\nREADY:fake\n",
        "error": b"STARTING:initialize_device\nERROR:allocation failed\n",
        "eof": b"STARTING:import_torch\nimport failed\n",
        "timeout": b"STARTING:initialize_device\n",
        "partial_line": b"STARTING:initialize_device",
    }
    # One write exposes the old TextIOWrapper/select read-ahead failure. An
    # incomplete line must not block readline past the startup deadline either.
    code = f"import os,time; os.write(1, {payloads[outcome]!r}); "
    code += "raise SystemExit(7)" if outcome == "eof" else "time.sleep(60)"
    monkeypatch.setattr(helpers, "_build_sidecar_cmd", lambda *args: [sys.executable, "-u", "-c", code])
    popen = subprocess.Popen
    processes = []

    def capture_process(*args, **kwargs):
        proc = popen(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(helpers.subprocess, "Popen", capture_process)
    started = time.monotonic()
    if outcome == "ready":
        handle = helpers.start_gpu_oom_hog(startup_timeout_sec=2)
        helpers.stop_gpu_oom_hog(handle)
    else:
        error = TimeoutError if outcome in {"timeout", "partial_line"} else RuntimeError
        with pytest.raises(error, match="STARTING:"):
            helpers.start_gpu_oom_hog(startup_timeout_sec=1)
    assert time.monotonic() - started < 10
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdout is not None
    assert processes[0].stdout.closed


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
    command = helpers._build_sidecar_cmd(0, 0.0, 1, False, None)
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


def test_sidecar_accepts_small_reported_residual_after_allocator_rejection(tmp_path: Path) -> None:
    (tmp_path / "torch.py").write_text(
        """
_MIB = 1024 * 1024


class _DeviceProperties:
    total_memory = 800 * _MIB


class _Chunk:
    def __init__(self, elements):
        self._elements = elements

    def numel(self):
        return self._elements


class _Cuda:
    initial_free = 600 * _MIB
    rejection_limit = 570 * _MIB
    allocated = 0

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
        reporting_reserve = 3 * _MIB if _Cuda.allocated else 0
        return _Cuda.initial_free - _Cuda.allocated + reporting_reserve, 800 * _MIB


def empty(shape, **kwargs):
    requested = shape[0] * 2
    if _Cuda.allocated + requested > _Cuda.rejection_limit:
        raise RuntimeError("allocator full")
    _Cuda.allocated += requested
    return _Chunk(shape[0])


cuda = _Cuda()
float16 = object()
""",
        encoding="utf-8",
    )
    command = helpers._build_sidecar_cmd(0, 0.95, 1, True, 32 * 1024 * 1024)
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
    assert "free_after=34603008" in completed.stdout
    assert "READY:" in completed.stdout
    assert completed.stdout.rstrip().endswith("DONE")


@pytest.mark.parametrize(
    ("reporting_reserve_mib", "expected_returncode", "expected_status"),
    [
        (6, 0, "READY:"),
        (9, 2, "ERROR:"),
    ],
)
def test_sidecar_bounds_reported_residual_without_allocator_rejection(
    tmp_path: Path,
    reporting_reserve_mib: int,
    expected_returncode: int,
    expected_status: str,
) -> None:
    (tmp_path / "torch.py").write_text(
        f"""
_MIB = 1024 * 1024


class _DeviceProperties:
    total_memory = 1024 * _MIB


class _Chunk:
    def __init__(self, elements):
        self._elements = elements

    def numel(self):
        return self._elements


class _Cuda:
    initial_free = 800 * _MIB
    allocated = 0

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
        reporting_reserve = {reporting_reserve_mib} * _MIB if _Cuda.allocated else 0
        return _Cuda.initial_free - _Cuda.allocated + reporting_reserve, 1024 * _MIB


def empty(shape, **kwargs):
    requested = shape[0] * 2
    _Cuda.allocated += requested
    return _Chunk(shape[0])


cuda = _Cuda()
float16 = object()
""",
        encoding="utf-8",
    )
    command = helpers._build_sidecar_cmd(0, 0.95, 1, False, 32 * 1024 * 1024)
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

    expected_free_after = (32 + reporting_reserve_mib) * 1024 * 1024
    assert completed.returncode == expected_returncode, completed.stderr
    assert f"free_after={expected_free_after}" in completed.stdout
    assert expected_status in completed.stdout
    assert completed.stdout.rstrip().endswith("DONE") is (expected_returncode == 0)


@pytest.mark.parametrize(
    ("target_ratio", "rejection_limit_mib", "expected_returncode", "expected_status"),
    [
        (0.95, 9728, 0, "READY:"),
        (0.95, 9472, 2, "ERROR:"),
        (0.0, 0, 2, "ERROR:"),
    ],
)
def test_sidecar_requires_ratio_target_before_accepting_allocator_saturation(
    tmp_path: Path,
    target_ratio: float,
    rejection_limit_mib: int,
    expected_returncode: int,
    expected_status: str,
) -> None:
    """A coarse allocation rejection is ready only after positive ratio pressure."""
    (tmp_path / "torch.py").write_text(
        f"""
_MIB = 1024 * 1024


class _DeviceProperties:
    total_memory = 12 * 1024 * _MIB


class _Chunk:
    def __init__(self, elements):
        self._elements = elements

    def numel(self):
        return self._elements


class _Cuda:
    initial_free = 10 * 1024 * _MIB
    rejection_limit = {rejection_limit_mib} * _MIB
    allocated = 0

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
        return _Cuda.initial_free - _Cuda.allocated, 12 * 1024 * _MIB


def empty(shape, **kwargs):
    requested = shape[0] * 2
    if _Cuda.allocated + requested > _Cuda.rejection_limit:
        raise RuntimeError("next coarse allocation rejected")
    _Cuda.allocated += requested
    return _Chunk(shape[0])


cuda = _Cuda()
float16 = object()
""",
        encoding="utf-8",
    )
    command = helpers._build_sidecar_cmd(0, target_ratio, 1, False, 32 * 1024 * 1024)
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

    assert completed.returncode == expected_returncode, completed.stderr
    assert expected_status in completed.stdout
    assert f"allocator_saturated={int(expected_returncode == 0)}" in completed.stdout
    if expected_returncode == 0:
        assert "free_after=536870912" in completed.stdout
        assert completed.stdout.rstrip().endswith("DONE")


def test_stage_config_device_spec_uses_inherited_platform_devices(tmp_path: Path) -> None:
    pytest.importorskip("vllm")
    base = tmp_path / "base.yaml"
    base.write_text(
        "\n".join(
            (
                "stages:",
                "  - stage_id: 0",
                '    devices: "0,1"',
                "platforms:",
                "  npu:",
                "    stages:",
                "      - stage_id: 0",
                '        devices: "4,5"',
                "  xpu:",
                "    stages:",
                "      - stage_id: 0",
                '        devices: "0,1,2,3,4,5,6,7"',
                "",
            )
        ),
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "\n".join(
            (
                "base_config: base.yaml",
                "stages:",
                "  - stage_id: 0",
                "    max_num_seqs: 2",
                "",
            )
        ),
        encoding="utf-8",
    )
    hunyuan = RELIABILITY_ROOT.parents[2] / "vllm_omni" / "deploy" / "hunyuan_image3_dit.yaml"

    assert helpers.stage_config_device_spec(base, None) == "0,1"
    assert helpers.stage_config_device_spec(base, "cuda") == "0,1"
    assert helpers.stage_config_device_spec(base, "npu") == "4,5"
    assert helpers.stage_config_device_spec(base, "xpu") == "0,1,2,3,4,5,6,7"
    assert helpers.stage_config_device_spec(overlay, "cuda") == "0,1"
    assert helpers.stage_config_device_spec(hunyuan, None) == "0,1,2,3"
    assert helpers.stage_config_device_spec(hunyuan, "cuda") == "0,1,2,3"
    assert helpers.stage_config_device_spec(hunyuan, "npu") == "0,1,2,3"
    assert helpers.stage_config_device_spec(hunyuan, "xpu") == "0,1,2,3,4,5,6,7"


def test_qwen_hip_oom_scenarios_force_eager_request_allocations() -> None:
    base = [{"test_name": "qwen", "server_params": {"model": "Qwen/Qwen3-Omni"}}]
    scenarios = helpers.reliability_scenarios_for_eager_oom(base, is_hip=True, stage_ids=(0, 1))

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
    scenarios = helpers.reliability_scenarios_for_eager_oom(base, is_hip=False, stage_ids=(0, 1))

    assert scenarios == base
    assert scenarios is not base


@pytest.mark.skipif(os.name == "nt", reason="SIGKILL is not a Windows signal")
def test_stuck_pid_is_escalated_to_sigkill() -> None:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('READY', flush=True); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        ready, _, _ = select.select([proc.stdout], [], [], 5)
        assert ready, "child did not install its SIGTERM handler within 5 seconds"
        assert proc.stdout.readline().strip() == "READY"
        os.kill(proc.pid, signal.SIGTERM)
        assert helpers.escalate_live_pids([proc.pid], grace_sec=0.4) is True
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()


@pytest.mark.parametrize("mapped", [True, False])
def test_linux_shm_snapshot_tracks_mapped_objects(tmp_path, monkeypatch, mapped):
    monkeypatch.setattr(helpers.sys, "platform", "linux")
    monkeypatch.setattr(helpers, "_LINUX_PROC_ROOT", tmp_path / "proc")
    monkeypatch.setattr(helpers, "_LINUX_SHM_ROOT", tmp_path / "shm")
    shm = tmp_path / "shm"
    shm.mkdir()
    owned = shm / "psm_owned"
    owned.write_bytes(b"payload")
    unrelated = shm / "psm_unrelated"
    unrelated.write_bytes(b"other task")
    proc = tmp_path / "proc" / "42"
    (proc / "fd").mkdir(parents=True)
    stat = owned.stat()
    # Device conversion is the OS seam; keep this fake /proc test portable.
    monkeypatch.setattr(helpers.os, "makedev", lambda major, minor: major, raising=False)
    (proc / "maps").write_text(f"1000-2000 rw-s 0000 {stat.st_dev:x}:0 {stat.st_ino} {owned}\n" if mapped else "")
    objects = helpers._capture_linux_shm_objects([42, 999])
    assert helpers._remaining_linux_shm_objects(objects) == ([str(owned)] if mapped else [])
    owned.unlink()
    assert helpers._remaining_linux_shm_objects(objects) == []
    assert unrelated.exists()


def test_linux_shm_reused_name_is_not_a_residual(tmp_path):
    shm = tmp_path / "psm_reused"
    shm.write_bytes(b"new object")
    stat = shm.stat()
    assert helpers._remaining_linux_shm_objects([(str(shm), stat.st_dev, stat.st_ino + 1)]) == []


def test_linux_fd_check_includes_pids_excluded_from_alive_list(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers.sys, "platform", "linux")
    monkeypatch.setattr(helpers, "_LINUX_PROC_ROOT", tmp_path)
    descriptors = tmp_path / "42" / "fd"
    descriptors.mkdir(parents=True)
    fd = descriptors / "3"
    fd.touch()
    assert helpers._linux_fd_counts([42, 999]) == {42: 1}
    fd.unlink()
    assert helpers._linux_fd_counts([42, 999]) == {}


@pytest.mark.parametrize("state", ["zombie", "zombie_with_threads", "gone", "running", "denied"])
def test_linux_fd_permission_error_requires_confirmed_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture, state: str
) -> None:
    monkeypatch.setattr(helpers.sys, "platform", "linux")
    monkeypatch.setattr(helpers, "_LINUX_PROC_ROOT", tmp_path)
    original_iterdir = Path.iterdir

    def denied_fd_table(path):
        if path == tmp_path / "42" / "fd":
            raise PermissionError("FD table denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", denied_fd_table)
    process = mocker.Mock()
    process.num_threads.return_value = 2 if state == "zombie_with_threads" else 1
    lookup = mocker.patch.object(helpers.psutil, "Process", return_value=process)
    if state == "gone":
        lookup.side_effect = helpers.psutil.NoSuchProcess(42)
    elif state == "denied":
        process.status.side_effect = helpers.psutil.AccessDenied(42)
    else:
        process.status.return_value = "zombie" if state == "zombie_with_threads" else state

    if state in {"zombie", "gone"}:
        assert helpers._linux_fd_counts([42]) == {}
    else:
        error = helpers.psutil.AccessDenied if state == "denied" else PermissionError
        with pytest.raises(error):
            helpers._linux_fd_counts([42])


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux zombie process state")
def test_linux_fd_check_accepts_a_real_unreaped_child() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 10
        while helpers.psutil.Process(child.pid).status() != helpers.psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline, "Child did not exit"
            time.sleep(0.01)
        assert helpers._linux_fd_counts([child.pid]) == {}
    finally:
        child.wait(timeout=5)


@pytest.mark.parametrize("num_threads", [1, 2])
def test_cleanup_checks_zombie_threads_with_readable_empty_fd_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture, num_threads: int
) -> None:
    monkeypatch.setattr(helpers.sys, "platform", "linux")
    monkeypatch.setattr(helpers, "_LINUX_PROC_ROOT", tmp_path)
    (tmp_path / "42" / "fd").mkdir(parents=True)
    process = mocker.Mock()
    process.status.return_value = helpers.psutil.STATUS_ZOMBIE
    process.num_threads.return_value = num_threads
    mocker.patch.object(helpers.psutil, "pid_exists", return_value=True)
    mocker.patch.object(helpers.psutil, "Process", return_value=process)
    server = mocker.Mock(reliability_fault_snapshot={"tree_pids": [42], "shm_objects": []})
    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(helpers.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(helpers.time, "sleep", lambda _: None)
    monkeypatch.setattr(helpers, "query_gpu_compute_pid_used_memory_mb", lambda: None)

    if num_threads > 1:
        with pytest.raises(AssertionError, match="residual server-tree processes remain alive"):
            helpers.assert_no_server_tree_process_residual_and_gpu_release(server, scenario="zombie", timeout_sec=1)
    else:
        helpers.assert_no_server_tree_process_residual_and_gpu_release(server, scenario="zombie", timeout_sec=1)


@pytest.mark.parametrize("residual", [None, "fd", "shm"])
def test_cleanup_retries_transient_fd_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture, residual: str | None
) -> None:
    owned = tmp_path / "owned"
    owned.touch()
    identity = owned.stat()
    objects = [(str(owned), identity.st_dev, identity.st_ino)] if residual == "shm" else []
    server = mocker.Mock(reliability_fault_snapshot={"tree_pids": [42], "shm_objects": objects})
    ticks = iter([0.0, 0.0, 0.5, 2.0])
    monkeypatch.setattr(helpers.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(helpers.time, "sleep", lambda _: None)
    monkeypatch.setattr(helpers, "list_alive_pids", lambda _: [])
    monkeypatch.setattr(helpers, "query_gpu_compute_pid_used_memory_mb", lambda: None)
    counts = {42: 1} if residual == "fd" else {}
    read_fds = mocker.Mock(side_effect=[PermissionError("exiting process"), counts])
    monkeypatch.setattr(helpers, "_linux_fd_counts", read_fds)

    if residual is None:
        helpers.assert_no_server_tree_process_residual_and_gpu_release(server, scenario="exit-race", timeout_sec=1)
    else:
        message = "file descriptors remain open" if residual == "fd" else "named shared memory/semaphores remain"
        with pytest.raises(AssertionError, match=message):
            helpers.assert_no_server_tree_process_residual_and_gpu_release(server, scenario="exit-race", timeout_sec=1)
    assert read_fds.call_count == 2


def test_cleanup_does_not_accept_unreadable_fd_tables(monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture) -> None:
    server = mocker.Mock(reliability_fault_snapshot={"tree_pids": [42]})
    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(helpers.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(helpers.time, "sleep", lambda _: None)
    monkeypatch.setattr(helpers, "list_alive_pids", lambda _: [])
    monkeypatch.setattr(helpers, "query_gpu_compute_pid_used_memory_mb", lambda: None)
    monkeypatch.setattr(helpers, "_linux_fd_counts", mocker.Mock(side_effect=PermissionError("still unreadable")))

    with pytest.raises(PermissionError, match="still unreadable"):
        helpers.assert_no_server_tree_process_residual_and_gpu_release(server, scenario="denied", timeout_sec=1)


@pytest.mark.parametrize("resource", ["fd", "shm"])
def test_cleanup_rejects_linux_resource_residual_after_process_exit(monkeypatch, resource):
    server = SimpleNamespace(reliability_fault_snapshot={"tree_pids": [42]})
    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(helpers.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(helpers.time, "sleep", lambda _: None)
    monkeypatch.setattr(helpers, "list_alive_pids", lambda _: [])
    monkeypatch.setattr(helpers, "query_gpu_compute_pid_used_memory_mb", lambda: None)
    monkeypatch.setattr(helpers, "_linux_fd_counts", lambda _: {42: 1} if resource == "fd" else {})
    monkeypatch.setattr(
        helpers, "_remaining_linux_shm_objects", lambda _: ["/dev/shm/owned"] if resource == "shm" else []
    )
    message = "file descriptors remain open" if resource == "fd" else "named shared memory/semaphores remain"
    with pytest.raises(AssertionError, match=message):
        helpers.assert_no_server_tree_process_residual_and_gpu_release(server, scenario="linux", timeout_sec=1)


def test_fault_snapshot_preserves_resources_of_previously_killed_stage(monkeypatch):
    server = SimpleNamespace(reliability_fault_snapshot={"tree_pids": [41, 42], "shm_objects": [("old", 1, 2)]})
    monkeypatch.setattr(helpers, "_resolve_runtime_worker_markers", lambda *args: ())
    monkeypatch.setattr(helpers, "_list_server_process_tree", lambda _: [41])
    monkeypatch.setattr(helpers, "_capture_linux_shm_objects", lambda _: [("new", 1, 3)])
    snapshot = helpers._capture_server_fault_snapshot(server)
    assert snapshot["tree_pids"] == [41, 42]
    assert set(snapshot["shm_objects"]) == {("old", 1, 2), ("new", 1, 3)}


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux /proc and /dev/shm")
@pytest.mark.parametrize("mapping_only", [False, True])
def test_linux_shm_snapshot_reads_real_proc(mapping_only):
    import mmap
    import tempfile

    with tempfile.NamedTemporaryFile(prefix="omni-reliability-", dir="/dev/shm", delete=False) as backing:
        path = Path(backing.name)
        backing.write(b"x" * 4096)
        backing.flush()
        mapping = mmap.mmap(backing.fileno(), 4096) if mapping_only else None
        try:
            if mapping_only:
                backing.close()
            objects = helpers._capture_linux_shm_objects([os.getpid()])
            captured = [item for item in objects if item[0] == str(path)]
            assert len(captured) == 1
            assert helpers._remaining_linux_shm_objects(captured) == [str(path)]
            path.unlink()
            assert helpers._remaining_linux_shm_objects(captured) == []
        finally:
            if mapping is not None:
                mapping.close()
            path.unlink(missing_ok=True)


@pytest.fixture
def cleanup():
    module = ModuleType("cleanup_under_test")
    module.__dict__.update(Any=Any, Sequence=Sequence, stat_module=stat, pytest=pytest)
    root = RELIABILITY_ROOT
    for path, names in [
        (
            root / "helpers.py",
            {
                "_remaining_linux_shm_objects",
                "assert_no_server_tree_process_residual_and_gpu_release",
                "capture_exclusive_fifo_objects",
            },
        ),
        (root / "scenarios/process_kill.py", {"_assert_cleanup"}),
    ]:
        nodes: list[ast.stmt] = [
            n
            for n in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        exec_ast_nodes(nodes, path, module.__dict__)
    return module


@pytest.mark.parametrize("allow_fifo", [False, True])
@pytest.mark.parametrize("resource", ["fifo", "shm", "fd", "gpu", "process"])
def test_fifo_exception_never_disables_other_resource_checks(cleanup, allow_fifo, resource):
    mode = stat.S_IFIFO if resource == "fifo" else stat.S_IFREG
    cleanup.Path = lambda name: SimpleNamespace(stat=lambda: SimpleNamespace(st_dev=1, st_ino=2, st_mode=mode))
    clock = [0.0]
    cleanup.time = SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    cleanup.logger = Mock()
    cleanup.list_alive_pids = lambda pids: [42] if resource == "process" else []
    cleanup.query_gpu_compute_pid_used_memory_mb = lambda: {42: 100} if resource == "gpu" else {}
    cleanup._linux_fd_counts = lambda pids: {42: 1} if resource == "fd" else {}
    server = SimpleNamespace(
        reliability_fault_snapshot={
            "tree_pids": [42],
            "shm_objects": [("owned", 1, 2)] if resource in {"fifo", "shm"} else [],
        }
    )
    if allow_fifo and resource == "fifo":
        cleanup.assert_no_server_tree_process_residual_and_gpu_release(
            server, scenario="tree", timeout_sec=1, allow_fifo_residual=True
        )
        assert clock[0] == 0.0
        cleanup.logger.warning.assert_called_once()
    else:
        with pytest.raises(AssertionError):
            cleanup.assert_no_server_tree_process_residual_and_gpu_release(
                server, scenario="strict", timeout_sec=1, allow_fifo_residual=allow_fifo
            )


@pytest.mark.parametrize(
    "scenario,allowed",
    [
        ("kill_serve_tree_no_load_sigkill", True),
        ("kill_serve_tree_with_load_sigkill", True),
        ("kill_serve_root_no_load_sigkill", False),
    ],
)
def test_only_tree_crash_selects_fifo_exception(cleanup, scenario, allowed):
    cleanup.worker_residual_timeout_after_kill_signal = lambda signal: 30
    cleanup.assert_no_server_tree_process_residual_and_gpu_release = Mock()
    server = object()
    cleanup._assert_cleanup(server, scenario=scenario, signal_name="SIGKILL")
    cleanup.assert_no_server_tree_process_residual_and_gpu_release.assert_called_once_with(
        server,
        scenario=scenario,
        timeout_sec=30,
        allow_fifo_residual=allowed,
        allowed_fifo_objects=(),
    )


@pytest.mark.parametrize("phase", ["no_load", "with_load"])
def test_root_crash_passes_only_captured_root_fifo_exceptions(cleanup, phase):
    root_fifo = ("root_fifo", 1, 2)
    server = SimpleNamespace(reliability_fault_snapshot={"killed_root_fifo_objects": [root_fifo]})
    cleanup.worker_residual_timeout_after_kill_signal = lambda signal: 30
    cleanup.assert_no_server_tree_process_residual_and_gpu_release = Mock()
    cleanup._assert_cleanup(server, scenario=f"kill_serve_root_{phase}_sigkill", signal_name="SIGKILL")
    kwargs = cleanup.assert_no_server_tree_process_residual_and_gpu_release.call_args.kwargs
    assert kwargs["allowed_fifo_objects"] == [root_fifo]
    assert kwargs["allow_fifo_residual"] is False


def test_partial_crash_exempts_only_target_exclusive_fifo_identities(cleanup):
    target = ("target", 1, 2)
    shared = ("shared", 1, 3)
    regular = ("data_segment", 1, 4)
    survivor = ("survivor", 1, 5)
    objects = [target, shared, regular, survivor]
    modes = {name: stat.S_IFIFO if name != "data_segment" else stat.S_IFREG for name, _, _ in objects}
    inodes = {name: inode for name, _, inode in objects}
    cleanup.Path = lambda name: SimpleNamespace(
        stat=lambda: SimpleNamespace(st_dev=1, st_ino=inodes[name], st_mode=modes[name])
    )
    cleanup._capture_linux_shm_objects = lambda pids: [target, shared, regular] if pids == [11] else [shared, survivor]
    allowed = cleanup.capture_exclusive_fifo_objects([11], [10, 11, 12])
    assert allowed == [target]
    assert cleanup._remaining_linux_shm_objects(objects, allowed_fifo_objects=allowed) == [
        "shared",
        "data_segment",
        "survivor",
    ]
    assert cleanup._remaining_linux_shm_objects(objects) == ["target", "shared", "data_segment", "survivor"]
    # A wrong inode must not exempt a different object with the same name.
    assert cleanup._remaining_linux_shm_objects([target], allowed_fifo_objects=[("target", 1, 999)]) == ["target"]
    assert cleanup._remaining_linux_shm_objects([regular], allowed_fifo_objects=[regular]) == ["data_segment"]


def test_tree_crash_preserves_trackers_and_keeps_them_in_snapshot():
    path = RELIABILITY_ROOT / "helpers.py"
    node = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "make_server_tree_kill_fault_injector"
    )
    module: Any = ModuleType("tree_injection")
    module.__dict__.update(Any=Any, Sequence=list, FaultInjector=Any, pytest=pytest)
    exec_ast_nodes([node], path, module.__dict__)
    snapshot = {"tree_pids": [10, 11, 12, 13, 14, 15]}
    server = SimpleNamespace(reliability_fault_snapshot=snapshot)
    commands = {
        10: "serve model",
        11: "python -c from multiprocessing.resource_tracker import main;main(63)",
        12: "vLLM-Omni::DiffusionWorker",
        13: "python -m joblib.externals.loky.backend.resource_tracker",
        14: "python -m multiprocessing.semaphore_tracker",
        15: "ordinary helper",
    }
    killed = []
    module._log_server_process_tree = lambda server: None
    module._capture_server_fault_snapshot = lambda server, **kwargs: snapshot
    module._safe_proc_info = lambda pid: ("python", commands[pid])
    module.psutil = SimpleNamespace(pid_exists=lambda pid: True)
    module.signal = SimpleNamespace(SIGKILL=9)
    module.os = SimpleNamespace(kill=lambda pid, sig: killed.append((pid, sig)))
    module.time = SimpleNamespace(sleep=lambda seconds: None)
    module.make_server_tree_kill_fault_injector()(server)
    assert killed == [(10, 9), (12, 9), (15, 9)]
    assert server.reliability_fault_snapshot["tree_pids"] == [10, 11, 12, 13, 14, 15]


def test_minicpm_fault_cli_pins_single_replica_without_changing_replica_suite():
    import json

    source = RELIABILITY_ROOT / "test_minicpmo_4_5.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniCPMO45")
    scenarios = next(
        n.value
        for n in model.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "scenarios" for t in n.targets)
    )
    namespace: dict[str, Any] = {"_MODEL": "model", "_NATIVE_CHAT_TEMPLATE": "template"}
    cases = eval(compile(ast.Expression(scenarios), str(source), "eval"), namespace)
    args = cases[0]["server_params"]["server_args"]
    overrides = json.loads(args[args.index("--stage-overrides") + 1])
    assert overrides == {str(stage): {"num_replicas": 1, "devices": "0"} for stage in range(3)}
    replica = (RELIABILITY_ROOT / "test_minicpmo_4_5_replica.py").read_text(encoding="utf-8")
    assert "--stage-overrides" not in replica
