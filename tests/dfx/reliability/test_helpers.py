# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import ast
import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.dfx.reliability import helpers

_THINKER_STAGE_PATTERN = "StageEngineCoreProc_stage0"

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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

    helpers.assert_no_server_tree_process_residual_and_gpu_release(
        server,
        scenario="rocm-no-telemetry",
        timeout_sec=1.0,
    )


def test_qwen_worker_pattern_is_the_thinker_stage() -> None:
    source = (Path(__file__).resolve().parent / "test_qwen3_omni.py").read_text(encoding="utf-8")
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
    hunyuan = Path(__file__).resolve().parents[3] / "vllm_omni" / "deploy" / "hunyuan_image3_dit.yaml"

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
