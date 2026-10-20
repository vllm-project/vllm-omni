# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from tests.dfx.reliability import helpers

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
