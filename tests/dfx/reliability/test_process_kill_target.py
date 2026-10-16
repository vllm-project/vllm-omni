# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen text probes must kill the thinker stage and see that PID die."""

from __future__ import annotations

import signal

import pytest

from tests.dfx.reliability import helpers
from tests.dfx.reliability.test_reliability_qwen3_omni import RUNTIME_WORKER_PATTERN

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_qwen_worker_pattern_is_the_thinker_stage() -> None:
    assert RUNTIME_WORKER_PATTERN == "StageEngineCoreProc_stage0"


def test_stage0_pattern_does_not_select_stage1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpers, "_list_server_process_tree", lambda _server: [1, 2, 3])
    names = {
        1: ("python", "serve"),
        2: ("VLLM::StageEngineCoreProc_stage1_replica0_DP0", "stage1"),
        3: ("VLLM::StageEngineCoreProc_stage0_replica0_DP0", "stage0"),
    }
    monkeypatch.setattr(helpers, "_safe_proc_info", lambda pid: names[pid])

    hits = helpers._pids_in_server_tree_substring_match(object(), RUNTIME_WORKER_PATTERN, limit=1)

    assert hits == [3]


def test_sigkill_records_dead_pid_and_health(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Proc:
        pid = 10

    class _Server:
        proc = _Proc()
        host = "127.0.0.1"
        port = 9

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
    monkeypatch.setattr(helpers, "get_health_raw", lambda *_args, **_kwargs: (503, b""))

    inject = helpers.make_process_kill_fault_injector(
        grep_patterns=RUNTIME_WORKER_PATTERN,
        signal_name="SIGKILL",
        limit=1,
        post_kill_wait_seconds=0.0,
    )
    inject(_Server())

    assert killed == [(20, signal.SIGKILL)]
    output = capsys.readouterr().out
    assert "pid=20" in output
    assert "dead=1" in output
    assert "post-kill health status=503" in output
