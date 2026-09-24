# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for OmniRunner startup rollback."""

from __future__ import annotations

import os

import pytest

import tests.helpers.runtime as runtime
from tests.helpers.runtime import OmniRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_runner_init_rolls_back_on_omni_startup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """``__exit__`` is skipped when construction raises; rollback must still run."""
    cleaned: list[str] = []
    monkeypatch.setattr(
        "tests.helpers.runtime.cleanup_test_environment",
        lambda: cleaned.append("env"),
    )
    monkeypatch.setattr(
        OmniRunner,
        "_cleanup_process",
        lambda self: cleaned.append("process"),
    )

    class _BoomOmni:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Orchestrator initialization failed")

    monkeypatch.setattr("vllm_omni.entrypoints.omni.Omni", _BoomOmni)

    with pytest.raises(RuntimeError, match="Orchestrator initialization failed"):
        with OmniRunner("fake-model"):
            raise AssertionError("context body must not run after a failed constructor")

    assert cleaned.count("process") == 1
    # Once at the start of ``__init__``, once from the constructor rollback.
    assert cleaned.count("env") == 2


class _FakeChildProcess:
    def __init__(self, pid: int, cmdline: list[str] | None = None, name: str = "python") -> None:
        self.pid = pid
        self._cmdline = cmdline or []
        self._name = name
        self.terminated = False
        self.killed = False

    def cmdline(self) -> list[str]:
        return self._cmdline

    def name(self) -> str:
        return self._name

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def test_runner_cleanup_only_owns_new_child_enginecore_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cleanup must ignore baseline children and processes outside the tree."""
    baseline = _FakeChildProcess(101, ["python", "enginecore-existing"])
    owned = _FakeChildProcess(202, ["python", "enginecore-owned"])
    foreign = _FakeChildProcess(303, ["python", "enginecore-foreign"])
    root = _FakeChildProcess(os.getpid())
    root.children = lambda recursive=True: [baseline, owned]  # type: ignore[attr-defined]
    process_calls: list[int] = []
    global_scan_calls: list[None] = []

    def fake_process(pid: int) -> _FakeChildProcess:
        process_calls.append(pid)
        if pid == os.getpid():
            return root
        if pid == owned.pid:
            return owned
        raise AssertionError(f"unrelated or baseline process was inspected: {pid}")

    monkeypatch.setattr(runtime.psutil, "Process", fake_process)

    def fake_process_iter(*args, **kwargs):
        global_scan_calls.append(None)
        return iter([foreign])

    monkeypatch.setattr(
        runtime.psutil,
        "process_iter",
        fake_process_iter,
    )
    monkeypatch.setattr(runtime.psutil, "wait_procs", lambda processes, timeout: (processes, []))
    runner = object.__new__(OmniRunner)
    runner._runner_baseline_child_pids = {baseline.pid}

    runner._cleanup_process()

    assert process_calls == [os.getpid(), owned.pid]
    assert not global_scan_calls
    assert not baseline.terminated
    assert owned.terminated
    assert not foreign.terminated


def test_runner_failed_startup_cleans_only_post_baseline_children(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructor rollback must use the same child-PID ownership boundary."""
    baseline = _FakeChildProcess(101, ["python", "enginecore-existing"])
    owned = _FakeChildProcess(202, ["python", "enginecore-owned"])
    root = _FakeChildProcess(os.getpid())
    children = [baseline]
    root.children = lambda recursive=True: list(children)  # type: ignore[attr-defined]
    process_calls: list[int] = []

    def fake_process(pid: int) -> _FakeChildProcess:
        process_calls.append(pid)
        if pid == os.getpid():
            return root
        if pid == owned.pid:
            return owned
        raise AssertionError(f"unrelated or baseline process was inspected: {pid}")

    monkeypatch.setattr(runtime.psutil, "Process", fake_process)
    monkeypatch.setattr(runtime.psutil, "wait_procs", lambda processes, timeout: (processes, []))
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)

    class _BoomOmni:
        def __init__(self, *args, **kwargs) -> None:
            children.append(owned)
            raise RuntimeError("Orchestrator initialization failed")

    monkeypatch.setattr("vllm_omni.entrypoints.omni.Omni", _BoomOmni)

    with pytest.raises(RuntimeError, match="Orchestrator initialization failed"):
        OmniRunner("fake-model")

    assert process_calls == [os.getpid(), os.getpid(), owned.pid]
    assert not baseline.terminated
    assert owned.terminated
