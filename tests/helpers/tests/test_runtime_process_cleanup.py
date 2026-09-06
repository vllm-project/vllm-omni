# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Safety contract for OmniRunner residual-process cleanup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.helpers import runtime as runtime_helpers
from tests.helpers.runtime import OmniRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_omni_runner_only_snapshots_owned_engine_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SimpleNamespace(
        pid=101,
        cmdline=lambda: ["vllm::EngineCore"],
        name=lambda: "python",
    )
    unrelated = SimpleNamespace(
        pid=202,
        cmdline=lambda: ["python", "unrelated.py"],
        name=lambda: "python",
    )
    root = SimpleNamespace(children=lambda recursive: [engine, unrelated])
    monkeypatch.setattr(runtime_helpers.psutil, "Process", lambda _pid: root)
    monkeypatch.setattr(
        runtime_helpers.psutil,
        "process_iter",
        lambda *_args, **_kwargs: pytest.fail("global process iteration is forbidden"),
    )

    runner = object.__new__(OmniRunner)

    assert runner._owned_engine_processes() == [engine]


def test_omni_runner_only_signals_explicit_owned_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Process:
        def __init__(self, pid: int, label: str) -> None:
            self.pid = pid
            self.label = label

        def terminate(self) -> None:
            events.append(f"terminate:{self.label}")

        def kill(self) -> None:
            events.append(f"kill:{self.label}")

    owned = Process(101, "owned")
    external = Process(202, "external")
    wait_calls = 0

    def wait_procs(processes, timeout):
        nonlocal wait_calls
        wait_calls += 1
        assert processes == [owned]
        if wait_calls == 1:
            assert timeout == 5
            return [], [owned]
        assert timeout == 3
        return [owned], []

    monkeypatch.setattr(runtime_helpers.psutil, "wait_procs", wait_procs)

    runner = object.__new__(OmniRunner)
    runner._cleanup_process([owned])

    assert events == ["terminate:owned", "kill:owned"]
    assert external.pid == 202


def test_omni_runner_cleanup_runs_when_graceful_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    owned = SimpleNamespace(pid=101)

    class Omni:
        def close(self) -> None:
            events.append("close")
            raise RuntimeError("injected close failure")

    runner = object.__new__(OmniRunner)
    runner.omni = Omni()
    setattr(runner, "_owned_engine_processes", lambda: [owned])
    setattr(
        runner,
        "_cleanup_process",
        lambda matched: events.append(f"cleanup:{matched[0].pid}"),
    )
    monkeypatch.setattr(
        runtime_helpers,
        "cleanup_test_environment",
        lambda: events.append("environment-cleanup"),
    )

    with pytest.raises(RuntimeError, match="injected close failure"):
        runner.__exit__(None, None, None)

    assert events == [
        "close",
        "cleanup:101",
        "environment-cleanup",
    ]
