# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for OmniRunner startup rollback and worker ownership."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Generator

import psutil
import pytest

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


@pytest.fixture
def live_workers() -> Generator[list[subprocess.Popen[bytes]], None, None]:
    workers: list[subprocess.Popen[bytes]] = []
    try:
        yield workers
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=10)
            if worker.stdout is not None:
                worker.stdout.close()


def _start_engine_worker(
    workers: list[subprocess.Popen[bytes]], *, ignore_terminate: bool = False
) -> subprocess.Popen[bytes]:
    # Engine workers are recognised by process title, as set by vLLM's ``set_process_title``.
    script = "import signal, sys, time, setproctitle; setproctitle.setproctitle(sys.argv[1]); "
    if ignore_terminate:
        script += "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    script += "print('ready', flush=True); time.sleep(90)"
    worker = subprocess.Popen([sys.executable, "-c", script, "VLLM::EngineCore"], stdout=subprocess.PIPE)
    workers.append(worker)
    assert worker.stdout is not None
    assert worker.stdout.readline() == b"ready\n"
    return worker


@pytest.mark.parametrize("close_raises", [False, True])
def test_runner_cleanup_preserves_unrelated_engine_workers(
    monkeypatch: pytest.MonkeyPatch,
    live_workers: list[subprocess.Popen[bytes]],
    close_raises: bool,
) -> None:
    existing = _start_engine_worker(live_workers)
    owned: list[subprocess.Popen[bytes]] = []
    late_unrelated: list[subprocess.Popen[bytes]] = []
    monkeypatch.setattr("tests.helpers.runtime.cleanup_test_environment", lambda: None)

    class _OwnedOmni:
        def __init__(self, *args: object, **kwargs: object) -> None:
            owned.append(_start_engine_worker(live_workers))

        def close(self) -> None:
            late_unrelated.append(_start_engine_worker(live_workers))
            if close_raises:
                raise RuntimeError("close failed")

    monkeypatch.setattr("vllm_omni.entrypoints.omni.Omni", _OwnedOmni)
    if close_raises:
        with pytest.raises(RuntimeError, match="close failed"), OmniRunner("fake-model"):
            pass
    else:
        with OmniRunner("fake-model"):
            pass

    assert owned[0].poll() is not None
    assert existing.poll() is None
    assert late_unrelated[0].poll() is None


def test_runner_startup_failure_cleans_only_new_workers(
    monkeypatch: pytest.MonkeyPatch, live_workers: list[subprocess.Popen[bytes]]
) -> None:
    unrelated = _start_engine_worker(live_workers)
    owned: list[subprocess.Popen[bytes]] = []
    monkeypatch.setattr("tests.helpers.runtime.cleanup_test_environment", lambda: None)

    class _FailingOmni:
        def __init__(self, *args: object, **kwargs: object) -> None:
            owned.append(_start_engine_worker(live_workers))
            raise RuntimeError("startup failed")

    monkeypatch.setattr("vllm_omni.entrypoints.omni.Omni", _FailingOmni)
    with pytest.raises(RuntimeError, match="startup failed"):
        OmniRunner("fake-model")

    assert owned[0].poll() is not None
    assert unrelated.poll() is None


def test_runner_startup_failure_cleans_reparented_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker whose parent died during startup is still stopped by the rollback."""
    worker_script = (
        "import sys, time, setproctitle; setproctitle.setproctitle(sys.argv[1]); "
        "print('ready', flush=True); time.sleep(90)"
    )
    # Starts a titled worker, prints its pid and exits, which orphans the worker.
    parent_script = (
        "import subprocess, sys; "
        "worker = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], stdout=subprocess.PIPE); "
        "worker.stdout.readline(); print(worker.pid, flush=True)"
    )
    orphans: list[psutil.Process] = []
    monkeypatch.setattr("tests.helpers.runtime.cleanup_test_environment", lambda: None)

    class _FailingOmni:
        def __init__(self, *args: object, **kwargs: object) -> None:
            done = subprocess.run(
                [sys.executable, "-c", parent_script, worker_script, "VLLM::EngineCore"],
                stdout=subprocess.PIPE,
                check=True,
                timeout=30,
            )
            orphans.append(psutil.Process(int(done.stdout)))
            raise RuntimeError("startup failed")

    monkeypatch.setattr("vllm_omni.entrypoints.omni.Omni", _FailingOmni)
    try:
        with pytest.raises(RuntimeError, match="startup failed"):
            OmniRunner("fake-model")
        _, alive = psutil.wait_procs(orphans, timeout=10)
        assert not alive
    finally:
        for orphan in orphans:
            try:
                orphan.kill()
            except psutil.NoSuchProcess:
                pass


def test_runner_cleanup_rejects_stale_process_identity(live_workers: list[subprocess.Popen[bytes]]) -> None:
    unrelated = _start_engine_worker(live_workers)
    runner = object.__new__(OmniRunner)
    runner._owned_engine_processes = {unrelated.pid: psutil.Process(unrelated.pid).create_time() + 1}

    runner._cleanup_process()

    assert unrelated.poll() is None


def test_runner_cleanup_kills_owned_worker_ignoring_terminate(
    monkeypatch: pytest.MonkeyPatch, live_workers: list[subprocess.Popen[bytes]]
) -> None:
    owned: list[subprocess.Popen[bytes]] = []
    monkeypatch.setattr("tests.helpers.runtime.cleanup_test_environment", lambda: None)

    class _StubbornOmni:
        def __init__(self, *args: object, **kwargs: object) -> None:
            owned.append(_start_engine_worker(live_workers, ignore_terminate=True))

        def close(self) -> None:
            pass

    monkeypatch.setattr("vllm_omni.entrypoints.omni.Omni", _StubbornOmni)
    with OmniRunner("fake-model"):
        assert owned[0].poll() is None

    assert owned[0].poll() is not None
