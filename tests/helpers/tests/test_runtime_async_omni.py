# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for the in-process ``AsyncOmni`` harness and the engine-worker reap helpers.

Covers ``AsyncOmniRunner`` teardown order (success, test failure, constructor
failure, ``shutdown()`` failure), the ``iter_async_omni`` parameter contract
behind ``async_omni_runner`` / ``async_omni``, and the PID snapshot / reap
helpers in ``tests.helpers.clean``. No engine is started.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from tests.helpers import clean as clean_mod
from tests.helpers import runtime as runtime_mod
from tests.helpers.runtime import AsyncOmniParams, AsyncOmniRunner, iter_async_omni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# AsyncOmniRunner lifecycle
# ---------------------------------------------------------------------------


class _Lifecycle:
    """Cleanup calls made by the runner, in order; ``snapshots`` feeds the PID snapshot."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.snapshots: list[list[int]] = []


@pytest.fixture
def lifecycle(monkeypatch: pytest.MonkeyPatch) -> _Lifecycle:
    state = _Lifecycle()

    def _snapshot() -> list[int]:
        state.events.append("snapshot")
        return state.snapshots.pop(0) if state.snapshots else []

    def _reap(pids, **kwargs) -> list[int]:
        state.events.append(("reap", list(pids)))
        return list(pids)

    monkeypatch.setattr(runtime_mod, "cleanup_test_environment", lambda: state.events.append("env"))
    monkeypatch.setattr(runtime_mod, "snapshot_engine_worker_pids", _snapshot)
    monkeypatch.setattr(runtime_mod, "reap_engine_worker_pids", _reap)
    return state


def _install_fake_async_omni(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle: _Lifecycle,
    *,
    shutdown_error: BaseException | None = None,
):
    class _FakeAsyncOmni:
        instances: list[Any] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.answer = 42
            _FakeAsyncOmni.instances.append(self)
            lifecycle.events.append("construct")

        def shutdown(self) -> None:
            lifecycle.events.append("shutdown")
            if shutdown_error is not None:
                raise shutdown_error

    monkeypatch.setattr("vllm_omni.entrypoints.async_omni.AsyncOmni", _FakeAsyncOmni)
    return _FakeAsyncOmni


def test_runner_init_rolls_back_on_async_omni_startup_failure(
    monkeypatch: pytest.MonkeyPatch, lifecycle: _Lifecycle
) -> None:
    """``__exit__`` is skipped when construction raises; workers spawned before the
    failure must still be reaped and the device reset."""
    lifecycle.snapshots.append([11, 12])

    class _BoomAsyncOmni:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Orchestrator initialization failed")

    monkeypatch.setattr("vllm_omni.entrypoints.async_omni.AsyncOmni", _BoomAsyncOmni)

    with pytest.raises(RuntimeError, match="Orchestrator initialization failed"):
        with AsyncOmniRunner("fake-model"):
            raise AssertionError("context body must not run after a failed constructor")

    # Once at the start of ``__init__``, once from the constructor rollback; no
    # ``shutdown`` because no engine instance exists.
    assert lifecycle.events == ["env", "snapshot", ("reap", [11, 12]), "env"]


def test_runner_teardown_order_and_engine_proxy(monkeypatch: pytest.MonkeyPatch, lifecycle: _Lifecycle) -> None:
    lifecycle.snapshots.extend([[21], [21, 22]])
    fake_cls = _install_fake_async_omni(monkeypatch, lifecycle)

    with AsyncOmniRunner("fake-model", deploy_config="deploy.yaml", enforce_eager=True) as runner:
        lifecycle.events.append("body")
        engine = fake_cls.instances[-1]
        assert runner.engine is engine
        # Attribute access falls through to the engine so tests can call
        # ``runner.generate(...)`` without unwrapping.
        assert runner.answer == 42
        assert engine.kwargs == {
            "model": "fake-model",
            "log_stats": False,
            "stage_init_timeout": 600,
            "init_timeout": 1800,
            "deploy_config": "deploy.yaml",
            "enforce_eager": True,
        }

    # PIDs are snapshotted again *before* shutdown (the constructor snapshot can
    # miss workers that were reparented later), then shutdown, reap, cleanup.
    assert lifecycle.events == [
        "env",
        "construct",
        "snapshot",
        "body",
        "snapshot",
        "shutdown",
        ("reap", [21, 22]),
        "env",
    ]

    # ``close()`` is idempotent and the engine reference is dropped.
    runner.close()
    assert lifecycle.events.count("shutdown") == 1
    assert lifecycle.events.count("env") == 2
    assert runner.engine is None
    with pytest.raises(AttributeError):
        _ = runner.answer


def test_runner_teardown_runs_when_body_raises(monkeypatch: pytest.MonkeyPatch, lifecycle: _Lifecycle) -> None:
    _install_fake_async_omni(monkeypatch, lifecycle)

    with pytest.raises(ValueError, match="body failed"):
        with AsyncOmniRunner("fake-model"):
            raise ValueError("body failed")

    assert lifecycle.events[-3:] == ["shutdown", ("reap", []), "env"]


def test_runner_reaps_and_cleans_when_shutdown_raises(monkeypatch: pytest.MonkeyPatch, lifecycle: _Lifecycle) -> None:
    lifecycle.snapshots.extend([[31], []])
    _install_fake_async_omni(monkeypatch, lifecycle, shutdown_error=RuntimeError("shutdown boom"))

    with pytest.raises(RuntimeError, match="shutdown boom"):
        with AsyncOmniRunner("fake-model"):
            pass

    assert lifecycle.events[-3:] == ["shutdown", ("reap", [31]), "env"]


# ---------------------------------------------------------------------------
# iter_async_omni (behind the async_omni_runner / async_omni fixtures)
# ---------------------------------------------------------------------------


class _FakeRunner:
    calls: list[dict[str, Any]] = []

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.closed = False
        _FakeRunner.calls.append({"model_name": model_name, **kwargs})

    def __enter__(self) -> _FakeRunner:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.closed = True


def _fake_request(param: Any, *, diffusion: bool) -> SimpleNamespace:
    marker = SimpleNamespace(name="diffusion") if diffusion else None
    return SimpleNamespace(
        param=param,
        fixturename="async_omni_runner_function",
        node=SimpleNamespace(get_closest_marker=lambda name: marker if name == "diffusion" else None),
    )


@pytest.fixture
def fixture_env(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Stub everything ``iter_async_omni`` touches; returns the Whisper-release log."""
    _FakeRunner.calls = []
    released: list[bool] = []
    monkeypatch.setattr(runtime_mod, "AsyncOmniRunner", _FakeRunner)
    monkeypatch.setattr(runtime_mod, "get_model_prefix", lambda: "/models/")
    monkeypatch.setattr(runtime_mod, "resolve_tiny_model_path", lambda model: f"{model}-tiny")
    monkeypatch.setattr(runtime_mod, "release_audio_transcriber", lambda: released.append(True))
    monkeypatch.setattr(
        "tests.helpers.stage_config.stage_config_path_for_run_level",
        lambda path, level: None if path is None else f"{path}@{level}",
    )
    return released


@pytest.mark.parametrize(
    ("run_level", "diffusion", "expected_model"),
    [
        ("core_model", True, "/models/org/model-tiny"),
        ("core_model", False, "/models/org/model"),
        ("advanced_model", True, "/models/org/model"),
    ],
)
def test_iter_async_omni_resolves_model_and_deploy_config(
    fixture_env: list[bool], run_level: str, diffusion: bool, expected_model: str
) -> None:
    params = AsyncOmniParams(model="org/model", deploy_config="deploy.yaml", extra_omni_kwargs={"enforce_eager": True})
    lock = threading.Lock()

    gen = iter_async_omni(_fake_request(params, diffusion=diffusion), run_level, lock)
    runner = next(gen)

    assert isinstance(runner, _FakeRunner)
    assert lock.locked(), "the fixture lock is held while the engine is alive"
    assert _FakeRunner.calls == [
        {"model_name": expected_model, "deploy_config": f"deploy.yaml@{run_level}", "enforce_eager": True}
    ]
    assert fixture_env == [True], "Whisper judge released before the engine starts"

    with pytest.raises(StopIteration):
        next(gen)
    assert runner.closed
    assert not lock.locked()
    assert fixture_env == [True, True], "Whisper judge released again after shutdown"


def test_iter_async_omni_defaults(fixture_env: list[bool]) -> None:
    request = _fake_request(AsyncOmniParams(model="org/model"), diffusion=False)
    gen = iter_async_omni(request, "core_model", threading.Lock())
    next(gen)
    assert _FakeRunner.calls == [{"model_name": "/models/org/model", "deploy_config": None}]
    with pytest.raises(StopIteration):
        next(gen)


@pytest.mark.parametrize("param", [("org/model", None), ["org/model"], "org/model", None])
def test_iter_async_omni_rejects_non_params(fixture_env: list[bool], param: Any) -> None:
    request = _fake_request(param, diffusion=False)
    if param is None:
        del request.param
    with pytest.raises(ValueError, match="AsyncOmniParams"):
        next(iter_async_omni(request, "core_model", threading.Lock()))
    assert _FakeRunner.calls == []


# ---------------------------------------------------------------------------
# Engine-worker snapshot / reap helpers
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, pid: int, title: str, *, error: BaseException | None = None) -> None:
        self.pid = pid
        self._title = title
        self._error = error
        self.killed = False

    def name(self) -> str:
        if self._error is not None:
            raise self._error
        return self._title.split()[0]

    def cmdline(self) -> list[str]:
        return self._title.split()

    def kill(self) -> None:
        self.killed = True


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("VLLM::EngineCore", True),
        ("VLLM::Worker_TP0", True),
        ("vLLM-Omni::DiffusionWorker_TP1", True),
        ("StageDiffusionProc", True),
        ("python -m pytest tests/", False),
        ("python -c import whisper", False),
    ],
)
def test_is_engine_worker_process_matches_engine_titles(title: str, expected: bool) -> None:
    assert clean_mod.is_engine_worker_process(_FakeProc(1, title)) is expected


def test_is_engine_worker_process_ignores_vanished_processes() -> None:
    proc = _FakeProc(1, "VLLM::EngineCore", error=psutil.NoSuchProcess(1))
    assert clean_mod.is_engine_worker_process(proc) is False


def test_snapshot_and_reap_engine_worker_pids(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _FakeProc(101, "vLLM-Omni::DiffusionWorker")
    core = _FakeProc(102, "VLLM::EngineCore")
    other = _FakeProc(103, "python -c import whisper")
    by_pid = {101: worker, 102: core, 103: other}
    waited: list[tuple[list[int], float]] = []

    def _process(pid: int | None = None):
        if pid is None:
            return SimpleNamespace(children=lambda recursive: [worker, core, other])
        if pid not in by_pid:
            raise psutil.NoSuchProcess(pid)
        return by_pid[pid]

    def _wait_procs(procs, timeout):
        waited.append(([p.pid for p in procs], timeout))
        return [], []

    monkeypatch.setattr(
        clean_mod,
        "psutil",
        SimpleNamespace(
            Process=_process, wait_procs=_wait_procs, Error=psutil.Error, NoSuchProcess=psutil.NoSuchProcess
        ),
    )

    pids = clean_mod.snapshot_engine_worker_pids()
    assert pids == [101, 102]

    del by_pid[102]  # EngineCore exited on its own after shutdown()
    assert clean_mod.reap_engine_worker_pids([*pids, 999]) == [101]
    assert worker.killed
    assert not other.killed
    assert waited == [([101], 3.0)]


def test_reap_engine_worker_pids_without_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        clean_mod,
        "psutil",
        SimpleNamespace(
            Process=lambda pid: (_ for _ in ()).throw(AssertionError("must not look up processes")),
            wait_procs=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not wait")),
            Error=psutil.Error,
            NoSuchProcess=psutil.NoSuchProcess,
        ),
    )
    assert clean_mod.reap_engine_worker_pids([]) == []
