# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for the in-process ``AsyncOmni`` harness and the engine-worker matcher.

Covers ``AsyncOmniRunner`` teardown order (success, test failure, constructor
failure, ``shutdown()`` failure), worker ownership on teardown, the
``iter_async_omni`` parameter contract behind ``async_omni_runner`` /
``async_omni``, and ``is_engine_worker_process`` in ``tests.helpers.clean``.
No engine is started.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Generator
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


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Cleanup calls made by the runner, in order."""
    calls: list[str] = []
    monkeypatch.setattr(runtime_mod, "cleanup_test_environment", lambda: calls.append("env"))
    monkeypatch.setattr(AsyncOmniRunner, "_cleanup_process", lambda self: calls.append("process"))
    return calls


def _install_fake_async_omni(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    shutdown_error: BaseException | None = None,
):
    class _FakeAsyncOmni:
        instances: list[Any] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.answer = 42
            _FakeAsyncOmni.instances.append(self)
            events.append("construct")

        def shutdown(self) -> None:
            events.append("shutdown")
            if shutdown_error is not None:
                raise shutdown_error

    monkeypatch.setattr("vllm_omni.entrypoints.async_omni.AsyncOmni", _FakeAsyncOmni)
    return _FakeAsyncOmni


def test_runner_init_rolls_back_on_async_omni_startup_failure(
    monkeypatch: pytest.MonkeyPatch, events: list[str]
) -> None:
    """``__exit__`` is skipped when construction raises; rollback must still run."""

    class _BoomAsyncOmni:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Orchestrator initialization failed")

    monkeypatch.setattr("vllm_omni.entrypoints.async_omni.AsyncOmni", _BoomAsyncOmni)

    with pytest.raises(RuntimeError, match="Orchestrator initialization failed"):
        with AsyncOmniRunner("fake-model"):
            raise AssertionError("context body must not run after a failed constructor")

    # Once at the start of ``__init__``, once from the constructor rollback; no
    # ``shutdown`` because no engine instance exists.
    assert events == ["env", "process", "env"]


def test_runner_teardown_order_and_engine_proxy(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    fake_cls = _install_fake_async_omni(monkeypatch, events)

    with AsyncOmniRunner("fake-model", deploy_config="deploy.yaml", enforce_eager=True) as runner:
        events.append("body")
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

    assert events == ["env", "construct", "body", "shutdown", "process", "env"]

    # ``close()`` is idempotent and the engine reference is dropped.
    runner.close()
    assert events.count("shutdown") == 1
    assert events.count("env") == 2
    assert runner.engine is None
    with pytest.raises(AttributeError):
        _ = runner.answer


def test_runner_teardown_runs_when_body_raises(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    _install_fake_async_omni(monkeypatch, events)

    with pytest.raises(ValueError, match="body failed"):
        with AsyncOmniRunner("fake-model"):
            raise ValueError("body failed")

    assert events[-3:] == ["shutdown", "process", "env"]


def test_runner_cleans_up_when_shutdown_raises(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    _install_fake_async_omni(monkeypatch, events, shutdown_error=RuntimeError("shutdown boom"))

    with pytest.raises(RuntimeError, match="shutdown boom"):
        with AsyncOmniRunner("fake-model"):
            pass

    assert events[-3:] == ["shutdown", "process", "env"]


# ---------------------------------------------------------------------------
# AsyncOmniRunner worker ownership (real subprocesses)
# ---------------------------------------------------------------------------


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
    workers: list[subprocess.Popen[bytes]], *, title: str = "enginecore"
) -> subprocess.Popen[bytes]:
    script = "import time; print('ready', flush=True); time.sleep(90)"
    worker = subprocess.Popen([sys.executable, "-c", script, title], stdout=subprocess.PIPE)
    workers.append(worker)
    assert worker.stdout is not None
    assert worker.stdout.readline() == b"ready\n"
    return worker


@pytest.mark.parametrize("shutdown_raises", [False, True])
def test_runner_cleanup_preserves_unrelated_engine_workers(
    monkeypatch: pytest.MonkeyPatch,
    live_workers: list[subprocess.Popen[bytes]],
    shutdown_raises: bool,
) -> None:
    existing = _start_engine_worker(live_workers)
    owned: list[subprocess.Popen[bytes]] = []
    late_unrelated: list[subprocess.Popen[bytes]] = []
    monkeypatch.setattr(runtime_mod, "cleanup_test_environment", lambda: None)

    class _OwnedAsyncOmni:
        def __init__(self, **kwargs: Any) -> None:
            owned.append(_start_engine_worker(live_workers))
            # Diffusion workers carry a different process title than EngineCore.
            owned.append(_start_engine_worker(live_workers, title="vLLM-Omni::DiffusionWorker_0"))

        def shutdown(self) -> None:
            late_unrelated.append(_start_engine_worker(live_workers))
            if shutdown_raises:
                raise RuntimeError("shutdown failed")

    monkeypatch.setattr("vllm_omni.entrypoints.async_omni.AsyncOmni", _OwnedAsyncOmni)
    if shutdown_raises:
        with pytest.raises(RuntimeError, match="shutdown failed"), AsyncOmniRunner("fake-model"):
            pass
    else:
        with AsyncOmniRunner("fake-model"):
            assert all(worker.poll() is None for worker in owned)

    assert all(worker.wait(timeout=10) is not None for worker in owned)
    assert existing.poll() is None
    assert late_unrelated[0].poll() is None


def test_runner_startup_failure_cleans_only_new_workers(
    monkeypatch: pytest.MonkeyPatch, live_workers: list[subprocess.Popen[bytes]]
) -> None:
    unrelated = _start_engine_worker(live_workers)
    owned: list[subprocess.Popen[bytes]] = []
    monkeypatch.setattr(runtime_mod, "cleanup_test_environment", lambda: None)

    class _FailingAsyncOmni:
        def __init__(self, **kwargs: Any) -> None:
            owned.append(_start_engine_worker(live_workers))
            raise RuntimeError("startup failed")

    monkeypatch.setattr("vllm_omni.entrypoints.async_omni.AsyncOmni", _FailingAsyncOmni)
    with pytest.raises(RuntimeError, match="startup failed"):
        AsyncOmniRunner("fake-model")

    assert owned[0].wait(timeout=10) is not None
    assert unrelated.poll() is None


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
# Engine-worker matcher
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, pid: int, title: str, *, error: BaseException | None = None) -> None:
        self.pid = pid
        self._title = title
        self._error = error

    def name(self) -> str:
        if self._error is not None:
            raise self._error
        return self._title.split()[0]

    def cmdline(self) -> list[str]:
        return self._title.split()


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
