# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""L1 contract tests for AsyncOmniRunner (RFC #8013 Phase A).

No GPU and no real engine: ``AsyncOmni`` and the cleanup helpers are
monkeypatched, mirroring ``test_runtime_omni_runner.py``.
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tests.helpers import runtime as runtime_mod
from tests.helpers import stage_config as stage_config_mod
from tests.helpers.runtime import AsyncOmniParams, AsyncOmniRunner, iter_async_omni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeAsyncOmni:
    """Records the kwargs it was built with and its shutdown calls."""

    instances: list[_FakeAsyncOmni] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.shutdown_calls = 0
        self.magic = 7
        _FakeAsyncOmni.instances.append(self)

    def shutdown(self, timeout: float | None = None) -> None:
        self.shutdown_calls += 1
        sink = getattr(self, "event_sink", None)
        if sink is not None:
            sink.append("shutdown")


@pytest.fixture(autouse=True)
def _reset_instances():
    _FakeAsyncOmni.instances.clear()
    yield
    _FakeAsyncOmni.instances.clear()


@pytest.fixture
def _tracked_cleanup(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    events: list[str] = []
    monkeypatch.setattr(_FakeAsyncOmni, "event_sink", events, raising=False)
    monkeypatch.setattr(
        "tests.helpers.runtime.cleanup_test_environment",
        lambda: events.append("env"),
    )
    monkeypatch.setattr(
        "tests.helpers.runtime.reap_leftover_engine_children",
        lambda: events.append("reap"),
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni.AsyncOmni",
        _FakeAsyncOmni,
    )
    return events


def test_runner_init_rolls_back_on_async_omni_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``__exit__`` is skipped when construction raises; rollback must still run."""

    class _BoomAsyncOmni:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Async orchestrator initialization failed")

    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni.AsyncOmni",
        _BoomAsyncOmni,
    )
    cleaned: list[str] = []
    monkeypatch.setattr(
        "tests.helpers.runtime.cleanup_test_environment",
        lambda: cleaned.append("env"),
    )
    monkeypatch.setattr(
        "tests.helpers.runtime.reap_leftover_engine_children",
        lambda: cleaned.append("reap"),
    )

    with pytest.raises(RuntimeError, match="Async orchestrator initialization failed"):
        with AsyncOmniRunner("fake-model"):
            raise AssertionError("context body must not run after a failed constructor")

    assert cleaned.count("reap") == 1
    # Once at the start of ``__init__``, once from the constructor rollback.
    assert cleaned.count("env") == 2


def test_runner_teardown_order_on_clean_exit(_tracked_cleanup: list[str]) -> None:
    with AsyncOmniRunner("fake-model", deploy_config=None, max_num_seqs=1) as runner:
        engine = runner.engine
        assert isinstance(engine, _FakeAsyncOmni)
        assert engine.kwargs["model"] == "fake-model"
        assert engine.kwargs["max_num_seqs"] == 1

    assert engine.shutdown_calls == 1
    assert _tracked_cleanup == ["env", "shutdown", "reap", "env"]


def test_runner_teardown_runs_on_body_exception(_tracked_cleanup: list[str]) -> None:
    with pytest.raises(ValueError, match="boom"):
        with AsyncOmniRunner("fake-model"):
            raise ValueError("boom")

    assert _tracked_cleanup == ["env", "shutdown", "reap", "env"]
    assert _FakeAsyncOmni.instances[0].shutdown_calls == 1


def test_runner_teardown_reaps_and_cleans_when_shutdown_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed/hung shutdown is the case the reap exists for; it must not skip cleanup."""

    class _CrashOnShutdownOmni:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def shutdown(self, timeout: float | None = None) -> None:
            raise RuntimeError("shutdown hang")

    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni.AsyncOmni",
        _CrashOnShutdownOmni,
    )
    events: list[str] = []
    monkeypatch.setattr(
        "tests.helpers.runtime.cleanup_test_environment",
        lambda: events.append("env"),
    )
    monkeypatch.setattr(
        "tests.helpers.runtime.reap_leftover_engine_children",
        lambda: events.append("reap"),
    )

    with pytest.raises(RuntimeError, match="shutdown hang"):
        with AsyncOmniRunner("fake-model"):
            pass

    assert events == ["env", "reap", "env"]


def test_runner_delegates_attributes_to_engine(_tracked_cleanup: list[str]) -> None:
    with AsyncOmniRunner("fake-model") as runner:
        assert runner.magic == 7
        assert runner.engine is _FakeAsyncOmni.instances[0]
        with pytest.raises(AttributeError):
            _ = runner.no_such_attribute


def test_async_omni_params_defaults() -> None:
    params = AsyncOmniParams(model="tiny/Qwen-Image")
    assert params.model == "tiny/Qwen-Image"
    assert params.deploy_config is None
    assert params.extra_omni_kwargs is None


def _iter_request(params: AsyncOmniParams) -> SimpleNamespace:
    return SimpleNamespace(
        param=params,
        node=SimpleNamespace(get_closest_marker=lambda name: None),
    )


def test_iter_async_omni_cycle_applies_prefix_run_level_and_extras(
    monkeypatch: pytest.MonkeyPatch,
    _tracked_cleanup: list[str],
) -> None:
    """The generator wires model prefix + run-level deploy rewrite and passes extras through."""
    monkeypatch.setenv("MODEL_PREFIX", "/models")
    monkeypatch.setattr(runtime_mod, "_whisper_device_free_around", nullcontext)
    rewrites: list[tuple[str | None, str]] = []

    def _rewrite(path: str | None, level: str) -> str:
        rewrites.append((path, level))
        return f"rewritten-{level}.yaml"

    monkeypatch.setattr(
        stage_config_mod,
        "stage_config_path_for_run_level",
        _rewrite,
    )

    gen = iter_async_omni(
        _iter_request(
            AsyncOmniParams(
                model="tiny/Qwen-Image",
                deploy_config="orig.yaml",
                extra_omni_kwargs={"max_num_seqs": 4},
            )
        ),
        "core_model",
        threading.Lock(),
    )
    runner = next(gen)
    engine = runner.engine

    assert rewrites == [("orig.yaml", "core_model")]
    assert engine.kwargs["model"] == "/models/tiny/Qwen-Image"
    assert engine.kwargs["deploy_config"] == "rewritten-core_model.yaml"
    assert engine.kwargs["max_num_seqs"] == 4

    with pytest.raises(StopIteration):
        next(gen)
    assert engine.shutdown_calls == 1
    assert _tracked_cleanup == ["env", "shutdown", "reap", "env"]


def test_iter_async_omni_rejects_reserved_extra_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    _tracked_cleanup: list[str],
) -> None:
    monkeypatch.setattr(runtime_mod, "_whisper_device_free_around", nullcontext)

    gen = iter_async_omni(
        _iter_request(
            AsyncOmniParams(
                model="tiny/Qwen-Image",
                extra_omni_kwargs={"deploy_config": "rogue.yaml"},
            )
        ),
        "core_model",
        threading.Lock(),
    )

    with pytest.raises(ValueError, match="deploy_config"):
        next(gen)
    assert not _FakeAsyncOmni.instances


def test_reap_is_scoped_to_the_current_process_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """A matching process outside this process tree must be left alone.

    pytest-xdist loadgroup workers own sibling engines on a shared host;
    the reap helper must only ever see this process's own children.
    """
    import os

    from tests.helpers import clean as clean_mod

    terminated: list[int] = []

    class FakeProc:
        def __init__(self, pid: int, cmdline: list[str]) -> None:
            self.pid = pid
            self._cmdline = cmdline

        def cmdline(self) -> list[str]:
            return self._cmdline

        def name(self) -> str:
            return "python"

        def terminate(self) -> None:
            terminated.append(self.pid)

    child = FakeProc(101, ["python", "-m", "vllm-omni::engine"])
    sibling = FakeProc(202, ["python", "-m", "vllm-omni::engine"])

    class FakeRoot:
        def children(self, recursive: bool = True) -> list[FakeProc]:
            return [child]

    class FakePsutil:
        NoSuchProcess = clean_mod.psutil.NoSuchProcess
        AccessDenied = clean_mod.psutil.AccessDenied

        @staticmethod
        def Process(pid: int) -> FakeRoot:
            assert pid == os.getpid()
            return FakeRoot()

        @staticmethod
        def wait_procs(procs: list, timeout: float | None = None) -> tuple[list, list]:
            return list(procs), []

    monkeypatch.setattr(clean_mod, "psutil", FakePsutil)
    clean_mod.reap_leftover_engine_children()
    assert terminated == [101]
    assert not hasattr(sibling, "_terminated")
