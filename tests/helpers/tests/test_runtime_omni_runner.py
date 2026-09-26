# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for OmniRunner startup rollback and teardown."""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers import runtime as runtime_mod
from tests.helpers.runtime import OmniRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def lifecycle(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[list[int]]]:
    """Record the runner's cleanup calls in order; ``snapshots`` feeds the PID snapshot."""
    events: list[Any] = []
    snapshots: list[list[int]] = []

    def _snapshot() -> list[int]:
        events.append("snapshot")
        return snapshots.pop(0) if snapshots else []

    def _reap(pids, **kwargs) -> list[int]:
        events.append(("reap", list(pids)))
        return list(pids)

    monkeypatch.setattr(runtime_mod, "cleanup_test_environment", lambda: events.append("env"))
    monkeypatch.setattr(runtime_mod, "snapshot_engine_worker_pids", _snapshot)
    monkeypatch.setattr(runtime_mod, "reap_engine_worker_pids", _reap)
    return events, snapshots


def test_runner_init_rolls_back_on_omni_startup_failure(
    monkeypatch: pytest.MonkeyPatch, lifecycle: tuple[list[Any], list[list[int]]]
) -> None:
    """``__exit__`` is skipped when construction raises; rollback must still run."""
    events, snapshots = lifecycle
    snapshots.append([11])

    class _BoomOmni:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Orchestrator initialization failed")

    monkeypatch.setattr("vllm_omni.entrypoints.omni.Omni", _BoomOmni)

    with pytest.raises(RuntimeError, match="Orchestrator initialization failed"):
        with OmniRunner("fake-model"):
            raise AssertionError("context body must not run after a failed constructor")

    # Once at the start of ``__init__``, once from the constructor rollback, which
    # also reaps the workers the failed constructor left behind.
    assert events == ["env", "snapshot", ("reap", [11]), "env"]


def test_runner_exit_closes_reaps_and_cleans(
    monkeypatch: pytest.MonkeyPatch, lifecycle: tuple[list[Any], list[list[int]]]
) -> None:
    events, snapshots = lifecycle
    snapshots.extend([[21], [21, 22]])

    class _FakeOmni:
        def __init__(self, **kwargs) -> None:
            events.append("construct")

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr("vllm_omni.entrypoints.omni.Omni", _FakeOmni)

    with OmniRunner("fake-model") as runner:
        events.append("body")
        assert isinstance(runner.omni, _FakeOmni)

    # Workers are snapshotted after construction and again before ``close()`` so
    # daemon workers reparented in between are still reaped.
    assert events == ["env", "construct", "snapshot", "body", "snapshot", "close", ("reap", [21, 22]), "env"]


def test_runner_exit_reaps_and_cleans_when_close_raises(
    monkeypatch: pytest.MonkeyPatch, lifecycle: tuple[list[Any], list[list[int]]]
) -> None:
    events, snapshots = lifecycle
    snapshots.extend([[31], []])

    class _FakeOmni:
        def __init__(self, **kwargs) -> None:
            pass

        def close(self) -> None:
            events.append("close")
            raise RuntimeError("close boom")

    monkeypatch.setattr("vllm_omni.entrypoints.omni.Omni", _FakeOmni)

    with pytest.raises(RuntimeError, match="close boom"):
        with OmniRunner("fake-model"):
            pass

    assert events[-3:] == ["close", ("reap", [31]), "env"]
