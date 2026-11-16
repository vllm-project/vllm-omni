# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU unit tests for parent-death signal arming (#10398)."""

from __future__ import annotations

import ctypes
import logging
import platform
import signal

import pytest

from vllm_omni.engine import stage_init_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeLibc:
    def __init__(self) -> None:
        self.rc = 0
        self.calls: list[tuple[int, int]] = []

    def prctl(self, option: int, sig: int) -> int:
        self.calls.append((option, sig))
        return self.rc


@pytest.fixture
def death_signal_env(monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeLibc, list[tuple[int, int]]]:
    """Provide a stable Linux parent and record prctl/self-signal calls."""
    fake = _FakeLibc()
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(ctypes, "CDLL", lambda *_a, **_k: fake)
    monkeypatch.setattr(ctypes, "get_errno", lambda: 1)
    monkeypatch.setattr(stage_init_utils.os, "getppid", lambda: 321)
    monkeypatch.setattr(stage_init_utils.os, "getpid", lambda: 654)
    monkeypatch.setattr(stage_init_utils.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    return fake, killed


def test_set_death_signal_noop_on_non_linux(monkeypatch, death_signal_env) -> None:
    fake, killed = death_signal_env
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    stage_init_utils.set_death_signal(signal.SIGKILL)

    assert fake.calls == []
    assert killed == []


def test_set_death_signal_self_kills_when_already_orphaned(monkeypatch, death_signal_env) -> None:
    fake, killed = death_signal_env
    monkeypatch.setattr(stage_init_utils.os, "getppid", lambda: 1)

    stage_init_utils.set_death_signal(signal.SIGKILL)

    assert fake.calls == []
    assert killed == [(654, signal.SIGKILL)]


def test_set_death_signal_self_kills_if_parent_dies_during_prctl(monkeypatch, death_signal_env) -> None:
    fake, killed = death_signal_env
    ppids = iter([321, 1])
    monkeypatch.setattr(stage_init_utils.os, "getppid", lambda: next(ppids))

    stage_init_utils.set_death_signal(signal.SIGKILL)

    assert fake.calls == [(1, signal.SIGKILL)]
    assert killed == [(654, signal.SIGKILL)]


def test_set_death_signal_logs_prctl_failure_without_self_kill_if_parent_alive(death_signal_env, caplog) -> None:
    fake, killed = death_signal_env
    fake.rc = -1

    with caplog.at_level(logging.WARNING, logger=stage_init_utils.logger.name):
        stage_init_utils.set_death_signal(signal.SIGTERM)

    assert fake.calls == [(1, signal.SIGTERM)]
    assert killed == []
    assert any("PR_SET_PDEATHSIG" in rec.message for rec in caplog.records)


def test_set_death_signal_self_kills_on_prctl_failure_if_parent_gone(monkeypatch, death_signal_env) -> None:
    fake, killed = death_signal_env
    fake.rc = -1
    ppids = iter([321, 1])
    monkeypatch.setattr(stage_init_utils.os, "getppid", lambda: next(ppids))

    stage_init_utils.set_death_signal(signal.SIGKILL)

    assert fake.calls == [(1, signal.SIGKILL)]
    assert killed == [(654, signal.SIGKILL)]


def test_set_death_signal_success_when_parent_stable(death_signal_env) -> None:
    fake, killed = death_signal_env

    stage_init_utils.set_death_signal(signal.SIGKILL)

    assert fake.calls == [(1, signal.SIGKILL)]
    assert killed == []
