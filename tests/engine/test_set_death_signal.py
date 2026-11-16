# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU unit tests for parent-death signal arming (#10398)."""

from __future__ import annotations

import signal

import pytest

from vllm_omni.engine import stage_init_utils


class _FakeLibc:
    def __init__(self, *, rc: int = 0) -> None:
        self.rc = rc
        self.calls: list[tuple[int, int]] = []

    def prctl(self, option: int, sig: int) -> int:
        self.calls.append((option, sig))
        return self.rc


def test_set_death_signal_noop_on_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "system", lambda: "Darwin")
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(stage_init_utils.os, "getppid", lambda: 42)
    monkeypatch.setattr(
        stage_init_utils.os,
        "kill",
        lambda pid, sig: killed.append((pid, sig)),
    )

    stage_init_utils.set_death_signal(signal.SIGKILL)

    assert killed == []


def test_set_death_signal_self_kills_when_already_orphaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "system", lambda: "Linux")
    monkeypatch.setattr(stage_init_utils.os, "getppid", lambda: 1)
    monkeypatch.setattr(stage_init_utils.os, "getpid", lambda: 999)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        stage_init_utils.os,
        "kill",
        lambda pid, sig: killed.append((pid, sig)),
    )

    stage_init_utils.set_death_signal(signal.SIGKILL)

    assert killed == [(999, signal.SIGKILL)]


def test_set_death_signal_self_kills_if_parent_dies_during_prctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "system", lambda: "Linux")
    fake = _FakeLibc(rc=0)
    monkeypatch.setattr(ctypes, "CDLL", lambda *_a, **_k: fake)
    monkeypatch.setattr(ctypes, "get_errno", lambda: 0)

    ppids = iter([4242, 1])  # alive before prctl, init afterward
    monkeypatch.setattr(stage_init_utils.os, "getppid", lambda: next(ppids))
    monkeypatch.setattr(stage_init_utils.os, "getpid", lambda: 777)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        stage_init_utils.os,
        "kill",
        lambda pid, sig: killed.append((pid, sig)),
    )

    stage_init_utils.set_death_signal(signal.SIGKILL)

    assert fake.calls == [(1, signal.SIGKILL)]
    assert killed == [(777, signal.SIGKILL)]


def test_set_death_signal_logs_prctl_failure_without_self_kill_if_parent_alive(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import ctypes
    import logging
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "system", lambda: "Linux")
    fake = _FakeLibc(rc=-1)
    monkeypatch.setattr(ctypes, "CDLL", lambda *_a, **_k: fake)
    monkeypatch.setattr(ctypes, "get_errno", lambda: 1)
    monkeypatch.setattr(stage_init_utils.os, "getppid", lambda: 55)
    monkeypatch.setattr(stage_init_utils.os, "getpid", lambda: 88)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        stage_init_utils.os,
        "kill",
        lambda pid, sig: killed.append((pid, sig)),
    )

    with caplog.at_level(logging.WARNING, logger=stage_init_utils.logger.name):
        stage_init_utils.set_death_signal(signal.SIGTERM)

    assert fake.calls == [(1, signal.SIGTERM)]
    assert killed == []
    assert any("PR_SET_PDEATHSIG" in rec.message for rec in caplog.records)


def test_set_death_signal_self_kills_on_prctl_failure_if_parent_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "system", lambda: "Linux")
    fake = _FakeLibc(rc=-1)
    monkeypatch.setattr(ctypes, "CDLL", lambda *_a, **_k: fake)
    monkeypatch.setattr(ctypes, "get_errno", lambda: 1)

    ppids = iter([100, 1])
    monkeypatch.setattr(stage_init_utils.os, "getppid", lambda: next(ppids))
    monkeypatch.setattr(stage_init_utils.os, "getpid", lambda: 200)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        stage_init_utils.os,
        "kill",
        lambda pid, sig: killed.append((pid, sig)),
    )

    stage_init_utils.set_death_signal(signal.SIGKILL)

    assert killed == [(200, signal.SIGKILL)]


def test_set_death_signal_success_when_parent_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes
    import platform as platform_mod

    monkeypatch.setattr(platform_mod, "system", lambda: "Linux")
    fake = _FakeLibc(rc=0)
    monkeypatch.setattr(ctypes, "CDLL", lambda *_a, **_k: fake)
    monkeypatch.setattr(stage_init_utils.os, "getppid", lambda: 321)
    monkeypatch.setattr(stage_init_utils.os, "getpid", lambda: 654)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        stage_init_utils.os,
        "kill",
        lambda pid, sig: killed.append((pid, sig)),
    )

    stage_init_utils.set_death_signal(signal.SIGKILL)

    assert fake.calls == [(1, signal.SIGKILL)]
    assert killed == []
