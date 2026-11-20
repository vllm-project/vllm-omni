# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import logging

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.code2wav_stats import (
    CODE2WAV_STATS_ENV,
    Code2WavSolveStats,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_stats_are_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CODE2WAV_STATS_ENV, raising=False)
    stats = Code2WavSolveStats()

    assert stats.enabled is False
    stats.record("cfm", batch=4, frames=32)
    assert stats.summary()["solves"] == 0
    assert stats.summary()["frames"] == {}


def test_stats_count_solve_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CODE2WAV_STATS_ENV, "1")
    stats = Code2WavSolveStats()

    assert stats.enabled is True
    for _ in range(3):
        stats.record("cfm", batch=16, frames=32)
    stats.record("cfm", batch=16, frames=4, ragged=True)
    stats.record("hift", batch=16, frames=128)

    summary = stats.summary()
    assert summary["solves"] == 5
    assert summary["frames"][("cfm", 32)] == 3
    assert summary["frames"][("cfm", 4)] == 1
    assert summary["frames"][("hift", 128)] == 1
    assert summary["batches"][("cfm", 16)] == 4


def test_duty_uses_solve_durations_not_inter_record_gaps(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Solve durations, not gaps, make up the busy time.

    Review follow-up (PR #9733): accumulating now-last_ts telescoped every
    solve into the next one, so busy always equalled wall and duty read a
    flat 100% however idle stage 2 was. The call sites now time the solve
    itself (start/stop around the CFM/HiFT call) and hand the duration in.
    """
    monkeypatch.setenv(CODE2WAV_STATS_ENV, "1")
    clock = iter([0.0, 1.0, 1.1, 2.0])
    stats = Code2WavSolveStats(flush_every=3, clock=lambda: next(clock))

    with caplog.at_level(logging.INFO):
        stats.record("cfm", batch=4, frames=32, duration=0.2)  # busy 0.0-0.2, idle to 1.0
        stats.record("cfm", batch=4, frames=32, duration=0.05)  # busy 1.0-1.05
        stats.record("cfm", batch=4, frames=32, duration=0.05)  # busy 1.1-1.15, flush

    # wall runs from the first solve's start (-0.2 s) to the last record
    # (1.1 s) = 1.3 s, busy = 0.3 s -> duty 23%, not the flat 100% the gap
    # accumulation reported for the same timeline.
    assert "busy=0.3s" in caplog.text
    assert "duty=23%" in caplog.text


def test_duty_stays_bounded_for_back_to_back_solves(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A saturated window reports at most 100% duty.

    Review follow-up (PR #9733): record() runs after the solve, so anchoring
    the window at the first record's timestamp put the first solve's duration
    into busy against a wall clock that started after it -- back-to-back
    solves reported duty > 100%. The window now starts at the first solve's
    start.
    """
    monkeypatch.setenv(CODE2WAV_STATS_ENV, "1")
    clock = iter([0.1, 0.2, 0.3])
    stats = Code2WavSolveStats(flush_every=3, clock=lambda: next(clock))

    with caplog.at_level(logging.INFO):
        # Three 0.1 s solves back to back; record runs at their ends.
        for _ in range(3):
            stats.record("hift", batch=1, frames=8, duration=0.1)

    # busy = 0.3 s over wall = 0.3 s -> duty 100%, not the 150% the
    # end-anchored window reported.
    assert "busy=0.3s" in caplog.text
    assert "duty=100%" in caplog.text


def test_flush_reports_window_and_starts_a_new_one(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(CODE2WAV_STATS_ENV, "1")
    stats = Code2WavSolveStats(flush_every=4)

    with caplog.at_level(logging.INFO):
        for _ in range(4):
            stats.record("cfm", batch=8, frames=32)

    assert "duty=" in caplog.text
    assert "cfm(n=4 frames={32:4}" in caplog.text
    window = stats.summary()
    assert window["solves"] == 4
    assert window["pending"] == 0
    assert window["frames"] == {}

    # The next window counts from zero again but keeps the running total.
    with caplog.at_level(logging.INFO):
        for _ in range(4):
            stats.record("cfm", batch=8, frames=4)

    assert stats.summary()["solves"] == 8
    assert "frames={4:4}" in caplog.text
