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
