# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import threading

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.code2wav_watchdog import (
    CODE2WAV_WATCHDOG_TIMEOUT_ENV,
    Code2WavWatchdog,
    Code2WavWatchdogReport,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_completed_watch_does_not_report() -> None:
    reports: list[Code2WavWatchdogReport] = []
    reported = threading.Event()

    def reporter(report: Code2WavWatchdogReport) -> None:
        reports.append(report)
        reported.set()

    watchdog = Code2WavWatchdog(0.05, reporter=reporter)
    try:
        with watchdog.watch("decode/encoder", {"request_ids": ["req-0"]}):
            pass
        assert not reported.wait(0.1)
        assert reports == []
    finally:
        watchdog.close()


def test_watchdog_reports_phase_details_and_owner_stack() -> None:
    reports: list[Code2WavWatchdogReport] = []
    reported = threading.Event()

    def reporter(report: Code2WavWatchdogReport) -> None:
        reports.append(report)
        reported.set()

    watchdog = Code2WavWatchdog(0.05, reporter=reporter)
    try:
        with watchdog.watch("decode/cfm", {"request_ids": ["req-1"], "batch_size": 1}):
            assert reported.wait(1.0)
        assert len(reports) == 1
        report = reports[0]
        assert report.phase == "decode/cfm"
        assert report.details["request_ids"] == ["req-1"]
        assert report.details["batch_size"] == 1
        assert report.elapsed_s >= report.timeout_s
        assert "test_watchdog_reports_phase_details_and_owner_stack" in report.stack
    finally:
        watchdog.close()


def test_phase_update_resets_deadline() -> None:
    reported = threading.Event()
    reports: list[Code2WavWatchdogReport] = []

    def reporter(report: Code2WavWatchdogReport) -> None:
        reports.append(report)
        reported.set()

    watchdog = Code2WavWatchdog(0.3, reporter=reporter)
    try:
        with watchdog.watch("decode/encoder"):
            assert not reported.wait(0.05)
            watchdog.phase("decode/hift", {"batch_size": 2})
            assert not reported.wait(0.1)
            assert reported.wait(0.4)
        assert len(reports) == 1
        assert reports[0].phase == "decode/hift"
        assert reports[0].details["batch_size"] == 2
    finally:
        watchdog.close()


def test_phase_update_replaces_previous_phase_details() -> None:
    reported = threading.Event()
    reports: list[Code2WavWatchdogReport] = []

    def reporter(report: Code2WavWatchdogReport) -> None:
        reports.append(report)
        reported.set()

    watchdog = Code2WavWatchdog(0.05, reporter=reporter)
    try:
        with watchdog.watch("decode/encoder", {"codec_frames": 12}):
            watchdog.phase("decode/cfm", {"batch_size": 2})
            assert reported.wait(1.0)
        assert len(reports) == 1
        assert reports[0].phase == "decode/cfm"
        assert reports[0].details == {"batch_size": 2}
    finally:
        watchdog.close()


def test_close_stops_monitor_thread() -> None:
    watchdog = Code2WavWatchdog(30.0)
    thread = watchdog._thread
    assert thread is not None and thread.is_alive()

    watchdog.close()

    assert not thread.is_alive()


def test_environment_timeout_is_opt_in_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CODE2WAV_WATCHDOG_TIMEOUT_ENV, raising=False)
    disabled = Code2WavWatchdog.from_environment()
    assert disabled.enabled is False

    monkeypatch.setenv(CODE2WAV_WATCHDOG_TIMEOUT_ENV, "1.5")
    enabled = Code2WavWatchdog.from_environment()
    try:
        assert enabled.enabled is True
        assert enabled.timeout_s == 1.5
    finally:
        enabled.close()

    monkeypatch.setenv(CODE2WAV_WATCHDOG_TIMEOUT_ENV, "not-a-number")
    with pytest.raises(ValueError, match=CODE2WAV_WATCHDOG_TIMEOUT_ENV):
        Code2WavWatchdog.from_environment()
