# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.utils import startup_timing


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, tuple[object, ...]]] = []

    def info(self, message: str, *args: object) -> None:
        self.records.append((message, args))


@pytest.mark.core_model
@pytest.mark.cpu
def test_startup_span_logs_success_with_sorted_labels(monkeypatch):
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(startup_timing.time, "perf_counter", lambda: next(ticks))
    logger = _RecordingLogger()

    with startup_timing.startup_span(logger, "worker.device_init", rank=1, backend="cuda"):
        pass

    assert logger.records == [
        (
            "[StartupTiming] phase=%s duration_s=%.6f status=%s backend=%s rank=%s",
            ("worker.device_init", 2.5, "ok", "cuda", 1),
        )
    ]


@pytest.mark.core_model
@pytest.mark.cpu
def test_startup_span_logs_failure_without_swallowing_exception(monkeypatch):
    ticks = iter([3.0, 3.25])
    monkeypatch.setattr(startup_timing.time, "perf_counter", lambda: next(ticks))
    logger = _RecordingLogger()

    with pytest.raises(RuntimeError, match="failed"):
        with startup_timing.startup_span(logger, "model.construct"):
            raise RuntimeError("failed")

    assert logger.records == [
        (
            "[StartupTiming] phase=%s duration_s=%.6f status=%s",
            ("model.construct", 0.25, "error"),
        )
    ]


@pytest.mark.core_model
@pytest.mark.cpu
def test_log_process_checkpoint_uses_process_age(monkeypatch):
    monkeypatch.setattr(startup_timing, "process_age_seconds", lambda: 4.75)
    logger = _RecordingLogger()

    startup_timing.log_process_checkpoint(logger, "worker.process_to_init", rank=0)

    assert logger.records == [
        (
            "[StartupTiming] phase=%s duration_s=%.6f status=%s checkpoint=%s rank=%s",
            ("worker.process_to_init", 4.75, "ok", True, 0),
        )
    ]


@pytest.mark.core_model
@pytest.mark.cpu
def test_log_process_checkpoint_skips_unsupported_platform(monkeypatch):
    monkeypatch.setattr(startup_timing, "process_age_seconds", lambda: None)
    logger = _RecordingLogger()

    startup_timing.log_process_checkpoint(logger, "worker.process_to_init")

    assert logger.records == []
