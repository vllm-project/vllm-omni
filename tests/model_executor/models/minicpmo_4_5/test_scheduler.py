# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

import vllm_omni  # noqa: F401 - apply vLLM request patches before scheduler import
from vllm_omni.model_executor.models.minicpmo_4_5.scheduler import (
    MiniCPMO45Code2WavScheduler,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _AdapterStub:
    def __init__(self, *, ready: int, pending: int) -> None:
        self._finished_load_reqs = {f"ready-{index}" for index in range(ready)}
        self._pending_load_reqs = deque(f"pending-{index}" for index in range(pending))


class _FakeClock:
    def __init__(
        self,
        adapter: _AdapterStub,
        *,
        release_after_s: float | None = None,
        additional_ready: int = 0,
    ) -> None:
        self.adapter = adapter
        self.release_after_s = release_after_s
        self.additional_ready = additional_ready
        self.now = 10.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.now += duration
        if self.release_after_s is not None and sum(self.sleeps) >= self.release_after_s:
            while self.adapter._pending_load_reqs:
                request_id = self.adapter._pending_load_reqs.popleft()
                self.adapter._finished_load_reqs.add(request_id)
            for index in range(self.additional_ready):
                self.adapter._finished_load_reqs.add(f"released-{index}")
            self.release_after_s = None


def _make_scheduler(*, wait_ms: float, ready: int, pending: int) -> MiniCPMO45Code2WavScheduler:
    scheduler = MiniCPMO45Code2WavScheduler.__new__(MiniCPMO45Code2WavScheduler)
    scheduler._code2wav_batch_wait_s = wait_ms / 1000.0
    scheduler.chunk_transfer_adapter = _AdapterStub(ready=ready, pending=pending)
    scheduler.max_num_running_reqs = 4
    scheduler.requests = {f"request-{index}": object() for index in range(min(4, ready + pending))}
    return scheduler


def test_disabled_wait_returns_without_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(wait_ms=0.0, ready=1, pending=3)
    clock = _FakeClock(scheduler.chunk_transfer_adapter)
    monkeypatch.setattr("vllm_omni.model_executor.models.minicpmo_4_5.scheduler.time", clock)

    scheduler._wait_for_ready_chunk_batch()

    assert clock.sleeps == []


def test_single_request_returns_without_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(wait_ms=2.0, ready=1, pending=0)
    clock = _FakeClock(scheduler.chunk_transfer_adapter)
    monkeypatch.setattr("vllm_omni.model_executor.models.minicpmo_4_5.scheduler.time", clock)

    scheduler._wait_for_ready_chunk_batch()

    assert clock.sleeps == []


def test_wait_exits_when_pending_cohort_becomes_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(wait_ms=2.0, ready=1, pending=3)
    clock = _FakeClock(scheduler.chunk_transfer_adapter, release_after_s=0.0006)
    monkeypatch.setattr("vllm_omni.model_executor.models.minicpmo_4_5.scheduler.time", clock)

    scheduler._wait_for_ready_chunk_batch()

    assert len(scheduler.chunk_transfer_adapter._finished_load_reqs) == 4
    assert 0.0006 <= sum(clock.sleeps) < 0.002


def test_wait_uses_live_scheduler_cohort_when_recv_queue_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _make_scheduler(wait_ms=2.0, ready=1, pending=0)
    scheduler.requests = {f"request-{index}": object() for index in range(4)}
    clock = _FakeClock(
        scheduler.chunk_transfer_adapter,
        release_after_s=0.0006,
        additional_ready=3,
    )
    monkeypatch.setattr("vllm_omni.model_executor.models.minicpmo_4_5.scheduler.time", clock)

    scheduler._wait_for_ready_chunk_batch()

    assert len(scheduler.chunk_transfer_adapter._finished_load_reqs) == 4
    assert 0.0006 <= sum(clock.sleeps) < 0.002


def test_wait_falls_back_at_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler(wait_ms=1.0, ready=1, pending=3)
    clock = _FakeClock(scheduler.chunk_transfer_adapter)
    monkeypatch.setattr("vllm_omni.model_executor.models.minicpmo_4_5.scheduler.time", clock)

    scheduler._wait_for_ready_chunk_batch()

    assert len(scheduler.chunk_transfer_adapter._finished_load_reqs) == 1
    assert sum(clock.sleeps) == pytest.approx(0.001)


def test_wait_config_rejects_negative_values() -> None:
    model_config = SimpleNamespace(
        stage_connector_config={
            "extra": {
                "code2wav_batch_wait_ms": -0.1,
            }
        }
    )

    with pytest.raises(ValueError, match="code2wav_batch_wait_ms must be non-negative"):
        MiniCPMO45Code2WavScheduler._batch_wait_seconds(model_config)


def test_wait_config_reads_connector_extra() -> None:
    model_config = SimpleNamespace(
        stage_connector_config={
            "extra": {
                "code2wav_batch_wait_ms": 1.5,
            }
        }
    )

    assert MiniCPMO45Code2WavScheduler._batch_wait_seconds(model_config) == pytest.approx(0.0015)
