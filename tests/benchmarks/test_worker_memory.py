# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract checks; no accelerator context or model is required."""

from dataclasses import dataclass, field

import pytest

from benchmarks import worker_memory
from benchmarks.worker_memory import WorkerMemoryExtension

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class MemoryBackend:
    stats: dict[str, int] = field(
        default_factory=lambda: {
            "allocated_bytes.all.current": 20,
            "reserved_bytes.all.current": 40,
            "allocated_bytes.all.peak": 100,
            "reserved_bytes.all.peak": 200,
        }
    )
    calls: list[str] = field(default_factory=list)

    def synchronize(self) -> None:
        self.calls.append("synchronize")

    def reset_peak_memory_stats(self) -> None:
        self.calls.append("reset")
        self.stats["allocated_bytes.all.peak"] = self.stats["allocated_bytes.all.current"]
        self.stats["reserved_bytes.all.peak"] = self.stats["reserved_bytes.all.current"]

    def memory_stats(self) -> dict[str, int]:
        self.calls.append("read")
        return self.stats.copy()


@pytest.fixture
def memory_backend(monkeypatch: pytest.MonkeyPatch) -> MemoryBackend:
    backend = MemoryBackend()
    monkeypatch.setattr(worker_memory.torch.accelerator, "synchronize", backend.synchronize)
    monkeypatch.setattr(
        worker_memory.torch.accelerator.memory, "reset_peak_memory_stats", backend.reset_peak_memory_stats
    )
    monkeypatch.setattr(worker_memory.torch.accelerator.memory, "memory_stats", backend.memory_stats)
    return backend


def test_unscoped_read_fails_without_touching_accelerator(memory_backend: MemoryBackend) -> None:
    with pytest.raises(RuntimeError, match="benchmark_memory_reset"):
        WorkerMemoryExtension().benchmark_memory_read()
    assert memory_backend.calls == []


@pytest.mark.parametrize("scope", ["", "   "])
def test_empty_scope_fails_before_reset(memory_backend: MemoryBackend, scope: str) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        WorkerMemoryExtension().benchmark_memory_reset(scope)
    assert memory_backend.calls == []


def test_reset_starts_an_explicit_peak_interval(memory_backend: MemoryBackend) -> None:
    extension = WorkerMemoryExtension()
    first = extension.benchmark_memory_reset("serving-1")
    assert memory_backend.calls == ["synchronize", "reset", "read"]
    assert first["scope"] == "serving-1"
    assert first["pid"] > 0
    assert first["observed_time_ns"] >= first["reset_time_ns"]
    assert first["peak_allocated_bytes"] == 20
    assert first["peak_reserved_bytes"] == 40

    memory_backend.stats["allocated_bytes.all.peak"] = 70
    memory_backend.stats["reserved_bytes.all.peak"] = 120
    memory_backend.stats["allocated_bytes.all.current"] = 25
    current = extension.benchmark_memory_read()
    assert memory_backend.calls[-2:] == ["synchronize", "read"]
    assert current["reset_time_ns"] == first["reset_time_ns"]
    assert current["allocated_bytes"] == 25
    assert current["peak_allocated_bytes"] == 70
    assert current["peak_reserved_bytes"] == 120

    restarted = extension.benchmark_memory_reset("serving-2")
    assert restarted["scope"] == "serving-2"
    assert restarted["peak_allocated_bytes"] == 25
    assert restarted["peak_reserved_bytes"] == 40


def test_missing_counters_fail_instead_of_reporting_zero(memory_backend: MemoryBackend) -> None:
    extension = WorkerMemoryExtension()
    extension.benchmark_memory_reset("supported-backend")
    del memory_backend.stats["reserved_bytes.all.peak"]
    with pytest.raises(RuntimeError, match="reserved_bytes.all.peak"):
        extension.benchmark_memory_read()


def test_scopes_are_owned_by_the_worker_instance(memory_backend: MemoryBackend) -> None:
    first = WorkerMemoryExtension()
    first.benchmark_memory_reset("worker-1")
    second = WorkerMemoryExtension()
    with pytest.raises(RuntimeError, match="benchmark_memory_reset"):
        second.benchmark_memory_read()
