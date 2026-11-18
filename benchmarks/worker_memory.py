# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in allocator diagnostics through vLLM's worker-extension RPC surface.

Call reset/read only between completed request batches. Both synchronize the
worker's current accelerator, so these calls do not belong in a latency run.
The reported peaks cover the explicit reset interval, not all of startup.
"""

import os
import time
from typing import TypedDict

import torch


class WorkerMemorySnapshot(TypedDict):
    pid: int
    scope: str
    reset_time_ns: int
    observed_time_ns: int
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


class WorkerMemoryExtension:
    """Model-neutral diagnostic methods injected into an AR worker on opt-in."""

    _benchmark_memory_scope: str | None = None
    _benchmark_memory_reset_time_ns: int | None = None

    def benchmark_memory_reset(self, scope: str) -> WorkerMemorySnapshot:
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("A nonempty allocator measurement scope is required")
        torch.accelerator.synchronize()
        torch.accelerator.memory.reset_peak_memory_stats()
        self._benchmark_memory_scope = scope
        self._benchmark_memory_reset_time_ns = time.time_ns()
        return self._benchmark_memory_snapshot()

    def benchmark_memory_read(self) -> WorkerMemorySnapshot:
        if self._benchmark_memory_scope is None:
            raise RuntimeError("Call benchmark_memory_reset before reading allocator peaks")
        torch.accelerator.synchronize()
        return self._benchmark_memory_snapshot()

    def _benchmark_memory_snapshot(self) -> WorkerMemorySnapshot:
        scope = self._benchmark_memory_scope
        reset_time_ns = self._benchmark_memory_reset_time_ns
        if scope is None or reset_time_ns is None:
            raise RuntimeError("Allocator measurement scope has not been initialized")
        stats = torch.accelerator.memory.memory_stats()
        required = (
            "allocated_bytes.all.current",
            "reserved_bytes.all.current",
            "allocated_bytes.all.peak",
            "reserved_bytes.all.peak",
        )
        missing = sorted(set(required).difference(stats))
        if missing:
            raise RuntimeError(f"Allocator backend does not report required counters: {missing}")
        return WorkerMemorySnapshot(
            pid=os.getpid(),
            scope=scope,
            reset_time_ns=reset_time_ns,
            observed_time_ns=time.time_ns(),
            allocated_bytes=int(stats["allocated_bytes.all.current"]),
            reserved_bytes=int(stats["reserved_bytes.all.current"]),
            peak_allocated_bytes=int(stats["allocated_bytes.all.peak"]),
            peak_reserved_bytes=int(stats["reserved_bytes.all.peak"]),
        )
