# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DiffusionEngine._rpc_lock contention counters.

Verifies that get_rpc_lock_stats() reflects:

1. count + total + mean grow on every collective_rpc invocation
2. timeout-path acquisitions are also counted (so the contention
   surface is observable when callers give up)
3. max_wait_ms tracks the worst-case acquisition
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _load_engine_module():
    """Load diffusion_engine.py with mocked heavy dependencies."""
    engine_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "vllm_omni",
            "diffusion",
            "diffusion_engine.py",
        )
    )

    mocks = {
        "vllm_omni": MagicMock(),
        "vllm_omni.diffusion": types.ModuleType("vllm_omni.diffusion"),
        "vllm_omni.diffusion.data": MagicMock(),
        "vllm_omni.diffusion.executor": MagicMock(),
        "vllm_omni.diffusion.executor.abstract": MagicMock(),
        "vllm_omni.diffusion.registry": MagicMock(),
        "vllm_omni.diffusion.request": MagicMock(),
        "vllm_omni.diffusion.sched": MagicMock(),
        "vllm_omni.diffusion.sched.interface": MagicMock(),
        "vllm_omni.diffusion.worker": MagicMock(),
        "vllm_omni.diffusion.worker.utils": MagicMock(),
        "vllm_omni.inputs": MagicMock(),
        "vllm_omni.inputs.data": MagicMock(),
        "vllm_omni.outputs": MagicMock(),
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.engine": types.ModuleType("vllm.v1.engine"),
        "vllm.v1.engine.exceptions": MagicMock(),
        "vllm.logger": MagicMock(init_logger=lambda name: MagicMock()),
        "PIL": MagicMock(),
        "PIL.Image": MagicMock(),
    }
    with patch.dict(sys.modules, mocks):
        spec = importlib.util.spec_from_file_location("diffusion_engine", engine_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


DiffusionEngine = _load_engine_module().DiffusionEngine

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_engine() -> DiffusionEngine:
    """Build a DiffusionEngine without going through the heavy __init__."""
    engine = object.__new__(DiffusionEngine)
    engine._rpc_lock = threading.RLock()
    engine._rpc_lock_stats_lock = threading.Lock()
    engine._rpc_lock_wait_ms_total = 0.0
    engine._rpc_lock_wait_count = 0
    engine._rpc_lock_wait_max_ms = 0.0
    engine.executor = SimpleNamespace(collective_rpc=lambda **kw: "ok")
    return engine


class TestRpcLockStats:
    def test_initial_stats_are_zero(self) -> None:
        engine = _make_engine()
        stats = engine.get_rpc_lock_stats()
        assert stats == {"count": 0, "total_wait_ms": 0.0, "mean_wait_ms": 0.0, "max_wait_ms": 0.0}

    def test_successful_calls_increment_counters(self) -> None:
        engine = _make_engine()
        for _ in range(5):
            engine.collective_rpc(method="dummy")
        stats = engine.get_rpc_lock_stats()
        assert stats["count"] == 5
        assert stats["total_wait_ms"] >= 0.0
        assert stats["mean_wait_ms"] == stats["total_wait_ms"] / 5
        assert stats["max_wait_ms"] >= 0.0

    def test_timeout_path_is_counted(self) -> None:
        """If a caller times out waiting for the lock, the wait must
        still appear in stats so contention is observable even when
        callers give up."""
        engine = _make_engine()
        held = threading.Event()
        release = threading.Event()

        def hold_lock():
            with engine._rpc_lock:
                held.set()
                release.wait(timeout=2.0)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        held.wait(timeout=2.0)

        try:
            engine.collective_rpc(method="dummy", timeout=0.05)
        except TimeoutError:
            pass
        else:
            raise AssertionError("collective_rpc should have timed out")
        finally:
            release.set()
            holder.join(timeout=2.0)

        stats = engine.get_rpc_lock_stats()
        assert stats["count"] == 1, "Timed-out acquisition must still be counted"
        # 50ms requested timeout; allow generous slack for CI scheduling
        # jitter (GIL hand-off + RLock.acquire(timeout=) granularity can
        # shave ~10-15ms off the observed wait on busy runners).
        assert stats["total_wait_ms"] >= 25.0, (
            f"Expected wait near the 50ms timeout, got {stats['total_wait_ms']:.2f} ms"
        )
        assert stats["max_wait_ms"] == stats["total_wait_ms"]

    def test_max_tracks_worst_case(self) -> None:
        engine = _make_engine()
        # Cheap call first, then a slower one (artificial wait via lock holder).
        engine.collective_rpc(method="dummy")
        cheap_max = engine.get_rpc_lock_stats()["max_wait_ms"]

        held = threading.Event()
        release = threading.Event()

        def hold_lock():
            with engine._rpc_lock:
                held.set()
                release.wait(timeout=2.0)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        held.wait(timeout=2.0)

        def slow_caller():
            engine.collective_rpc(method="dummy")

        t = threading.Thread(target=slow_caller)
        t.start()
        time.sleep(0.08)
        release.set()
        holder.join(timeout=2.0)
        t.join(timeout=2.0)

        stats = engine.get_rpc_lock_stats()
        assert stats["count"] == 2
        assert stats["max_wait_ms"] >= cheap_max, "max_wait_ms must be monotonic non-decreasing"
        # Slow caller waited ~80ms behind the holder before lock release;
        # 30ms floor leaves room for CI scheduling jitter while still
        # asserting we measured the contention rather than a near-zero
        # acquisition.
        assert stats["max_wait_ms"] >= 30.0, (
            f"Expected slow call to register a max ~80ms, got {stats['max_wait_ms']:.2f} ms"
        )

    def test_concurrent_increments_are_consistent(self) -> None:
        """count must equal the number of completed collective_rpc calls
        across threads (no lost increments under contention)."""
        engine = _make_engine()
        n_threads = 8
        per_thread = 25

        def caller():
            for _ in range(per_thread):
                engine.collective_rpc(method="dummy")

        threads = [threading.Thread(target=caller) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        stats = engine.get_rpc_lock_stats()
        assert stats["count"] == n_threads * per_thread
