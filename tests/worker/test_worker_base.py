# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Characterization tests for ``vllm_omni.worker.base.OmniGPUWorkerBase``.

Pins the CURRENT behaviour of ``determine_available_memory`` (BOTH the NVML /
process-scoped arm AND the profiling-fallback arm — this base still carries the
NVML branch), plus the
memory-pool / sleep / wake-up plumbing that a later change may touch.

Workers are constructed via ``object.__new__`` with only the attributes each
method reads; module-level collaborators are monkeypatched. Pure CPU.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.worker.base as base
from vllm_omni.worker.base import OmniGPUWorkerBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

GIB = 1024**3


def _fake_memory_profiling(*, non_torch: int, torch_peak: int):
    """A stand-in for ``vllm.utils.mem_utils.memory_profiling`` context manager."""

    @contextmanager
    def _mp(snapshot, weights_memory):  # noqa: ARG001 - signature parity only
        # total_consumed mirrors upstream vllm.utils.mem_utils.MemoryProfilingResult
        # (added by upstream 58b2012aa2) and is read by OmniGPUWorkerBase.
        yield SimpleNamespace(
            non_torch_increase=non_torch,
            torch_peak_increase=torch_peak,
            total_consumed=torch_peak + non_torch,
        )

    return _mp


def _make_worker(*, requested_memory: int, kv_cache_memory_bytes: int = 0, model_memory_usage: int = 0):
    worker = object.__new__(OmniGPUWorkerBase)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=kv_cache_memory_bytes)
    worker.model_runner = SimpleNamespace(
        profile_run=lambda: None,
        model_memory_usage=model_memory_usage,
    )
    worker.init_snapshot = object()  # unused once memory_profiling is faked
    worker.requested_memory = requested_memory
    worker.local_rank = 0
    return worker


# --------------------------------------------------------------------------- #
# determine_available_memory                                                  #
# --------------------------------------------------------------------------- #
def test_determine_available_memory_nvml_arm(monkeypatch):
    """NVML available -> available = requested - per-process memory."""
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))
    monkeypatch.setattr(base, "is_process_scoped_memory_available", lambda: True)
    monkeypatch.setattr(base, "detect_pid_host", lambda: True)
    monkeypatch.setattr(base, "get_process_gpu_memory", lambda local_rank: 8 * GIB)

    out = OmniGPUWorkerBase.determine_available_memory(worker)

    # NVML arm ignores profiled usage; uses process_memory directly.
    assert out == 30 * GIB - 8 * GIB


def test_determine_available_memory_profiling_fallback_arm(monkeypatch):
    """NVML unavailable -> available = requested - (weights + peak + non_torch)."""
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))
    monkeypatch.setattr(base, "is_process_scoped_memory_available", lambda: False)
    # detect_pid_host is short-circuited by the `and`, but pin it anyway.
    monkeypatch.setattr(base, "detect_pid_host", lambda: True)

    out = OmniGPUWorkerBase.determine_available_memory(worker)

    profiled = 10 * GIB + 2 * GIB + 1 * GIB
    assert out == 30 * GIB - profiled


def test_determine_available_memory_falls_back_when_not_pid_host(monkeypatch):
    """detect_pid_host False also routes to the profiling fallback."""
    worker = _make_worker(requested_memory=20 * GIB, model_memory_usage=5 * GIB)
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=0, torch_peak=0))
    monkeypatch.setattr(base, "is_process_scoped_memory_available", lambda: True)
    monkeypatch.setattr(base, "detect_pid_host", lambda: False)

    out = OmniGPUWorkerBase.determine_available_memory(worker)

    assert out == 20 * GIB - 5 * GIB


def test_determine_available_memory_clamps_to_zero(monkeypatch):
    """Over-subscription clamps the KV budget to 0, never negative."""
    worker = _make_worker(requested_memory=5 * GIB, model_memory_usage=0)
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=0, torch_peak=0))
    monkeypatch.setattr(base, "is_process_scoped_memory_available", lambda: True)
    monkeypatch.setattr(base, "detect_pid_host", lambda: True)
    monkeypatch.setattr(base, "get_process_gpu_memory", lambda local_rank: 8 * GIB)

    out = OmniGPUWorkerBase.determine_available_memory(worker)

    assert out == 0


def test_determine_available_memory_kv_cache_short_circuit(monkeypatch):
    """A pre-set kv_cache_memory_bytes short-circuits profiling and is returned."""
    worker = _make_worker(requested_memory=30 * GIB, kv_cache_memory_bytes=7 * GIB)
    profile_calls = []
    worker.model_runner.profile_run = lambda: profile_calls.append(True)
    # is_rocm gates only an extra synchronize on the short-circuit path.
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(is_rocm=lambda: False))

    out = OmniGPUWorkerBase.determine_available_memory(worker)

    assert out == 7 * GIB
    assert profile_calls == [True]  # profile_run still runs before returning


def test_determine_available_memory_kv_short_circuit_syncs_on_rocm(monkeypatch):
    """On ROCm the short-circuit adds an accelerator synchronize (a later change may remove it)."""
    worker = _make_worker(requested_memory=30 * GIB, kv_cache_memory_bytes=7 * GIB)
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(is_rocm=lambda: True))
    synced = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: synced.append(True))

    out = OmniGPUWorkerBase.determine_available_memory(worker)

    assert out == 7 * GIB
    assert synced == [True]  # ROCm path synchronizes; non-ROCm path (above) does not


# --------------------------------------------------------------------------- #
# _maybe_get_memory_pool_context                                              #
# --------------------------------------------------------------------------- #
def test_memory_pool_context_disabled_returns_nullcontext():
    worker = object.__new__(OmniGPUWorkerBase)
    worker.rank = 0
    worker.cache_config = SimpleNamespace(enable_sleep_mode=False)
    # No vllm_config attribute -> v1_config_enabled stays False.

    ctx = OmniGPUWorkerBase._maybe_get_memory_pool_context(worker, "weights")

    assert isinstance(ctx, type(nullcontext()))


def test_memory_pool_context_enabled_uses_cumem_pool_with_tag(monkeypatch):
    import vllm.device_allocator.cumem as cumem_mod

    worker = object.__new__(OmniGPUWorkerBase)
    worker.rank = 0
    worker.cache_config = SimpleNamespace(enable_sleep_mode=True)
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(synchronize=lambda: None))

    recorded = {}

    class FakeAllocator:
        @classmethod
        def get_instance(cls):
            return cls()

        def use_memory_pool(self, tag):
            recorded["tag"] = tag
            return "POOL_CTX"

    monkeypatch.setattr(cumem_mod, "CuMemAllocator", FakeAllocator)

    ctx = OmniGPUWorkerBase._maybe_get_memory_pool_context(worker, "weights")

    assert ctx == "POOL_CTX"
    assert recorded["tag"] == "weights"


# --------------------------------------------------------------------------- #
# sleep / wake_up                                                             #
# --------------------------------------------------------------------------- #
def _fake_platform():
    return SimpleNamespace(
        get_current_memory_usage=lambda device: 0,
        empty_cache=lambda: None,
        synchronize=lambda: None,
    )


@pytest.mark.parametrize(
    ("level", "expected_tags"),
    [(1, ("weights",)), (2, tuple())],
)
def test_sleep_offload_tags_by_level(monkeypatch, level, expected_tags):
    import vllm.device_allocator.cumem as cumem_mod

    worker = object.__new__(OmniGPUWorkerBase)
    worker.rank = 0
    worker.device = "cuda:0"
    order: list[str] = []
    platform = SimpleNamespace(
        get_current_memory_usage=lambda device: 0,
        empty_cache=lambda: order.append("empty_cache"),
        synchronize=lambda: order.append("sync"),
        collect=lambda: None,
    )
    monkeypatch.setattr(base, "current_omni_platform", platform)

    calls = {}

    class FakeAllocator:
        @classmethod
        def get_instance(cls):
            return cls()

        def sleep(self, offload_tags):
            order.append("allocator_sleep")
            calls["offload_tags"] = offload_tags

    monkeypatch.setattr(cumem_mod, "CuMemAllocator", FakeAllocator)

    assert OmniGPUWorkerBase.sleep(worker, level=level) is True
    assert calls["offload_tags"] == expected_tags
    assert order[0] == "sync"
    assert "allocator_sleep" in order
    assert order.index("sync") < order.index("allocator_sleep")
    assert "empty_cache" not in order


def test_wake_up_forwards_tags_to_allocator(monkeypatch):
    import vllm.device_allocator.cumem as cumem_mod

    worker = object.__new__(OmniGPUWorkerBase)
    worker.rank = 0
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(synchronize=lambda: None))

    calls = {}

    class FakeAllocator:
        @classmethod
        def get_instance(cls):
            return cls()

        def wake_up(self, tags):
            calls["tags"] = tags

    monkeypatch.setattr(cumem_mod, "CuMemAllocator", FakeAllocator)

    assert OmniGPUWorkerBase.wake_up(worker, tags=["weights"]) is True
    assert calls["tags"] == ["weights"]
