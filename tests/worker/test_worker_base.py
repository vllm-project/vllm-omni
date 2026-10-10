# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for ``vllm_omni.worker.base.OmniGPUWorkerBase``.

Pins the behaviour of ``determine_available_memory`` (the KV budget comes from
upstream's ``non_kv_cache_memory`` aggregate; device-level profiling is the only
path — the NVML / process-scoped arm was removed with parallel stage init, which
coordinates concurrent same-device measurement via admission + SH/EX device locks
instead), plus the memory-pool / sleep / wake-up plumbing that a later change may
touch.

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


def _fake_memory_profiling(
    *, non_torch: int, torch_peak: int, total_consumed: int | None = None, transient_headroom: int = 0
):
    """A stand-in for ``vllm.utils.mem_utils.memory_profiling`` context manager.

    ``non_kv_cache_memory`` mirrors upstream's aggregate: ``total_consumed +
    transient_peak_headroom``. ``total_consumed`` defaults to the profiling-window
    delta (``torch_peak + non_torch``) for tests where nothing predates the
    window; pass it explicitly to exercise persistent runner buffers, the case
    #7839 reports.
    """

    @contextmanager
    def _mp(snapshot, weights_memory):  # noqa: ARG001 - signature parity only
        consumed = torch_peak + non_torch if total_consumed is None else total_consumed
        yield SimpleNamespace(
            non_torch_increase=non_torch,
            torch_peak_increase=torch_peak,
            total_consumed=consumed,
            transient_peak_headroom=transient_headroom,
            non_kv_cache_memory=consumed + transient_headroom,
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
def test_no_process_scoped_collaborators_remain():
    """The NVML arm is gone from base.py, so the method CANNOT take a
    process-scoped path on any host: its collaborators are not even imported.
    (The removal itself happened in the parallel-stage-init feature commit;
    tests/engine/test_parallel_stage_init.py pins the same invariant.)"""
    assert not hasattr(base, "is_process_scoped_memory_available")
    assert not hasattr(base, "detect_pid_host")
    assert not hasattr(base, "get_process_gpu_memory")


def test_determine_available_memory_uses_non_kv_cache_memory(monkeypatch):
    """The budget comes from upstream's ``non_kv_cache_memory`` aggregate, which
    covers allocations made before the profiling window (runner buffers), not the
    legacy weights + peak + non-torch sum.

    Replays #7839's accounting example (GiB): 30 requested, 10 weights, 2 of
    persistent runner buffers allocated after the initial snapshot, another 1
    persistent + 1 non-torch during the window, transient headroom of 2.
    -> total_consumed = 14, non_kv_cache_memory = 16, budget = 14.
    The legacy sum (10+3+1=14) would leave 16, over-allocating the 2 GiB.
    """
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    monkeypatch.setattr(
        base,
        "memory_profiling",
        _fake_memory_profiling(
            non_torch=1 * GIB,
            torch_peak=3 * GIB,
            total_consumed=14 * GIB,
            transient_headroom=2 * GIB,
        ),
    )

    out = OmniGPUWorkerBase.determine_available_memory(worker)

    assert out == 14 * GIB
    assert worker.total_consumed == 14 * GIB
    assert worker.peak_activation_memory == 2 * GIB  # transient headroom, not torch_peak_increase


def test_determine_available_memory_populates_total_consumed(monkeypatch):
    """total_consumed mirrors upstream's MemoryProfilingResult field."""
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))

    OmniGPUWorkerBase.determine_available_memory(worker)

    assert worker.total_consumed == 3 * GIB


def test_determine_available_memory_clamps_to_zero(monkeypatch):
    """Over-subscription clamps the KV budget to 0, never negative."""
    worker = _make_worker(requested_memory=5 * GIB, model_memory_usage=8 * GIB)
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=3 * GIB, torch_peak=3 * GIB))

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


def test_cuda_profiler_is_wired_for_omni_worker(monkeypatch):
    import vllm.profiler.wrapper as profiler_wrapper

    def fake_gpu_worker_init(worker, *args, **kwargs):
        worker.vllm_config = SimpleNamespace(profiler_config=SimpleNamespace(profiler="cuda"))
        worker.local_rank = 0
        worker.rank = 0
        worker.profiler = None

    class FakeCudaProfiler:
        def __init__(self, profiler_config):
            self.profiler_config = profiler_config

    monkeypatch.setattr(base.GPUWorker, "__init__", fake_gpu_worker_init)
    monkeypatch.setattr(profiler_wrapper, "CudaProfilerWrapper", FakeCudaProfiler)

    worker = OmniGPUWorkerBase()

    assert isinstance(worker.profiler, FakeCudaProfiler)
    assert worker.profiler.profiler_config.profiler == "cuda"
