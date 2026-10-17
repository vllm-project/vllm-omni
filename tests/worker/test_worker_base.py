# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Characterization tests for ``vllm_omni.worker.base.OmniGPUWorkerBase``.

Pins the CURRENT behaviour of ``determine_available_memory`` (device-level
profiling is the only path — the NVML / process-scoped arm was removed with
parallel stage init, which coordinates concurrent same-device measurement via
admission + SH/EX device locks instead), plus the
memory-pool / sleep / wake-up plumbing that a later change may touch.

Workers are constructed via ``object.__new__`` with only the attributes each
method reads; module-level collaborators are monkeypatched. Pure CPU.
"""

import math
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

import vllm_omni.worker.base as base
from vllm_omni.worker.base import OmniGPUWorkerBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

GIB = 1024**3


@pytest.fixture(autouse=True)
def _fresh_once_loggers():
    """`logger.info_once` caches on (message, args) for the process, so a test
    that asserts on the guidance has to start from an empty cache."""
    import vllm.logger as vllm_logger

    for name in ("_print_info_once", "_print_warning_once", "_print_debug_once"):
        getattr(vllm_logger, name).cache_clear()
    yield
    for name in ("_print_info_once", "_print_warning_once", "_print_debug_once"):
        getattr(vllm_logger, name).cache_clear()


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
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        gpu_memory_utilization=0.8,
    )
    worker.model_runner = SimpleNamespace(
        profile_run=lambda: None,
        model_memory_usage=model_memory_usage,
    )
    worker.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE))
    # Read only by the graph-reserve guidance log; memory_profiling is faked.
    worker.init_snapshot = SimpleNamespace(total_memory=40 * GIB, free_memory=40 * GIB)
    # Not capped: the guidance branch that trades the reserve against
    # utilization only applies when the budget follows it.
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


def test_determine_available_memory_profiling_path(monkeypatch):
    """available = requested - (weights + peak + non_torch); the only path."""
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))

    out = OmniGPUWorkerBase.determine_available_memory(worker)

    profiled = 10 * GIB + 2 * GIB + 1 * GIB
    assert out == 30 * GIB - profiled


def test_determine_available_memory_populates_total_consumed(monkeypatch):
    """total_consumed mirrors upstream's MemoryProfilingResult field."""
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))

    OmniGPUWorkerBase.determine_available_memory(worker)

    assert worker.total_consumed == 3 * GIB


def test_determine_available_memory_clamps_to_zero(monkeypatch):
    """Over-subscription clamps the KV budget to 0, never negative."""
    worker = _make_worker(requested_memory=5 * GIB, model_memory_usage=8 * GIB)
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=0, torch_peak=0))

    out = OmniGPUWorkerBase.determine_available_memory(worker)

    assert out == 0


@pytest.mark.parametrize("apply_estimate", [True, False])
def test_determine_available_memory_reserves_cudagraph_memory(monkeypatch, apply_estimate):
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    calls = []
    worker.model_runner.profile_run = lambda: calls.append("model")

    def profile_graphs():
        calls.append("graphs")
        return 4 * GIB

    worker.model_runner.profile_cudagraph_memory = profile_graphs
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(is_cuda_alike=lambda: True))
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1" if apply_estimate else "0")

    @contextmanager
    def profile_memory(snapshot, weights_memory):
        calls.append("profile-enter")
        with _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB)(snapshot, weights_memory) as result:
            yield result
        calls.append("profile-exit")

    monkeypatch.setattr(base, "memory_profiling", profile_memory)

    for _ in range(2):
        out = worker.determine_available_memory()
        assert out == (13 if apply_estimate else 17) * GIB
        assert worker.cudagraph_memory_estimate == (4 if apply_estimate else 0) * GIB
        # Warmup adds the actual graph memory separately when suggesting a KV budget.
        assert worker.peak_activation_memory == 2 * GIB
        assert worker.total_consumed == 3 * GIB
    # With the reserve off there is no extra capture pass at all.
    assert calls == ["profile-enter", "model", "profile-exit", *(["graphs"] if apply_estimate else [])] * 2


@pytest.mark.parametrize(
    ("cuda_alike", "mode"),
    [(True, CUDAGraphMode.NONE), (False, CUDAGraphMode.FULL)],
)
def test_determine_available_memory_skips_unavailable_cudagraphs(monkeypatch, cuda_alike, mode):
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    worker.vllm_config.compilation_config.cudagraph_mode = mode
    worker.cudagraph_memory_estimate = 4 * GIB

    def profile_graphs():
        pytest.fail("CUDA graphs are not captured for this configuration")

    worker.model_runner.profile_cudagraph_memory = profile_graphs
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(is_cuda_alike=lambda: cuda_alike))
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))

    assert worker.determine_available_memory() == 17 * GIB
    assert worker.cudagraph_memory_estimate == 0


def test_determine_available_memory_cudagraph_estimate_can_exhaust_budget(monkeypatch):
    worker = _make_worker(requested_memory=15 * GIB, model_memory_usage=10 * GIB)
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    worker.model_runner.profile_cudagraph_memory = lambda: 4 * GIB
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(is_cuda_alike=lambda: True))
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))

    assert worker.determine_available_memory() == 0


def test_a_negative_graph_estimate_does_not_raise_the_budget(monkeypatch):
    """The upstream profiler reads device free-memory deltas, so memory freed
    elsewhere on the device during capture can make it negative."""
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    worker.model_runner.profile_cudagraph_memory = lambda: -2 * GIB
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(is_cuda_alike=lambda: True))
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))

    assert worker.determine_available_memory() == 17 * GIB
    assert worker.cudagraph_memory_estimate == 0


def test_reserving_graph_memory_tells_the_operator_how_to_restore_the_budget(monkeypatch, caplog):
    """The reserve shrinks KV on every stage that captures graphs, so the log
    has to name the utilization that restores the previous size (upstream
    prints the same guidance; the override used to drop it).

    0.8 of the 40 GiB device is exactly this budget, so it is not capped and
    the utilization trade applies.
    """
    worker = _make_worker(requested_memory=32 * GIB, model_memory_usage=10 * GIB)
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    worker.model_runner.profile_cudagraph_memory = lambda: 4 * GIB
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(is_cuda_alike=lambda: True))
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))

    with caplog.at_level("INFO", logger=base.logger.name):
        worker.determine_available_memory()

    message = "\n".join(record.getMessage() for record in caplog.records)
    # 4 GiB of a 40 GiB device is 0.1 of the utilization fraction, so 0.8 goes to 0.9.
    assert "from 0.8000 to 0.9000" in message
    assert message.count("Reserved") == 1


def _reserve_message(monkeypatch, caplog, *, utilization, free, estimate=4 * GIB, requested=None):
    total = 40 * GIB
    worker = _make_worker(requested_memory=requested or math.ceil(total * utilization), model_memory_usage=10 * GIB)
    worker.cache_config.gpu_memory_utilization = utilization
    worker.init_snapshot.free_memory = free
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    worker.model_runner.profile_cudagraph_memory = lambda: estimate
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(is_cuda_alike=lambda: True))
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))
    with caplog.at_level("INFO", logger=base.logger.name):
        worker.determine_available_memory()
    return "\n".join(record.getMessage() for record in caplog.records)


def test_a_suggestion_the_free_memory_would_cap_is_not_offered(monkeypatch, caplog):
    """The current 32 GiB budget fits in 33 GiB free, but 0.9 of 40 GiB would be
    capped to 33 GiB, i.e. 29 GiB after the reserve instead of the previous 32."""
    message = _reserve_message(monkeypatch, caplog, utilization=0.8, free=33 * GIB)
    assert "no --gpu-memory-utilization restores the previous KV size" in message
    assert "raise --gpu-memory-utilization" not in message


def test_a_suggestion_that_exactly_fits_the_free_memory_is_offered(monkeypatch, caplog):
    message = _reserve_message(monkeypatch, caplog, utilization=0.8, free=36 * GIB)
    assert "from 0.8000 to 0.9000" in message


def test_a_pool_larger_than_the_utilization_headroom_is_not_restorable(monkeypatch, caplog):
    # 0.95 + 0.1 would exceed the whole device.
    message = _reserve_message(monkeypatch, caplog, utilization=0.95, free=40 * GIB)
    assert "no --gpu-memory-utilization restores the previous KV size" in message


def test_disabling_the_reserve_skips_the_graph_profile(monkeypatch, caplog):
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    worker.model_runner.profile_cudagraph_memory = lambda: pytest.fail("the reserve is disabled")
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(is_cuda_alike=lambda: True))
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))

    with caplog.at_level("INFO", logger=base.logger.name):
        assert worker.determine_available_memory() == 17 * GIB

    assert worker.cudagraph_memory_estimate == 0
    assert "CUDA graph" not in "\n".join(record.getMessage() for record in caplog.records)


def test_no_graph_capture_says_nothing_about_the_reserve(monkeypatch, caplog):
    """A stage with graphs disabled must not be told to change utilization.

    The platform is patched CUDA-alike so `cudagraph_mode` is the gate under
    test; otherwise the first operand short-circuits on the CPU lane.
    """
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(is_cuda_alike=lambda: True))
    worker.model_runner.profile_cudagraph_memory = lambda: pytest.fail("graphs are disabled for this stage")
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))

    with caplog.at_level("INFO", logger=base.logger.name):
        worker.determine_available_memory()

    assert "CUDA graph" not in "\n".join(record.getMessage() for record in caplog.records)


def _fake_device_memory(monkeypatch, free_values):
    """Device free memory read before and after a capture."""
    frees = iter(free_values)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *args, **kwargs: (next(frees), 40 * GIB))


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.NONE])
def test_auxiliary_graphs_are_measured_released_and_reserved(monkeypatch, caplog, mode):
    """Model-owned graphs are captured once while profiling, measured, and
    released before the KV cache is sized; warmup captures them for real. The
    hook owns its graphs whatever the runner's cudagraph mode is."""
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    worker.vllm_config.compilation_config.cudagraph_mode = mode
    calls = []
    worker.model_runner.profile_run = lambda: calls.append("model")

    def profile_graphs():
        calls.append("graphs")
        return 4 * GIB

    worker.model_runner.profile_cudagraph_memory = profile_graphs
    worker.model_runner.model = SimpleNamespace(
        capture_auxiliary_graphs=lambda: calls.append("auxiliary"),
        release_auxiliary_graphs=lambda: calls.append("release"),
    )
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(is_cuda_alike=lambda: True))
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))
    _fake_device_memory(monkeypatch, [20 * GIB, 18 * GIB, 20 * GIB])

    with caplog.at_level("INFO", logger=base.logger.name):
        out = worker.determine_available_memory()

    graphs = 4 * GIB if mode == CUDAGraphMode.FULL else 0
    assert out == 30 * GIB - 13 * GIB - graphs - 2 * GIB
    assert calls == ["model", *(["graphs"] if graphs else []), "auxiliary", "release"]
    # capture_model() reports only the runner's graphs, so the estimate it is
    # compared with leaves the auxiliary pool out; the persistent total keeps it.
    assert worker.cudagraph_memory_estimate == graphs
    assert worker.total_consumed == 3 * GIB + 2 * GIB
    reserved = "\n".join(record.getMessage() for record in caplog.records)
    assert f"Reserved {base.format_gib(graphs + 2 * GIB)} GiB for CUDA graphs" in reserved
    assert f"({base.format_gib(0)} GiB still held after release)" in reserved


def test_disabling_the_reserve_leaves_auxiliary_graphs_to_warmup(monkeypatch):
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    worker.model_runner.profile_cudagraph_memory = lambda: pytest.fail("the reserve is disabled")
    worker.model_runner.model = SimpleNamespace(
        capture_auxiliary_graphs=lambda: pytest.fail("warmup captures these when the reserve is off")
    )
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(is_cuda_alike=lambda: True))
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))

    assert worker.determine_available_memory() == 17 * GIB


@pytest.mark.parametrize(
    "hooks",
    [{}, {"capture_auxiliary_graphs": lambda: pytest.fail("cannot be released, so warmup captures it only")}],
)
def test_auxiliary_graphs_that_cannot_be_released_are_not_measured(monkeypatch, hooks):
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    worker.model_runner.model = SimpleNamespace(**hooks)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *args, **kwargs: pytest.fail("nothing to measure"))

    assert worker._profile_auxiliary_graph_memory() == 0


def test_auxiliary_graphs_are_released_when_capture_fails(monkeypatch):
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    released = []

    def capture():
        raise RuntimeError("capture failed")

    worker.model_runner.model = SimpleNamespace(
        capture_auxiliary_graphs=capture, release_auxiliary_graphs=lambda: released.append(True)
    )
    _fake_device_memory(monkeypatch, [20 * GIB])

    with pytest.raises(RuntimeError, match="capture failed"):
        worker._profile_auxiliary_graph_memory()
    assert released == [True]


@pytest.mark.parametrize(
    "runner_path", ["gpu_ar_model_runner.GPUARModelRunner", "gpu_generation_model_runner.GPUGenerationModelRunner"]
)
def test_the_upstream_graph_profiler_is_callable_with_no_arguments(runner_path):
    """`determine_available_memory` calls this inherited method at startup on
    every CUDA stage, through the two runner classes the Omni workers build. A
    rename or a new required argument upstream has to fail in this CPU lane,
    not at the first GPU engine start."""
    import importlib
    import inspect

    module_name, class_name = runner_path.split(".")
    runner_cls = getattr(importlib.import_module(f"vllm_omni.worker.{module_name}"), class_name)

    signature = inspect.signature(runner_cls.profile_cudagraph_memory)
    required = [
        name
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.default is inspect.Parameter.empty
    ]
    assert required == []


def test_a_capped_budget_is_not_told_to_raise_utilization(monkeypatch, caplog):
    """`request_memory_tolerant` caps the budget to the free memory when Omni
    stages share a GPU. The reserve still applies, but no utilization restores
    it, so the log must not send the operator after one."""
    worker = _make_worker(requested_memory=30 * GIB, model_memory_usage=10 * GIB)
    # 0.8 of 40 GiB is 32 GiB; this stage only got 30 GiB.
    worker.requested_memory = 30 * GIB
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    worker.model_runner.profile_cudagraph_memory = lambda: 4 * GIB
    monkeypatch.setattr(base, "current_omni_platform", SimpleNamespace(is_cuda_alike=lambda: True))
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")
    monkeypatch.setattr(base, "memory_profiling", _fake_memory_profiling(non_torch=1 * GIB, torch_peak=2 * GIB))

    with caplog.at_level("INFO", logger=base.logger.name):
        budget = worker.determine_available_memory()

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert budget == 13 * GIB
    assert "no --gpu-memory-utilization restores the previous KV size" in message
    assert "raise --gpu-memory-utilization" not in message


def test_determine_available_memory_kv_cache_short_circuit(monkeypatch):
    """A pre-set kv_cache_memory_bytes short-circuits profiling and is returned."""
    worker = _make_worker(requested_memory=30 * GIB, kv_cache_memory_bytes=7 * GIB)
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    profile_calls = []
    worker.model_runner.profile_run = lambda: profile_calls.append(True)

    def profile_graphs():
        pytest.fail("An explicit KV budget must skip CUDA graph memory profiling")

    worker.model_runner.profile_cudagraph_memory = profile_graphs
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


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("vllm_omni.platforms.xpu.worker.xpu_ar_worker", "XPUARWorker"),
        ("vllm_omni.platforms.xpu.worker.xpu_generation_worker", "XPUGenerationWorker"),
    ],
)
def test_xpu_workers_reuse_omni_gpu_worker_base(module_name, class_name):
    module = pytest.importorskip(module_name)
    worker_cls = getattr(module, class_name)

    mro = worker_cls.__mro__
    assert mro.index(OmniGPUWorkerBase) < mro.index(module.XPUWorker)
    for method in ("_maybe_get_memory_pool_context", "sleep", "wake_up", "handle_sleep_task", "handle_wake_task"):
        assert getattr(worker_cls, method) is getattr(OmniGPUWorkerBase, method)
