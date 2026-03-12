# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import pytest


# Find repo root by looking for pyproject.toml marker
def _find_repo_root(start: Path) -> Path:
    """Walk up from start to find repo root (contains pyproject.toml)."""
    current = start.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(f"Could not find repo root from {start}")


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())


def _ensure_module(name: str) -> ModuleType:
    """Get or create a module stub without polluting existing modules."""
    if name in sys.modules:
        return sys.modules[name]
    module = ModuleType(name)
    sys.modules[name] = module
    return module


def _stub_scheduler_dependencies() -> None:
    """Create minimal stubs for vllm/vllm_omni dependencies needed by the scheduler."""
    _ensure_module("vllm")
    _ensure_module("vllm.compilation")
    cuda_graph = _ensure_module("vllm.compilation.cuda_graph")
    if not hasattr(cuda_graph, "CUDAGraphStat"):
        cuda_graph.CUDAGraphStat = type("CUDAGraphStat", (), {})

    _ensure_module("vllm.distributed")
    kv_events = _ensure_module("vllm.distributed.kv_events")
    if not hasattr(kv_events, "KVEventBatch"):
        kv_events.KVEventBatch = type("KVEventBatch", (), {})
    _ensure_module("vllm.distributed.kv_transfer")
    _ensure_module("vllm.distributed.kv_transfer.kv_connector")
    _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1")
    kv_metrics = _ensure_module("vllm.distributed.kv_transfer.kv_connector.v1.metrics")
    if not hasattr(kv_metrics, "KVConnectorStats"):
        kv_metrics.KVConnectorStats = type("KVConnectorStats", (), {})

    logger_mod = _ensure_module("vllm.logger")
    if not hasattr(logger_mod, "init_logger"):
        logger_mod.init_logger = logging.getLogger

    _ensure_module("vllm.v1")
    _ensure_module("vllm.v1.core")
    kv_cache = _ensure_module("vllm.v1.core.kv_cache_manager")
    if not hasattr(kv_cache, "KVCacheBlocks"):
        kv_cache.KVCacheBlocks = type("KVCacheBlocks", (), {})

    sched_interface = _ensure_module("vllm.v1.core.sched.interface")
    if not hasattr(sched_interface, "PauseState"):
        sched_interface.PauseState = type("PauseState", (), {"PAUSED_ALL": object(), "UNPAUSED": object()})

    sched_output = _ensure_module("vllm.v1.core.sched.output")
    if not hasattr(sched_output, "SchedulerOutput"):
        sched_output.SchedulerOutput = type("SchedulerOutput", (), {})

    request_queue = _ensure_module("vllm.v1.core.sched.request_queue")
    if not hasattr(request_queue, "create_request_queue"):
        request_queue.create_request_queue = lambda policy: []

    scheduler_mod = _ensure_module("vllm.v1.core.sched.scheduler")
    if not hasattr(scheduler_mod, "Scheduler"):
        scheduler_mod.Scheduler = type("Scheduler", (), {})

    sched_utils = _ensure_module("vllm.v1.core.sched.utils")
    if not hasattr(sched_utils, "remove_all"):
        sched_utils.remove_all = lambda seq, items: [x for x in seq if x not in items]

    engine_mod = _ensure_module("vllm.v1.engine")
    if not hasattr(engine_mod, "EngineCoreEventType"):
        engine_mod.EngineCoreEventType = type("EngineCoreEventType", (), {"SCHEDULED": object()})
    if not hasattr(engine_mod, "EngineCoreOutput"):
        engine_mod.EngineCoreOutput = type("EngineCoreOutput", (), {})
    if not hasattr(engine_mod, "EngineCoreOutputs"):
        engine_mod.EngineCoreOutputs = type("EngineCoreOutputs", (), {})

    perf_mod = _ensure_module("vllm.v1.metrics.perf")
    if not hasattr(perf_mod, "PerfStats"):
        perf_mod.PerfStats = type("PerfStats", (), {})

    request_mod = _ensure_module("vllm.v1.request")
    if not hasattr(request_mod, "Request"):
        request_mod.Request = type("Request", (), {})
    if not hasattr(request_mod, "RequestStatus"):
        request_mod.RequestStatus = type("RequestStatus", (), {"FINISHED_STOPPED": object()})

    spec_decode = _ensure_module("vllm.v1.spec_decode.metrics")
    if not hasattr(spec_decode, "SpecDecodingStats"):
        spec_decode.SpecDecodingStats = type("SpecDecodingStats", (), {})

    _ensure_module("vllm_omni")
    _ensure_module("vllm_omni.core")
    _ensure_module("vllm_omni.core.sched")
    omni_sched_output = _ensure_module("vllm_omni.core.sched.output")
    if not hasattr(omni_sched_output, "OmniCachedRequestData"):
        omni_sched_output.OmniCachedRequestData = type("OmniCachedRequestData", (), {})
    if not hasattr(omni_sched_output, "OmniNewRequestData"):
        omni_sched_output.OmniNewRequestData = type("OmniNewRequestData", (), {})

    _ensure_module("vllm_omni.distributed")
    _ensure_module("vllm_omni.distributed.omni_connectors")
    _ensure_module("vllm_omni.distributed.omni_connectors.transfer_adapter")
    chunk_adapter = _ensure_module("vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter")
    if not hasattr(chunk_adapter, "OmniChunkTransferAdapter"):
        chunk_adapter.OmniChunkTransferAdapter = type("OmniChunkTransferAdapter", (), {})

    outputs_mod = _ensure_module("vllm_omni.outputs")
    if not hasattr(outputs_mod, "OmniModelRunnerOutput"):
        outputs_mod.OmniModelRunnerOutput = type("OmniModelRunnerOutput", (), {})

    _ensure_module("vllm_omni.diffusion")
    # Load the actual memory_profiling module
    mem_prof_path = _REPO_ROOT / "vllm_omni" / "diffusion" / "memory_profiling.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("vllm_omni.diffusion.memory_profiling", mem_prof_path)
    mem_prof = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mem_prof)
    sys.modules["vllm_omni.diffusion.memory_profiling"] = mem_prof


def _load_scheduler_module() -> ModuleType:
    """Load the scheduler module with stubs, with cleanup."""
    import importlib.util

    _stub_scheduler_dependencies()

    sched_path = _REPO_ROOT / "vllm_omni" / "core" / "sched" / "omni_generation_scheduler.py"
    spec = importlib.util.spec_from_file_location("vllm_omni.core.sched.omni_generation_scheduler", sched_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scheduler_module() -> Generator[ModuleType, None, None]:
    """Load scheduler module for testing, with cleanup."""
    # Save any existing module state
    original_modules = {}
    for name in list(sys.modules.keys()):
        if name.startswith("vllm_omni.") or name.startswith("vllm."):
            original_modules[name] = sys.modules.pop(name)

    # Load and yield
    module = _load_scheduler_module()
    yield module

    # Cleanup: restore original state
    for name in list(sys.modules.keys()):
        if name.startswith("vllm_omni.") or name.startswith("vllm."):
            sys.modules.pop(name, None)
    for name, mod in original_modules.items():
        sys.modules[name] = mod


class DummyRequest:
    def __init__(self):
        self.request_id = "req-1"
        self.num_computed_tokens = 3
        self.prompt_token_ids = [1, 2, 3, 4, 5]


def test_log_allocation_failure_includes_memory_snapshot(monkeypatch, caplog, scheduler_module):
    scheduler = object.__new__(scheduler_module.OmniGenerationScheduler)
    scheduler._memory_profiling_enabled = True

    monkeypatch.setattr(
        scheduler_module,
        "capture_cuda_memory_snapshot",
        lambda: {
            "device": 0,
            "allocated_bytes": 1024,
            "reserved_bytes": 2048,
            "max_allocated_bytes": 4096,
            "max_reserved_bytes": 8192,
        },
    )
    monkeypatch.setattr(
        scheduler_module,
        "format_cuda_memory_snapshot",
        lambda snapshot: "cuda:0 allocated=0.00GiB reserved=0.00GiB max_allocated=0.00GiB max_reserved=0.00GiB",
    )

    with caplog.at_level(logging.WARNING):
        scheduler._log_allocation_failure(DummyRequest(), required_tokens=2, token_budget=1)

    assert "Diffusion scheduler allocation failed" in caplog.text
    assert "request_id=req-1" in caplog.text
    assert "token_budget=1" in caplog.text


def test_log_allocation_failure_noop_when_disabled(caplog, scheduler_module):
    scheduler = object.__new__(scheduler_module.OmniGenerationScheduler)
    scheduler._memory_profiling_enabled = False

    with caplog.at_level(logging.WARNING):
        scheduler._log_allocation_failure(DummyRequest(), required_tokens=2, token_budget=1)

    assert caplog.text == ""
