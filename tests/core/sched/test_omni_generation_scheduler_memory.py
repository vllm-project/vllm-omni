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


def _is_vllm_related(name: str) -> bool:
    return name == "vllm" or name.startswith("vllm.") or name == "vllm_omni" or name.startswith("vllm_omni.")


def _create_stub_package(name: str) -> ModuleType:
    """Create a package-like module stub with ``__path__`` so Python can import submodules.

    If the module already exists in ``sys.modules`` it is returned unchanged.
    Intermediate parent packages are created and the child is set as an attribute
    on the parent so that ``from . import sub`` works inside stub packages.
    """
    if name in sys.modules:
        mod = sys.modules[name]
        if not hasattr(mod, "__path__"):
            mod.__path__ = []  # type: ignore[attr-defined]
        return mod
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        subname = ".".join(parts[:i])
        if subname not in sys.modules:
            mod = ModuleType(subname)
            mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[subname] = mod
        else:
            mod = sys.modules[subname]
            if not hasattr(mod, "__path__"):
                mod.__path__ = []  # type: ignore[attr-defined]
        # Attach child to parent so relative imports work.
        if i > 1:
            parent_name = ".".join(parts[: i - 1])
            parent = sys.modules[parent_name]
            attr = parts[i - 1]
            if not hasattr(parent, attr):
                setattr(parent, attr, mod)
    return sys.modules[name]


def _stub_scheduler_dependencies(
    original_modules: dict[str, ModuleType | None],
) -> dict[str, ModuleType]:
    """Create minimal stubs for vllm/vllm_omni dependencies.

    For each module name that already exists in ``sys.modules``, the existing
    reference is stored in ``original_modules`` so the caller can restore it.
    For names that do not exist, ``None`` is stored.

    Returns a dict of newly-created stub modules (excluding pre-existing ones)
    so the caller can remove them after loading the scheduler.
    """
    created: dict[str, ModuleType] = {}

    def ensure(name: str) -> ModuleType:
        """Get or create a package-like stub for *name*."""
        if name in sys.modules:
            original_modules[name] = sys.modules[name]
        else:
            original_modules[name] = None
            stub = _create_stub_package(name)
            sys.modules[name] = stub
            created[name] = stub
        return sys.modules[name]

    # ── vllm ──────────────────────────────────────────────────────────────────
    ensure("vllm")
    ensure("vllm.compilation")
    cuda_graph = ensure("vllm.compilation.cuda_graph")
    if not hasattr(cuda_graph, "CUDAGraphStat"):
        cuda_graph.CUDAGraphStat = type("CUDAGraphStat", (), {})

    ensure("vllm.distributed")
    kv_events = ensure("vllm.distributed.kv_events")
    if not hasattr(kv_events, "KVEventBatch"):
        kv_events.KVEventBatch = type("KVEventBatch", (), {})
    ensure("vllm.distributed.kv_transfer")
    ensure("vllm.distributed.kv_transfer.kv_connector")
    ensure("vllm.distributed.kv_transfer.kv_connector.v1")
    kv_metrics = ensure("vllm.distributed.kv_transfer.kv_connector.v1.metrics")
    if not hasattr(kv_metrics, "KVConnectorStats"):
        kv_metrics.KVConnectorStats = type("KVConnectorStats", (), {})

    logger_mod = ensure("vllm.logger")
    if not hasattr(logger_mod, "init_logger"):
        logger_mod.init_logger = logging.getLogger

    ensure("vllm.v1")
    ensure("vllm.v1.core")
    kv_cache = ensure("vllm.v1.core.kv_cache_manager")
    if not hasattr(kv_cache, "KVCacheBlocks"):
        kv_cache.KVCacheBlocks = type("KVCacheBlocks", (), {})

    ensure("vllm.v1.core.sched")
    sched_interface = ensure("vllm.v1.core.sched.interface")
    if not hasattr(sched_interface, "PauseState"):
        sched_interface.PauseState = type("PauseState", (), {"PAUSED_ALL": object(), "UNPAUSED": object()})
    sched_output = ensure("vllm.v1.core.sched.output")
    if not hasattr(sched_output, "SchedulerOutput"):
        sched_output.SchedulerOutput = type("SchedulerOutput", (), {})
    request_queue = ensure("vllm.v1.core.sched.request_queue")
    if not hasattr(request_queue, "create_request_queue"):
        request_queue.create_request_queue = lambda policy: []
    scheduler_mod = ensure("vllm.v1.core.sched.scheduler")
    if not hasattr(scheduler_mod, "Scheduler"):
        scheduler_mod.Scheduler = type("Scheduler", (), {})
    sched_utils = ensure("vllm.v1.core.sched.utils")
    if not hasattr(sched_utils, "remove_all"):
        sched_utils.remove_all = lambda seq, items: [x for x in seq if x not in items]

    # Missing parent stubs fixed: create vllm.v1.metrics and vllm.v1.spec_decode
    # before their sub-modules (Copilot/Codex review feedback)
    ensure("vllm.v1.metrics")
    perf_mod = ensure("vllm.v1.metrics.perf")
    if not hasattr(perf_mod, "PerfStats"):
        perf_mod.PerfStats = type("PerfStats", (), {})
    ensure("vllm.v1.spec_decode")
    spec_decode = ensure("vllm.v1.spec_decode.metrics")
    if not hasattr(spec_decode, "SpecDecodingStats"):
        spec_decode.SpecDecodingStats = type("SpecDecodingStats", (), {})

    engine_mod = ensure("vllm.v1.engine")
    if not hasattr(engine_mod, "EngineCoreEventType"):
        engine_mod.EngineCoreEventType = type("EngineCoreEventType", (), {"SCHEDULED": object()})
    if not hasattr(engine_mod, "EngineCoreOutput"):
        engine_mod.EngineCoreOutput = type("EngineCoreOutput", (), {})
    if not hasattr(engine_mod, "EngineCoreOutputs"):
        engine_mod.EngineCoreOutputs = type("EngineCoreOutputs", (), {})

    request_mod = ensure("vllm.v1.request")
    if not hasattr(request_mod, "Request"):
        request_mod.Request = type("Request", (), {})
    if not hasattr(request_mod, "RequestStatus"):
        request_mod.RequestStatus = type("RequestStatus", (), {"FINISHED_STOPPED": object()})

    # ── vllm_omni ─────────────────────────────────────────────────────────────
    ensure("vllm_omni")
    ensure("vllm_omni.core")
    ensure("vllm_omni.core.sched")
    omni_sched_output = ensure("vllm_omni.core.sched.output")
    if not hasattr(omni_sched_output, "OmniCachedRequestData"):
        omni_sched_output.OmniCachedRequestData = type("OmniCachedRequestData", (), {})
    if not hasattr(omni_sched_output, "OmniNewRequestData"):
        omni_sched_output.OmniNewRequestData = type("OmniNewRequestData", (), {})

    ensure("vllm_omni.distributed")
    ensure("vllm_omni.distributed.omni_connectors")
    ensure("vllm_omni.distributed.omni_connectors.transfer_adapter")
    chunk_adapter = ensure("vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter")
    if not hasattr(chunk_adapter, "OmniChunkTransferAdapter"):
        chunk_adapter.OmniChunkTransferAdapter = type("OmniChunkTransferAdapter", (), {})

    outputs_mod = ensure("vllm_omni.outputs")
    if not hasattr(outputs_mod, "OmniModelRunnerOutput"):
        outputs_mod.OmniModelRunnerOutput = type("OmniModelRunnerOutput", (), {})

    ensure("vllm_omni.diffusion")
    # Provide the real memory_profiling module (fixed: was loading from a
    # non-existent path; now uses the actual module in the repo)
    import importlib.util

    mem_prof_path = _REPO_ROOT / "vllm_omni" / "diffusion" / "memory_profiling.py"
    spec = importlib.util.spec_from_file_location("vllm_omni.diffusion.memory_profiling", mem_prof_path)
    mem_prof = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mem_prof)
    sys.modules["vllm_omni.diffusion.memory_profiling"] = mem_prof

    return created


def _load_scheduler_module(
    original_modules: dict[str, ModuleType | None],
) -> tuple[ModuleType, dict[str, ModuleType]]:
    """Load the scheduler module with stubs, with cleanup.

    Creates stubs for any missing vllm/vllm_omni modules, loads the scheduler,
    then removes all vllm/vllm_omni entries from sys.modules that were created
    by this function (leaving pre-existing real modules intact).

    Returns:
        A tuple of (loaded_scheduler_module, created_stub_modules).
    """
    import importlib.util

    created = _stub_scheduler_dependencies(original_modules)

    sched_path = _REPO_ROOT / "vllm_omni" / "core" / "sched" / "omni_generation_scheduler.py"
    spec = importlib.util.spec_from_file_location("vllm_omni.core.sched.omni_generation_scheduler", sched_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    # Remove only the stub modules this function created (not pre-existing
    # real vllm/vllm_omni packages that were in sys.modules before).
    for name in created:
        sys.modules.pop(name, None)
    # Also remove memory_profiling stub if it was placed by us.
    sys.modules.pop("vllm_omni.diffusion.memory_profiling", None)

    # Restore pre-existing originals that we temporarily overwrote.
    for name, mod in original_modules.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod

    return module, created


@pytest.fixture
def scheduler_module() -> Generator[ModuleType, None, None]:
    """Load scheduler module for testing, with cleanup.

    Saves all vllm/vllm_omni modules (including bare 'vllm' parent) before
    loading, then restores them afterward so this test does not affect others.
    """
    original_modules: dict[str, ModuleType | None] = {}
    for name in list(sys.modules.keys()):
        if _is_vllm_related(name):
            original_modules[name] = sys.modules.pop(name)

    module, _created_stubs = _load_scheduler_module(original_modules)
    yield module

    # Remove any vllm/vllm_omni modules that exist now (should only be the
    # scheduler module we just loaded).
    for name in list(sys.modules.keys()):
        if _is_vllm_related(name):
            sys.modules.pop(name, None)
    # Restore original state.
    for name, mod in original_modules.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
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
        lambda snapshot: ("cuda:0 allocated=0.00GiB reserved=0.00GiB max_allocated=0.00GiB max_reserved=0.00GiB"),
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
