# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional memory probes work without CUDA snapshot support on NPU workers."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("is_310p", [False, True])
def test_npu_worker_memory_probe_is_unsupported(monkeypatch, is_310p):
    # Load the production class while replacing only the hardware dependencies.
    # The upstream worker deliberately has no optional Omni memory RPC method.
    upstream = ModuleType("vllm_ascend.worker.worker")
    upstream_worker = type("NPUWorker", (), {})
    monkeypatch.setattr(upstream, "NPUWorker", upstream_worker, raising=False)
    upstream_310p = ModuleType("vllm_ascend._310p.worker_310p")
    upstream_worker_310p = type("NPUWorker310", (), {})
    monkeypatch.setattr(upstream_310p, "NPUWorker310", upstream_worker_310p, raising=False)
    soc = ModuleType("vllm_omni.platforms.npu._310p")
    monkeypatch.setattr(soc, "is_310p", lambda: is_310p, raising=False)
    for module in (upstream, upstream_310p, soc):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    source = Path(__file__).resolve().parents[3] / "vllm_omni/platforms/npu/worker/base.py"
    spec = importlib.util.spec_from_file_location("_npu_worker_memory_probe", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    worker = object.__new__(module.OmniNPUWorkerBase)
    expected_base = upstream_worker_310p if is_310p else upstream_worker
    assert isinstance(worker, expected_base)
    # RPC dispatch resolves a method on the worker. Unsupported optional stats
    # should return None, without requiring a runner or touching device memory.
    assert getattr(worker, "get_stage_post_warmup_memory_stats")() is None
