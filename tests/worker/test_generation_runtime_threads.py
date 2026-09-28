# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker


@pytest.mark.parametrize("omp,internal,expected", [(None, False, [1]), ("4", False, []), ("96", True, [1])])
def test_mrv2_generation_restores_upstream_runtime_thread_policy(monkeypatch, omp, internal, expected):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("VLLM_OMP_NUM_THREADS_SET_BY_VLLM", raising=False)
    if omp is not None:
        monkeypatch.setenv("OMP_NUM_THREADS", omp)
    if internal:
        monkeypatch.setenv("VLLM_OMP_NUM_THREADS_SET_BY_VLLM", "1")
    events = []
    monkeypatch.setattr("vllm_omni.worker.gpu_generation_worker.freeze_gc_heap", lambda: events.append("freeze"))
    monkeypatch.setattr(torch, "get_num_threads", lambda: 96)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: events.append(n))
    worker = object.__new__(GPUGenerationWorker)
    worker.use_v2_model_runner = True
    worker.model_runner = SimpleNamespace(profile_run=lambda: events.append("profile"))
    timing = worker.compile_or_warm_up_model()
    assert events == ["profile", "freeze", *expected]
    assert timing.encoder == 0


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
