# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("generation", [False, True])
@pytest.mark.parametrize("enabled,fail", [(False, False), (True, False), (True, True)])
def test_auxiliary_capture_finishes_before_worker_readiness(monkeypatch, generation, enabled, fail):
    from vllm.v1.worker.gpu_worker import Worker

    from vllm_omni.worker.base import OmniGPUWorkerBase
    from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker

    events: list[str] = []

    def capture():
        assert events == ["warmup"]
        events.append("capture")
        if fail:
            raise RuntimeError("auxiliary capture failed")

    monkeypatch.setattr(Worker, "compile_or_warm_up_model", lambda _: events.append("warmup"))
    worker = object.__new__(GPUGenerationWorker if generation else OmniGPUWorkerBase)
    worker.use_v2_model_runner = generation
    model = SimpleNamespace(capture_auxiliary_graphs=capture) if enabled else SimpleNamespace()
    worker.model_runner = SimpleNamespace(model=model, profile_run=lambda: events.append("warmup"))
    if fail:
        with pytest.raises(RuntimeError, match="auxiliary capture failed"):
            worker.compile_or_warm_up_model()
    else:
        worker.compile_or_warm_up_model()
        events.append("ready")
    assert events == ["warmup"] + (["capture"] if enabled else []) + ([] if fail else ["ready"])
