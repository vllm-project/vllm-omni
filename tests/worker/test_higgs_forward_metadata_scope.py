# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("fail", [False, True])
def test_runner_closes_metadata_scope_on_return_and_error(monkeypatch, fail):
    events = []

    def begin(**kw):
        events.append("begin")

    def end():
        events.append("end")

    def forward(*args, **kw):
        assert events == ["begin"]
        events.extend(["warmup", "capture"])
        if fail:
            raise RuntimeError("capture failed")
        return torch.ones(2)

    r = object.__new__(OmniGPUModelRunner)
    r.model = SimpleNamespace(
        supports_omni_decode_step_metadata=True, update_decode_step_metadata=begin, finish_decode_step_forward=end
    )
    r.input_batch = SimpleNamespace(req_ids=["a", "b"])
    r._build_model_kwargs_extra = lambda: {}
    monkeypatch.setattr(GPUModelRunner, "_model_forward", forward)
    if fail:
        with pytest.raises(RuntimeError, match="capture failed"):
            r._model_forward()
    else:
        torch.testing.assert_close(r._model_forward(), torch.ones(2))
    assert events == ["begin", "warmup", "capture", "end"]


@pytest.mark.parametrize("requires_tails", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_cpu_tail_metadata_requires_model_capability(monkeypatch, requires_tails, async_scheduling):
    observed = {}
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(
        supports_omni_decode_step_metadata=True,
        requires_cpu_input_tail_ids=requires_tails,
        update_decode_step_metadata=lambda **kwargs: observed.update(kwargs),
    )
    ids = torch.tensor([10, 11, 12])
    runner.use_async_scheduling = async_scheduling
    runner.input_ids = SimpleNamespace(gpu=ids, cpu=ids)
    runner.query_start_loc = SimpleNamespace(cpu=torch.tensor([0, 2, 3]))
    runner.input_batch = SimpleNamespace(req_ids=["a", "b"])
    runner._build_model_kwargs_extra = lambda: {}
    monkeypatch.setattr(GPUModelRunner, "_model_forward", lambda *args, **kwargs: torch.ones(2))
    runner._model_forward(input_ids=ids)
    expected = [11, 12] if requires_tails and not async_scheduling else None
    assert observed["cpu_input_tail_ids"] == expected


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
