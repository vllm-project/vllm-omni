# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional sampler-owned step dispatch and legacy runner compatibility."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.interface import SupportsSamplingStep, supports_step_execution
from vllm_omni.diffusion.worker import diffusion_model_runner as runner_module
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_taomate_keeps_separate_scheduler_updates(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.taomate_h3.pipeline import TaoMateH3Pipeline

    assert MiniMaxH3Pipeline.denoise_step_updates_state is True
    assert TaoMateH3Pipeline.denoise_step_updates_state is False
    pipeline = object.__new__(TaoMateH3Pipeline)
    pipeline.calls = []
    for name in ("prepare_encode", "denoise_step", "step_scheduler", "post_decode"):
        monkeypatch.setattr(pipeline, name, getattr(_LegacyPipeline, name).__get__(pipeline))
    state = StepRequestState(request_id="ok", sampling=SimpleNamespace())
    state.latents = torch.zeros(1, 1)
    state.timesteps = torch.ones(1)
    states = [state]
    monkeypatch.setattr(runner_module, "set_forward_context", lambda **kw: nullcontext())
    runner = SimpleNamespace(
        pipeline=pipeline,
        od_config=SimpleNamespace(parallel_config=SimpleNamespace(use_hsdp=False), streaming_output=False),
        vllm_config=None,
        state_cache={state.request_id: state},
        _update_states=lambda _: (states, []),
        _prepare_batch_inputs=lambda *_: (states, SimpleNamespace(states=states), []),
        _prepare_output_for_transport=lambda output, _: output,
        _attach_stepwise_metadata=lambda *_: None,
        _maybe_send_stage_payload=lambda *_: None,
        _update_states_after=lambda *_: None,
    )
    output = runner_module.DiffusionModelRunner._execute_stepwise_core(
        runner, SimpleNamespace(), record_output_peak_memory=False
    )
    assert state.step_index == 1
    assert output["ok"].result.error is None
    assert pipeline.calls == ["predict", "update", "decode"]


class _LegacyPipeline:
    supports_step_execution = True

    def __init__(self):
        self.calls = []

    def prepare_encode(self, state):
        return state

    def denoise_step(self, input_batch, *, states):
        self.calls.append("predict")
        return torch.zeros(len(states), 1)

    def step_scheduler(self, state, noise_pred):
        self.calls.append("update")
        state.step_index += 1

    def post_decode(self, state):
        self.calls.append("decode")
        return DiffusionOutput(output=torch.zeros(1))


class _SamplingPipeline(_LegacyPipeline):
    denoise_step_updates_state = True

    def denoise_step(self, input_batch, *, states):
        self.calls.append("sample")
        results: dict[str, Exception | None] = {}
        for state in states:
            if state.request_id == "bad":
                results[state.request_id] = ValueError("prediction failed")
            else:
                state.step_index += 1
                results[state.request_id] = None
        return results


@pytest.mark.parametrize("sampler_owned", [False, True])
def test_runner_dispatch_advances_once_and_decodes(monkeypatch, sampler_owned):
    pipeline = _SamplingPipeline() if sampler_owned else _LegacyPipeline()
    assert supports_step_execution(pipeline)
    assert isinstance(pipeline, SupportsSamplingStep) is sampler_owned
    states = [StepRequestState(request_id="ok", sampling=SimpleNamespace())]
    if sampler_owned:
        states.append(StepRequestState(request_id="bad", sampling=SimpleNamespace()))
    for state in states:
        state.latents = torch.zeros(1, 1)
        state.timesteps = torch.ones(1)
    monkeypatch.setattr(runner_module, "set_forward_context", lambda **kw: nullcontext())
    runner = SimpleNamespace(
        pipeline=pipeline,
        od_config=SimpleNamespace(parallel_config=SimpleNamespace(use_hsdp=False), streaming_output=False),
        vllm_config=None,
        state_cache={s.request_id: s for s in states},
        _update_states=lambda _: (states, []),
        _prepare_batch_inputs=lambda *_: (states, SimpleNamespace(states=states), []),
        _prepare_output_for_transport=lambda output, _: output,
        _attach_stepwise_metadata=lambda *_: None,
        _maybe_send_stage_payload=lambda *_: None,
        _update_states_after=lambda *_: None,
    )
    output = runner_module.DiffusionModelRunner._execute_stepwise_core(
        runner, SimpleNamespace(), record_output_peak_memory=False
    )
    assert states[0].step_index == 1
    assert output["ok"].finished
    assert pipeline.calls == (["sample", "decode"] if sampler_owned else ["predict", "update", "decode"])
    if sampler_owned:
        assert states[1].step_index == 0
        assert output["bad"].finished
        assert "prediction failed" in output["bad"].result.error
        assert "bad" not in runner.state_cache


@pytest.mark.parametrize("updates_state", [False, True])
def test_sampling_only_pipeline_requires_state_update_flag(updates_state):
    pipeline = SimpleNamespace(
        supports_step_execution=True,
        denoise_step_updates_state=updates_state,
        prepare_encode=lambda s: s,
        denoise_step=lambda batch, **kwargs: {},
        post_decode=lambda s: None,
    )
    assert supports_step_execution(pipeline) is updates_state
    assert isinstance(pipeline, SupportsSamplingStep)


def test_profiler_reports_denoise_step_as_diffuse(monkeypatch):
    from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "is_available", lambda: False)
    pipeline_type = type(
        "StepProfiler",
        (DiffusionPipelineProfilerMixin,),
        {"_PROFILER_TARGETS": [], "forward": lambda self: None, "denoise_step": lambda self: "ok"},
    )
    pipeline = pipeline_type()
    pipeline.setup_diffusion_pipeline_profiler(enable_diffusion_pipeline_profiler=True)
    assert pipeline.denoise_step() == "ok"
    assert set(pipeline.stage_durations) == {"StepProfiler.diffuse"}
