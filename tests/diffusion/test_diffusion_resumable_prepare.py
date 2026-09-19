# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runner tests for the resumable prepare phase of step execution."""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.models.interface import (
    supports_resumable_prepare,
    supports_step_execution,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _StepPipeline:
    """Step-execution pipeline whose prepare phase is atomic."""

    supports_step_execution = True

    def __init__(self, *, num_steps: int = 2):
        self.num_steps = num_steps
        self.prepare_calls = 0
        self.denoise_calls = 0
        self.decode_calls = 0

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_calls += 1
        state.timesteps = torch.zeros(self.num_steps)
        state.latents = torch.zeros(1, 2)
        return state

    def denoise_step(self, input_batch, **kwargs):
        del kwargs
        self.denoise_calls += 1
        return torch.zeros_like(input_batch.latents)

    def step_scheduler(self, state, noise_pred, **kwargs):
        del noise_pred, kwargs
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        self.decode_calls += 1
        return DiffusionOutput(output=torch.tensor([state.step_index], dtype=torch.float32))


class _ResumablePipeline(_StepPipeline):
    """Prepare phase that is a loop, like a unified model's AR decode."""

    supports_resumable_prepare = True

    def __init__(self, *, prepare_steps: int = 3, num_steps: int = 2, fail_at: int | None = None):
        super().__init__(num_steps=num_steps)
        self.prepare_steps = prepare_steps
        self.fail_at = fail_at
        self.prepare_step_calls = 0

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_calls += 1
        state.extra["remaining"] = self.prepare_steps
        state.timesteps = torch.zeros(self.num_steps)
        state.latents = torch.zeros(1, 2)
        return state

    def prepare_steps_remaining(self, state):
        remaining = state.extra.get("remaining", 0)
        return remaining or None

    def prepare_step(self, state):
        self.prepare_step_calls += 1
        if self.fail_at is not None and self.prepare_step_calls == self.fail_at:
            raise RuntimeError("prepare step blew up")
        state.extra["remaining"] -= 1


class _TextOnlyPipeline(_ResumablePipeline):
    """Prepare phase produces the whole output; there is nothing to denoise."""

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_calls += 1
        state.extra["remaining"] = self.prepare_steps
        return state

    def denoise_step(self, input_batch, **kwargs):
        raise AssertionError("a request with no denoise schedule must not reach denoise_step")

    def post_decode(self, state, **kwargs):
        del kwargs
        self.decode_calls += 1
        return DiffusionOutput(output={"payload": {"text": "done"}})


class _FakePeakMemoryPlatform:
    """Stands in for the device so the memory path is exercised on CPU."""

    def __init__(self, reserved_mb: float):
        self._reserved_mb = reserved_mb

    def is_available(self) -> bool:
        return True

    def reset_peak_memory_stats(self) -> None:
        return None

    def max_memory_reserved(self) -> int:
        return int(self._reserved_mb * 1024**2)

    def max_memory_allocated(self) -> int:
        return int(self._reserved_mb * 1024**2)


def _make_vllm_config():
    @contextlib.contextmanager
    def set_priority(*args, **kwargs):
        yield

    return SimpleNamespace(
        kernel_config=SimpleNamespace(ir_op_priority=SimpleNamespace(set_priority=set_priority)),
        compilation_config=SimpleNamespace(ir_enable_torch_wrap=True),
    )


def _make_runner(pipeline):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = _make_vllm_config()
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
        parallel_config=SimpleNamespace(use_hsdp=False),
        streaming_output=False,
        step_execution=True,
    )
    runner.device = torch.device("cpu")
    runner.pipeline = pipeline
    runner.cache_backend = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.input_batch = None
    runner.kv_transfer_manager = SimpleNamespace(
        receive_multi_kv_cache_distributed=lambda req, cfg_kv_collect_func=None, target_device=None: None
    )
    return runner


def _new_output(request_id="req-1", step_id=0):
    req = OmniDiffusionRequest(
        prompt="a prompt",
        request_id=request_id,
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
    )
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[NewRequestData(request_id=request_id, req=req)],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )


def _cached_output(request_id="req-1", step_id=1):
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(request_ids=[request_id]),
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )


class TestCapability:
    def test_resumable_prepare_is_separate_from_step_execution(self):
        atomic = _StepPipeline()
        resumable = _ResumablePipeline()
        assert supports_step_execution(atomic) is True
        assert supports_resumable_prepare(atomic) is False
        assert supports_step_execution(resumable) is True
        assert supports_resumable_prepare(resumable) is True


class TestResumablePrepare:
    def test_prepare_runs_one_step_per_invocation_before_any_denoise(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        pipeline = _ResumablePipeline(prepare_steps=3)
        runner = _make_runner(pipeline)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        output = DiffusionModelRunner.execute_stepwise(runner, _new_output())
        assert pipeline.prepare_step_calls == 1
        assert pipeline.denoise_calls == 0
        assert output.get_request_output("req-1").finished is False

        output = DiffusionModelRunner.execute_stepwise(runner, _cached_output(step_id=2))
        assert pipeline.prepare_step_calls == 2
        assert pipeline.denoise_calls == 0
        assert output.get_request_output("req-1").finished is False

        # The step that ends the prepare phase denoises in the same tick, so the
        # request does not lose one to the handover.
        DiffusionModelRunner.execute_stepwise(runner, _cached_output(step_id=3))
        assert pipeline.prepare_step_calls == 3
        assert pipeline.denoise_calls == 1

    def test_a_finished_prepare_phase_releases_the_previous_batch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        pipeline = _ResumablePipeline(prepare_steps=3)
        runner = _make_runner(pipeline)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        DiffusionModelRunner.execute_stepwise(runner, _new_output())

        # Nothing denoised, so no batch may be left holding the latents and the
        # states of whatever ran before.
        assert runner.input_batch is None

    def test_a_request_that_asks_for_no_prepare_steps_denoises_immediately(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        pipeline = _ResumablePipeline(prepare_steps=0)
        runner = _make_runner(pipeline)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        DiffusionModelRunner.execute_stepwise(runner, _new_output())

        assert pipeline.prepare_step_calls == 0
        assert pipeline.denoise_calls == 1

    def test_the_memory_profile_keeps_going_until_a_denoise_step_runs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        pipeline = _ResumablePipeline(prepare_steps=4)
        runner = _make_runner(pipeline)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
        monkeypatch.setattr(model_runner_module.current_omni_platform, "synchronize", lambda: None)

        request = OmniDiffusionRequest(
            prompt="a prompt",
            request_id="profile-1",
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
        )
        runner.profile_run([request])

        # Sizing the budget from a prepare step alone would miss the denoise
        # allocations the profile exists to measure.
        assert pipeline.prepare_step_calls == 4
        assert pipeline.denoise_calls == 1

    def test_atomic_prepare_still_denoises_on_the_first_invocation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        pipeline = _StepPipeline()
        runner = _make_runner(pipeline)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        DiffusionModelRunner.execute_stepwise(runner, _new_output())

        assert pipeline.prepare_calls == 1
        assert pipeline.denoise_calls == 1

    def test_prepare_only_request_finishes_without_a_denoise_step(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        pipeline = _TextOnlyPipeline(prepare_steps=2)
        runner = _make_runner(pipeline)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        first = DiffusionModelRunner.execute_stepwise(runner, _new_output())
        assert first.get_request_output("req-1").finished is False
        assert pipeline.decode_calls == 0

        second = DiffusionModelRunner.execute_stepwise(runner, _cached_output())
        request_output = second.get_request_output("req-1")
        assert request_output.finished is True
        assert request_output.result.output == {"payload": {"text": "done"}}
        assert pipeline.decode_calls == 1
        assert "req-1" not in runner.state_cache

    def test_prepare_only_request_reports_its_peak_memory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        pipeline = _TextOnlyPipeline(prepare_steps=1)
        runner = _make_runner(pipeline)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
        monkeypatch.setattr(model_runner_module, "current_omni_platform", _FakePeakMemoryPlatform(512.0))

        output = DiffusionModelRunner.execute_stepwise(runner, _new_output())

        request_output = output.get_request_output("req-1")
        assert request_output.finished is True
        assert request_output.result.peak_memory_mb == pytest.approx(512.0)

    def test_prepare_step_failure_is_terminal_for_that_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        pipeline = _ResumablePipeline(prepare_steps=3, fail_at=2)
        runner = _make_runner(pipeline)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        DiffusionModelRunner.execute_stepwise(runner, _new_output())
        output = DiffusionModelRunner.execute_stepwise(runner, _cached_output())

        request_output = output.get_request_output("req-1")
        assert request_output.finished is True
        assert "prepare step blew up" in request_output.result.error
        assert "req-1" not in runner.state_cache
        assert pipeline.denoise_calls == 0
