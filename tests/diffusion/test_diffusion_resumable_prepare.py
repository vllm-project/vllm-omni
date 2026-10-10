# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runner tests for the resumable prepare phase of step execution."""

from __future__ import annotations

import contextlib
import gc
import queue
import threading
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm import forward_context as vllm_forward_context
from vllm.config import VllmConfig
from vllm.config.vllm import get_current_vllm_config_or_none

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm_omni.diffusion.models.interface import (
    supports_resumable_prepare,
    supports_step_execution,
)
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import SenseNovaU1ForCausalLM
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.sched.step_scheduler import StepScheduler
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, WorkerWrapperBase
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
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


class _ZeroWhenDonePipeline(_ResumablePipeline):
    """Reports a finished prepare phase as ``0`` rather than ``None``."""

    def prepare_steps_remaining(self, state):
        return state.extra.get("remaining", 0)


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


@pytest.mark.parametrize("phase", ["prepare", "denoise"])
def test_public_abort_releases_step_state_after_the_wave_without_another_request(phase):
    entered = threading.Event()
    resume = threading.Event()
    state_refs = []

    class Pipeline(_ResumablePipeline):
        def prepare_encode(self, state, **kwargs):
            super().prepare_encode(state, **kwargs)
            state_refs.append(weakref.ref(state))

        def wait_for_abort(self):
            entered.set()
            assert resume.wait(10), "test did not release the active wave"

        def prepare_step(self, state):
            if phase == "prepare":
                self.wait_for_abort()
            super().prepare_step(state)

        def denoise_step(self, input_batch, **kwargs):
            self.wait_for_abort()
            return super().denoise_step(input_batch, **kwargs)

    pipeline = Pipeline(prepare_steps=3 if phase == "prepare" else 0)
    runner = _make_runner(pipeline)
    runner.vllm_config = VllmConfig()
    runner.od_config.custom_pipeline_args = {"pipeline_class": Pipeline}
    runner.od_config.max_num_seqs = 1
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = runner
    worker.lora_manager = None
    worker._step_lora_state = {}
    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.worker = worker
    executor = object.__new__(UniProcDiffusionExecutor)
    executor.od_config = runner.od_config
    executor.driver_worker = wrapper
    executor._closed = False
    executor._is_failed = False
    executor._warned_about_timeout = False
    engine = object.__new__(DiffusionEngine)
    engine.od_config = runner.od_config
    engine.execution_mode = DiffusionExecutionMode.STEP_BATCH
    engine.scheduler = StepScheduler()
    engine.scheduler.initialize(engine.od_config)
    engine.executor = executor
    engine._init_runtime_state()
    engine._init_execute_fn()
    engine.stop_event = threading.Event()
    outputs: queue.Queue[tuple[str, DiffusionOutput]] = queue.Queue()
    engine._put_output = lambda request_id, output: outputs.put((request_id, output))
    request = _new_output("abort-last").scheduled_new_reqs[0].req
    engine.scheduler.add_request(request)
    thread = threading.Thread(target=engine._busy_loop)
    thread.start()
    try:
        assert entered.wait(10), "request did not reach its active wave"
        engine.abort([request.request_id, request.request_id])
        # Public abort enqueues control work. It cannot destroy a state that
        # the worker is still using inside prepare or denoise.
        assert request.request_id in runner.state_cache
        assert request.request_id in worker._step_lora_state
        if phase == "denoise":
            assert runner.input_batch is not None
        resume.set()
        request_id, output = outputs.get(timeout=10)
        assert request_id == request.request_id
        assert output.aborted
        assert not engine.scheduler.has_requests()
        assert runner.state_cache == {}
        assert worker._step_lora_state == {}
        assert runner.input_batch is None
        gc.collect()
        assert state_refs[0]() is None
        assert pipeline.prepare_calls == 1
        assert pipeline.prepare_step_calls == (1 if phase == "prepare" else 0)
        assert pipeline.denoise_calls == (0 if phase == "prepare" else 1)
    finally:
        resume.set()
        engine.stop_event.set()
        with engine._cv:
            engine._cv.notify_all()
        thread.join(timeout=10)
        assert not thread.is_alive()


def test_step_release_preserves_live_requests_and_native_kv_ownership():
    runner = _make_runner(_StepPipeline())
    ended, live = object(), object()
    runner.state_cache = {"ended": ended, "live": live}
    runner.input_batch = SimpleNamespace(request_ids=["ended", "live"], states=[ended, live])
    runner.diffusion_kv_backend = SimpleNamespace(remove_diffusion_kv_requests=Mock())
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = runner
    worker._step_lora_state = {"ended": (None, 1.0), "live": (None, 0.5)}

    assert worker.release_step_requests(["ended", "missing", "ended"]) == 1
    assert runner.state_cache == {"live": live}
    assert worker._step_lora_state == {"live": (None, 0.5)}
    assert runner.input_batch is None
    assert worker.release_step_requests(["ended", "missing"]) == 0
    runner.diffusion_kv_backend.remove_diffusion_kv_requests.assert_not_called()


def test_step_release_uses_the_all_rank_control_rpc_and_propagates_errors():
    executor = object.__new__(UniProcDiffusionExecutor)
    executor.collective_rpc = Mock(side_effect=RuntimeError("nonzero rank cleanup failed"))
    with pytest.raises(RuntimeError, match="nonzero rank cleanup failed"):
        executor.release_step_requests(["ended", "ended"])
    # Omitting unique_reply_rank uses the shared control protocol that executes
    # on every rank and gathers their statuses, including nonzero-rank errors.
    executor.collective_rpc.assert_called_once_with("release_step_requests", args=(["ended"],))


def test_worker_releases_a_normal_terminal_request_without_another_wave():
    runner = _make_runner(_ResumablePipeline(prepare_steps=0, num_steps=1))
    runner.vllm_config = VllmConfig()
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = runner
    worker.lora_manager = None
    worker._step_lora_state = {}
    output = worker.execute_stepwise(_new_output("last-completed"))
    assert output.get_request_output("last-completed").finished
    assert runner.state_cache == {}
    assert runner.input_batch is None
    assert worker._step_lora_state == {}


@pytest.mark.parametrize("failure", ["lora", "profiler_enter", "profiler_exit", "profiler_step"])
def test_worker_retires_only_the_failed_wave_even_before_runner_execution(failure):
    runner = _make_runner(_StepPipeline())
    failed, live = object(), object()
    runner.state_cache = {"failed": failed, "live": live}
    runner.input_batch = SimpleNamespace(request_ids=["failed", "live"], states=[failed, live])
    runner.diffusion_kv_backend = SimpleNamespace(remove_diffusion_kv_requests=Mock())
    runner.execute_stepwise = Mock(
        return_value=BatchRunnerOutput.from_list([RunnerOutput(request_id="failed", finished=False)])
    )
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = runner
    worker._step_lora_state = {"failed": (object(), 1.0), "live": (None, 0.5)}
    error = RuntimeError(f"injected {failure} failure")
    worker.lora_manager = SimpleNamespace(set_active_adapter=Mock(side_effect=error if failure == "lora" else None))

    @contextmanager
    def profiler_context():
        if failure == "profiler_enter":
            raise error
        yield
        if failure == "profiler_exit":
            raise error

    worker.profiler = SimpleNamespace(
        annotate_context_manager=lambda _name: profiler_context(),
        step=Mock(side_effect=error if failure == "profiler_step" else None),
    )
    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        worker.execute_stepwise(_cached_output("failed"))

    assert runner.state_cache == {"live": live}
    assert worker._step_lora_state == {"live": (None, 0.5)}
    assert runner.input_batch is None
    runner.diffusion_kv_backend.remove_diffusion_kv_requests.assert_not_called()
    if failure in {"lora", "profiler_enter"}:
        runner.execute_stepwise.assert_not_called()


class TestCapability:
    def test_resumable_prepare_is_separate_from_step_execution(self):
        atomic = _StepPipeline()
        resumable = _ResumablePipeline()
        assert supports_step_execution(atomic) is True
        assert supports_resumable_prepare(atomic) is False
        assert supports_step_execution(resumable) is True
        assert supports_resumable_prepare(resumable) is True


class TestResumablePrepare:
    @pytest.mark.parametrize("stage", ["prepare_encode", "prepare_step", "post_decode"])
    def test_prepare_model_calls_receive_the_native_moe_context(self, stage):
        calls: list[str] = []
        model = SimpleNamespace(has_moe=True)

        class Pipeline(_TextOnlyPipeline):
            def check_model_context(self, current_stage):
                if stage != current_stage:
                    return
                assert get_current_vllm_config_or_none() is runner.vllm_config
                assert get_forward_context().vllm_config is runner.vllm_config
                with SenseNovaU1ForCausalLM._vllm_forward_context(model):
                    context = vllm_forward_context.get_forward_context()
                    assert context.no_compile_layers is runner.vllm_config.compilation_config.static_forward_context
                    calls.append(current_stage)

            def prepare_encode(self, state, **kwargs):
                self.check_model_context("prepare_encode")
                return super().prepare_encode(state, **kwargs)

            def prepare_step(self, state):
                self.check_model_context("prepare_step")
                return super().prepare_step(state)

            def post_decode(self, state, **kwargs):
                self.check_model_context("post_decode")
                return super().post_decode(state, **kwargs)

        pipeline = Pipeline(prepare_steps=2)
        runner = _make_runner(pipeline)
        runner.vllm_config = VllmConfig()
        previous_config = get_current_vllm_config_or_none()
        previous_native_context = vllm_forward_context.is_forward_context_available()
        previous_omni_context = get_forward_context() if is_forward_context_available() else None

        first = runner.execute_stepwise(_new_output()).get_request_output("req-1")
        assert first.finished is False
        second = runner.execute_stepwise(_cached_output()).get_request_output("req-1")
        assert second.finished is True
        assert second.result.error is None
        assert calls == ([stage, stage] if stage == "prepare_step" else [stage])
        assert get_current_vllm_config_or_none() is previous_config
        assert vllm_forward_context.is_forward_context_available() is previous_native_context
        assert is_forward_context_available() is (previous_omni_context is not None)
        if previous_omni_context is not None:
            assert get_forward_context() is previous_omni_context

    @pytest.mark.parametrize("pipeline_cls", [_ResumablePipeline, _ZeroWhenDonePipeline])
    def test_prepare_runs_one_step_per_invocation_before_any_denoise(
        self,
        monkeypatch: pytest.MonkeyPatch,
        pipeline_cls,
    ):
        pipeline = pipeline_cls(prepare_steps=3)
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
        pipeline = _ResumablePipeline(prepare_steps=0, num_steps=1)
        runner = _make_runner(pipeline)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        # Completion releases its batch immediately, before another request.
        output = DiffusionModelRunner.execute_stepwise(runner, _new_output("req-1"))
        assert output.get_request_output("req-1").finished is True
        assert runner.input_batch is None

        # Nothing denoises on the next tick, so no batch may be left holding the
        # latents and the states of the request that ran before.
        pipeline.prepare_steps = 3
        DiffusionModelRunner.execute_stepwise(runner, _new_output("req-2", step_id=1))
        assert pipeline.prepare_step_calls == 1
        assert runner.input_batch is None

    def test_a_request_past_prepare_keeps_denoising_while_another_prepares(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        pipeline = _ResumablePipeline(prepare_steps=0, num_steps=4)
        runner = _make_runner(pipeline)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
        DiffusionModelRunner.execute_stepwise(runner, _new_output("req-a"))
        assert pipeline.denoise_calls == 1

        pipeline.prepare_steps = 3
        both = _new_output("req-b", step_id=1)
        both.scheduled_cached_reqs = CachedRequestData(request_ids=["req-a"])
        both.num_running_reqs = 2
        output = DiffusionModelRunner.execute_stepwise(runner, both)

        # req-b spends this tick on one prepare step; req-a does not wait for it.
        assert pipeline.prepare_step_calls == 1
        assert pipeline.denoise_calls == 2
        assert runner.state_cache["req-a"].step_index == 2
        assert output.get_request_output("req-b").finished is False

    @pytest.mark.parametrize("pipeline_cls", [_ResumablePipeline, _ZeroWhenDonePipeline])
    def test_a_request_that_asks_for_no_prepare_steps_denoises_immediately(
        self,
        monkeypatch: pytest.MonkeyPatch,
        pipeline_cls,
    ):
        pipeline = pipeline_cls(prepare_steps=0)
        runner = _make_runner(pipeline)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        DiffusionModelRunner.execute_stepwise(runner, _new_output())

        assert pipeline.prepare_step_calls == 0
        assert pipeline.denoise_calls == 1

    @pytest.mark.parametrize("num_steps", [1, 2])
    def test_the_memory_profile_keeps_going_until_a_denoise_step_runs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        num_steps,
    ):
        pipeline = _ResumablePipeline(prepare_steps=4, num_steps=num_steps)
        runner = _make_runner(pipeline)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
        monkeypatch.setattr(model_runner_module.current_omni_platform, "synchronize", lambda: None)

        request = OmniDiffusionRequest(
            prompt="a prompt",
            request_id="profile-1",
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=num_steps),
        )
        runner.profile_run([request])

        # Sizing the budget from a prepare step alone would miss the denoise
        # allocations the profile exists to measure.
        assert pipeline.prepare_step_calls == 4
        assert pipeline.denoise_calls == 1
        assert runner.state_cache == {}
        assert runner.input_batch is None

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

    def test_prepare_only_request_sends_its_stage_payload_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        pipeline = _TextOnlyPipeline(prepare_steps=2)
        runner = _make_runner(pipeline)
        sent = []
        runner._maybe_send_stage_payload = lambda reqs, outputs: sent.append(
            ([req.request_id for req in reqs], [out.output for out in outputs])
        )
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

        DiffusionModelRunner.execute_stepwise(runner, _new_output())
        assert sent == []

        DiffusionModelRunner.execute_stepwise(runner, _cached_output())
        assert sent == [(["req-1"], [{"payload": {"text": "done"}}])]

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
