# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for micro-step level diffusion execution across runner / worker / executor / engine."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
    Rank0Layout,
    RankTask,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.utils import RunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _FakePPGroup:
    def __init__(self, rank_in_group: int = 0, world_size: int = 1):
        self.rank_in_group = rank_in_group
        self.world_size = world_size
        self.is_first_rank = rank_in_group == 0
        self.is_last_rank = rank_in_group == world_size - 1
        self.prev_rank = (rank_in_group - 1) % world_size
        self.next_rank = (rank_in_group + 1) % world_size
        self.reset_calls = 0

    def reset_buffer(self) -> None:
        self.reset_calls += 1


class _MicroStepPipeline:
    supports_step_execution = True
    supports_micro_step_execution = True

    def __init__(self, num_steps: int = 1):
        self.num_steps = num_steps
        self.prepare_calls = 0
        self.set_buffer_calls = 0
        self.denoise_calls = 0
        self.scheduler_calls = 0
        self.decode_calls = 0
        self.prefetch_calls = 0
        self.is_buffer_setup = False

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_calls += 1
        state.timesteps = [torch.tensor(float(i)) for i in range(self.num_steps)]
        state.latents = torch.zeros((1, 1, 1, 1, 1))
        state.scheduler = SimpleNamespace(timesteps=list(state.timesteps))
        return state

    def set_pp_recv_dict_buffers(self, state, **kwargs):
        del state, kwargs
        self.set_buffer_calls += 1
        self.is_buffer_setup = True

    def denoise_step(self, state, **kwargs):
        del state, kwargs
        self.denoise_calls += 1
        return torch.tensor([1.0])

    def step_scheduler(self, state, noise_pred, **kwargs):
        del noise_pred, kwargs
        self.scheduler_calls += 1
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        self.decode_calls += 1
        # One batched decode covers all rows on this rank; output keeps the
        # per-row layout so _merge_chunk_outputs can stitch the temporal axis.
        b = state.latents.shape[0] if state.latents.ndim > 0 else 1
        return DiffusionOutput(output=torch.ones(b, 1, 1, 1, 1, dtype=torch.float32))

    def prefetch_its(self, state, **kwargs):
        del state, kwargs
        self.prefetch_calls += 1


class _InterruptingMicroStepPipeline(_MicroStepPipeline):
    interrupt = True

    def denoise_step(self, state, **kwargs):
        del state, kwargs
        self.denoise_calls += 1
        return None

    def step_scheduler(self, state, noise_pred, **kwargs):
        del state, noise_pred, kwargs
        raise AssertionError("step_scheduler should not run after interrupt")

    def post_decode(self, state, **kwargs):
        del state, kwargs
        raise AssertionError("post_decode should not run after interrupt")


def _make_micro_request(
    req_id: str = "req-1",
    *,
    num_inference_steps: int = 1,
    num_chunks: int = 1,
):
    return SimpleNamespace(
        prompts=["a prompt"],
        request_ids=[req_id],
        sampling_params=SimpleNamespace(
            generator=None,
            seed=None,
            generator_device=None,
            num_inference_steps=num_inference_steps,
            num_chunks=num_chunks,
            num_frames=1,
            slo_fps=None,
            slo_max_batch=8,
            slo_ema_alpha=0.3,
            lora_request=None,
        ),
    )


def _make_runner(pp_size: int = 1, num_steps: int = 1):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.device = torch.device("cpu")
    runner.pipeline = _MicroStepPipeline(num_steps=num_steps)
    runner.cache_backend = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace()
    runner._fake_pp_group = _FakePPGroup(world_size=pp_size)
    return runner


def _make_layout(
    *,
    n_finished: int = 0,
    n_circulating: int = 0,
    n_new: int = 0,
    finished_idxs: list[int] | None = None,
    new_idxs: list[int] | None = None,
) -> Rank0Layout:
    return Rank0Layout(
        n_finished=n_finished,
        n_circulating=n_circulating,
        n_new=n_new,
        finished_idxs=finished_idxs or [],
        new_idxs=new_idxs or [],
    )


def _make_micro_scheduler_output(
    *,
    req=None,
    sched_req_id: str = "req-1",
    step_id: int = 0,
    assignment=None,
    is_new: bool = True,
    finished_req_ids=None,
    rank0_layout: Rank0Layout | None = None,
):
    if assignment is None:
        assignment = [RankTask(sched_req_id=sched_req_id, chunk_indices=[0])]
    rank0_layouts = {sched_req_id: rank0_layout} if rank0_layout is not None else None
    if is_new and req is not None:
        new_reqs = [NewRequestData(sched_req_id=sched_req_id, req=req)]
        cached_reqs = CachedRequestData.make_empty()
    else:
        new_reqs = []
        cached_reqs = CachedRequestData(sched_req_ids=[sched_req_id])
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached_reqs,
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=1,
        num_waiting_reqs=0,
        per_rank_assignment=assignment,
        rank0_layouts=rank0_layouts,
    )


def _patch_runtime(monkeypatch, runner) -> None:
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "get_pp_group", lambda: runner._fake_pp_group)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestRunner:
    """DiffusionModelRunner.execute_micro_step (PP=1)."""

    def test_completes_single_chunk_request(self, monkeypatch):
        runner = _make_runner(pp_size=1, num_steps=1)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=1, num_chunks=1)

        out0 = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                req=req, step_id=0,
                assignment=[RankTask(sched_req_id="req-1", chunk_indices=[0])],
                rank0_layout=_make_layout(n_new=1, new_idxs=[0]),
            ),
        )
        assert out0.req_id == "req-1"
        assert out0.finished is False
        assert "req-1" in runner.state_cache

        out1 = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                sched_req_id="req-1", step_id=1,
                assignment=[None], is_new=False,
                rank0_layout=_make_layout(n_finished=1, finished_idxs=[0]),
            ),
        )
        assert out1.finished is True
        assert out1.result is not None
        assert out1.result.output is not None
        assert "req-1" not in runner.state_cache

        assert runner.pipeline.prepare_calls == 1
        assert runner.pipeline.denoise_calls == 1
        assert runner.pipeline.scheduler_calls == 1
        assert runner.pipeline.decode_calls == 1

    def test_completes_multi_chunk_request(self, monkeypatch):
        runner = _make_runner(pp_size=1, num_steps=1)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=1, num_chunks=2)

        DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                req=req, step_id=0,
                assignment=[RankTask(sched_req_id="req-1", chunk_indices=[0])],
                rank0_layout=_make_layout(n_new=1, new_idxs=[0]),
            ),
        )
        out1 = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                sched_req_id="req-1", step_id=1,
                assignment=[RankTask(sched_req_id="req-1", chunk_indices=[1])],
                is_new=False,
                rank0_layout=_make_layout(n_finished=1, n_new=1, finished_idxs=[0], new_idxs=[1]),
            ),
        )
        assert out1.finished is False

        out2 = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                sched_req_id="req-1", step_id=2,
                assignment=[None], is_new=False,
                rank0_layout=_make_layout(n_finished=1, finished_idxs=[1]),
            ),
        )
        assert out2.finished is True
        assert out2.result is not None
        assert "req-1" not in runner.state_cache

        assert runner.pipeline.prepare_calls == 1
        assert runner.pipeline.denoise_calls == 2
        assert runner.pipeline.decode_calls == 2

    def test_re_admits_circulating_chunk(self, monkeypatch):
        runner = _make_runner(pp_size=1, num_steps=2)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=2, num_chunks=1)

        out0 = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                req=req, step_id=0,
                assignment=[RankTask(sched_req_id="req-1", chunk_indices=[0])],
                rank0_layout=_make_layout(n_new=1, new_idxs=[0]),
            ),
        )
        assert out0.finished is False

        out1 = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                sched_req_id="req-1", step_id=1,
                assignment=[RankTask(sched_req_id="req-1", chunk_indices=[0])],
                is_new=False,
                rank0_layout=_make_layout(n_circulating=1),
            ),
        )
        assert out1.finished is False
        assert runner.pipeline.denoise_calls == 2

        out2 = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                sched_req_id="req-1", step_id=2,
                assignment=[None], is_new=False,
                rank0_layout=_make_layout(n_finished=1, finished_idxs=[0]),
            ),
        )
        assert out2.finished is True
        assert runner.pipeline.decode_calls == 1

    def test_empty_layout_is_a_no_op(self, monkeypatch):
        runner = _make_runner(pp_size=1, num_steps=1)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=1, num_chunks=1)

        DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                req=req, step_id=0,
                assignment=[RankTask(sched_req_id="req-1", chunk_indices=[0])],
                rank0_layout=_make_layout(n_new=1, new_idxs=[0]),
            ),
        )
        denoise_calls_before = runner.pipeline.denoise_calls

        out = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                sched_req_id="req-1", step_id=1,
                assignment=[None], is_new=False,
                rank0_layout=_make_layout(),
            ),
        )
        assert out.req_id == "req-1"
        assert out.finished is False
        assert runner.pipeline.denoise_calls == denoise_calls_before
        assert runner.pipeline.decode_calls == 0

    def test_interrupt_marks_request_as_aborted(self, monkeypatch):
        runner = _make_runner(pp_size=1, num_steps=1)
        runner.pipeline = _InterruptingMicroStepPipeline(num_steps=1)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=1, num_chunks=1)

        out = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                req=req, step_id=0,
                assignment=[RankTask(sched_req_id="req-1", chunk_indices=[0])],
                rank0_layout=_make_layout(n_new=1, new_idxs=[0]),
            ),
        )
        assert out.req_id == "req-1"
        assert out.result is not None
        assert out.result.error == "micro-step denoise interrupted"
        assert runner.pipeline.denoise_calls == 1
        assert runner.pipeline.scheduler_calls == 0
        assert runner.pipeline.decode_calls == 0

    def test_rejects_missing_per_rank_assignment(self):
        runner = _make_runner(pp_size=1)
        req = _make_micro_request()
        sched_output = _make_micro_scheduler_output(req=req)
        sched_output.per_rank_assignment = None

        with pytest.raises(ValueError, match="per_rank_assignment"):
            DiffusionModelRunner.execute_micro_step(runner, sched_output)

    def test_rejects_cache_backend(self):
        runner = _make_runner(pp_size=1)
        runner.od_config = SimpleNamespace(
            cache_backend="teacache",
            parallel_config=SimpleNamespace(use_hsdp=False),
        )
        req = _make_micro_request()

        with pytest.raises(ValueError, match="cache_backend"):
            DiffusionModelRunner.execute_micro_step(runner, _make_micro_scheduler_output(req=req))

    def test_stamps_micro_step_wall_ns_on_rank0(self, monkeypatch):
        runner = _make_runner(pp_size=1, num_steps=1)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=1, num_chunks=1)

        out = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                req=req, step_id=0,
                assignment=[RankTask(sched_req_id="req-1", chunk_indices=[0])],
                rank0_layout=_make_layout(n_new=1, new_idxs=[0]),
            ),
        )
        assert out.micro_step_wall_ns is not None
        assert out.micro_step_wall_ns >= 0

    def test_batch_two_runs_one_fused_forward(self, monkeypatch):
        runner = _make_runner(pp_size=1, num_steps=1)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=1, num_chunks=2)

        out = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                req=req, step_id=0,
                assignment=[RankTask(sched_req_id="req-1", chunk_indices=[0, 1])],
                rank0_layout=_make_layout(n_new=2, new_idxs=[0, 1]),
            ),
        )

        assert runner.pipeline.denoise_calls == 1
        assert runner.pipeline.scheduler_calls == 1
        assert out.req_id == "req-1"
        assert out.finished is False
        assert out.micro_step_wall_ns is not None

    def test_batch_two_decodes_both_chunks_when_drain_completes(self, monkeypatch):
        runner = _make_runner(pp_size=1, num_steps=1)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=1, num_chunks=2)

        DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                req=req, step_id=0,
                assignment=[RankTask(sched_req_id="req-1", chunk_indices=[0, 1])],
                rank0_layout=_make_layout(n_new=2, new_idxs=[0, 1]),
            ),
        )
        out = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(
                sched_req_id="req-1", step_id=1,
                assignment=[None], is_new=False,
                rank0_layout=_make_layout(n_finished=2, finished_idxs=[0, 1]),
            ),
        )

        assert out.finished is True
        assert runner.pipeline.decode_calls == 1
        assert "req-1" not in runner.state_cache


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class TestWorker:
    """DiffusionWorker.execute_micro_step"""

    def test_delegates_to_model_runner(self):
        worker = object.__new__(DiffusionWorker)
        expected = RunnerOutput(req_id="req-1")
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req=SimpleNamespace(sampling_params=SimpleNamespace(lora_request=None)))
            ]
        )
        worker.lora_manager = None
        worker.model_runner = SimpleNamespace(
            execute_micro_step=lambda arg: expected if arg is scheduler_output else None
        )
        worker._get_profiler = lambda: None

        output = DiffusionWorker.execute_micro_step(worker, scheduler_output)
        assert output is expected

    def test_clears_active_lora(self):
        worker = object.__new__(DiffusionWorker)
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req=SimpleNamespace(sampling_params=SimpleNamespace(lora_request=None)))
            ]
        )
        calls: list = []

        class _FakeLoRAManager:
            def set_active_adapter(self, adapter):
                calls.append(adapter)

        worker.lora_manager = _FakeLoRAManager()
        worker.model_runner = SimpleNamespace(execute_micro_step=lambda _: RunnerOutput(req_id="req-1"))
        worker._get_profiler = lambda: None

        DiffusionWorker.execute_micro_step(worker, scheduler_output)
        assert calls == [None]

    def test_rejects_lora_requests(self):
        worker = object.__new__(DiffusionWorker)
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req=SimpleNamespace(sampling_params=SimpleNamespace(lora_request=object())))
            ]
        )
        worker.lora_manager = None
        worker.model_runner = SimpleNamespace(execute_micro_step=lambda _: RunnerOutput(req_id="req-1"))
        worker._get_profiler = lambda: None

        with pytest.raises(ValueError, match="does not support LoRA"):
            DiffusionWorker.execute_micro_step(worker, scheduler_output)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class TestSupportedPipelines:
    """Micro-step protocol membership checks."""

    def test_stub_pipeline_satisfies_protocol(self):
        from vllm_omni.diffusion.models.interface import (
            SupportsMicroStepExecution,
            SupportsStepExecution,
            supports_micro_step_execution,
            supports_step_execution,
        )

        pipeline = _MicroStepPipeline()
        assert isinstance(pipeline, SupportsMicroStepExecution) is True
        assert supports_micro_step_execution(pipeline) is True
        # Micro-step protocol extends step protocol.
        assert isinstance(pipeline, SupportsStepExecution) is True
        assert supports_step_execution(pipeline) is True

    def test_wan22_supports_micro_step_execution(self):
        from vllm_omni.diffusion.models.interface import (
            SupportsMicroStepExecution,
            supports_micro_step_execution,
        )
        from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline

        # Avoid loading weights; protocol membership is a class-contract check.
        pipeline = object.__new__(Wan22Pipeline)

        assert pipeline.supports_step_execution is True
        assert pipeline.supports_micro_step_execution is True
        assert supports_micro_step_execution(pipeline) is True
        assert isinstance(pipeline, SupportsMicroStepExecution) is True


class TestExecutor:
    """MultiprocDiffusionExecutor.execute_micro_step collects rank-0's reply."""

    def test_passes_through_runner_output(self, mocker: MockerFixture):
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._ensure_open = lambda: None
        expected = RunnerOutput(req_id="req-1", finished=True)
        rpc = mocker.Mock(return_value=expected)
        executor.collective_rpc = rpc

        sched_output = _make_micro_scheduler_output(req=_make_micro_request())
        output = MultiprocDiffusionExecutor.execute_micro_step(executor, sched_output)

        assert output is expected
        _, kwargs = rpc.call_args
        assert kwargs.get("unique_reply_rank") == 0
        assert kwargs.get("exec_all_ranks") is True

    def test_rejects_unexpected_reply_type(self, mocker: MockerFixture):
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._ensure_open = lambda: None
        executor.collective_rpc = mocker.Mock(return_value="not a runner output")

        sched_output = _make_micro_scheduler_output(req=_make_micro_request())
        with pytest.raises(RuntimeError, match="Unexpected response type"):
            MultiprocDiffusionExecutor.execute_micro_step(executor, sched_output)