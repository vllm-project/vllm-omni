# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for micro-step (temporal PP) execution across runner / worker / executor / engine."""

from __future__ import annotations

import copy
import queue
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.data import DiffusionOutput, DiffusionParallelConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import StreamBatchScheduler
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
    RankTask,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _FakeScheduler:
    """Minimal scheduler deepcopyable and tracks step_index."""

    def __init__(self):
        self._step_index = 0

    def step(self, noise_pred, t, latents, return_dict=False):
        del t, return_dict
        self._step_index += 1
        return (latents + noise_pred,)


class _MicroStepPipeline:
    """Minimal pipeline stub supporting micro-step execution."""

    supports_step_execution = True

    def __init__(self):
        self.prepare_calls = 0
        self.denoise_calls = 0
        self.scheduler_calls = 0
        self.decode_calls = 0
        self.sync_calls = 0
        self.scheduler = _FakeScheduler()

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_calls += 1
        n = state.sampling.num_inference_steps
        state.timesteps = [torch.tensor(float(n - i)) for i in range(n)]
        state.latents = torch.zeros((1, 1, 2, 2, 2))  # [B, C, T, H, W] video-like
        state.step_index = 0
        state.scheduler = copy.deepcopy(self.scheduler)
        return state

    def denoise_step(self, state, **kwargs):
        del kwargs
        self.denoise_calls += 1
        return torch.ones_like(state.latents)

    def step_scheduler(self, state, noise_pred, **kwargs):
        del noise_pred, kwargs
        self.scheduler_calls += 1
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        self.decode_calls += 1
        # Produce a per-chunk video tensor uniquely tagged by decode call count
        # so we can verify concatenation order downstream.
        return DiffusionOutput(output=torch.full((1, 1, 2, 2, 2), float(self.decode_calls)))

    def sync_pp_send(self):
        self.sync_calls += 1


def _make_pp_group(rank: int, world_size: int) -> SimpleNamespace:
    """Mock PP group for the runner's get_pp_group() call."""
    return SimpleNamespace(
        rank_in_group=rank,
        is_first_rank=(rank == 0),
        is_last_rank=(rank == world_size - 1),
    )


def _make_runner(pp_size: int = 1, pp_rank: int = 0) -> DiffusionModelRunner:
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        parallel_config=SimpleNamespace(use_hsdp=False, pipeline_parallel_size=pp_size),
    )
    runner.device = torch.device("cpu")
    runner.pipeline = _MicroStepPipeline()
    runner.cache_backend = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace()
    runner._pp_group = _make_pp_group(pp_rank, pp_size)
    return runner


def _install_pp_group_stub(monkeypatch, rank: int, world_size: int) -> None:
    """Replace get_pp_group in the runner module with a constant stub."""
    monkeypatch.setattr(
        model_runner_module,
        "get_pp_group",
        lambda: _make_pp_group(rank, world_size),
    )


def _make_micro_step_request(
    num_chunks: int = 1, num_inference_steps: int = 2
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=["a prompt"],
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            num_chunks=num_chunks,
            seed=42,
        ),
        request_ids=["req-1"],
    )


def _make_micro_step_scheduler_output(
    task: RankTask | None,
    pp_size: int,
    req: OmniDiffusionRequest | None = None,
    step_id: int = 0,
    finished_req_ids: set[str] | None = None,
) -> DiffusionSchedulerOutput:
    """Scheduler output with a single-rank assignment (the rest idle)."""
    assignment: list[RankTask | None] = [None] * pp_size
    if task is not None:
        # For the runner we're simulating, the task is at the rank the runner reports.
        # Tests set up pp_rank separately via monkeypatch; task is placed at rank 0 by default.
        assignment[0] = task

    new_reqs = []
    cached = CachedRequestData.make_empty()
    if req is not None:
        new_reqs = [NewRequestData(sched_req_id="req-1", req=req)]
    else:
        cached = CachedRequestData(sched_req_ids=["req-1"])

    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached,
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=1,
        num_waiting_reqs=0,
        per_rank_assignment=assignment,
    )


def _make_engine(scheduler, execute_fn=None, stream_batch: bool = True) -> DiffusionEngine:
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(model_class_name="Wan22Pipeline")
    engine.pre_process_func = None
    engine.post_process_func = None
    engine.scheduler = scheduler
    engine.execute_fn = execute_fn
    engine.stream_batch = stream_batch
    engine.step_execution = True
    engine._rpc_lock = threading.RLock()
    engine.abort_queue = queue.Queue()
    return engine


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestMicroStepRunner:
    """DiffusionModelRunner.execute_micro_step"""

    def test_single_chunk_completes_and_returns_merged(self, monkeypatch):
        runner = _make_runner(pp_size=1, pp_rank=0)
        _install_pp_group_stub(monkeypatch, rank=0, world_size=1)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
        req = _make_micro_step_request(num_chunks=1, num_inference_steps=2)

        # Step 0.
        out0 = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_step_scheduler_output(
                RankTask(sched_req_id="req-1", chunk_idx=0, step_index=0), pp_size=1, req=req,
            ),
        )
        assert out0.chunk_idx == 0
        assert out0.step_index == 1
        assert out0.chunk_completed is False
        assert out0.finished is False
        assert out0.result is None

        # Step 1 (completes the chunk — with single rank, single chunk, this finishes the request).
        out1 = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_step_scheduler_output(
                RankTask(sched_req_id="req-1", chunk_idx=0, step_index=1), pp_size=1,
            ),
        )
        assert out1.chunk_idx == 0
        assert out1.step_index == 2
        assert out1.chunk_completed is True
        assert out1.finished is True
        assert out1.result is not None
        assert runner.pipeline.decode_calls == 1
        # State cache should be cleared once the request completes.
        assert "req-1" not in runner.state_cache

    def test_multi_chunk_produces_concatenated_result(self, monkeypatch):
        runner = _make_runner(pp_size=1, pp_rank=0)
        _install_pp_group_stub(monkeypatch, rank=0, world_size=1)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
        req = _make_micro_step_request(num_chunks=2, num_inference_steps=1)

        # Chunk 0, step 0 (completes chunk 0).
        DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_step_scheduler_output(
                RankTask(sched_req_id="req-1", chunk_idx=0, step_index=0), pp_size=1, req=req,
            ),
        )
        # Chunk 1, step 0 (completes chunk 1 → merges → finishes request).
        final = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_step_scheduler_output(
                RankTask(sched_req_id="req-1", chunk_idx=1, step_index=0), pp_size=1,
            ),
        )
        assert final.finished is True
        assert final.result is not None
        # Two chunks concatenated along time dim (dim 2): [1, 1, 4, 2, 2].
        assert final.result.output.shape == (1, 1, 4, 2, 2)
        # First chunk's frames tagged 1.0, second chunk's tagged 2.0.
        assert torch.all(final.result.output[:, :, :2] == 1.0)
        assert torch.all(final.result.output[:, :, 2:] == 2.0)
        assert runner.pipeline.decode_calls == 2

    def test_idle_rank_returns_early_and_syncs(self, monkeypatch):
        runner = _make_runner(pp_size=2, pp_rank=1)
        _install_pp_group_stub(monkeypatch, rank=1, world_size=2)
        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
        req = _make_micro_step_request(num_chunks=1, num_inference_steps=2)

        # Assignment puts task at rank 0 only; rank 1 (this runner) is idle.
        sched_output = _make_micro_step_scheduler_output(
            RankTask(sched_req_id="req-1", chunk_idx=0, step_index=0), pp_size=2, req=req,
        )
        out = DiffusionModelRunner.execute_micro_step(runner, sched_output)

        assert out.chunk_idx is None
        assert out.step_index is None
        assert out.finished is False
        # Idle path still drains pending sends.
        assert runner.pipeline.sync_calls == 1
        # prepare_encode must run on idle ranks so shared state is ready when
        # the rank later receives a chunk.
        assert runner.pipeline.prepare_calls == 1

    def test_rejects_missing_per_rank_assignment(self, monkeypatch):
        runner = _make_runner(pp_size=1, pp_rank=0)
        _install_pp_group_stub(monkeypatch, rank=0, world_size=1)
        req = _make_micro_step_request()

        sched_output = DiffusionSchedulerOutput(
            step_id=0,
            scheduled_new_reqs=[NewRequestData(sched_req_id="req-1", req=req)],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            finished_req_ids=set(),
            num_running_reqs=1,
            num_waiting_reqs=0,
            per_rank_assignment=None,
        )

        with pytest.raises(ValueError, match="per_rank_assignment"):
            DiffusionModelRunner.execute_micro_step(runner, sched_output)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class TestMicroStepWorker:
    """DiffusionWorker.execute_micro_step"""

    def test_delegates_to_model_runner(self):
        worker = object.__new__(DiffusionWorker)
        expected = RunnerOutput(req_id="req-1", chunk_idx=0, step_index=1)
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req=SimpleNamespace(sampling_params=SimpleNamespace(lora_request=None))
                )
            ]
        )
        worker.lora_manager = None
        worker.model_runner = SimpleNamespace(
            execute_micro_step=lambda arg: expected if arg is scheduler_output else None
        )

        output = DiffusionWorker.execute_micro_step(worker, scheduler_output)
        assert output is expected

    def test_rejects_lora_requests(self):
        worker = object.__new__(DiffusionWorker)
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req=SimpleNamespace(sampling_params=SimpleNamespace(lora_request=object()))
                )
            ]
        )
        worker.lora_manager = None
        worker.model_runner = SimpleNamespace(execute_micro_step=lambda arg: RunnerOutput(req_id="req-1"))

        with pytest.raises(ValueError, match="does not support LoRA"):
            DiffusionWorker.execute_micro_step(worker, scheduler_output)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class TestMicroStepExecutor:
    """MultiprocDiffusionExecutor.execute_micro_step"""

    def test_passes_through_runner_output_and_uses_first_pp_rank(self, mocker: MockerFixture):
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._ensure_open = lambda: None
        executor.od_config = SimpleNamespace(
            parallel_config=DiffusionParallelConfig(pipeline_parallel_size=4),
        )
        expected = RunnerOutput(req_id="req-1", chunk_idx=0, step_index=1, chunk_completed=False)
        rpc_mock = mocker.Mock(return_value=expected)
        executor.collective_rpc = rpc_mock

        sched_output = DiffusionSchedulerOutput(
            step_id=0,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(sched_req_ids=["req-1"]),
            finished_req_ids=set(),
            num_running_reqs=1,
            num_waiting_reqs=0,
            per_rank_assignment=[None, None, None, RankTask("req-1", 0, 0)],
        )

        result = MultiprocDiffusionExecutor.execute_micro_step(executor, sched_output)

        assert result is expected
        # Reply is collected from the first PP rank (index 0).
        rpc_mock.assert_called_once()
        kwargs = rpc_mock.call_args.kwargs
        assert kwargs["unique_reply_rank"] == 0
        assert kwargs["exec_all_ranks"] is True


# ---------------------------------------------------------------------------
# Engine (full loop: scheduler + executor + engine)
# ---------------------------------------------------------------------------


class TestMicroStepEngine:
    """Full stream-batch flow through DiffusionEngine.add_req_and_wait_for_response."""

    def _make_scheduler(self, pp_size: int = 1) -> StreamBatchScheduler:
        scheduler = StreamBatchScheduler()
        scheduler.initialize(
            SimpleNamespace(parallel_config=DiffusionParallelConfig(pipeline_parallel_size=pp_size))
        )
        return scheduler

    def _make_execute_fn(self, num_chunks: int, num_steps: int):
        """Simulate the executor: advance each micro-step's last-rank chunk."""
        completed = {"n": 0}

        def execute_fn(sched_output):
            assignment = sched_output.per_rank_assignment
            if assignment is None:
                return RunnerOutput(req_id="")
            task = assignment[-1]  # last rank's slot
            if task is None:
                req_id = sched_output.scheduled_req_ids[0] if sched_output.scheduled_req_ids else ""
                return RunnerOutput(req_id=req_id)

            new_step = task.step_index + 1
            chunk_completed = new_step >= num_steps
            finished = False
            result = None
            if chunk_completed:
                completed["n"] += 1
                if completed["n"] >= num_chunks:
                    finished = True
                    result = DiffusionOutput(output=torch.tensor([float(completed["n"])]))
            return RunnerOutput(
                req_id=task.sched_req_id,
                step_index=new_step,
                chunk_idx=task.chunk_idx,
                chunk_completed=chunk_completed,
                finished=finished,
                result=result,
            )

        return execute_fn

    def test_single_chunk_completes(self):
        scheduler = self._make_scheduler(pp_size=1)
        engine = _make_engine(scheduler, execute_fn=self._make_execute_fn(num_chunks=1, num_steps=2))
        request = _make_micro_step_request(num_chunks=1, num_inference_steps=2)

        output = engine.add_req_and_wait_for_response(request)

        assert output.error is None
        assert output.aborted is False
        assert torch.equal(output.output, torch.tensor([1.0]))

    def test_multi_chunk_completes(self):
        scheduler = self._make_scheduler(pp_size=1)
        engine = _make_engine(scheduler, execute_fn=self._make_execute_fn(num_chunks=3, num_steps=2))
        request = _make_micro_step_request(num_chunks=3, num_inference_steps=2)

        output = engine.add_req_and_wait_for_response(request)

        assert output.error is None
        assert torch.equal(output.output, torch.tensor([3.0]))  # completed 3 chunks

    def test_execute_fn_exception_returns_error(self):
        scheduler = self._make_scheduler(pp_size=1)

        def failing(_):
            raise RuntimeError("gpu on fire")

        engine = _make_engine(scheduler, execute_fn=failing)
        output = engine.add_req_and_wait_for_response(_make_micro_step_request())

        assert output.output is None
        assert "gpu on fire" in output.error

    def test_pipeline_fills_with_pp_gt_1(self):
        """With PP>1, scheduler drives warmup/steady/cooldown; engine sees final merged output."""
        pp_size = 3
        num_chunks = 4
        num_steps = 2
        scheduler = self._make_scheduler(pp_size=pp_size)
        engine = _make_engine(
            scheduler, execute_fn=self._make_execute_fn(num_chunks=num_chunks, num_steps=num_steps)
        )
        request = _make_micro_step_request(num_chunks=num_chunks, num_inference_steps=num_steps)

        output = engine.add_req_and_wait_for_response(request)

        assert output.error is None
        assert torch.equal(output.output, torch.tensor([float(num_chunks)]))