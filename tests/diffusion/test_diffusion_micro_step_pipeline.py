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
        self.group_prev_rank = (rank_in_group - 1) % world_size
        self.group_next_rank = (rank_in_group + 1) % world_size
        self.reset_calls = 0

    def reset_buffer(self) -> None:
        self.reset_calls += 1


class _MicroStepPipeline:
    """Mock implementing the SupportsMicroStepExecution protocol.

    A fixed ladder runs chunk 0 in one ``prepare_first_chunk`` micro-step, then the
    steady ladder (``prepare_chunks`` + ``denoise_step`` + ``step_scheduler`` +
    ``decode_chunks``) carries chunks 1..N-1. The merged output appears once every
    chunk has been decoded.
    """

    supports_micro_step_execution = True

    def __init__(self, num_steps: int = 1):
        self.num_steps = num_steps
        self.prepare_encode_calls = 0
        self.set_buffer_calls = 0
        self.first_chunk_calls = 0
        self.prepare_chunks_calls = 0
        self.denoise_calls = 0
        self.scheduler_calls = 0
        self.decode_calls = 0
        self.prefetch_calls = 0
        self.decoded_chunks = 0

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.prepare_encode_calls += 1
        state.timesteps = torch.zeros(self.num_steps)
        state.latents = None
        state.scheduler = SimpleNamespace()
        state.step_index = 0
        return state

    def set_pp_recv_dict_buffers(self, state, **kwargs):
        del state, kwargs
        self.set_buffer_calls += 1

    def prepare_first_chunk(self, state, **kwargs):
        del kwargs
        self.first_chunk_calls += 1
        return self._record_decode(state)  # first always decodes chunk 0

    def prepare_chunks(self, state, **kwargs):
        del state, kwargs
        self.prepare_chunks_calls += 1

    def denoise_step(self, state, **kwargs):
        del state, kwargs
        self.denoise_calls += 1
        return torch.tensor([1.0])

    def step_scheduler(self, state, noise_pred, **kwargs):
        del noise_pred, kwargs
        self.scheduler_calls += 1
        state.step_index += 1

    def decode_chunks(self, state, **kwargs):
        del kwargs
        self.decode_calls += 1
        # Decode only when the deepest slot holds a real chunk (a dummy bottom
        # slot is skipped, matching the real pipeline).
        if state.extra["slot_chunks"][-1] is None:
            return None
        return self._record_decode(state)

    def prefetch_tensors(self, state, batch_size: int = 1, **kwargs):
        del state, batch_size, kwargs
        self.prefetch_calls += 1

    def _record_decode(self, state):
        self.decoded_chunks += 1
        if self.decoded_chunks >= state.sampling.num_chunks:
            return DiffusionOutput(output=torch.ones(1, 1, 1, 1, 1, dtype=torch.float32))
        return None


class _InterruptingMicroStepPipeline(_MicroStepPipeline):
    interrupt = True

    def denoise_step(self, state, **kwargs):
        del state, kwargs
        self.denoise_calls += 1
        return None

    def step_scheduler(self, state, noise_pred, **kwargs):
        del state, noise_pred, kwargs
        raise AssertionError("step_scheduler should not run after interrupt")

    def decode_chunks(self, state, **kwargs):
        del state, kwargs
        raise AssertionError("decode_chunks should not run after interrupt")


def _make_micro_request(
    req_id: str = "req-1",
    *,
    num_inference_steps: int = 1,
    num_chunks: int = 1,
    chunk_frames: int = 1,
):
    return SimpleNamespace(
        prompts=["a prompt"],
        request_id=req_id,
        sampling_params=SimpleNamespace(
            generator=None,
            seed=None,
            generator_device=None,
            num_inference_steps=num_inference_steps,
            chunk_frames=chunk_frames,
            num_chunks=num_chunks,
            num_frames=num_chunks * chunk_frames,
            lora_request=None,
        ),
    )


def _make_runner(pp_size: int = 1, num_steps: int = 1, rank: int = 0):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        parallel_config=SimpleNamespace(use_hsdp=False),
        enable_dynamic_block_schedule=False,
    )
    runner.device = torch.device("cpu")
    runner.pipeline = _MicroStepPipeline(num_steps=num_steps)
    runner.cache_backend = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace()
    runner._fake_pp_group = _FakePPGroup(rank_in_group=rank, world_size=pp_size)
    return runner


def _make_micro_scheduler_output(
    *,
    req=None,
    request_id: str = "req-1",
    step_id: int = 0,
    slot_chunks: list[int | None] | None = None,
    is_last: bool = False,
    is_new: bool = True,
    finished_req_ids=None,
):
    if slot_chunks is None:
        slot_chunks = [0]
    assignment = [RankTask(request_id=request_id, slot_chunks=slot_chunks, is_last=is_last)]
    if is_new and req is not None:
        new_reqs = [NewRequestData(request_id=request_id, req=req)]
        cached_reqs = CachedRequestData.make_empty()
    else:
        new_reqs = []
        cached_reqs = CachedRequestData(request_ids=[request_id])
    return DiffusionSchedulerOutput(
        step_id=step_id,
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached_reqs,
        finished_req_ids=set() if finished_req_ids is None else set(finished_req_ids),
        num_running_reqs=1,
        num_waiting_reqs=0,
        assignment=assignment,
    )


def _make_pp_micro_scheduler_output(
    *,
    req=None,
    request_id: str = "req-1",
    per_rank: list[list[int | None]],
    is_last: bool = False,
    is_new: bool = True,
):
    """Scheduler output whose assignment carries one RankTask per PP rank."""
    assignment = [RankTask(request_id=request_id, slot_chunks=list(sc), is_last=is_last) for sc in per_rank]
    if is_new and req is not None:
        new_reqs = [NewRequestData(request_id=request_id, req=req)]
        cached_reqs = CachedRequestData.make_empty()
    else:
        new_reqs = []
        cached_reqs = CachedRequestData(request_ids=[request_id])
    return DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached_reqs,
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
        assignment=assignment,
    )


def _patch_runtime(monkeypatch, runner) -> None:
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "get_pp_group", lambda: runner._fake_pp_group)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "synchronize", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestRunner:
    """DiffusionModelRunner.execute_micro_step (PP=1)."""

    def test_first_single_chunk_completes(self, monkeypatch):
        runner = _make_runner(pp_size=1, num_steps=1)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=1, num_chunks=1)

        out = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(req=req, slot_chunks=[0], is_last=True),
        )
        assert out.request_id == "req-1"
        assert out.finished is True
        assert out.result is not None
        assert out.result.output is not None
        assert "req-1" not in runner.state_cache

        # First-chunk path only: prepare_encode + prepare_first_chunk, no steady hooks.
        assert runner.pipeline.prepare_encode_calls == 1
        assert runner.pipeline.set_buffer_calls == 1
        assert runner.pipeline.first_chunk_calls == 1
        assert runner.pipeline.prepare_chunks_calls == 0
        assert runner.pipeline.denoise_calls == 0
        assert runner.pipeline.decode_calls == 0

    def test_multi_chunk_first_then_steady(self, monkeypatch):
        runner = _make_runner(pp_size=1, num_steps=1)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=1, num_chunks=2)

        out0 = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(req=req, slot_chunks=[0], is_last=False),
        )
        assert out0.finished is False
        assert out0.result is None  # chunk 0 decoded, but 2 chunks total
        assert "req-1" in runner.state_cache

        out1 = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(request_id="req-1", slot_chunks=[1], is_last=True, is_new=False),
        )
        assert out1.finished is True
        assert out1.result is not None
        assert out1.result.output is not None
        assert "req-1" not in runner.state_cache

        assert runner.pipeline.first_chunk_calls == 1
        assert runner.pipeline.prepare_chunks_calls == 1
        assert runner.pipeline.denoise_calls == 1
        assert runner.pipeline.scheduler_calls == 1
        assert runner.pipeline.decode_calls == 1

    def test_three_chunks_two_steps(self, monkeypatch):
        runner = _make_runner(pp_size=1, num_steps=2)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=2, num_chunks=3)

        steps = [
            ([0, None], False),  # first chunk 0
            ([1, None], False),  # steady: bottom dummy -> no decode
            ([2, 1], False),     # steady: decode chunk 1
            ([None, 2], True),   # steady: decode chunk 2 (final)
        ]
        outs = []
        for i, (slot_chunks, is_last) in enumerate(steps):
            outs.append(
                DiffusionModelRunner.execute_micro_step(
                    runner,
                    _make_micro_scheduler_output(
                        req=req if i == 0 else None,
                        request_id="req-1",
                        slot_chunks=slot_chunks,
                        is_last=is_last,
                        is_new=(i == 0),
                    ),
                )
            )

        assert [o.finished for o in outs] == [False, False, False, True]
        assert outs[-1].result is not None
        assert outs[-1].result.output is not None
        assert "req-1" not in runner.state_cache

        assert runner.pipeline.first_chunk_calls == 1
        assert runner.pipeline.prepare_chunks_calls == 3
        assert runner.pipeline.denoise_calls == 3
        assert runner.pipeline.scheduler_calls == 3
        assert runner.pipeline.decode_calls == 3
        assert runner.pipeline.decoded_chunks == 3  # chunk 0 (first) + chunks 1, 2 (steady)

    def test_steady_skips_decode_when_bottom_is_dummy(self, monkeypatch):
        runner = _make_runner(pp_size=1, num_steps=2)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=2, num_chunks=3)

        DiffusionModelRunner.execute_micro_step(
            runner, _make_micro_scheduler_output(req=req, slot_chunks=[0, None], is_last=False)
        )
        # Steady micro-step whose deepest slot is empty: decode runs but emits nothing.
        out = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(request_id="req-1", slot_chunks=[1, None], is_last=False, is_new=False),
        )
        assert out.finished is False
        assert out.result is None
        assert runner.pipeline.decode_calls == 1
        assert runner.pipeline.decoded_chunks == 1  # only chunk 0 (first) so far

    def test_interrupt_marks_request_as_aborted(self, monkeypatch):
        runner = _make_runner(pp_size=1, num_steps=1)
        runner.pipeline = _InterruptingMicroStepPipeline(num_steps=1)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=1, num_chunks=2)

        # A steady micro-step (no chunk 0) so denoise_step runs and interrupts.
        out = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_micro_scheduler_output(req=req, slot_chunks=[1], is_last=False),
        )
        assert out.request_id == "req-1"
        assert out.result is not None
        assert out.result.error == "micro-step denoise interrupted"
        assert runner.pipeline.denoise_calls == 1
        assert runner.pipeline.scheduler_calls == 0

    def test_rejects_missing_assignment(self):
        runner = _make_runner(pp_size=1)
        req = _make_micro_request()
        sched_output = _make_micro_scheduler_output(req=req)
        sched_output.assignment = None

        with pytest.raises(ValueError, match="assignment"):
            DiffusionModelRunner.execute_micro_step(runner, sched_output)

    def test_rejects_cache_backend(self):
        runner = _make_runner(pp_size=1)
        runner.od_config = SimpleNamespace(
            cache_backend="teacache",
            parallel_config=SimpleNamespace(use_hsdp=False),
            enable_dynamic_block_schedule=False,
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
            _make_micro_scheduler_output(req=req, slot_chunks=[0], is_last=True),
        )
        assert out.micro_step_wall_ns is not None
        assert out.micro_step_wall_ns >= 0


class TestRunnerPipelineParallel:
    """Per-rank orchestration of execute_micro_step under PP>1."""

    # A 3-rank steady ladder (chunks 1..3 in flight; no chunk 0 => not first).
    PER_RANK = [[3], [2], [1]]

    def _run_rank(self, monkeypatch, rank: int):
        runner = _make_runner(pp_size=3, num_steps=1, rank=rank)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=1, num_chunks=4)
        out = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_pp_micro_scheduler_output(req=req, per_rank=self.PER_RANK, is_last=False),
        )
        return runner.pipeline, out

    def test_first_rank_admits_and_prefetches(self, monkeypatch):
        pipe, out = self._run_rank(monkeypatch, rank=0)
        assert out.finished is False
        assert out.result is None  # not the last rank -> no decoded output
        assert pipe.prepare_chunks_calls == 1  # first rank rolls + admits
        assert pipe.denoise_calls == 1
        assert pipe.scheduler_calls == 1
        assert pipe.decode_calls == 0
        assert pipe.prefetch_calls == 1

    def test_middle_rank_only_denoises(self, monkeypatch):
        pipe, out = self._run_rank(monkeypatch, rank=1)
        assert out.result is None
        assert pipe.prepare_chunks_calls == 0  # not first rank
        assert pipe.denoise_calls == 1
        assert pipe.scheduler_calls == 1
        assert pipe.decode_calls == 0  # not last rank
        assert pipe.prefetch_calls == 1

    def test_last_rank_decodes(self, monkeypatch):
        pipe, _ = self._run_rank(monkeypatch, rank=2)
        assert pipe.prepare_chunks_calls == 0
        assert pipe.denoise_calls == 1
        assert pipe.scheduler_calls == 1
        assert pipe.decode_calls == 1  # last rank decodes the deepest slot
        assert pipe.prefetch_calls == 1

    def test_first_runs_on_every_rank_without_prefetch(self, monkeypatch):
        # Chunk 0's first micro-step: all ranks run prepare_first_chunk, and the
        # steady hooks (incl. prefetch) are skipped entirely.
        runner = _make_runner(pp_size=3, num_steps=2, rank=1)
        _patch_runtime(monkeypatch, runner)
        req = _make_micro_request(num_inference_steps=2, num_chunks=2)
        out = DiffusionModelRunner.execute_micro_step(
            runner,
            _make_pp_micro_scheduler_output(
                req=req, per_rank=[[0, None], [0, None], [0, None]], is_last=False
            ),
        )
        assert out.finished is False
        assert runner.pipeline.first_chunk_calls == 1
        assert runner.pipeline.prepare_chunks_calls == 0
        assert runner.pipeline.denoise_calls == 0
        assert runner.pipeline.scheduler_calls == 0
        assert runner.pipeline.prefetch_calls == 0


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class TestWorker:
    """DiffusionWorker.execute_micro_step"""

    def test_delegates_to_model_runner(self):
        worker = object.__new__(DiffusionWorker)
        expected = RunnerOutput(request_id="req-1")
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
        worker.model_runner = SimpleNamespace(execute_micro_step=lambda _: RunnerOutput(request_id="req-1"))
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
        worker.model_runner = SimpleNamespace(execute_micro_step=lambda _: RunnerOutput(request_id="req-1"))
        worker._get_profiler = lambda: None

        with pytest.raises(ValueError, match="does not support LoRA"):
            DiffusionWorker.execute_micro_step(worker, scheduler_output)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class TestExecutor:
    """MultiprocDiffusionExecutor.execute_micro_step collects rank-0's reply."""

    def test_passes_through_runner_output(self, mocker: MockerFixture):
        executor = object.__new__(MultiprocDiffusionExecutor)
        executor._ensure_open = lambda: None
        executor.od_config = SimpleNamespace(reply_rank=0)
        expected = RunnerOutput(request_id="req-1", finished=True)
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
        executor.od_config = SimpleNamespace(reply_rank=0)
        executor.collective_rpc = mocker.Mock(return_value="not a runner output")

        sched_output = _make_micro_scheduler_output(req=_make_micro_request())
        with pytest.raises(RuntimeError, match="Unexpected response type"):
            MultiprocDiffusionExecutor.execute_micro_step(executor, sched_output)
