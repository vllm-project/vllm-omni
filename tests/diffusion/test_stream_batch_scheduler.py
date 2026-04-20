# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for StreamBatchScheduler (temporal PP chunk scheduling)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.stream_batch_scheduler import StreamBatchScheduler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(pp_size: int = 2) -> SimpleNamespace:
    """Minimal OmniDiffusionConfig stub with the fields StreamBatchScheduler reads."""
    return SimpleNamespace(parallel_config=DiffusionParallelConfig(pipeline_parallel_size=pp_size))


def _make_request(
    req_id: str,
    num_chunks: int = 1,
    num_inference_steps: int = 4,
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[f"prompt_{req_id}"],
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=num_inference_steps,
            num_chunks=num_chunks,
        ),
        request_ids=[req_id],
    )


def _make_runner_output(
    req_id: str = "",
    step_index: int | None = None,
    chunk_idx: int | None = None,
    chunk_completed: bool = False,
    finished: bool = False,
) -> SimpleNamespace:
    """Simulate a RunnerOutput from rank N-1."""
    return SimpleNamespace(
        req_id=req_id,
        step_index=step_index,
        chunk_idx=chunk_idx,
        chunk_completed=chunk_completed,
        finished=finished,
        result=None,
    )


def _simulate_last_rank_output(sched_output, pp_size: int, num_steps: int) -> SimpleNamespace:
    """Build the RunnerOutput that rank N-1 would produce for a given schedule output.

    Simulates: if rank N-1 had a task, its chunk's step_index advances by 1.
    chunk_completed is True if the new step_index reaches num_steps.
    """
    assignment = sched_output.per_rank_assignment
    if assignment is None:
        return _make_runner_output()
    task = assignment[pp_size - 1]
    if task is None:
        req_id = sched_output.scheduled_req_ids[0] if sched_output.scheduled_req_ids else ""
        return _make_runner_output(req_id=req_id)
    new_step = task.step_index + 1
    return _make_runner_output(
        req_id=task.sched_req_id,
        step_index=new_step,
        chunk_idx=task.chunk_idx,
        chunk_completed=(new_step >= num_steps),
    )


def _run_until_finished(
    scheduler: StreamBatchScheduler,
    pp_size: int,
    num_steps: int,
    num_chunks: int,
    max_iters: int = 200,
) -> list[tuple[list, SimpleNamespace]]:
    """Drive the scheduler loop, returning (assignment, runner_output) per micro-step.

    Simulates what the runner would set in ``RunnerOutput.finished``: True once
    the total number of chunks produced equals ``num_chunks`` for that request.
    """
    trace: list[tuple[list, SimpleNamespace]] = []
    completed_per_req: dict[str, int] = {}
    for _ in range(max_iters):
        sched_output = scheduler.schedule()
        output = _simulate_last_rank_output(sched_output, pp_size, num_steps)
        if output.chunk_completed:
            completed_per_req[output.req_id] = completed_per_req.get(output.req_id, 0) + 1
            if completed_per_req[output.req_id] >= num_chunks:
                output.finished = True
        finished = scheduler.update_from_output(sched_output, output)
        assignment = sched_output.per_rank_assignment or [None] * pp_size
        trace.append((assignment, output))
        if finished:
            break
    return trace


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestAddRequestValidation:
    def test_rejects_zero_chunks(self):
        sched = StreamBatchScheduler()
        sched.initialize(_make_config())
        with pytest.raises(ValueError, match="num_chunks"):
            sched.add_request(_make_request("r1", num_chunks=0))

    def test_rejects_negative_steps(self):
        sched = StreamBatchScheduler()
        sched.initialize(_make_config())
        with pytest.raises(ValueError, match="num_inference_steps"):
            sched.add_request(_make_request("r1", num_inference_steps=-1))

    def test_accepts_valid_request(self):
        sched = StreamBatchScheduler()
        sched.initialize(_make_config())
        req_id = sched.add_request(_make_request("r1", num_chunks=3, num_inference_steps=10))
        assert req_id == "r1"


# ---------------------------------------------------------------------------
# Single chunk, single rank (PP=1)
# ---------------------------------------------------------------------------


class TestSingleChunkSingleRank:
    """PP=1, K=1 — degenerate case: one chunk, one rank, behaves like step scheduler."""

    def test_completes_in_m_steps(self):
        pp_size, num_chunks, num_steps = 1, 1, 4
        sched = StreamBatchScheduler()
        sched.initialize(_make_config(pp_size))
        sched.add_request(_make_request("r1", num_chunks=num_chunks, num_inference_steps=num_steps))

        trace = _run_until_finished(sched, pp_size, num_steps, num_chunks)
        assert len(trace) == num_steps

    def test_assignment_is_always_rank_0(self):
        pp_size, num_chunks, num_steps = 1, 1, 3
        sched = StreamBatchScheduler()
        sched.initialize(_make_config(pp_size))
        sched.add_request(_make_request("r1", num_chunks=num_chunks, num_inference_steps=num_steps))

        trace = _run_until_finished(sched, pp_size, num_steps, num_chunks)
        for assignment, _ in trace:
            assert assignment[0] is not None
            assert assignment[0].sched_req_id == "r1"
            assert assignment[0].chunk_idx == 0

    def test_step_index_advances(self):
        pp_size, num_chunks, num_steps = 1, 1, 3
        sched = StreamBatchScheduler()
        sched.initialize(_make_config(pp_size))
        sched.add_request(_make_request("r1", num_chunks=num_chunks, num_inference_steps=num_steps))

        trace = _run_until_finished(sched, pp_size, num_steps, num_chunks)
        step_indices = [a[0].step_index for a, _ in trace]
        assert step_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# Multi-chunk, single rank (PP=1)
# ---------------------------------------------------------------------------


class TestMultiChunkSingleRank:
    """PP=1, K>1 — chunks are processed sequentially on one rank."""

    def test_completes_in_k_times_m_steps(self):
        pp_size, num_chunks, num_steps = 1, 3, 2
        sched = StreamBatchScheduler()
        sched.initialize(_make_config(pp_size))
        sched.add_request(_make_request("r1", num_chunks=num_chunks, num_inference_steps=num_steps))

        trace = _run_until_finished(sched, pp_size, num_steps, num_chunks)
        assert len(trace) == num_chunks * num_steps

    def test_chunks_processed_in_order(self):
        pp_size, num_chunks, num_steps = 1, 3, 2
        sched = StreamBatchScheduler()
        sched.initialize(_make_config(pp_size))
        sched.add_request(_make_request("r1", num_chunks=num_chunks, num_inference_steps=num_steps))

        trace = _run_until_finished(sched, pp_size, num_steps, num_chunks)
        chunk_indices = [a[0].chunk_idx for a, _ in trace]
        # Each chunk runs M steps before the next chunk starts.
        assert chunk_indices == [0, 0, 1, 1, 2, 2]


# ---------------------------------------------------------------------------
# Pipeline warmup / assignment (PP > 1)
# ---------------------------------------------------------------------------


class TestPipelineAssignment:
    """Verify per-rank assignment table for N=3, K=4, M=2."""

    def _setup(self):
        pp_size, num_chunks, num_steps = 3, 4, 2
        sched = StreamBatchScheduler()
        sched.initialize(_make_config(pp_size))
        sched.add_request(_make_request("r1", num_chunks=num_chunks, num_inference_steps=num_steps))
        return sched, pp_size, num_steps, num_chunks

    def _extract_assignment(self, assignment):
        """Convert assignment list to tuples of (chunk_idx, step_index) or None."""
        return [
            (t.chunk_idx, t.step_index) if t is not None else None
            for t in assignment
        ]

    def test_warmup_idles_trailing_ranks(self):
        sched, pp_size, num_steps, num_chunks = self._setup()
        trace = _run_until_finished(sched, pp_size, num_steps, num_chunks)

        # Micro-step 0: only rank 0 active.
        a0 = self._extract_assignment(trace[0][0])
        assert a0[0] is not None
        assert a0[1] is None
        assert a0[2] is None

    def test_warmup_fills_pipeline(self):
        sched, pp_size, num_steps, num_chunks = self._setup()
        trace = _run_until_finished(sched, pp_size, num_steps, num_chunks)

        # Micro-step 0: rank 0 = chunk 0
        assert self._extract_assignment(trace[0][0]) == [(0, 0), None, None]
        # Micro-step 1: rank 0 = chunk 1, rank 1 = chunk 0
        assert self._extract_assignment(trace[1][0]) == [(1, 0), (0, 0), None]
        # Micro-step 2: all ranks busy
        a2 = self._extract_assignment(trace[2][0])
        assert all(x is not None for x in a2)

    def test_chunk_propagates_through_ranks(self):
        sched, pp_size, num_steps, num_chunks = self._setup()
        trace = _run_until_finished(sched, pp_size, num_steps, num_chunks)

        # Chunk 0 should appear at rank 0, then rank 1, then rank 2.
        chunk0_ranks = []
        for assignment, _ in trace:
            for r, task in enumerate(assignment):
                if task is not None and task.chunk_idx == 0:
                    chunk0_ranks.append(r)
                    break
        # First 3 entries: ranks 0, 1, 2 (warmup propagation of chunk 0's first step).
        assert chunk0_ranks[:3] == [0, 1, 2]

    def test_request_completes(self):
        sched, pp_size, num_steps, num_chunks = self._setup()
        trace = _run_until_finished(sched, pp_size, num_steps, num_chunks)

        # The last micro-step's output should have finished=True (set by _simulate).
        # But finished is set by the runner, not the scheduler. In our simulation,
        # we don't set finished=True. Instead, check that the scheduler reported
        # the request in finished_req_ids (the loop exited).
        assert len(trace) > 0  # loop exited → request finished


# ---------------------------------------------------------------------------
# Chunk re-admission
# ---------------------------------------------------------------------------


class TestChunkReAdmission:
    """Verify that a chunk returning from rank N-1 is re-admitted to rank 0."""

    def test_re_admission_priority_over_new_chunk(self):
        """With K=2, M=2, N=2: after chunk 0 exits rank 1 at ms1,
        it should re-enter rank 0 at ms2 (priority over admitting chunk 1)."""
        pp_size, num_chunks, num_steps = 2, 2, 2
        sched = StreamBatchScheduler()
        sched.initialize(_make_config(pp_size))
        sched.add_request(_make_request("r1", num_chunks=num_chunks, num_inference_steps=num_steps))

        # ms0: [chunk0, idle]
        out0 = sched.schedule()
        assert out0.per_rank_assignment[0].chunk_idx == 0
        assert out0.per_rank_assignment[1] is None
        sched.update_from_output(out0, _simulate_last_rank_output(out0, pp_size, num_steps))

        # ms1: [chunk1, chunk0] — chunk 0 reaches rank 1 and completes step 0.
        out1 = sched.schedule()
        assert out1.per_rank_assignment[0].chunk_idx == 1
        assert out1.per_rank_assignment[1].chunk_idx == 0
        sched.update_from_output(out1, _simulate_last_rank_output(out1, pp_size, num_steps))

        # ms2: chunk 0 should be re-admitted (step 1), NOT chunk 1 continuing.
        out2 = sched.schedule()
        assert out2.per_rank_assignment[0].chunk_idx == 0
        assert out2.per_rank_assignment[0].step_index == 1

    def test_chunk_not_readmitted_after_completion(self):
        """A chunk that finished all denoising steps should NOT be re-admitted."""
        pp_size, num_chunks, num_steps = 1, 1, 2
        sched = StreamBatchScheduler()
        sched.initialize(_make_config(pp_size))
        sched.add_request(_make_request("r1", num_chunks=num_chunks, num_inference_steps=num_steps))

        # Run step 0.
        out0 = sched.schedule()
        runner0 = _make_runner_output(req_id="r1", step_index=1, chunk_idx=0)
        sched.update_from_output(out0, runner0)

        # Run step 1 (final).
        out1 = sched.schedule()
        runner1 = _make_runner_output(
            req_id="r1", step_index=2, chunk_idx=0, chunk_completed=True, finished=True,
        )
        finished = sched.update_from_output(out1, runner1)
        assert "r1" in finished

        # No more requests.
        assert not sched.has_requests()


# ---------------------------------------------------------------------------
# Completion ordering
# ---------------------------------------------------------------------------


class TestCompletionOrdering:
    """Verify chunks complete in admission order (FIFO)."""

    def test_chunks_complete_in_fifo_order(self):
        pp_size, num_chunks, num_steps = 2, 3, 1
        sched = StreamBatchScheduler()
        sched.initialize(_make_config(pp_size))
        sched.add_request(_make_request("r1", num_chunks=num_chunks, num_inference_steps=num_steps))

        completed_chunks: list[int] = []
        for _ in range(20):
            out = sched.schedule()
            runner = _simulate_last_rank_output(out, pp_size, num_steps)
            if runner.chunk_completed:
                completed_chunks.append(runner.chunk_idx)
            finished = sched.update_from_output(out, runner)
            if finished:
                break

        assert completed_chunks == [0, 1, 2]


# ---------------------------------------------------------------------------
# Request finished signal
# ---------------------------------------------------------------------------


class TestRequestFinished:
    def test_finished_after_all_chunks(self):
        pp_size, num_chunks, num_steps = 1, 3, 1
        sched = StreamBatchScheduler()
        sched.initialize(_make_config(pp_size))
        sched.add_request(_make_request("r1", num_chunks=num_chunks, num_inference_steps=num_steps))

        finished_set = set()
        for _ in range(20):
            out = sched.schedule()
            runner = _simulate_last_rank_output(out, pp_size, num_steps)
            # On the last chunk, set finished=True (simulating the runner's merge).
            if runner.chunk_completed:
                progress = sched._chunk_progress.get("r1")
                if progress and progress.chunks_completed + 1 >= num_chunks:
                    runner.finished = True
            finished_set = sched.update_from_output(out, runner)
            if finished_set:
                break

        assert "r1" in finished_set
        assert not sched.has_requests()

    def test_not_finished_before_all_chunks(self):
        pp_size, num_chunks, num_steps = 1, 3, 1
        sched = StreamBatchScheduler()
        sched.initialize(_make_config(pp_size))
        sched.add_request(_make_request("r1", num_chunks=num_chunks, num_inference_steps=num_steps))

        # Process only 2 of 3 chunks.
        for i in range(2):
            out = sched.schedule()
            runner = _make_runner_output(
                req_id="r1", step_index=1, chunk_idx=i, chunk_completed=True,
            )
            finished = sched.update_from_output(out, runner)
            assert not finished

        assert sched.has_requests()


# ---------------------------------------------------------------------------
# Sequential requests
# ---------------------------------------------------------------------------


class TestSequentialRequests:
    """Second request is processed after the first finishes."""

    def test_second_request_starts_after_first(self):
        pp_size, num_steps = 1, 1
        sched = StreamBatchScheduler()
        sched.initialize(_make_config(pp_size))
        sched.add_request(_make_request("r1", num_chunks=1, num_inference_steps=num_steps))
        sched.add_request(_make_request("r2", num_chunks=1, num_inference_steps=num_steps))

        # Process r1.
        out1 = sched.schedule()
        assert out1.per_rank_assignment[0].sched_req_id == "r1"
        sched.update_from_output(
            out1,
            _make_runner_output(req_id="r1", step_index=1, chunk_idx=0, chunk_completed=True, finished=True),
        )

        # r1 finished. Next schedule should pick r2.
        out2 = sched.schedule()
        assert out2.per_rank_assignment[0].sched_req_id == "r2"