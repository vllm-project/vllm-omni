"""Temporal-pipeline-parallel scheduler for streaming chunked diffusion.

Each ``schedule()`` call corresponds to one micro-step. The pipeline is modeled
as ``pp_size`` per-rank chunk queues plus a transient ``returning`` queue.
At each schedule(), chunks at rank N-1 drain (finished -> finished_head,
otherwise -> returning), queues shift one rank, and rank 0 receives the
returning chunks plus B fresh admits.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import _BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    Rank0Layout,
    RankTask,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import RunnerOutput

logger = init_logger(__name__)


@dataclass
class _InFlightChunk:
    """One chunk of an active request currently in the pipeline."""

    chunk_idx: int
    steps_done: int = 0


@dataclass
class _ChunkProgress:
    """Per-request chunk-level scheduling state."""

    sched_req_id: str
    num_chunks: int
    num_steps: int
    pp_size: int
    chunks_admitted: int = 0
    # chunks_at[r] = chunks that will be processed by rank r at the current step.
    chunks_at: list[deque[_InFlightChunk]] = field(default_factory=list)
    returning: deque[_InFlightChunk] = field(default_factory=deque)


@dataclass
class _SLOReqState:
    """Per-request SLO state, owned by ``_SLOController``."""

    slo_fps: float
    max_batch: int
    ema_alpha: float
    chunk_frames: int
    batch_size: int = 1
    latency_ema_ns: float | None = None
    slack_streak: int = 0
    violation_streak: int = 0


class _SLOController:
    """AIMD controller for the stream admission rate B, tracked per request.

    Driven by per-micro-step wall-clock latency observations on rank 0.

    Maintains an EMA of micro-step latency; halves B on sustained budget violations and
    increments B by 1 on sustained slack.
    """

    SLACK_THRESHOLD_RATIO = 0.25
    SLACK_STREAK_TARGET = 4
    VIOLATION_STREAK_TARGET = 2

    def __init__(self) -> None:
        self._reqs: dict[str, _SLOReqState] = {}

    def register(
        self,
        sched_req_id: str,
        slo_fps: float | None,
        max_batch: int,
        ema_alpha: float,
        chunk_frames: int,
    ) -> None:
        if slo_fps is None or slo_fps <= 0:
            return
        self._reqs[sched_req_id] = _SLOReqState(
            slo_fps=float(slo_fps),
            max_batch=max(1, max_batch),
            ema_alpha=ema_alpha,
            chunk_frames=max(1, chunk_frames),
        )

    def unregister(self, sched_req_id: str) -> None:
        self._reqs.pop(sched_req_id, None)

    def batch_size(self, sched_req_id: str) -> int:
        st = self._reqs.get(sched_req_id)
        return st.batch_size if st is not None else 1

    def observe(self, sched_req_id: str, latency_ns: int | None) -> None:
        st = self._reqs.get(sched_req_id)
        if st is None or latency_ns is None or latency_ns <= 0:
            return

        if st.latency_ema_ns is None:
            st.latency_ema_ns = float(latency_ns)
        else:
            a = st.ema_alpha
            st.latency_ema_ns = a * float(latency_ns) + (1.0 - a) * st.latency_ema_ns

        budget = (st.batch_size * st.chunk_frames / st.slo_fps) * 1e9
        ema = st.latency_ema_ns

        if ema > budget:
            st.violation_streak += 1
            st.slack_streak = 0
            if st.violation_streak >= self.VIOLATION_STREAK_TARGET:
                new_b = max(1, st.batch_size // 2)
                if new_b != st.batch_size:
                    logger.info(
                        f"SLO[{sched_req_id}]: halving B {st.batch_size} -> {new_b} "
                        f"(ema={ema/1e6:.2f}ms budget={budget/1e6:.2f}ms)"
                    )
                st.batch_size = new_b
                st.violation_streak = 0
            return

        st.violation_streak = 0
        headroom_ratio = (budget - ema) / budget
        if headroom_ratio >= self.SLACK_THRESHOLD_RATIO:
            st.slack_streak += 1
            if st.slack_streak >= self.SLACK_STREAK_TARGET and st.batch_size < st.max_batch:
                st.batch_size += 1
                logger.info(
                    f"SLO[{sched_req_id}]: B -> {st.batch_size} "
                    f"(ema={ema/1e6:.2f}ms budget={budget/1e6:.2f}ms)"
                )
                st.slack_streak = 0
        else:
            st.slack_streak = 0


class StreamBatchScheduler(_BaseScheduler):
    """Temporal-PP scheduler driving chunked-streaming diffusion requests.

    Per micro-step:
      1. Promote waiting requests (handled by the base class).
      2. Drain rank N-1 of last step: finished chunks -> finished_head (decode
         layout for rank 0), others -> returning queue.
      3. Shift per-rank queues by one (rank r <- rank r-1).
      4. Rank 0 = returning + B fresh admits (unconditional re-admit).
      5. Emit per-rank assignment and the per-request Rank0Layout.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pp_size: int = 1
        self._chunk_progress: dict[str, _ChunkProgress] = {}
        self._slo: _SLOController = _SLOController()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        super().initialize(od_config)
        self.pp_size = od_config.parallel_config.pipeline_parallel_size

    def _reset_scheduler_state(self) -> None:
        self._chunk_progress.clear()
        self._slo = _SLOController()

    def _pop_extra_request_state(self, sched_req_id: str) -> None:
        self._chunk_progress.pop(sched_req_id, None)
        self._slo.unregister(sched_req_id)

    # ── Request admission ──────────────────────────────────────────────────

    def add_request(self, request: OmniDiffusionRequest) -> str:
        num_chunks = request.sampling_params.num_chunks
        num_steps = request.sampling_params.num_inference_steps
        if num_chunks is None or num_chunks <= 0:
            raise ValueError(f"num_chunks must be a positive int, got {num_chunks!r}")
        if num_steps is None or num_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be a positive int, got {num_steps!r}"
            )
        return super().add_request(request)

    # ── Scheduling ─────────────────────────────────────────────────────────

    def schedule(self) -> DiffusionSchedulerOutput:
        base_output = super().schedule()

        for new_req in base_output.scheduled_new_reqs:
            self._init_chunk_progress(new_req.sched_req_id, new_req.req)

        rank0_layouts: dict[str, Rank0Layout] = {}
        for progress in self._chunk_progress.values():
            rank0_layouts[progress.sched_req_id] = self._advance_chunk_pipeline_for(progress)

        if self._chunk_progress:
            base_output.per_rank_assignment = self._build_assignment()
            base_output.rank0_layouts = rank0_layouts

        return base_output

    def _init_chunk_progress(self, sched_req_id: str, req: OmniDiffusionRequest) -> None:
        sampling = req.sampling_params
        num_chunks = sampling.num_chunks
        num_steps = sampling.num_inference_steps
        assert num_chunks is not None and num_steps is not None
        self._chunk_progress[sched_req_id] = _ChunkProgress(
            sched_req_id=sched_req_id,
            num_chunks=num_chunks,
            num_steps=num_steps,
            pp_size=self.pp_size,
            chunks_at=[deque() for _ in range(self.pp_size)],
        )

        chunk_frames = max(1, sampling.num_frames)
        self._slo.register(
            sched_req_id=sched_req_id,
            slo_fps=sampling.slo_fps,
            max_batch=sampling.slo_max_batch,
            ema_alpha=sampling.slo_ema_alpha,
            chunk_frames=chunk_frames,
        )

        logger.debug(
            "StreamBatchScheduler initialized chunk progress for %s "
            "(num_chunks=%d, num_steps=%d, chunk_frames=%d, slo_fps=%s, pp_size=%d)",
            sched_req_id, num_chunks, num_steps, chunk_frames, sampling.slo_fps, self.pp_size,
        )

    def _advance_chunk_pipeline_for(self, progress: _ChunkProgress) -> Rank0Layout:
        """Advance the per-rank queues by one micro-step and return rank 0's layout."""

        pp = progress.pp_size

        # 1. Drain last rank from previous step
        finished_idxs: list[int] = []
        n_finished = 0
        n_circulating = 0
        last = progress.chunks_at[pp - 1]
        while last:
            chunk = last.popleft()
            chunk.steps_done += 1
            if chunk.steps_done >= progress.num_steps:
                finished_idxs.append(chunk.chunk_idx)
                n_finished += 1
            else:
                progress.returning.append(chunk)
                n_circulating += 1

        # 2. Shift: rank r receives what rank r-1 had
        for r in range(pp - 1, 0, -1):
            progress.chunks_at[r] = progress.chunks_at[r - 1]
        progress.chunks_at[0] = deque()

        # 3. Rank 0 = returning + B fresh admits
        while progress.returning:
            progress.chunks_at[0].append(progress.returning.popleft())

        new_idxs: list[int] = []
        budget = self._slo.batch_size(progress.sched_req_id)
        admitted = 0
        while admitted < budget and progress.chunks_admitted < progress.num_chunks:
            idx = progress.chunks_admitted
            progress.chunks_at[0].append(_InFlightChunk(chunk_idx=idx))
            progress.chunks_admitted += 1
            new_idxs.append(idx)
            admitted += 1

        return Rank0Layout(
            n_finished=n_finished,
            n_circulating=n_circulating,
            n_new=len(new_idxs),
            finished_idxs=finished_idxs,
            new_idxs=new_idxs,
        )

    def _build_assignment(self) -> list[RankTask | None]:
        assignment: list[RankTask | None] = [None] * self.pp_size
        for progress in self._chunk_progress.values():
            for r in range(self.pp_size):
                queue = progress.chunks_at[r]
                if not queue:
                    continue
                indices = [c.chunk_idx for c in queue]
                if assignment[r] is None:
                    assignment[r] = RankTask(
                        sched_req_id=progress.sched_req_id,
                        chunk_indices=indices,
                    )
                else:
                    assignment[r].chunk_indices.extend(indices)
        return assignment

    # ── Output processing ──────────────────────────────────────────────────

    def update_from_output(
        self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput
    ) -> set[str]:
        if not self._chunk_progress:
            return set()

        if output.micro_step_wall_ns is not None:
            self._slo.observe(output.req_id, output.micro_step_wall_ns)

        terminal: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}

        progress = self._chunk_progress.get(output.req_id)
        if progress is not None:
            err = output.result.error if output.result is not None else None
            if err is not None:
                terminal[output.req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[output.req_id] = err
            elif output.finished:
                terminal[output.req_id] = DiffusionRequestStatus.FINISHED_COMPLETED

        return self._finalize_update_from_output(sched_output, terminal, terminal_errors)