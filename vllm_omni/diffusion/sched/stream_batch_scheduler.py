"""Temporal-pipeline-parallel scheduler for streaming chunked diffusion.

Each ``schedule()`` call corresponds to one micro-step. At any micro-step,
each PP rank processes the chunks at the denoising step ``r = current -
entered_rank0_at`` from the active requests' in-flight chunks. Chunks are
admitted to rank 0 in order, propagate through ranks under NCCL FIFO
ordering, and exit at rank N-1 in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import _BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    RankTask,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import RunnerOutput

logger = init_logger(__name__)


@dataclass
class _InFlightChunk:
    """One chunk of an active request, tracked through the temporal pipeline."""

    chunk_idx: int
    is_active: bool = True
    is_completed: bool = False
    entered_rank0_at: int = -1


@dataclass
class _ChunkProgress:
    """Per-request chunk-level scheduling state."""
    sched_req_id: str
    num_chunks: int
    num_steps: int
    chunks_admitted: int = 0
    in_flight: list[_InFlightChunk] = field(default_factory=list)


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
    """AIMD controller for the stream batch size, tracked per request.

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
                    logger.info(f"SLO[{sched_req_id}]: halving batch_size {st.batch_size} -> {new_b} (ema={ema/1e6:.2}ms budget={budget/1e6:.2}ms)")
                st.batch_size = new_b
                st.violation_streak = 0
            return

        st.violation_streak = 0
        headroom_ratio = (budget - ema) / budget
        if headroom_ratio >= self.SLACK_THRESHOLD_RATIO:
            st.slack_streak += 1
            if st.slack_streak >= self.SLACK_STREAK_TARGET and st.batch_size < st.max_batch:
                st.batch_size += 1
                logger.info(f"SLO[{sched_req_id}]: increasing batch_size -> {st.batch_size} (ema={ema/1e6:.2}ms budget={budget/1e6:.2}ms)")
                st.slack_streak = 0
        else:
            st.slack_streak = 0


class StreamBatchScheduler(_BaseScheduler):
    """Temporal-PP scheduler driving chunked-streaming diffusion requests.

    Per micro-step:
      1. Promote waiting requests (handled by the base class).
      2. Admit returning chunks to rank 0 first (FIFO across active requests),
         then admit fresh chunks, until ``batch_size`` chunks have entered
         rank 0 this micro-step or no admittable chunks remain.
      3. Build the per-rank assignment table from in-pipeline chunks'
         positions ``r = current_micro_step - entered_rank0_at``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pp_size: int = 1
        self._global_micro_step: int = 0
        self._chunk_progress: dict[str, _ChunkProgress] = {}
        self._slo: _SLOController = _SLOController()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        super().initialize(od_config)
        self.pp_size = od_config.parallel_config.pipeline_parallel_size

    def _reset_scheduler_state(self) -> None:
        self._global_micro_step = 0
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

        self._advance_chunk_pipeline()

        if self._chunk_progress:
            base_output.per_rank_assignment = self._build_assignment()

        self._global_micro_step += 1
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
        )

        chunk_frames = max(1, sampling.num_frames)
        self._slo.register(
            sched_req_id=sched_req_id,
            slo_fps=sampling.slo_fps,
            max_batch=sampling.slo_max_batch,
            ema_alpha=sampling.slo_ema_alpha,
            chunk_frames=chunk_frames,
        )
        
        logger.debug(f"""StreamBatchScheduler initialized chunk progress for {sched_req_id}
        (num_chunks={num_chunks}, num_steps={num_steps}, chunk_frames={chunk_frames}, slo_fps={sampling.slo_fps}, pp_size={self.pp_size})""")

    def _advance_chunk_pipeline(self) -> None:
        """Admit returning + new chunks to rank 0 this micro-step."""
        if not self._chunk_progress:
            return

        m = self._global_micro_step

        # 1. Re-admit every returning chunk.
        for progress in self._chunk_progress.values():
            for chunk in progress.in_flight:
                if not chunk.is_active:
                    chunk.is_active = True
                    chunk.entered_rank0_at = m

        # 2. Admit up to ``B_req`` fresh chunks per request.
        for progress in self._chunk_progress.values():
            budget = self._slo.batch_size(progress.sched_req_id)
            admitted = 0
            while (
                progress.chunks_admitted < progress.num_chunks
                and admitted < budget
            ):
                progress.in_flight.append(_InFlightChunk(
                    chunk_idx=progress.chunks_admitted,
                    entered_rank0_at=m,
                ))
                progress.chunks_admitted += 1
                admitted += 1

    def _build_assignment(self) -> list[list[RankTask]]:
        assignment: list[list[RankTask]] = [[] for _ in range(self.pp_size)]
        for progress in self._chunk_progress.values():
            for chunk in progress.in_flight:
                if not chunk.is_active:
                    continue
                r = self._global_micro_step - chunk.entered_rank0_at
                if 0 <= r < self.pp_size:
                    assignment[r].append(RankTask(
                        sched_req_id=progress.sched_req_id,
                        chunk_idx=chunk.chunk_idx,
                    ))
        return assignment

    # ── Output processing ──────────────────────────────────────────────────

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        if not self._chunk_progress or sched_output.per_rank_assignment is None:
            return set()

        per_task = [output] + list(output.extra_task_outputs or [])

        if output.micro_step_wall_ns is not None:
            self._slo.observe(output.req_id, output.micro_step_wall_ns)

        terminal: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}

        for task_out in per_task:
            progress = self._chunk_progress.get(task_out.req_id)
            if progress is None:
                continue
            err = task_out.result.error if task_out.result is not None else None
            if err is not None:
                terminal[task_out.req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[task_out.req_id] = err
                continue
            chunk = self._find_chunk(progress, task_out.chunk_idx) if task_out.chunk_idx is not None else None
            if chunk is not None:
                chunk.is_completed = task_out.chunk_completed
            if task_out.finished:
                terminal[task_out.req_id] = DiffusionRequestStatus.FINISHED_COMPLETED

        # Roll last-rank chunks off the pipeline / mark them inactive.
        for last_task in (sched_output.per_rank_assignment[-1] if sched_output.per_rank_assignment else []):
            progress = self._chunk_progress.get(last_task.sched_req_id)
            if progress is None:
                continue
            last_chunk = self._find_chunk(progress, last_task.chunk_idx)
            if last_chunk is None:
                continue
            if last_chunk.is_completed:
                progress.in_flight = [
                    c for c in progress.in_flight if c.chunk_idx != last_chunk.chunk_idx
                ]
            else:
                last_chunk.is_active = False

        return self._finalize_update_from_output(sched_output, terminal, terminal_errors)

    @staticmethod
    def _find_chunk(progress: _ChunkProgress, chunk_idx: int) -> _InFlightChunk | None:
        for chunk in progress.in_flight:
            if chunk.chunk_idx == chunk_idx:
                return chunk
        return None