"""Temporal-pipeline-parallel scheduler for streaming chunked diffusion.

Each ``schedule()`` call corresponds to one micro-step. The pipeline is modeled
as ``pp_size`` per-rank chunk queues. At each schedule(), chunks at rank N-1
drain (finished -> Layout finished slice, otherwise -> circulating back to
rank 0), queues shift one rank, and rank 0 receives the circulating chunks
plus B fresh admits up to the request's output chunk target.
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
    Layout,
    RankTask,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import RunnerOutput

logger = init_logger(__name__)


@dataclass
class _InFlightChunk:
    chunk_idx: int
    steps_done: int = 0


@dataclass
class _Progress:
    sched_req_id: str
    pp_size: int
    chunk_frames: int
    num_chunks: int
    num_steps: int

    next_chunk_idx: int = 0
    batch_size: int = 0

    # chunks that will be processed by rank r at the current micro-step
    chunks_at: list[deque[_InFlightChunk]] = field(default_factory=list)
    # rank r's layout — constructed at rank 0 and shifted forward each step
    layouts_at: list[Layout] = field(default_factory=list)


@dataclass
class _SLOReqState:
    slo_fps: float
    max_batch: int
    chunk_frames: int
    batch_size: int = 1
    warmed_up: bool = False


class _SLOController:
    """Per-step B_target adjustment for per-request chunk admission."""

    SLACK_HEADROOM = 0.2

    def __init__(self) -> None:
        self._reqs: dict[str, _SLOReqState] = {}

    def register(
        self,
        sched_req_id: str,
        slo_fps: float | None,
        max_batch: int,
        chunk_frames: int,
    ) -> None:
        if slo_fps is None or slo_fps <= 0:
            return
        self._reqs[sched_req_id] = _SLOReqState(
            slo_fps=float(slo_fps),
            max_batch=max(1, max_batch),
            chunk_frames=max(1, chunk_frames),
        )

    def get_target(self, sched_req_id: str) -> int:
        st = self._reqs.get(sched_req_id)
        return st.batch_size if st is not None else 1

    def mark_warmed_up(self, sched_req_id: str) -> None:
        st = self._reqs.get(sched_req_id)
        if st is not None:
            st.warmed_up = True

    def observe(self, sched_req_id: str, latency_ns: int | None, b_current: int | None) -> None:
        st = self._reqs.get(sched_req_id)
        if st is None or not st.warmed_up or latency_ns is None or latency_ns <= 0 or b_current is None or b_current <= 0:
            return

        budget = (b_current * st.chunk_frames / st.slo_fps) * 1e9
        if latency_ns > budget:
            new_b = max(1, st.batch_size - 1)
        elif latency_ns < budget * (1.0 - self.SLACK_HEADROOM) and st.batch_size < st.max_batch:
            new_b = st.batch_size + 1
        else:
            return

        if new_b != st.batch_size:
            logger.info(
                "SLO[%s]: B_target %d -> %d (latency=%.2fms budget=%.2fms)",
                sched_req_id, st.batch_size, new_b, latency_ns / 1e6, budget / 1e6,
            )
            st.batch_size = new_b

    def unregister(self, sched_req_id: str) -> None:
        self._reqs.pop(sched_req_id, None)


class StreamBatchScheduler(_BaseScheduler):
    """Temporal-PP scheduler driving chunked-streaming diffusion requests.

    Per micro-step:
      1. Promote waiting requests (handled by the base class).
      2. Drain rank N-1: finished chunks -> finished slice in
         Layout, otherwise -> circulating back to rank 0.
      3. Shift per-rank queues by one (rank r <- rank r-1).
      4. Rank 0 = circulating + B fresh admits, where
         `B = min(B_target, output_chunks_remaining)`.
      5. Emit per-rank assignment with Layout attached to every RankTask.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pp_size: int = 1
        self._progress: dict[str, _Progress] = {}
        self._slo: _SLOController = _SLOController()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        super().initialize(od_config)
        self.pp_size = od_config.parallel_config.pipeline_parallel_size
        # TODO: support multiple requests
        self.max_num_running_reqs = 1

    def _reset_scheduler_state(self) -> None:
        self._progress.clear()
        self._slo = _SLOController()

    def _pop_extra_request_state(self, sched_req_id: str) -> None:
        self._progress.pop(sched_req_id, None)
        self._slo.unregister(sched_req_id)

    # ── Request admission ──────────────────────────────────────────────────

    def add_request(self, request: OmniDiffusionRequest) -> str:
        sampling = request.sampling_params
        if sampling.chunk_frames is None or sampling.chunk_frames <= 0:
            raise ValueError(
                f"chunk_frames must be a positive int when stream_batch=True, got {sampling.chunk_frames}"
            )
        if sampling.num_chunks is None or sampling.num_chunks <= 0:
            raise ValueError(f"num_chunks must be a positive int, got {sampling.num_chunks}")
        if sampling.num_inference_steps is None or sampling.num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be a positive int, got {sampling.num_inference_steps}"
            )
        return super().add_request(request)

    # ── Scheduling ─────────────────────────────────────────────────────────

    def schedule(self) -> DiffusionSchedulerOutput:
        base_output = super().schedule()

        for new_req in base_output.scheduled_new_reqs:
            self._init_progress(new_req.sched_req_id, new_req.req)

        for progress in self._progress.values():
            self._advance_chunk_pipeline(progress)

        if self._progress:
            base_output.assignment = self._build_assignment()

        logger.info(
            "StreamBatchScheduler schedule: %d running req(s), assignment=%s",
            len(self._running), base_output.assignment,
        )

        return base_output

    def _init_progress(self, sched_req_id: str, req: OmniDiffusionRequest) -> None:
        sampling = req.sampling_params
        chunk_frames = sampling.chunk_frames
        num_chunks = sampling.num_chunks
        num_steps = sampling.num_inference_steps

        self._progress[sched_req_id] = _Progress(
            sched_req_id=sched_req_id,
            chunk_frames=chunk_frames,
            num_chunks=num_chunks,
            num_steps=num_steps,
            pp_size=self.pp_size,
            chunks_at=[deque() for _ in range(self.pp_size)],
            layouts_at=[
                Layout(circulating_idxs=[], finished_idxs=[], new_idxs=[])
                for _ in range(self.pp_size)
            ],
        )

        self._slo.register(
            sched_req_id=sched_req_id,
            slo_fps=sampling.slo_fps,
            max_batch=sampling.slo_max_batch,
            chunk_frames=chunk_frames,
        )

        logger.debug(
            "StreamBatchScheduler initialized progress for %s "
            "(chunk_frames=%d, num_chunks=%d, num_steps=%d, slo_fps=%s, pp_size=%d)",
            sched_req_id, chunk_frames, num_chunks, num_steps, sampling.slo_fps, self.pp_size,
        )

    def _advance_chunk_pipeline(self, progress: _Progress) -> None:
        """Advance the per-rank queues and layouts by one micro-step."""

        pp = progress.pp_size

        # 1. Drain last rank from previous step
        finished_idxs: list[int] = []
        circulating: list[_InFlightChunk] = []
        last = progress.chunks_at[pp - 1]
        while last:
            chunk = last.popleft()
            chunk.steps_done += 1
            if chunk.steps_done >= progress.num_steps:
                finished_idxs.append(chunk.chunk_idx)
            else:
                circulating.append(chunk)

        # 2. Shift chunks and layouts: rank r receives what rank r-1 had
        for r in range(pp - 1, 0, -1):
            progress.chunks_at[r] = progress.chunks_at[r - 1]
            progress.layouts_at[r] = progress.layouts_at[r - 1]
        progress.chunks_at[0] = deque()

        # 3. Rank 0 = circulating + B fresh admits
        for chunk in circulating:
            progress.chunks_at[0].append(chunk)

        output_chunks_remaining = progress.num_chunks - progress.next_chunk_idx
        b_target = self._slo.get_target(progress.sched_req_id)
        batch_size = min(b_target, output_chunks_remaining)

        new_idxs: list[int] = []
        for _ in range(batch_size):
            chunk_idx = progress.next_chunk_idx
            progress.next_chunk_idx += 1
            progress.chunks_at[0].append(_InFlightChunk(chunk_idx=chunk_idx))
            new_idxs.append(chunk_idx)
        progress.batch_size = batch_size

        # 4. Set rank 0's layout for this step.
        progress.layouts_at[0] = Layout(
            circulating_idxs=[c.chunk_idx for c in circulating],
            finished_idxs=finished_idxs,
            new_idxs=new_idxs,
        )

        if finished_idxs:
            self._slo.mark_warmed_up(progress.sched_req_id)

    def _build_assignment(self) -> list[RankTask]:
        assert len(self._progress) <= 1 #TODO: support multiple requests
        assignment: list[RankTask] = []
        for progress in self._progress.values():
            for r in range(self.pp_size):
                queue = progress.chunks_at[r]
                assignment.append(RankTask(
                    sched_req_id=progress.sched_req_id,
                    chunk_indices=[c.chunk_idx for c in queue],
                    layout=progress.layouts_at[r],
                ))
        return assignment

    # ── Output processing ──────────────────────────────────────────────────

    def update_from_output(
        self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput
    ) -> set[str]:
        sched_req_ids = sched_output.scheduled_req_ids
        if not sched_req_ids:
            return set()
        
        assert len(sched_req_ids) == 1, "Multiple scheduled requests not supported"
        
        sched_req_id = output.req_id

        assert sched_req_id == sched_req_ids[0]

        progress = self._progress.get(sched_req_id)
        if progress is not None and output.micro_step_wall_ns is not None:
            self._slo.observe(sched_req_id, output.micro_step_wall_ns, progress.batch_size)

        terminal: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}

        if progress is not None:
            err = output.result.error if output.result is not None else None
            if err is not None:
                terminal[sched_req_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = err
            elif output.finished:
                terminal[sched_req_id] = DiffusionRequestStatus.FINISHED_COMPLETED

        return self._finalize_update_from_output(sched_output, terminal, terminal_errors)