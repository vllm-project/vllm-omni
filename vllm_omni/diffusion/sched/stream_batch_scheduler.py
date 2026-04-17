"""Temporal-pipeline-parallel scheduler for streaming chunked diffusion.

Each ``schedule()`` call corresponds to
one micro-step. At any micro-step, each PP rank processes a different
``(chunk, step_index)`` pair drawn from the active requests' in-flight
chunks. Chunks are admitted to rank 0 in order, propagate through ranks under
NCCL FIFO ordering, and exit at rank N-1 in the same order.
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
    step_index: int = 0         
    in_pipeline: bool = True     # currently flowing through ranks (True between admission and exit)
    entered_rank0_at: int = -1   # micro-step at which the chunk last entered rank 0


@dataclass
class _ChunkProgress:
    """Per-request chunk-level scheduling state."""
    sched_req_id: str
    num_chunks: int              # total chunks to produce for this request
    num_steps: int               # denoising steps per chunk
    chunks_admitted: int = 0
    chunks_completed: int = 0
    in_flight: list[_InFlightChunk] = field(default_factory=list)


class StreamBatchScheduler(_BaseScheduler):
    """Temporal-PP scheduler driving chunked-streaming diffusion requests.

    Per micro-step:
      1. Promote waiting requests up to ``max_num_running_reqs`` (handled by the base class).
      2. Re-admit at most one returning chunk to rank 0 (FIFO across all active requests).
      3. If rank 0 is still free and admission budget remains, admit a new chunk.
      4. Build the per-rank assignment table from in-pipeline chunks' positions.

    A chunk that entered rank 0 at micro-step ``m₀`` is at rank
    ``r = current_micro_step - m₀`` while ``0 ≤ r < pp_size``. After ``r ==
    pp_size - 1``, rank N-1's ``step_scheduler`` runs the ODE; the chunk's
    latents are sent back to rank 0; the chunk leaves the pipeline and may be
    re-admitted on the next micro-step (until it has run all
    ``num_steps`` denoising steps).
    """

    def __init__(self) -> None:
        super().__init__()
        self.pp_size: int = 1            # set in initialize()
        self.B: int = 1                  # intra-rank batch 
        self._global_micro_step: int = 0
        self._chunk_progress: dict[str, _ChunkProgress] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        super().initialize(od_config)
        self.pp_size = od_config.parallel_config.pipeline_parallel_size

    def _reset_scheduler_state(self) -> None:
        self._global_micro_step = 0
        self._chunk_progress.clear()

    def _pop_extra_request_state(self, sched_req_id: str) -> None:
        self._chunk_progress.pop(sched_req_id, None)

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
        # Base class promotes waiting → running and fills scheduled_new_reqs / step_id.
        base_output = super().schedule()

        # Initialize chunk-progress state for any newly promoted requests.
        for new_req in base_output.scheduled_new_reqs:
            self._init_chunk_progress(new_req.sched_req_id, new_req.req)

        # Re-admit a returning chunk; otherwise admit a new chunk if rank 0 is free.
        self._advance_chunk_pipeline()

        # Build the per-rank assignment from current in-pipeline chunks.
        if self._chunk_progress:
            base_output.per_rank_assignment = self._build_assignment()
        # else: no active request → executor sees per_rank_assignment=None and idles.

        self._global_micro_step += 1
        return base_output

    def _init_chunk_progress(self, sched_req_id: str, req: OmniDiffusionRequest) -> None:
        num_chunks = req.sampling_params.num_chunks
        num_steps = req.sampling_params.num_inference_steps
        assert num_chunks is not None and num_steps is not None  # validated in add_request()
        self._chunk_progress[sched_req_id] = _ChunkProgress(
            sched_req_id=sched_req_id,
            num_chunks=num_chunks,
            num_steps=num_steps,
        )
        logger.debug(
            "StreamBatchScheduler initialized chunk progress for %s "
            "(num_chunks=%d, num_steps=%d, pp_size=%d)",
            sched_req_id, num_chunks, num_steps, self.pp_size,
        )

    def _advance_chunk_pipeline(self) -> None:
        """Admit at most one chunk to rank 0 this micro-step.

        Re-admission of a returning chunk takes priority over admitting a new
        chunk so that FIFO order is preserved (an admitted chunk's latents
        always re-enter rank 0 before any later-admitted chunk's first entry).
        Admission order across requests follows ``_chunk_progress`` insertion
        order, which matches the order the base scheduler promoted them.
        """
        if not self._chunk_progress:
            return

        # 1. Try to re-admit a returning chunk (FIFO oldest-first across requests).
        for progress in self._chunk_progress.values():
            for chunk in progress.in_flight:
                if not chunk.in_pipeline and chunk.step_index < progress.num_steps:
                    chunk.in_pipeline = True
                    chunk.entered_rank0_at = self._global_micro_step
                    return  # rank 0 is now taken

        # 2. Otherwise admit a new chunk from the first request with budget.
        for progress in self._chunk_progress.values():
            if progress.chunks_admitted < progress.num_chunks:
                new_chunk = _InFlightChunk(
                    chunk_idx=progress.chunks_admitted,
                    step_index=0,
                    in_pipeline=True,
                    entered_rank0_at=self._global_micro_step,
                )
                progress.in_flight.append(new_chunk)
                progress.chunks_admitted += 1
                return

    def _build_assignment(self) -> list[RankTask | None]:
        assignment: list[RankTask | None] = [None] * self.pp_size
        for progress in self._chunk_progress.values():
            for chunk in progress.in_flight:
                if not chunk.in_pipeline:
                    continue
                r = self._global_micro_step - chunk.entered_rank0_at
                if 0 <= r < self.pp_size:
                    assert assignment[r] is None, (
                        f"two chunks would be assigned to rank {r} at micro-step "
                        f"{self._global_micro_step}: existing={assignment[r]}, "
                        f"new req={progress.sched_req_id} chunk_idx={chunk.chunk_idx}"
                    )
                    assignment[r] = RankTask(
                        sched_req_id=progress.sched_req_id,
                        chunk_idx=chunk.chunk_idx,
                        step_index=chunk.step_index,
                    )
        return assignment

    # ── Output processing ──────────────────────────────────────────────────

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        if not self._chunk_progress or sched_output.per_rank_assignment is None:
            return set()

        # Read per-chunk fields from RunnerOutput (set by execute_micro_step)
        # to advance chunk state — same pattern as StepScheduler reading
        # step_index / finished.
        terminal: dict[str, DiffusionRequestStatus] = {}

        if output.chunk_idx is not None:
            progress = self._chunk_progress.get(output.req_id)
            if progress is not None:
                chunk = self._find_chunk(progress, output.chunk_idx)
                if chunk is not None:
                    chunk.step_index = output.step_index
                    chunk.in_pipeline = False
                    if output.chunk_completed:
                        progress.in_flight = [
                            c for c in progress.in_flight if c.chunk_idx != output.chunk_idx
                        ]
                        progress.chunks_completed += 1

        if output.finished:
            terminal[output.req_id] = DiffusionRequestStatus.FINISHED_COMPLETED

        return self._finalize_update_from_output(sched_output, terminal)

    @staticmethod
    def _find_chunk(progress: _ChunkProgress, chunk_idx: int) -> _InFlightChunk | None:
        for chunk in progress.in_flight:
            if chunk.chunk_idx == chunk_idx:
                return chunk
        return None