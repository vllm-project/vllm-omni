"""Temporal-pipeline-parallel scheduler for streaming chunked diffusion."""

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
class _Progress:
    request_id: str
    pp_size: int
    num_chunks: int
    num_steps: int
    next_admit_idx: int = 1  # chunk 0 is consumed by the first-chunk micro-step
    num_decoded: int = 0
    first_pending: bool = True
    is_first: bool = False

    layout: list[list[int | None]] = field(default_factory=list)  # [rank][step] = chunk_idx or None


class StreamBatchScheduler(_BaseScheduler):
    """Temporal-PP scheduler driving chunked-streaming diffusion requests.

    Per micro-step:
      1. Promote waiting requests (handled by the base class).
      2. Drain rank N-1: circulate non-finished chunks back to rank 0.
      3. Shift per-rank queues by one (rank r <- rank r-1).
      4. Rank 0 admits the next chunk (one per micro-step).
      5. Emit per-rank assignment to every RankTask.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pp_size: int = 1
        self._progress: dict[str, _Progress] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        super().initialize(od_config)
        self.pp_size = od_config.parallel_config.pipeline_parallel_size
        # TODO: support multiple requests
        self.max_num_running_reqs = 1

    def _reset_scheduler_state(self) -> None:
        self._progress.clear()

    def _pop_extra_request_state(self, request_id: str) -> None:
        self._progress.pop(request_id, None)

    # ── Request admission ──────────────────────────────────────────────────

    def add_request(self, request: OmniDiffusionRequest) -> str:
        sampling = request.sampling_params
        if sampling.chunk_frames is None or sampling.chunk_frames <= 0:
            raise ValueError(f"chunk_frames must be a positive int when stream_batch=True, got {sampling.chunk_frames}")
        if sampling.num_chunks is None or sampling.num_chunks <= 0:
            raise ValueError(f"num_chunks must be a positive int, got {sampling.num_chunks}")
        if sampling.num_inference_steps is None or sampling.num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be a positive int, got {sampling.num_inference_steps}")
        return super().add_request(request)

    # ── Scheduling ─────────────────────────────────────────────────────────

    def schedule(self) -> DiffusionSchedulerOutput:
        base_output = super().schedule()

        for new_req in base_output.scheduled_new_reqs:
            self._init_progress(new_req.request_id, new_req.req)

        for progress in self._progress.values():
            self._advance_chunk_pipeline(progress)

        if self._progress:
            base_output.assignment = self._build_assignment()

        logger.debug(
            "StreamBatchScheduler: %d running req(s), assignment=%s",
            len(self._running),
            base_output.assignment,
        )

        return base_output

    def _init_progress(self, request_id: str, req: OmniDiffusionRequest) -> None:
        sampling = req.sampling_params
        num_chunks = sampling.num_chunks
        num_steps = sampling.num_inference_steps

        self._progress[request_id] = _Progress(
            request_id=request_id,
            num_chunks=num_chunks,
            num_steps=num_steps,
            pp_size=self.pp_size,
            layout=[[None] * num_steps for _ in range(self.pp_size)],
        )

    def _advance_chunk_pipeline(self, progress: _Progress) -> None:
        pp = progress.pp_size
        ns = progress.num_steps

        # Chunk 0 is denoised completely in a single first-chunk micro-step
        progress.is_first = progress.first_pending
        if progress.first_pending:
            progress.first_pending = False
            return

        # The last rank's deepest slot finished its final step on the micro-step
        # that just ran; count its decode before rolling it off.
        tail = progress.layout[pp - 1]
        if tail[ns - 1] is not None:
            progress.num_decoded += 1

        admit = progress.next_admit_idx if progress.next_admit_idx < progress.num_chunks else None
        if admit is not None:
            progress.next_admit_idx += 1

        rolled: list[int | None] = [admit, *tail[: ns - 1]]
        for r in range(pp - 1, 0, -1):
            progress.layout[r] = progress.layout[r - 1]
        progress.layout[0] = rolled

    def _build_assignment(self) -> list[RankTask]:
        assert len(self._progress) <= 1  # TODO: support multiple requests
        assignment: list[RankTask] = []
        for progress in self._progress.values():
            ns = progress.num_steps
            if progress.is_first:
                # Chunk 0 at slot 0; runner detects it via ``0 in slot_chunks``.
                slot_chunks: list[int | None] = [0, *([None] * (ns - 1))]
                is_last = progress.num_chunks == 1
                for _ in range(self.pp_size):
                    assignment.append(RankTask(request_id=progress.request_id, slot_chunks=list(slot_chunks), is_last=is_last))
                continue
            # Final micro-step: last rank's deepest slot holds the last chunk.
            is_last = progress.layout[self.pp_size - 1][ns - 1] == progress.num_chunks - 1
            for r in range(self.pp_size):
                assignment.append(
                    RankTask(
                        request_id=progress.request_id,
                        slot_chunks=list(progress.layout[r]),
                        is_last=is_last,
                    )
                )
        return assignment

    # ── Output processing ──────────────────────────────────────────────────

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        request_ids = sched_output.scheduled_request_ids
        if not request_ids:
            return set()

        assert len(request_ids) == 1, "Multiple scheduled requests not supported"

        request_id = output.request_id

        assert request_id == request_ids[0]

        progress = self._progress.get(request_id)
        terminal: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}

        if progress is not None:
            err = output.result.error if output.result is not None else None
            if err is not None:
                terminal[request_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[request_id] = err
            elif output.finished:
                terminal[request_id] = DiffusionRequestStatus.FINISHED_COMPLETED

        return self._finalize_update_from_output(sched_output, terminal, terminal_errors)
