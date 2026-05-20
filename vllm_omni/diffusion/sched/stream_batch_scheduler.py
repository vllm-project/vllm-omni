"""Temporal-pipeline-parallel scheduler for streaming chunked diffusion.

Each ``schedule()`` call corresponds to one micro-step. The pipeline is modeled
as ``pp_size`` per-rank chunk queues plus a transient ``returning`` queue.
At each schedule(), chunks at rank N-1 drain (finished -> Rank0Layout finished
slice, otherwise -> returning), queues shift one rank, and rank 0 receives the
returning chunks plus B fresh admits drawn from the source video frames in
``prompts[0]["multi_modal_data"]["video"]``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
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


def _video_frame_count(request: OmniDiffusionRequest) -> int:
    """Number of frames currently in ``prompts[0]["multi_modal_data"]["video"]``."""
    if not request.prompts:
        return 0
    prompt = request.prompts[0]
    if isinstance(prompt, str):
        return 0
    multi_modal = prompt.get("multi_modal_data") or {}
    video = multi_modal.get("video")
    if video is None:
        return 0
    if isinstance(video, torch.Tensor):
        return int(video.shape[0])
    if isinstance(video, list):
        return len(video)
    raise TypeError(
        f"multi_modal_data['video'] must be a Tensor or list of Tensors; got {type(video).__name__}."
    )


@dataclass
class _InFlightChunk:
    chunk_idx: int
    steps_done: int = 0


@dataclass
class _Progress:
    sched_req_id: str
    pp_size: int
    chunk_frames: int
    num_frames: int
    num_steps: int

    frames_committed: int = 0
    next_chunk_idx: int = 0
    batch_size: int = 0
    
    # chunks that will be processed by rank r at the current micro-step
    chunks_at: list[deque[_InFlightChunk]] = field(default_factory=list) 

    @property
    def output_chunks_target(self) -> int:
        return self.num_frames // self.chunk_frames


@dataclass
class _SLOReqState:
    slo_fps: float
    max_batch: int
    chunk_frames: int
    batch_size: int = 1


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

    def observe(self, sched_req_id: str, latency_ns: int | None, b_current: int | None) -> None:
        st = self._reqs.get(sched_req_id)
        if st is None or latency_ns is None or latency_ns <= 0 or b_current is None or b_current <= 0:
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
         Rank0Layout, otherwise -> returning queue.
      3. Shift per-rank queues by one (rank r <- rank r-1).
      4. Rank 0 = returning + B fresh admits, where
         `B = min(B_target, queue_chunks_available, output_chunks_remaining)`.
      5. Emit per-rank assignment and the per-request Rank0Layout. Flip req state
         RUNNING -> BLOCKED when admission is starved on input.
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
        if sampling.num_frames is None or sampling.num_frames <= 0:
            raise ValueError(f"num_frames must be a positive int, got {sampling.num_frames}")
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

        rank0_layouts: dict[str, Rank0Layout] = {}
        for progress in self._progress.values():
            rank0_layouts[progress.sched_req_id] = self._advance_chunk_pipeline(progress)

        if self._progress:
            base_output.assignment = self._build_assignment()
            base_output.rank0_layouts = rank0_layouts

        return base_output

    def _init_progress(self, sched_req_id: str, req: OmniDiffusionRequest) -> None:
        sampling = req.sampling_params
        chunk_frames = sampling.chunk_frames
        num_frames = sampling.num_frames
        num_steps = sampling.num_inference_steps

        self._progress[sched_req_id] = _Progress(
            sched_req_id=sched_req_id,
            chunk_frames=chunk_frames,
            num_frames=num_frames,
            num_steps=num_steps,
            pp_size=self.pp_size,
            chunks_at=[deque() for _ in range(self.pp_size)],
        )

        self._slo.register(
            sched_req_id=sched_req_id,
            slo_fps=sampling.slo_fps,
            max_batch=sampling.slo_max_batch,
            chunk_frames=chunk_frames,
        )

        logger.debug(
            "StreamBatchScheduler initialized progress for %s "
            "(chunk_frames=%d, num_frames=%d, num_steps=%d, slo_fps=%s, pp_size=%d)",
            sched_req_id, chunk_frames, num_frames, num_steps, sampling.slo_fps, self.pp_size,
        )

    def _advance_chunk_pipeline(self, progress: _Progress) -> Rank0Layout:
        """Advance the per-rank queues by one micro-step and return rank 0's layout."""

        pp = progress.pp_size

        # 1. Drain last rank from previous step
        finished_idxs: list[int] = []
        circulating = []
        last = progress.chunks_at[pp - 1]
        while last:
            chunk = last.popleft()
            chunk.steps_done += 1
            if chunk.steps_done >= progress.num_steps:
                finished_idxs.append(chunk.chunk_idx)
            else:
                circulating.append(chunk)

        # 2. Shift: rank r receives what rank r-1 had
        for r in range(pp - 1, 0, -1):
            progress.chunks_at[r] = progress.chunks_at[r - 1]
        progress.chunks_at[0] = deque()

        # 3. Rank 0 = circulating + B fresh admits
        for chunk in circulating:
            progress.chunks_at[0].append(chunk)

        state = self.get_request_state(progress.sched_req_id)
        available_frames = _video_frame_count(state.req) if state is not None else 0
        queue_chunks = max(0, (available_frames - progress.frames_committed) // progress.chunk_frames)
        output_chunks_remaining = progress.output_chunks_target - progress.next_chunk_idx
        b_target = self._slo.get_target(progress.sched_req_id)
        batch_size = min(b_target, queue_chunks, output_chunks_remaining)

        new_idxs: list[int] = []
        for _ in range(batch_size):
            chunk_idx = progress.next_chunk_idx
            progress.next_chunk_idx += 1
            progress.frames_committed += progress.chunk_frames
            progress.chunks_at[0].append(_InFlightChunk(chunk_idx=chunk_idx))
            new_idxs.append(chunk_idx)
        progress.batch_size = batch_size

        # 4. Flip RUNNING -> BLOCKED if input-starved and we still owe output.
        if (
            batch_size == 0
            and output_chunks_remaining > 0
            and queue_chunks == 0
            and progress.sched_req_id in self._running
        ):
            self.block_request(progress.sched_req_id)
            logger.debug(
                "StreamBatchScheduler: %s BLOCKED on input "
                "(committed_frames=%d, target_frames=%d, available_frames=%d)",
                progress.sched_req_id, progress.frames_committed, progress.num_frames, available_frames,
            )

        return Rank0Layout(
            n_circulating=len(circulating),
            finished_idxs=finished_idxs,
            new_idxs=new_idxs,
        )

    def _build_assignment(self) -> list[RankTask | None]:
        assert len(self._progress) <= 1 #TODO: support multiple requests
        assignment: list[RankTask | None] = [None] * self.pp_size
        for progress in self._progress.values():
            for r in range(self.pp_size):
                queue = progress.chunks_at[r]
                if not queue:
                    continue
                assignment[r] = RankTask(
                    sched_req_id=progress.sched_req_id,
                    chunk_indices=[c.chunk_idx for c in queue],
                )
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