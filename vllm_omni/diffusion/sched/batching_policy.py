# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionBatchingConfig
from vllm_omni.diffusion.sched.interface import SchedulerRequestState

if TYPE_CHECKING:
    from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler

logger = init_logger(__name__)

# DRR score normalization reference: the packed-token count of one 512x512
# image ((512 / 16) ** 2 = 1024 tokens), the same unit as the MixFusion chunk
# envelope. Costs carry the CFG weight; capacities do not.
_DRR_REFERENCE_UNITS = 1024.0


class DiffusionBatchingPolicy:
    """Pure queries that steer the scheduler's admission loop.

    The scheduler owns admission bookkeeping: budget accumulation, the
    request-count cap, the batch-compatibility gate and the dispatch of
    ``_admit_state`` results. A policy only supplies the candidate order, the
    per-request numbers and an optional reservation target, so new policies
    plug in without changing ``BaseScheduler``.
    """

    # FIFO stops at the first incompatible waiting request to preserve arrival
    # order. Policies that may skip incompatible candidates to fill the batch
    # (DRR) set this to False.
    stops_at_incompatible_head: bool = True

    # Whether the scheduler's admission loop should take the token budget from
    # this policy. A policy that does not budget by tokens leaves admission
    # bounded only by the request-count cap.
    consumes_token_budget: bool = False

    def on_schedule_start(self) -> None:
        """Hook called once at the beginning of a scheduler cycle."""

    def order_candidates(self, scheduler: BaseScheduler, now: float) -> list[SchedulerRequestState]:
        """Return the waiting requests to consider this step, in admission order."""
        candidates: list[SchedulerRequestState] = []
        for request_id in scheduler._waiting:
            state = scheduler._request_states.get(request_id)
            if state is None or state.is_finished():
                continue
            candidates.append(state)
        return candidates

    def request_tokens(self, state: SchedulerRequestState) -> int:
        """Capacity units ``state`` consumes against the admission budget."""
        raise NotImplementedError("This batching policy does not consume an admission budget.")

    def request_cost(self, state: SchedulerRequestState) -> int:
        """Weighted cost used to score admission order; defaults to capacity."""
        return self.request_tokens(state)

    def reservation_target(self, candidates: list[SchedulerRequestState], now: float) -> SchedulerRequestState | None:
        """Return a candidate whose budget gap must be held this step, if any."""
        return None

    def oversized_aged_target(
        self,
        candidates: list[SchedulerRequestState],
        now: float,
        budget: int,
    ) -> SchedulerRequestState | None:
        """Return an aged candidate that can never fit ``budget``, if any.

        Such a request can only ever run alone, so callers stop admitting new
        work and let the running batch drain for it.
        """
        return None

    def count_cap(self, hard_cap: int) -> int:
        """Request-count cap for this step, given the configured ``max_num_seqs``."""
        return max(1, int(hard_cap))


class FixedDiffusionBatchingPolicy(DiffusionBatchingPolicy):
    """Preserve the existing fixed ``max_num_seqs`` behavior."""


class ComputeBudgetDiffusionBatchingPolicy(DiffusionBatchingPolicy):
    """DRR admission: order by cost-normalized aging, fill a token budget."""

    stops_at_incompatible_head = False
    consumes_token_budget = True

    def __init__(
        self,
        config: DiffusionBatchingConfig,
        *,
        request_count_cap: int | None = None,
    ) -> None:
        self.config = config
        # An explicit ``max_num_seqs`` is a literal request-count cap; None
        # (the unset sentinel) lets the token budget alone bound the batch.
        self.request_count_cap = max(1, int(request_count_cap)) if request_count_cap is not None else None

    def order_candidates(self, scheduler: BaseScheduler, now: float) -> list[SchedulerRequestState]:
        """Order waiting requests by DRR score, descending.

        Ties fall back to enqueue time (FIFO) then request_id for
        determinism. Finished or popped states are skipped.
        """
        alpha = float(self.config.drr_alpha)
        beta = float(self.config.drr_beta)
        scored: list[tuple[float, float, str, SchedulerRequestState]] = []
        for request_id in scheduler._waiting:
            state = scheduler._request_states.get(request_id)
            if state is None or state.is_finished():
                continue
            age = max(0.0, now - state.queued_at) if state.queued_at else 0.0
            norm_cost = max(self.request_cost(state) / _DRR_REFERENCE_UNITS, 1e-6)
            score = (age**alpha) / (norm_cost**beta)
            scored.append((score, state.queued_at, request_id, state))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in scored]

    def request_tokens(self, state: SchedulerRequestState) -> int:
        """Packed patch tokens the request occupies against the admission budget."""
        info = getattr(state.req, "mixfusion_info", None)
        if info is not None:
            # Prepared image requests already carry the model-resolved grid.
            return max(1, int(info.num_images)) * max(1, int(info.image_tokens))
        sampling = state.req.sampling_params
        key = state.sampling_params_key
        if key is None:
            height = getattr(sampling, "height", None)
            width = getattr(sampling, "width", None)
            num_frames = getattr(sampling, "num_frames", None)
        else:
            # Mixfusion shape relaxation nulls the key's resolution so mixed
            # sizes can share a batch; fall back to the request's own one.
            height = key.height if key.height is not None else getattr(sampling, "height", None)
            width = key.width if key.width is not None else getattr(sampling, "width", None)
            num_frames = key.num_frames
        if height is None or width is None:
            raise ValueError(
                f"Request {state.request_id!r} has no resolved output size; DRR admission needs "
                "sampling_params.height/width or a MixFusion-enabled image request."
            )
        num_images = getattr(sampling, "num_outputs_per_prompt", None) or 1
        return self.config.spatial_tokens(height, width, num_frames, num_images)

    def request_cost(self, state: SchedulerRequestState) -> int:
        """Token cost weighted by CFG, used only to score admission order."""
        tokens = self.request_tokens(state)
        if self._uses_true_cfg(state):
            return max(1, int(tokens * float(self.config.cfg_compute_multiplier)))
        return tokens

    def _uses_true_cfg(self, state: SchedulerRequestState) -> bool:
        key = state.sampling_params_key
        if key is not None:
            return key.do_classifier_free_guidance or (key.true_cfg_scale is not None and key.true_cfg_scale > 1)
        return self.config.uses_true_cfg(state.req.sampling_params, state.req.prompts)

    def reservation_target(self, candidates: list[SchedulerRequestState], now: float) -> SchedulerRequestState | None:
        """Pick the oldest waiting request that has aged past the reserve threshold.

        Anti-starvation: an aged waiting request gets the head slot even when
        it is not the score head. The waiting score is cost normalized, so with
        beta > 0 an old large request can sit behind younger cheap ones
        indefinitely; keying the reservation on the score head then never keeps
        a slot for it. Compatibility is deliberately not part of the selection:
        an aged request that cannot join the current batch still needs the gap
        kept, or filling it would only prolong the batch.
        """
        reserve_time = max(0.0, float(self.config.drr_reserve_time_s))
        target: SchedulerRequestState | None = None
        for state in candidates:
            age = max(0.0, now - state.queued_at) if state.queued_at else 0.0
            if age > reserve_time and (target is None or (state.queued_at or 0.0) < (target.queued_at or 0.0)):
                target = state
        return target

    def oversized_aged_target(
        self,
        candidates: list[SchedulerRequestState],
        now: float,
        budget: int,
    ) -> SchedulerRequestState | None:
        """Oldest aged candidate costlier than ``budget``; it can only run alone."""
        reserve_time = max(0.0, float(self.config.drr_reserve_time_s))
        target: SchedulerRequestState | None = None
        for state in candidates:
            if self.request_tokens(state) <= budget:
                continue
            age = max(0.0, now - state.queued_at) if state.queued_at else 0.0
            if age > reserve_time and (target is None or (state.queued_at or 0.0) < (target.queued_at or 0.0)):
                target = state
        return target

    def count_cap(self, hard_cap: int) -> int:
        """Request-count bound for budgeted admission.

        The token budget is the real admission constraint, so a literal count
        cap only applies when ``max_num_seqs`` was set. With the unset ``None``
        sentinel no literal count cap applies; the budget decides how many
        requests fit.
        """
        if self.request_count_cap is not None:
            return max(1, int(hard_cap))
        return 1 << 30


class DiffusionBatchProfilerPolicy(DiffusionBatchingPolicy):
    """Gradually increases batch cap so users can collect sweet-spot logs."""

    def __init__(self, config: DiffusionBatchingConfig) -> None:
        self.config = config
        self._schedule_count = 0
        self._current_cap = 1
        logger.info(
            "DiffusionBatchProfiler enabled (profiler_interval=%s)",
            self.config.profiler_interval,
        )

    def on_schedule_start(self) -> None:
        interval = max(1, int(self.config.profiler_interval))
        self._schedule_count += 1
        self._current_cap = self._schedule_count // interval + 1

    def count_cap(self, hard_cap: int) -> int:
        hard_cap = max(1, int(hard_cap))
        cap = min(hard_cap, self._current_cap)
        logger.info("[DiffusionBatchProfiler] max_num_seqs=%s", cap)
        return cap


def create_diffusion_batching_policy(
    config: DiffusionBatchingConfig,
    *,
    request_count_cap: int | None = None,
    token_budget: int | None = None,
) -> DiffusionBatchingPolicy:
    if config.uses_compute_budget:
        logger.info(
            "ComputeBudgetDiffusionBatchingPolicy enabled (token_budget=%s, request_count_cap=%s)",
            token_budget,
            request_count_cap,
        )
        return ComputeBudgetDiffusionBatchingPolicy(config, request_count_cap=request_count_cap)
    if config.uses_profiler:
        return DiffusionBatchProfilerPolicy(config)
    if config.policy in ("auto", "fixed"):
        return FixedDiffusionBatchingPolicy()
    raise ValueError(
        f"Unknown diffusion batching policy {config.policy!r}; "
        "expected one of 'auto', 'fixed', 'compute_budget', 'profile', 'profiler'."
    )
