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

# DRR cost-normalization reference: one 512x512 image in packed latents
# ((512 / 16) ** 2 = 1024 compute units).
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

    def count_cap(self, hard_cap: int) -> int:
        """Request-count cap for this step, given the configured ``max_num_seqs``."""
        return max(1, int(hard_cap))


class FixedDiffusionBatchingPolicy(DiffusionBatchingPolicy):
    """Preserve the existing fixed ``max_num_seqs`` behavior."""


class ComputeBudgetDiffusionBatchingPolicy(DiffusionBatchingPolicy):
    """DRR admission: order by cost-normalized aging, fill a compute budget."""

    stops_at_incompatible_head = False

    def __init__(
        self,
        config: DiffusionBatchingConfig,
        *,
        request_budget_units: int | None = None,
    ) -> None:
        self.config = config
        # An explicit ``max_num_seqs > 1`` pins the admission budget to
        # ``max_num_seqs * reference_compute_units``. None means the scheduler
        # saw the default/unset ``max_num_seqs <= 1`` sentinel, so the budget
        # comes from ``compute_unit_budget`` (optionally warmup-calibrated).
        self.request_budget_units = max(1, int(request_budget_units)) if request_budget_units is not None else None

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
        config = self.config
        key = state.sampling_params_key
        if key is None:
            return config.request_compute_units(state.req.sampling_params, state.req.prompts)
        # Mixfusion shape relaxation nulls the key's resolution so mixed sizes
        # can share a batch; fall back to the request's own resolution instead
        # of the config default, which would overcost smaller requests.
        sampling = state.req.sampling_params
        height = key.height if key.height is not None else getattr(sampling, "height", None)
        width = key.width if key.width is not None else getattr(sampling, "width", None)
        units = config.spatial_compute_units(height, width, key.num_frames)
        if key.do_classifier_free_guidance or (key.true_cfg_scale is not None and key.true_cfg_scale > 1):
            units = int(units * float(config.cfg_compute_multiplier))
        return max(1, units)

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

    def count_cap(self, hard_cap: int) -> int:
        """Request-count bound for budgeted admission.

        The compute-unit sum is the real admission constraint, so the count
        bound only matters when the caller pinned ``max_num_seqs`` explicitly.
        With an unset ``max_num_seqs <= 1`` sentinel no literal count cap
        applies; the budget decides how many requests fit.
        """
        if self.request_budget_units is not None:
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
    request_budget_units: int | None = None,
) -> DiffusionBatchingPolicy:
    if config.uses_compute_budget:
        budget_source = "max_num_seqs" if request_budget_units is not None else "compute_unit_budget"
        budget = (
            max(1, int(request_budget_units))
            if request_budget_units is not None
            else config.effective_compute_unit_budget()
        )
        logger.info(
            "ComputeBudgetDiffusionBatchingPolicy enabled (%s=%s)",
            budget_source,
            budget,
        )
        return ComputeBudgetDiffusionBatchingPolicy(config, request_budget_units=request_budget_units)
    if config.uses_profiler:
        return DiffusionBatchProfilerPolicy(config)
    if config.policy in ("auto", "fixed"):
        return FixedDiffusionBatchingPolicy()
    raise ValueError(
        f"Unknown diffusion batching policy {config.policy!r}; "
        "expected one of 'auto', 'fixed', 'compute_budget', 'profile', 'profiler'."
    )
