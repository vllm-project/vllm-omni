# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionBatchingConfig
from vllm_omni.diffusion.sched.interface import SchedulerRequestState

if TYPE_CHECKING:
    from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler

logger = init_logger(__name__)

# iDRR cost-normalization reference: one 512x512 image in packed latents
# ((512 / 16) ** 2 = 1024 compute units).
_IDRR_REFERENCE_UNITS = 1024.0


class DiffusionBatchingPolicy(ABC):
    """Policy plugin that decides which waiting requests enter the batch.

    A policy owns both the request-count cap (``get_max_running_reqs``) and the
    admission order (``admit_waiting``). The scheduler only delegates, so new
    policies extend behavior without touching the scheduler itself.
    """

    def on_schedule_start(self) -> None:
        """Hook called once at the beginning of a scheduler cycle."""

    def admit_waiting(
        self,
        scheduler: "BaseScheduler",
        scheduled_new_reqs: list[Any],
        scheduled_cached_request_ids: list[str],
    ) -> None:
        """Admit waiting requests in arrival (FIFO) order.

        This is upstream vLLM's admission loop: walk ``_waiting`` from the head
        and let the scheduler try each request until a cap, a failed KV
        reservation, or a batch-compatibility gate stops it. Policies that
        reorder or budget the queue override this method.
        """
        deferred: list[str] = []
        waiting_budget = len(scheduler._waiting)
        while (
            scheduler._waiting
            and waiting_budget
            and len(scheduler._running) < scheduler.max_num_running_reqs
        ):
            waiting_budget -= 1
            state = scheduler._request_states.get(scheduler._waiting[0])
            if state is None:
                scheduler._waiting.popleft()
                continue
            if len(scheduler._running) >= scheduler._effective_max_running_reqs(state):
                break
            if not scheduler._can_schedule_waiting(state):
                break
            result = scheduler._admit_state(state, scheduled_new_reqs, scheduled_cached_request_ids)
            if result == "deferred":
                deferred.append(scheduler._waiting.popleft())
            elif result in ("stalled", "gated"):
                break
            # "admitted" / "error": continue with the next waiting request.
        scheduler._waiting.extendleft(reversed(deferred))

    @abstractmethod
    def get_max_running_reqs(
        self,
        hard_cap: int,
        candidate_state: SchedulerRequestState,
    ) -> int:
        """Return the effective request cap for admitting ``candidate_state``."""


class FixedDiffusionBatchingPolicy(DiffusionBatchingPolicy):
    """Preserve the existing fixed ``max_num_seqs`` behavior."""

    def get_max_running_reqs(
        self,
        hard_cap: int,
        candidate_state: SchedulerRequestState,
    ) -> int:
        del candidate_state
        return max(1, int(hard_cap))


class ComputeBudgetDiffusionBatchingPolicy(DiffusionBatchingPolicy):
    """iDRR admission: order by cost-normalized aging, fill a compute budget."""

    def __init__(self, config: DiffusionBatchingConfig) -> None:
        self.config = config

    def admit_waiting(
        self,
        scheduler: "BaseScheduler",
        scheduled_new_reqs: list[Any],
        scheduled_cached_request_ids: list[str],
    ) -> None:
        candidates = self.sorted_waiting_states(scheduler)
        if not candidates:
            return
        self._admit_by_compute_budget(scheduler, candidates, scheduled_new_reqs, scheduled_cached_request_ids)

    def _admit_by_compute_budget(
        self,
        scheduler: "BaseScheduler",
        candidates: list[SchedulerRequestState],
        scheduled_new_reqs: list[Any],
        scheduled_cached_request_ids: list[str],
    ) -> None:
        budget = self.config.effective_compute_unit_budget()
        running_units = self.running_compute_units(scheduler)
        reserve_time = max(0.0, float(self.config.idrr_reserve_time_s))
        now = time.perf_counter()
        # A request costlier than the whole budget must not deadlock admission:
        # when nothing is running there is no concurrency left to budget, so
        # the highest-scored head is admitted regardless of its cost. This also
        # covers a budget calibrated from cheaper reference requests (no CFG,
        # single frame) than the ones the server actually serves.
        if not scheduler._running and candidates:
            head = candidates[0]
            if not scheduler._can_schedule_waiting(head) or scheduler._admit_state(
                head, scheduled_new_reqs, scheduled_cached_request_ids
            ) == "stalled":
                return
            running_units += self.request_units(head)
            candidates = candidates[1:]

        # Anti-starvation: an aged waiting request gets the head slot even when
        # it is not the score head. The waiting score is cost normalized, so
        # with beta > 0 an old large request can sit behind younger cheap ones
        # indefinitely; keying the reservation on the score head then never
        # keeps a slot for it. Pick the oldest request that has aged past the
        # threshold: admit it directly if it both fits the remaining budget and
        # is compatible with the running batch, otherwise hold its gap so later
        # admissions cannot eat it.  Compatibility is deliberately not part of
        # the selection: an aged request that cannot join the current batch
        # still needs the gap kept, or filling it would only prolong the batch.
        remaining = budget - running_units
        reserve_state = None
        for state in candidates:
            age = max(0.0, now - state.queued_at) if state.queued_at else 0.0
            if age > reserve_time and (
                reserve_state is None or (state.queued_at or 0.0) < (reserve_state.queued_at or 0.0)
            ):
                reserve_state = state

        if reserve_state is not None:
            reserved_cost = self.request_units(reserve_state)
            if reserved_cost > remaining or not scheduler._can_schedule_waiting(reserve_state):
                # Reservation mode: keep a gap for the aged request and admit
                # only requests that fit into it, then stop admission so the
                # gap survives until the reserved request can be scheduled.
                gap = remaining - reserved_cost
                for state in candidates:
                    if state is reserve_state:
                        continue
                    if len(scheduler._running) >= scheduler.max_num_running_reqs:
                        break
                    if not scheduler._can_schedule_waiting(state):
                        continue
                    cost = self.request_units(state)
                    if cost > gap:
                        continue
                    if scheduler._admit_state(state, scheduled_new_reqs, scheduled_cached_request_ids) == "stalled":
                        break
                    gap -= cost
                return
            # The aged request fits: give it the head slot first, then admit in
            # score order for whatever budget is left. Without this, cheaper
            # younger requests in score order would consume the freed slot and
            # the aged request would miss it again.
            if (
                scheduler._admit_state(reserve_state, scheduled_new_reqs, scheduled_cached_request_ids)
                == "stalled"
            ):
                return
            running_units += reserved_cost
            candidates = [state for state in candidates if state is not reserve_state]

        # No aged request is blocked: admit in score order, skipping requests
        # that do not fit so cheaper ones can fill the remaining budget.
        index = 0
        while index < len(candidates) and len(scheduler._running) < scheduler.max_num_running_reqs:
            head = candidates[index]
            head_cost = self.request_units(head)
            if running_units + head_cost <= budget and scheduler._can_schedule_waiting(head):
                if (
                    scheduler._admit_state(head, scheduled_new_reqs, scheduled_cached_request_ids)
                    == "stalled"
                ):
                    break
                running_units += head_cost
            index += 1

    def sorted_waiting_states(self, scheduler: "BaseScheduler") -> list[SchedulerRequestState]:
        """Order waiting requests by iDRR score, descending.

        Ties fall back to enqueue time (FIFO) then request_id for
        determinism. Finished or popped states are skipped.
        """
        alpha = float(self.config.idrr_alpha)
        beta = float(self.config.idrr_beta)
        now = time.perf_counter()
        scored: list[tuple[float, float, str, SchedulerRequestState]] = []
        for request_id in scheduler._waiting:
            state = scheduler._request_states.get(request_id)
            if state is None or state.is_finished():
                continue
            age = max(0.0, now - state.queued_at) if state.queued_at else 0.0
            norm_cost = max(self.request_units(state) / _IDRR_REFERENCE_UNITS, 1e-6)
            score = (age**alpha) / (norm_cost**beta)
            scored.append((score, state.queued_at, request_id, state))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in scored]

    def running_compute_units(self, scheduler: "BaseScheduler") -> int:
        return sum(
            self.request_units(scheduler._request_states[request_id])
            for request_id in scheduler._running
            if request_id in scheduler._request_states
        )

    def request_units(self, state: SchedulerRequestState) -> int:
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

    def get_max_running_reqs(
        self,
        hard_cap: int,
        candidate_state: SchedulerRequestState,
    ) -> int:
        hard_cap = max(1, int(hard_cap))
        request_units = self.config.request_compute_units(
            candidate_state.req.sampling_params,
            candidate_state.req.prompts,
        )
        budget = self.config.effective_compute_unit_budget()
        dynamic_cap = (budget + request_units - 1) // request_units
        capped = max(1, min(hard_cap, dynamic_cap))

        if self.config.log_stats:
            logger.info(
                "[DiffusionBatching] request_compute_units=%s compute_unit_budget=%s "
                "dynamic_max_num_seqs=%s hard_max_num_seqs=%s effective_max_num_seqs=%s",
                request_units,
                budget,
                dynamic_cap,
                hard_cap,
                capped,
            )
        return capped


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

    def get_max_running_reqs(
        self,
        hard_cap: int,
        candidate_state: SchedulerRequestState,
    ) -> int:
        del candidate_state
        hard_cap = max(1, int(hard_cap))
        cap = min(hard_cap, self._current_cap)
        logger.info("[DiffusionBatchProfiler] max_num_seqs=%s", cap)
        return cap


def create_diffusion_batching_policy(config: DiffusionBatchingConfig) -> DiffusionBatchingPolicy:
    if config.uses_compute_budget:
        logger.info(
            "ComputeBudgetDiffusionBatchingPolicy enabled (compute_unit_budget=%s)",
            config.effective_compute_unit_budget(),
        )
        return ComputeBudgetDiffusionBatchingPolicy(config)
    if config.uses_profiler:
        return DiffusionBatchProfilerPolicy(config)
    return FixedDiffusionBatchingPolicy()
