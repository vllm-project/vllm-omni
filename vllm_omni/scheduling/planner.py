# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Online release-calendar beam planning.

The planner deliberately consumes only scheduler-visible state. It does not
know future arrivals or the final benchmark size, and it never changes a
request that has already started.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .config import TailAwareSchedulingConfig

_DEFERRED_LATENCY_PLACEHOLDER_S = 1.0e12


@dataclass(frozen=True)
class ReleaseCalendarRequest:
    request_id: str
    sequence: int
    arrival_time_s: float
    estimated_service_s_by_backend: tuple[float, ...]

    def estimate_on(self, backend_index: int) -> float:
        return self.estimated_service_s_by_backend[backend_index]


@dataclass(frozen=True)
class _BeamState:
    slots_s: tuple[float, ...]
    projected_latencies_s: tuple[float, ...]
    remaining: tuple[ReleaseCalendarRequest, ...]
    first_request_id: str
    prefix: tuple[str, ...]
    rollout_objective: tuple[float, float]


def percentile_type7(values: Sequence[float], quantile: float) -> float:
    """Hyndman-Fan type-7 percentile, matching NumPy's default."""

    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _request_risk(
    request: ReleaseCalendarRequest,
    *,
    dispatch_s: float,
    backend_index: int,
    depth: int,
    config: TailAwareSchedulingConfig,
) -> float:
    beta = config.band_risk_beta if config.band_min_pending <= depth <= config.band_max_pending else config.risk_beta
    return dispatch_s - request.arrival_time_s + beta * request.estimate_on(backend_index)


def _queue_band_select(
    requests: Sequence[ReleaseCalendarRequest],
    *,
    dispatch_s: float,
    backend_index: int,
    config: TailAwareSchedulingConfig,
) -> ReleaseCalendarRequest:
    depth = len(requests)
    return min(
        requests,
        key=lambda request: (
            -_request_risk(
                request,
                dispatch_s=dispatch_s,
                backend_index=backend_index,
                depth=depth,
                config=config,
            ),
            request.sequence,
            request.request_id,
        ),
    )


def _critical_candidates(
    remaining: tuple[ReleaseCalendarRequest, ...],
    *,
    dispatch_s: float,
    backend_index: int,
    config: TailAwareSchedulingConfig,
) -> tuple[ReleaseCalendarRequest, ...]:
    risks = {
        request.request_id: _request_risk(
            request,
            dispatch_s=dispatch_s,
            backend_index=backend_index,
            depth=len(remaining),
            config=config,
        )
        for request in remaining
    }

    def risk_key(request: ReleaseCalendarRequest) -> tuple[float, int, str]:
        return -risks[request.request_id], request.sequence, request.request_id

    by_risk = sorted(remaining, key=risk_key)
    minimum_risk = risks[by_risk[0].request_id] - config.beam_risk_slack_s
    eligible = [request for request in by_risk if risks[request.request_id] >= minimum_risk][: config.beam_branch_width]

    # Preserve the prototype's structural shortest alternative. This can add
    # one request beyond branch_width, while candidate_cap remains the hard
    # bound on total planner work.
    shortest = min(remaining, key=lambda request: (request.estimate_on(backend_index), *risk_key(request)))
    if risks[shortest.request_id] >= minimum_risk and all(
        request.request_id != shortest.request_id for request in eligible
    ):
        eligible.append(shortest)
    return tuple(eligible)


def _greedy_complete(
    *,
    now_s: float,
    slots_s: tuple[float, ...],
    projected_latencies_s: tuple[float, ...],
    remaining: tuple[ReleaseCalendarRequest, ...],
    config: TailAwareSchedulingConfig,
) -> tuple[float, float]:
    mutable_slots = list(slots_s)
    mutable_projected = list(projected_latencies_s)
    mutable_remaining = list(remaining)
    while mutable_remaining:
        backend_index = min(range(len(mutable_slots)), key=lambda index: (mutable_slots[index], index))
        dispatch_s = now_s + mutable_slots[backend_index]
        selected = _queue_band_select(
            mutable_remaining,
            dispatch_s=dispatch_s,
            backend_index=backend_index,
            config=config,
        )
        finish_s = dispatch_s + selected.estimate_on(backend_index)
        mutable_slots[backend_index] = finish_s - now_s
        mutable_projected.append(finish_s - selected.arrival_time_s)
        mutable_remaining.remove(selected)
    finite = [value for value in mutable_projected if value < _DEFERRED_LATENCY_PLACEHOLDER_S]
    mean_s = statistics.fmean(finite) if finite else math.inf
    return percentile_type7(mutable_projected, 0.95), mean_s


def _validated_release_calendar(
    release_calendar_s: Sequence[float] | None,
    pending: Sequence[ReleaseCalendarRequest],
) -> tuple[float, ...] | None:
    if release_calendar_s is None or not release_calendar_s:
        return None
    slots: list[float] = []
    for raw_slot in release_calendar_s:
        if (
            isinstance(raw_slot, bool)
            or not isinstance(raw_slot, (int, float))
            or not math.isfinite(raw_slot)
            or raw_slot < 0.0
        ):
            return None
        slots.append(float(raw_slot))
    for request in pending:
        estimates = request.estimated_service_s_by_backend
        if len(estimates) != len(slots):
            return None
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0
            for value in estimates
        ):
            return None
    return tuple(slots)


def plan_release_calendar(
    *,
    pending: Sequence[ReleaseCalendarRequest],
    now_s: float,
    release_calendar_s: Sequence[float] | None,
    first_backend_index: int,
    completed_latencies_s: Iterable[float],
    active_projected_latencies_s: Iterable[float],
    deferred_request_count: int,
    config: TailAwareSchedulingConfig,
) -> str:
    """Plan several visible release decisions, then execute only the first."""

    if not pending:
        raise ValueError("pending cannot be empty")
    completed = [
        float(value)
        for value in completed_latencies_s
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0.0
    ][-config.beam_history_size :]
    active = [
        float(value)
        for value in active_projected_latencies_s
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0.0
    ]
    deferred_count = max(int(deferred_request_count), 0)
    if not 0 <= first_backend_index < (len(release_calendar_s) if release_calendar_s is not None else 0):
        # The request estimates are still sufficient for a stable Queue-Band
        # fallback when callers pass a missing release calendar.
        fallback_backend = max(first_backend_index, 0)
        if pending and pending[0].estimated_service_s_by_backend:
            fallback_backend = min(
                fallback_backend,
                len(pending[0].estimated_service_s_by_backend) - 1,
            )
        return _queue_band_select(pending, dispatch_s=now_s, backend_index=fallback_backend, config=config).request_id
    slots = _validated_release_calendar(release_calendar_s, pending)
    if slots is None or not config.beam_min_pending <= len(pending) <= config.beam_max_pending:
        return _queue_band_select(
            pending, dispatch_s=now_s, backend_index=first_backend_index, config=config
        ).request_id

    initial_projected = tuple(
        [
            *completed,
            *active,
            *([_DEFERRED_LATENCY_PLACEHOLDER_S] * deferred_count),
        ]
    )
    base = _queue_band_select(
        pending,
        dispatch_s=now_s + slots[first_backend_index],
        backend_index=first_backend_index,
        config=config,
    )
    initial = _BeamState(
        slots_s=slots,
        projected_latencies_s=initial_projected,
        remaining=tuple(pending),
        first_request_id=base.request_id,
        prefix=(),
        # Only expanded children are ranked.
        rollout_objective=(math.inf, math.inf),
    )
    beam = [initial]
    candidate_count = 0
    cap_reached = False
    for depth_index in range(min(config.beam_horizon, len(pending))):
        children: list[_BeamState] = []
        for state in beam:
            if not state.remaining:
                children.append(state)
                continue
            backend_index = (
                first_backend_index
                if depth_index == 0
                else min(range(len(state.slots_s)), key=lambda index: (state.slots_s[index], index))
            )
            dispatch_s = now_s + state.slots_s[backend_index]
            for selected in _critical_candidates(
                state.remaining,
                dispatch_s=dispatch_s,
                backend_index=backend_index,
                config=config,
            ):
                if candidate_count >= config.beam_candidate_cap:
                    cap_reached = True
                    break
                child_slots = list(state.slots_s)
                finish_s = dispatch_s + selected.estimate_on(backend_index)
                child_slots[backend_index] = finish_s - now_s
                child_projected = (
                    *state.projected_latencies_s,
                    finish_s - selected.arrival_time_s,
                )
                child_remaining = tuple(
                    request for request in state.remaining if request.request_id != selected.request_id
                )
                child_objective = _greedy_complete(
                    now_s=now_s,
                    slots_s=tuple(child_slots),
                    projected_latencies_s=child_projected,
                    remaining=child_remaining,
                    config=config,
                )
                children.append(
                    _BeamState(
                        slots_s=tuple(child_slots),
                        projected_latencies_s=child_projected,
                        remaining=child_remaining,
                        first_request_id=(selected.request_id if not state.prefix else state.first_request_id),
                        prefix=(*state.prefix, selected.request_id),
                        rollout_objective=child_objective,
                    )
                )
                candidate_count += 1
            if cap_reached:
                break
        if not children:
            break
        children.sort(
            key=lambda state: (
                state.rollout_objective,
                state.first_request_id != base.request_id,
                state.prefix,
            )
        )
        beam = children[: config.beam_width]
        if cap_reached or not beam[0].remaining:
            break

    if not beam or not beam[0].prefix:
        return _queue_band_select(
            pending, dispatch_s=now_s, backend_index=first_backend_index, config=config
        ).request_id
    best = min(
        beam,
        key=lambda state: (
            state.rollout_objective,
            state.first_request_id != base.request_id,
            state.prefix,
        ),
    )
    return best.first_request_id
