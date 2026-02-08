# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduling policy abstractions for diffusion worker request ordering."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Protocol

from vllm_omni.diffusion.request import OmniDiffusionRequest


def req_debug_id(req: OmniDiffusionRequest) -> str:
    return req.request_ids[0] if req.request_ids else req.request_key


def req_step_index(req: OmniDiffusionRequest) -> int:
    if req.execution_state is None:
        return 0
    return int(req.execution_state.step_index)


def estimate_remaining_cost(req: OmniDiffusionRequest) -> float:
    """Simple estimator: remaining_steps * width * height."""
    total_steps = int(req.sampling_params.num_inference_steps or 0)
    done_steps = req_step_index(req)
    remaining_steps = max(0, total_steps - done_steps)
    width = int(req.sampling_params.width or 1024)
    height = int(req.sampling_params.height or 1024)
    return float(remaining_steps * width * height)


@dataclass
class PreemptionEvent:
    preempted_req: OmniDiffusionRequest
    selected_req: OmniDiffusionRequest
    preempted_cost: float
    selected_cost: float


@dataclass
class ArrivalDecision:
    current_req: OmniDiffusionRequest | None
    preemption: PreemptionEvent | None = None


class DiffusionSchedulingPolicy(Protocol):
    def on_request_arrival(
        self, req: OmniDiffusionRequest, current_req: OmniDiffusionRequest | None
    ) -> ArrivalDecision:
        ...

    def on_request_finish(self) -> OmniDiffusionRequest | None:
        ...

    def abort_request_ids(self, request_ids: list[str]) -> None:
        ...

    def is_aborted(self, request_id: str) -> bool:
        ...

    def consume_recent_dropped_request_ids(self) -> list[str]:
        ...


class TargetFreeGlobalReorderPolicy:
    """Global queue reorder policy with arrival/finish scheduling points."""

    def __init__(self) -> None:
        self._waiting_reqs: list[tuple[float, int, OmniDiffusionRequest]] = []
        self._queued_request_keys: set[str] = set()
        self._queue_seq: int = 0
        self._aborted_request_ids: set[str] = set()
        self._recent_dropped_request_ids: list[str] = []

    def on_request_arrival(
        self, req: OmniDiffusionRequest, current_req: OmniDiffusionRequest | None
    ) -> ArrivalDecision:
        req_id = req_debug_id(req)
        if req_id in self._aborted_request_ids:
            self._recent_dropped_request_ids.append(req_id)
            return ArrivalDecision(current_req=current_req)

        if current_req is None and not self._waiting_reqs:
            return ArrivalDecision(current_req=req)

        self._queue_request(req)
        candidate = self._pop_best_waiting_request()
        if candidate is None:
            return ArrivalDecision(current_req=current_req)
        if current_req is None:
            return ArrivalDecision(current_req=candidate)

        current_cost = estimate_remaining_cost(current_req)
        candidate_cost = estimate_remaining_cost(candidate)
        if candidate_cost < current_cost:
            self._queue_request(current_req)
            return ArrivalDecision(
                current_req=candidate,
                preemption=PreemptionEvent(
                    preempted_req=current_req,
                    selected_req=candidate,
                    preempted_cost=current_cost,
                    selected_cost=candidate_cost,
                ),
            )

        self._queue_request(candidate)
        return ArrivalDecision(current_req=current_req)

    def on_request_finish(self) -> OmniDiffusionRequest | None:
        return self._pop_best_waiting_request()

    def abort_request_ids(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            if isinstance(rid, str):
                self._aborted_request_ids.add(rid)

    def is_aborted(self, request_id: str) -> bool:
        return request_id in self._aborted_request_ids

    def consume_recent_dropped_request_ids(self) -> list[str]:
        dropped = self._recent_dropped_request_ids
        self._recent_dropped_request_ids = []
        return dropped

    def _queue_request(self, req: OmniDiffusionRequest) -> None:
        req_id = req_debug_id(req)
        if req_id in self._aborted_request_ids:
            self._recent_dropped_request_ids.append(req_id)
            return
        if req.request_key in self._queued_request_keys:
            return
        cost = estimate_remaining_cost(req)
        heapq.heappush(self._waiting_reqs, (cost, self._queue_seq, req))
        self._queued_request_keys.add(req.request_key)
        self._queue_seq += 1

    def _pop_best_waiting_request(self) -> OmniDiffusionRequest | None:
        while self._waiting_reqs:
            _, _, req = heapq.heappop(self._waiting_reqs)
            self._queued_request_keys.discard(req.request_key)
            req_id = req_debug_id(req)
            if req_id in self._aborted_request_ids:
                self._recent_dropped_request_ids.append(req_id)
                continue
            return req
        return None

