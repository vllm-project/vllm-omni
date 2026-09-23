# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native admission for one local diffusion stage with static replicas.

All methods run on the orchestrator's event loop. Only ``acquire`` awaits;
accounting, completion, cancellation and replica removal are atomic with
respect to each other. Backend execution and transport stay with StagePool.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .config import TailAwareSchedulingConfig
from .estimation import estimate_service_time_s


class TailAwareQueueFullError(ValueError):
    """The bounded central admission queue has no free entry."""


@dataclass(frozen=True)
class Decision:
    replica_id: int
    is_tail: bool = False


@dataclass
class _Request:
    request_id: str
    sequence: int
    arrival_s: float
    estimated_service_s: float
    future: asyncio.Future[Decision]
    deferred: bool = False
    decision: Decision | None = None
    bound_s: float | None = None
    terminal_error: BaseException | None = None


@dataclass
class _Replica:
    replica_id: int
    active: _Request | None = None
    latency_ema_s: float = 0.0


class TailAwareController:
    """Central admission and dispatch for one local diffusion stage.

    Each replica executes one request at a time without preemption. Selection
    only affects requests still waiting in the central queue.
    Removal fails in-flight work in StagePool; this object only releases its
    accounting and reroutes requests that have not yet been dispatched.
    """

    def __init__(self, replica_ids: list[int], config: TailAwareSchedulingConfig) -> None:
        if not replica_ids or len(set(replica_ids)) != len(replica_ids):
            raise ValueError("replica_ids must be nonempty and unique")
        if any(isinstance(replica_id, bool) or not isinstance(replica_id, int) for replica_id in replica_ids):
            raise ValueError("replica_ids must contain integers")
        if not config.enabled:
            raise ValueError("TailAwareController requires enabled scheduling")
        self.config = config
        self._replicas = {replica_id: _Replica(replica_id) for replica_id in replica_ids}
        self._requests: dict[str, _Request] = {}
        self._pending_normals: list[_Request] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._drain_handle: asyncio.Handle | None = None
        self._closed = False
        self.arrival_counter = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending_normals)

    @property
    def active_count(self) -> int:
        return len(self._requests) - self.pending_count

    async def acquire(self, request_id: str, sampling_params: Any, model_class_name: str) -> Decision:
        """Reserve an execution slot, or wait centrally until one is available."""
        loop = asyncio.get_running_loop()
        if self._loop is not None and loop is not self._loop:
            raise RuntimeError("TailAwareController must stay on one event loop")
        self._loop = loop
        if self._closed:
            raise RuntimeError("tail-aware controller is closed")
        if not self._replicas:
            raise RuntimeError("no diffusion replicas are available")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        if request_id in self._requests:
            raise ValueError(f"Duplicate in-flight request_id: {request_id!r}")
        if self.pending_count >= self.config.max_pending_requests:
            raise TailAwareQueueFullError("tail-aware pending request limit reached")
        estimate_s = estimate_service_time_s(sampling_params, model_class_name, self.config.hardware_profile)
        now_s = time.perf_counter()
        self.arrival_counter += 1
        request = _Request(
            request_id=request_id,
            sequence=self.arrival_counter,
            arrival_s=now_s,
            estimated_service_s=estimate_s,
            future=loop.create_future(),
        )
        self._requests[request_id] = request
        self._pending_normals.append(request)
        self._schedule_drain()
        try:
            decision = await request.future
            if request.terminal_error is not None:
                raise request.terminal_error
            return decision
        except BaseException:
            # Cancellation can arrive after the future was resolved but before
            # acquire returned. Release that reservation too, exactly once.
            if self._requests.get(request_id) is request:
                self.cancel(request_id)
            raise

    def complete(self, request_id: str, success: bool = True) -> None:
        """Release dispatched work once; only successes update replica latency."""
        request = self._requests.get(request_id)
        if request is None or request.decision is None:
            return
        now_s = time.perf_counter()
        replica = self._replicas.get(request.decision.replica_id)
        if success and replica is not None and request.bound_s is not None:
            elapsed = max(now_s - request.bound_s, 0.0)
            replica.latency_ema_s = (
                elapsed if replica.latency_ema_s <= 0 else 0.9 * replica.latency_ema_s + 0.1 * elapsed
            )
        self._forget(request)
        self._schedule_drain()

    def cancel(self, request_id: str) -> None:
        """Withdraw a pending request or release an active reservation once."""
        request = self._requests.get(request_id)
        if request is None:
            return
        request.terminal_error = asyncio.CancelledError()
        self._forget(request)
        if not request.future.done():
            request.future.cancel()
        self._schedule_drain()

    def remove_replica(self, replica_id: int) -> tuple[str, ...]:
        """Exclude a failed replica and return its formerly active request IDs.

        StagePool remains responsible for cancelling/failing backend execution.
        Pending work can use remaining replicas; active work is never silently
        retried, since generation may already have started.
        """
        replica = self._replicas.pop(replica_id, None)
        if replica is None:
            return ()
        active = replica.active
        if active is not None:
            active.terminal_error = RuntimeError(f"diffusion replica {replica_id} was removed")
            self._forget(active)
        if not self._replicas:
            for request in tuple(self._requests.values()):
                self._forget(request)
                if not request.future.done():
                    request.future.set_exception(RuntimeError("no diffusion replicas are available"))
        self._schedule_drain()
        return (active.request_id,) if active is not None else ()

    def close(self) -> None:
        """Cancel admission waiters and release all policy state on shutdown."""
        if self._closed:
            return
        self._closed = True
        if self._drain_handle is not None:
            self._drain_handle.cancel()
            self._drain_handle = None
        for request_id in tuple(self._requests):
            self.cancel(request_id)

    def _forget(self, request: _Request) -> None:
        self._requests.pop(request.request_id, None)
        if request.decision is None:
            self._pending_normals.remove(request)
            return
        replica = self._replicas.get(request.decision.replica_id)
        if replica is not None and replica.active is request:
            replica.active = None

    def _schedule_drain(self) -> None:
        if self._closed or not self.pending_count or self._drain_handle is not None:
            return
        assert self._loop is not None
        # Coalesce adjacent state changes before selecting the next request.
        self._drain_handle = self._loop.call_soon(self._drain)

    def _drain(self) -> None:
        self._drain_handle = None
        if self._closed:
            return
        # Task cancellation cancels an awaited future before its coroutine's
        # cleanup runs. Never bind such an entry during this intervening turn.
        for request in tuple(self._pending_normals):
            if request.future.cancelled():
                self._forget(request)
        while self._pending_normals:
            now_s = time.perf_counter()
            available = [replica for replica in self._replicas.values() if replica.active is None]
            if not available:
                break
            replica = min(available, key=lambda r: (r.latency_ema_s, r.replica_id))
            selected = self._select_request(replica.replica_id, now_s)
            self._bind(selected, replica, now_s)

    def _bind(self, request: _Request, replica: _Replica, now_s: float) -> None:
        assert replica.active is None
        self._pending_normals.remove(request)
        request.decision = Decision(replica.replica_id, request.deferred)
        request.bound_s = now_s
        replica.active = request
        request.future.set_result(request.decision)

    def _select_request(self, replica_id: int, now_s: float) -> _Request:
        depth = len(self._pending_normals)
        beta = (
            self.config.band_risk_beta
            if self.config.band_min_pending <= depth <= self.config.band_max_pending
            else self.config.risk_beta
        )
        return min(
            self._pending_normals,
            key=lambda request: (
                -(now_s - request.arrival_s + beta * request.estimated_service_s),
                request.sequence,
                request.request_id,
            ),
        )
