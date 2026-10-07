# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import random
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TypedDict

from .messages import ReplicaInfo


class Task(TypedDict, total=False):
    """Task structure passed to ``StagePool.pick`` / ``LoadBalancer.select``.

    Mirrors the dict built around a stage submission with request_id and any
    payload-related fields a future load-balancing policy might inspect.
    """

    request_id: str
    engine_inputs: Any
    sampling_params: Any
    # Explicit caller-provided relative work estimate. When present and
    # finite-positive, the cost-aware balancer uses it directly instead of
    # deriving an estimate from ``sampling_params``.
    estimated_cost: float


class LoadBalancingPolicy(str, Enum):
    """Enumeration for load balancing policies.

    These policies are used by :class:`LoadBalancer` implementations to route
    tasks to a subset of available replicas.
    """

    RANDOM = "random"
    ROUND_ROBIN = "round-robin"
    LEAST_QUEUE_LENGTH = "least-queue-length"
    COST_AWARE = "cost-aware"


class LoadBalancer(ABC):
    """Abstract base class for load balancers.

    Subclasses implement :meth:`select` to choose a replica for a given task.
    """

    @abstractmethod
    def select(self, task: Task, replicas: list[ReplicaInfo]) -> int:
        """Route a task to one of the available replicas.

        Args:
            task: The task to route. Not used by the random policy but reserved
                for future strategies that may inspect task metadata.
            replicas: List of available replicas to choose from.

        Returns:
            Index of the selected replica in ``replicas``.

        Raises:
            ValueError: If ``replicas`` is empty.
        """

        raise NotImplementedError

    def release(self, request_id: str) -> None:
        """Drop any per-request reservation tracked by this balancer.

        Default no-op. Stateful balancers (e.g.
        :class:`CostAwareBalancer`) override this to forget a request's
        assigned replica and estimated cost so it no longer contributes to
        the replica's effective load. Safe to call for unknown request ids.
        """
        return None


class RandomBalancer(LoadBalancer):
    """Load balancer that selects a replica uniformly at random."""

    def select(self, task: Task, replicas: list[ReplicaInfo]) -> int:  # noqa: ARG002
        if not replicas:
            raise ValueError("replicas must not be empty")

        return random.randrange(len(replicas))


class RoundRobinBalancer(LoadBalancer):
    """Load balancer that selects replicas in a round-robin fashion.

    This implementation keeps a running index modulo ``len(replicas)``. It
    therefore depends on the **order and stable meaning** of the ``replicas``
    list between calls. If the list length or ordering changes, the sequence
    of picks may skip or repeat entries relative to a fixed set of backends.

    Concurrency: a ``threading.Lock`` serializes updates to ``_next_index``
    for callers that invoke ``select`` from multiple threads or alongside
    threaded infrastructure (e.g. ZMQ receive threads).
    """

    def __init__(self, start_index: int = 0) -> None:
        self._next_index = start_index
        self._lock = threading.Lock()

    def select(self, task: Task, replicas: list[ReplicaInfo]) -> int:  # noqa: ARG002
        if not replicas:
            raise ValueError("replicas must not be empty")

        n = len(replicas)
        with self._lock:
            idx = self._next_index % n
            self._next_index = (self._next_index + 1) % n
        return idx


class LeastQueueLengthBalancer(LoadBalancer):
    """Select the replica with the smallest ``queue_length``.

    If multiple replicas share the same minimum queue length, one of them is
    chosen uniformly at random.

    Raises:
        ValueError: If any replica has a negative ``queue_length``.
    """

    def select(self, task: Task, replicas: list[ReplicaInfo]) -> int:  # noqa: ARG002
        if not replicas:
            raise ValueError("replicas must not be empty")

        queue_lengths = [rep.queue_length for rep in replicas]
        if any(q < 0 for q in queue_lengths):
            raise ValueError("queue_length must be non-negative for all replicas")
        min_q = min(queue_lengths)
        candidates = [i for i, q in enumerate(queue_lengths) if q == min_q]
        return random.choice(candidates)


def _safe_positive_int(value: Any) -> int | None:
    """Return ``value`` as a positive int, or ``None`` if it is missing/invalid."""
    if value is None or isinstance(value, bool):
        return None
    try:
        v = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if v <= 0:
        return None
    return v


def _safe_positive_float(value: Any) -> float | None:
    """Return ``value`` as a finite positive float, or ``None`` if invalid."""
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v


def estimate_request_cost(task: Task) -> float:
    """Estimate the relative work of a request for cost-aware routing.

    Precedence:

    1. An explicit ``task["estimated_cost"]`` (finite, positive) wins.
    2. For diffusion ``sampling_params``, estimate relative work from spatial
       size, frame count, inference steps, and outputs per prompt. Spatial
       work is normalized against a 512x512 baseline so the resulting number
       stays interpretable (a 512x512 single-frame single-step single-output
       request has cost ``1.0``).
    3. Otherwise (no usable signal), return ``1.0``.

    The function is pure and CPU-safe: it never touches tensors and only
    reads plain scalar attributes via ``getattr``. Any missing, non-finite,
    or non-positive component falls back to its neutral multiplier rather
    than raising.
    """
    explicit = task.get("estimated_cost")
    if explicit is not None:
        v = _safe_positive_float(explicit)
        if v is not None:
            return v

    sampling_params = task.get("sampling_params")
    if sampling_params is None:
        return 1.0

    # Spatial work: (h * w) / (512 * 512). Use height/width when both valid,
    # otherwise fall back to ``resolution`` treated as a square side.
    height = _safe_positive_int(getattr(sampling_params, "height", None))
    width = _safe_positive_int(getattr(sampling_params, "width", None))
    if height is None or width is None:
        resolution = _safe_positive_int(getattr(sampling_params, "resolution", None))
        if resolution is not None:
            height = width = resolution

    spatial = 1.0
    if height is not None and width is not None:
        try:
            spatial = (float(height) / 512.0) * (float(width) / 512.0)
        except OverflowError:
            spatial = 1.0
        if not math.isfinite(spatial) or spatial <= 0:
            spatial = 1.0

    num_frames = _safe_positive_int(getattr(sampling_params, "num_frames", None)) or 1
    num_inference_steps = _safe_positive_int(getattr(sampling_params, "num_inference_steps", None)) or 1
    num_outputs_per_prompt = _safe_positive_int(getattr(sampling_params, "num_outputs_per_prompt", None)) or 1

    try:
        cost = spatial * num_frames * num_inference_steps * num_outputs_per_prompt
    except OverflowError:
        return 1.0
    if not math.isfinite(cost) or cost <= 0:
        return 1.0
    return cost


class CostAwareBalancer(LoadBalancer):
    """Route by projected work, not by request count alone.

    This balancer is stateful: it tracks ``request_id -> (input_addr,
    estimated_cost)`` for every assignment it has made and not yet released.
    On each :meth:`select` it computes each replica's *effective load* as the
    sum of locally tracked outstanding work plus an estimate for requests the
    coordinator reports on the replica's queue but that this head did not
    assign (and therefore has no cost for)::

        unknown_count = max(queue_length - locally_tracked_count, 0)
        effective_load = locally_tracked_work + unknown_count * representative_cost

    ``representative_cost`` is the mean estimated cost of all locally tracked
    outstanding requests when any exist; otherwise it falls back to the
    incoming task's own estimated cost.

    The replica with the minimum effective load is chosen. Ties are broken
    first by smaller coordinator-reported ``queue_length`` and then uniformly
    at random (consistent with :class:`LeastQueueLengthBalancer`).

    Lifecycle notes:

    * Selection for a ``request_id`` is idempotent: any prior reservation for
      the same id is removed before the new one is recorded, so reselecting a
      request never double-counts it.
    * Reservations whose replica is no longer in the candidate list are pruned
      on every selection (the replica disappeared).
    * :meth:`release` forgets a request's reservation. StagePool calls this
      from its binding-cleanup paths so completed/aborted/failed requests stop
      contributing to load.
    * Only work assigned by *this* head has an exact cost; coordinator
      ``queue_length`` covers work from other heads using the representative
      cost approximation.

    Raises:
        ValueError: If ``replicas`` is empty or any replica has a negative
            ``queue_length``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # request_id -> (replica input_addr, estimated_cost)
        self._reservations: dict[str, tuple[str, float]] = {}
        self._request_ids_by_addr: dict[str, set[str]] = {}
        self._work_by_addr: dict[str, float] = {}
        self._count_by_addr: dict[str, int] = {}
        self._total_work = 0.0
        self._total_count = 0

    def _drop_reservation_locked(self, request_id: str) -> None:
        reservation = self._reservations.pop(request_id, None)
        if reservation is None:
            return
        addr, cost = reservation
        request_ids = self._request_ids_by_addr[addr]
        request_ids.remove(request_id)
        count = self._count_by_addr[addr]
        if count == 1:
            self._request_ids_by_addr.pop(addr)
            self._work_by_addr.pop(addr)
            self._count_by_addr.pop(addr)
        else:
            self._work_by_addr[addr] -= cost
            self._count_by_addr[addr] = count - 1
        self._total_work -= cost
        self._total_count -= 1

    def _reserve_locked(self, request_id: str, addr: str, cost: float) -> None:
        self._reservations[request_id] = (addr, cost)
        self._request_ids_by_addr.setdefault(addr, set()).add(request_id)
        self._work_by_addr[addr] = self._work_by_addr.get(addr, 0.0) + cost
        self._count_by_addr[addr] = self._count_by_addr.get(addr, 0) + 1
        self._total_work += cost
        self._total_count += 1

    def select(self, task: Task, replicas: list[ReplicaInfo]) -> int:
        if not replicas:
            raise ValueError("replicas must not be empty")

        queue_lengths = [rep.queue_length for rep in replicas]
        if any(q < 0 for q in queue_lengths):
            raise ValueError("queue_length must be non-negative for all replicas")

        raw_request_id = task.get("request_id")
        request_id = raw_request_id if isinstance(raw_request_id, str) and raw_request_id else None
        cost = estimate_request_cost(task)

        with self._lock:
            # Idempotent re-selection: drop any prior reservation for this id.
            if request_id is not None:
                self._drop_reservation_locked(request_id)

            # Prune reservations for replicas that are no longer present.
            present_addrs = {rep.input_addr for rep in replicas}
            stale_addrs = set(self._request_ids_by_addr).difference(present_addrs)
            for addr in stale_addrs:
                for stale_id in tuple(self._request_ids_by_addr[addr]):
                    self._drop_reservation_locked(stale_id)

            # Representative cost for coordinator-reported unknown work.
            if self._total_count:
                representative_cost = self._total_work / self._total_count
            else:
                representative_cost = cost

            effective_loads: list[float] = []
            for rep in replicas:
                addr = rep.input_addr
                tracked_work = self._work_by_addr.get(addr, 0.0)
                tracked_count = self._count_by_addr.get(addr, 0)
                unknown_count = max(rep.queue_length - tracked_count, 0)
                effective_loads.append(tracked_work + unknown_count * representative_cost)

            min_load = min(effective_loads)
            candidates = [i for i, load in enumerate(effective_loads) if load == min_load]

            # Deterministic secondary: prefer the replica with the smaller
            # coordinator-reported queue length among effective-load ties.
            if len(candidates) > 1:
                min_q = min(queue_lengths[i] for i in candidates)
                candidates = [i for i in candidates if queue_lengths[i] == min_q]

            chosen = random.choice(candidates)
            if request_id is not None:
                self._reserve_locked(request_id, replicas[chosen].input_addr, cost)
            return chosen

    def release(self, request_id: str) -> None:
        """Forget the reservation for ``request_id`` (no-op if unknown)."""
        with self._lock:
            self._drop_reservation_locked(request_id)


__all__ = [
    "Task",
    "LoadBalancingPolicy",
    "LoadBalancer",
    "RandomBalancer",
    "RoundRobinBalancer",
    "LeastQueueLengthBalancer",
    "CostAwareBalancer",
    "estimate_request_cost",
]
