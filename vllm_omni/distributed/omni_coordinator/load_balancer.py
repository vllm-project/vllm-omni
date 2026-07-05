# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

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


class LoadBalancingPolicy(str, Enum):
    """Enumeration for load balancing policies.

    These policies are used by :class:`LoadBalancer` implementations to route
    tasks to a subset of available replicas.
    """

    RANDOM = "random"
    ROUND_ROBIN = "round-robin"
    LEAST_QUEUE_LENGTH = "least-queue-length"


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


__all__ = [
    "Task",
    "LoadBalancingPolicy",
    "LoadBalancer",
    "RandomBalancer",
    "RoundRobinBalancer",
    "LeastQueueLengthBalancer",
]
