# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import random
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TypedDict

from .messages import InstanceInfo


class Task(TypedDict, total=False):
    """Task structure passed from async_omni (stage.submit(task)).

    Mirrors the dict built in AsyncOmni with request_id, engine_inputs,
    sampling_params. Future load-balancing policies may use these fields.
    """

    request_id: str
    engine_inputs: Any
    sampling_params: Any


class LoadBalancingPolicy(str, Enum):
    """Enumeration for load balancing policies.

    These policies are used by :class:`LoadBalancer` implementations to route
    tasks to a subset of available instances.

    TODO(NumberWan): Map enum values to balancer classes when OmniCoordinator
    integration lands. Tracked in https://github.com/vllm-project/vllm-omni/pull/2448
    """

    RANDOM = "random"
    ROUND_ROBIN = "round-robin"
    LEAST_QUEUE_LENGTH = "least-queue-length"


class LoadBalancer(ABC):
    """Abstract base class for load balancers.

    Subclasses implement :meth:`select` to choose an instance for a given task.
    """

    @abstractmethod
    def select(self, task: Task, instances: list[InstanceInfo]) -> int:
        """Route a task to one of the available instances.

        Args:
            task: The task to route. Not used by the random policy but reserved
                for future strategies that may inspect task metadata.
            instances: List of available instances to choose from.

        Returns:
            Index of the selected instance in ``instances``.

        Raises:
            ValueError: If ``instances`` is empty.
        """

        raise NotImplementedError


class RandomBalancer(LoadBalancer):
    """Load balancer that selects an instance uniformly at random.

    It intentionally ignores the task payload and chooses a random index from
    the provided instance list. More sophisticated policies (e.g. round-robin,
    least-queue-length) can be implemented as additional subclasses of
    :class:`LoadBalancer`.
    """

    def select(self, task: Task, instances: list[InstanceInfo]) -> int:  # noqa: ARG002
        if not instances:
            raise ValueError("instances must not be empty")

        return random.randrange(len(instances))


class RoundRobinBalancer(LoadBalancer):
    """Load balancer that selects instances in a round-robin fashion.

    This implementation keeps a running index modulo ``len(instances)``. It
    therefore depends on the **order and stable meaning** of the ``instances``
    list between calls. If the list length or ordering changes, the sequence
    of picks may skip or repeat entries relative to a fixed set of backends.

    When instance membership changes dynamically, callers should reset routing
    state—for example by constructing a new ``RoundRobinBalancer`` or resetting
    ``_next_index``—similar to rebuilding ``itertools.cycle`` after mutating
    the instance list (as in vLLM's disaggregated proxy examples).

    Concurrency: ``select`` is synchronous and is expected to run on the
    coordinator asyncio event loop thread without ``await`` inside this
    method, so a single invocation is not interleaved with another on that
    thread. A :class:`threading.Lock` still serializes updates to
    ``_next_index`` for callers that might invoke ``select`` from multiple
    threads or alongside threaded infrastructure (e.g. ZMQ receive threads).
    """

    def __init__(self, start_index: int = 0) -> None:
        self._next_index = start_index
        self._lock = threading.Lock()

    def select(self, task: Task, instances: list[InstanceInfo]) -> int:  # noqa: ARG002
        if not instances:
            raise ValueError("instances must not be empty")

        n = len(instances)
        with self._lock:
            idx = self._next_index % n
            self._next_index = (self._next_index + 1) % n
        return idx


class LeastQueueLengthBalancer(LoadBalancer):
    """Select the instance with the smallest ``queue_length``.

    If multiple instances share the same minimum queue length, one of them is
    chosen uniformly at random.

    Raises:
        ValueError: If any instance has a negative ``queue_length``.
    """

    def select(self, task: Task, instances: list[InstanceInfo]) -> int:  # noqa: ARG002
        if not instances:
            raise ValueError("instances must not be empty")

        queue_lengths = [inst.queue_length for inst in instances]
        if any(q < 0 for q in queue_lengths):
            raise ValueError("queue_length must be non-negative for all instances")
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
