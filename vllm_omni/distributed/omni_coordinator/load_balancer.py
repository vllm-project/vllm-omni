# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import random
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

    Only ``RANDOM`` is implemented. Additional policies (e.g. round-robin,
    least-connections) can be added in the future.
    """

    RANDOM = "random"


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

    This is the initial and only policy supported. It intentionally ignores
    the task payload and chooses a random index from the provided instance
    list. More sophisticated policies (e.g. round-robin, least-connections)
    can be implemented as additional subclasses of :class:`LoadBalancer`.
    """

    def select(self, task: Task, instances: list[InstanceInfo]) -> int:  # noqa: ARG002
        if not instances:
            raise ValueError("instances must not be empty")

        return random.randrange(len(instances))


__all__ = [
    "Task",
    "LoadBalancingPolicy",
    "LoadBalancer",
    "RandomBalancer",
]
