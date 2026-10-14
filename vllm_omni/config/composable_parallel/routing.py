# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed routing-pattern hierarchy for declarative parallel strategies.

A routing pattern describes how a request batch is distributed across the
workers of one mesh axis (e.g. broadcast to every tensor-parallel rank, hash
partition across data-parallel ranks). The hierarchy is *closed*: subclasses
may only be defined in this module, so the set of patterns is a fixed,
exhaustively matchable contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RoutingPattern:
    """Base class for closed routing patterns. Subclass only within this module."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__module__ != __name__:
            raise TypeError("RoutingPattern is closed to external subclassing")

    def pattern_kind(self) -> str:
        raise NotImplementedError


class RoutingKeyError(KeyError):
    """Raised when a routing key is missing from a request."""

    def __init__(self, request_id: object, missing_key: str) -> None:
        self.request_id = request_id
        self.missing_key = missing_key
        super().__init__(f"Missing routing key {missing_key!r} for request_id={request_id!r}")


@dataclass(frozen=True)
class Broadcast(RoutingPattern):
    """Every worker in the group sees the whole batch (TP, dense EP)."""

    def pattern_kind(self) -> str:
        return "broadcast"


@dataclass(frozen=True)
class PartitionByHash(RoutingPattern):
    """Stable hash partition by request key (DP, HSDP)."""

    key: str = "request_id"

    def pattern_kind(self) -> str:
        return "partition_by_hash"


@dataclass(frozen=True)
class ShardSequence(RoutingPattern):
    """Shard along a sequence dimension (SP-Ulysses, SP-Ring, CP)."""

    dim: int = 1

    def pattern_kind(self) -> str:
        return "shard_sequence"


@dataclass(frozen=True)
class ShardSpatial(RoutingPattern):
    """Shard along a spatial grid (VAE-PP)."""

    grid: tuple[int, int] = (1, 1)

    def pattern_kind(self) -> str:
        return "shard_spatial"


@dataclass(frozen=True)
class RouteByExpert(RoutingPattern):
    """Sparse MoE expert routing."""

    router_id: str = "default"

    def pattern_kind(self) -> str:
        return "route_by_expert"


@dataclass(frozen=True)
class PipelineMicrobatch(RoutingPattern):
    """Pipeline parallelism microbatch routing (PP)."""

    microbatch_size: int = 1

    def pattern_kind(self) -> str:
        return "pipeline_microbatch"


@dataclass(frozen=True)
class DuplicateWithCondTag(RoutingPattern):
    """Duplicate batch with a conditional tag (CFG)."""

    flag_name: str = "guidance_branch"

    def pattern_kind(self) -> str:
        return "duplicate_with_cond_tag"


@dataclass(frozen=True)
class RouteByStage(RoutingPattern):
    """Route requests by pipeline stage policy (stage replicas).

    The ``random``/``round_robin``/``least_queue`` policies describe stateless
    load balancing that a runtime (e.g. an engine's built-in DP load balancer,
    or the omni StagePool balancer) can realize. ``hash`` describes
    deterministic, key-stable routing that a load balancer cannot honor on its
    own (see ``PartitionByHash``).
    """

    routing_policy: Literal["random", "round_robin", "least_queue", "hash"] = "round_robin"

    def pattern_kind(self) -> str:
        return "route_by_stage"


ROUTING_PATTERN_VARIANTS: tuple[type[RoutingPattern], ...] = (
    Broadcast,
    PartitionByHash,
    ShardSequence,
    ShardSpatial,
    RouteByExpert,
    PipelineMicrobatch,
    DuplicateWithCondTag,
    RouteByStage,
)

# Optional callable slot for Combine-style custom routing reducers in adapters.
RouterFn = Callable[..., object]
