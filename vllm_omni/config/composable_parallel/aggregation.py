# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed aggregation-pattern hierarchy for declarative parallel strategies.

An aggregation pattern describes how the per-worker results of one mesh axis
are recombined into a single logical result (e.g. take one tensor-parallel
rank's output, union disjoint data-parallel shards). The hierarchy is *closed*:
subclasses may only be defined in this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class AggregationPattern:
    """Base class for closed aggregation patterns. Subclass only within this module."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__module__ != __name__:
            raise TypeError("AggregationPattern is closed to external subclassing")

    def pattern_kind(self) -> str:
        raise NotImplementedError


class AggregationConflictError(ValueError):
    """Raised when two aggregation inputs conflict for the same request id."""


@dataclass(frozen=True)
class TakeRank(AggregationPattern):
    """Select one rank's result (TP — all ranks agree)."""

    rank: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise TypeError(f"TakeRank.rank must be int, got {type(self.rank).__name__}")

    def pattern_kind(self) -> str:
        return "take_rank"


@dataclass(frozen=True)
class Union(AggregationPattern):
    """Union of disjoint replica results (DP, HSDP)."""

    def pattern_kind(self) -> str:
        return "union"


@dataclass(frozen=True)
class GatherDim(AggregationPattern):
    """Gather along a dimension and materialize on one designated rank.

    Default designated rank is axis-rank 0 unless adapter overrides.
    """

    dim: int = 1

    def pattern_kind(self) -> str:
        return "gather_dim"


@dataclass(frozen=True)
class AllGather(AggregationPattern):
    """All ranks materialize the same aggregate result across the axis."""

    dim: int | None = None

    def pattern_kind(self) -> str:
        return "all_gather"


@dataclass(frozen=True)
class StitchSpatial(AggregationPattern):
    """Stitch spatial patches (VAE-PP)."""

    grid: tuple[int, int] = (1, 1)

    def pattern_kind(self) -> str:
        return "stitch_spatial"


@dataclass(frozen=True)
class StitchPipeline(AggregationPattern):
    """Stitch pipeline stage outputs (PP)."""

    def pattern_kind(self) -> str:
        return "stitch_pipeline"


@dataclass(frozen=True)
class Combine(AggregationPattern):
    """Custom reducer combination (CFG). Adapter resolves reducer_id."""

    reducer_id: str = "default"

    def pattern_kind(self) -> str:
        return "combine"


@dataclass(frozen=True)
class FanInByStage(AggregationPattern):
    """Fan-in results routed by stage (stage replicas)."""

    def pattern_kind(self) -> str:
        return "fan_in_by_stage"


AGGREGATION_PATTERN_VARIANTS: tuple[type[AggregationPattern], ...] = (
    TakeRank,
    Union,
    GatherDim,
    AllGather,
    StitchSpatial,
    StitchPipeline,
    Combine,
    FanInByStage,
)

ReducerFn = Callable[..., object]
