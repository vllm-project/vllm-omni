# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composable parallel strategies for vLLM-Omni.

A small, runtime-agnostic vocabulary for declaring a stage's parallel layout
(:class:`StrategySpec` + mesh/routing/aggregation patterns), plus the logic to
translate a per-role stack of specs into concrete engine sizing and overlay it
onto merged stage configs.

The spec/pattern *types* are deliberately pure data with no engine imports; the
translation and apply logic lives alongside them and consumes vLLM-Omni's own
config objects.
"""

from vllm_omni.config.composable_parallel.aggregation import (
    AGGREGATION_PATTERN_VARIANTS,
    AggregationConflictError,
    AggregationPattern,
    AllGather,
    Combine,
    FanInByStage,
    GatherDim,
    StitchPipeline,
    StitchSpatial,
    TakeRank,
    Union,
)
from vllm_omni.config.composable_parallel.apply import (
    StrategyApplyError,
    StrategyApplyResult,
    apply_strategy_specs,
    check_device_layout,
)
from vllm_omni.config.composable_parallel.routing import (
    ROUTING_PATTERN_VARIANTS,
    Broadcast,
    DuplicateWithCondTag,
    PartitionByHash,
    PipelineMicrobatch,
    RouteByExpert,
    RouteByStage,
    RoutingKeyError,
    RoutingPattern,
    ShardSequence,
    ShardSpatial,
)
from vllm_omni.config.composable_parallel.spec import (
    MESH_AXIS_KINDS,
    KernelSpec,
    LayerHookSpec,
    MeshAxisKind,
    MeshAxisSpec,
    SpecMergeConflictError,
    StrategySpec,
)
from vllm_omni.config.composable_parallel.translator import (
    AxisTranslationError,
    OmniParallelConfig,
    translate_strategy_stack,
)

__all__ = [
    # spec types
    "MeshAxisKind",
    "MESH_AXIS_KINDS",
    "MeshAxisSpec",
    "LayerHookSpec",
    "KernelSpec",
    "StrategySpec",
    "SpecMergeConflictError",
    # routing patterns
    "RoutingPattern",
    "RoutingKeyError",
    "Broadcast",
    "PartitionByHash",
    "ShardSequence",
    "ShardSpatial",
    "RouteByExpert",
    "PipelineMicrobatch",
    "DuplicateWithCondTag",
    "RouteByStage",
    "ROUTING_PATTERN_VARIANTS",
    # aggregation patterns
    "AggregationPattern",
    "AggregationConflictError",
    "TakeRank",
    "Union",
    "GatherDim",
    "AllGather",
    "StitchSpatial",
    "StitchPipeline",
    "Combine",
    "FanInByStage",
    "AGGREGATION_PATTERN_VARIANTS",
    # translation
    "OmniParallelConfig",
    "translate_strategy_stack",
    "AxisTranslationError",
    # apply
    "apply_strategy_specs",
    "StrategyApplyResult",
    "check_device_layout",
    "StrategyApplyError",
]
