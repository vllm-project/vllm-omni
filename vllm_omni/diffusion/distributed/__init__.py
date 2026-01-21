# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed utilities for vLLM-Omni diffusion models."""

from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelConfig,
    SequenceParallelInput,
    SequenceParallelModelPlan,
    SequenceParallelOutput,
    SequenceParallelPartialInput,
    get_sp_plan_from_model,
    validate_sp_plan,
)
from vllm_omni.diffusion.distributed.sp_sharding import (
    ShardingValidator,
    get_sharding_validator,
    sp_gather,
    sp_shard,
    sp_shard_with_padding,
)

__all__ = [
    # Config
    "SequenceParallelConfig",
    # Plan types
    "SequenceParallelInput",
    "SequenceParallelOutput",
    "SequenceParallelPartialInput",
    "SequenceParallelModelPlan",
    "validate_sp_plan",
    "get_sp_plan_from_model",
    # Sharding utilities
    "sp_shard",
    "sp_gather",
    "sp_shard_with_padding",
    "ShardingValidator",
    "get_sharding_validator",
]
