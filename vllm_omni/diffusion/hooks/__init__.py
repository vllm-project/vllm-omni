# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hook mechanism for model forward interception."""

from vllm_omni.diffusion.hooks.base import (
    HookRegistry,
    ModelHook,
    StateManager,
)
from vllm_omni.diffusion.hooks.mixfusion import (
    MixFusionExpandArgHook,
    MixFusionExpandRowsHook,
    MixFusionReplaceOutputHook,
    MixFusionRootHook,
    apply_mixfusion,
    remove_mixfusion,
)
from vllm_omni.diffusion.hooks.sequence_parallel import (
    SequenceParallelGatherHook,
    SequenceParallelSplitHook,
    apply_sequence_parallel,
    remove_sequence_parallel,
)

__all__ = [
    # Base hooks
    "StateManager",
    "ModelHook",
    "HookRegistry",
    # Sequence parallel hooks (corresponds to diffusers' context_parallel)
    "SequenceParallelSplitHook",
    "SequenceParallelGatherHook",
    "apply_sequence_parallel",
    "remove_sequence_parallel",
    # MixFusion packed-layout hooks (same mechanism as the SP hooks above)
    "MixFusionRootHook",
    "MixFusionReplaceOutputHook",
    "MixFusionExpandRowsHook",
    "MixFusionExpandArgHook",
    "apply_mixfusion",
    "remove_mixfusion",
]
