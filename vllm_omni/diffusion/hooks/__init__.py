# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hook mechanism for model forward interception."""

from vllm_omni.diffusion.hooks.base import (
    BaseState,
    HookRegistry,
    ModelHook,
    StateManager,
)
from vllm_omni.diffusion.hooks.sequence_parallel import (
    SequenceParallelGatherHook,
    SequenceParallelSplitHook,
    apply_sequence_parallel,
    disable_sequence_parallel_for_model,
    enable_sequence_parallel_for_model,
    remove_sequence_parallel,
)

__all__ = [
    # Base hooks
    "BaseState",
    "StateManager",
    "ModelHook",
    "HookRegistry",
    # Sequence parallel hooks (corresponds to diffusers' context_parallel)
    "SequenceParallelSplitHook",
    "SequenceParallelGatherHook",
    "apply_sequence_parallel",
    "remove_sequence_parallel",
    "enable_sequence_parallel_for_model",
    "disable_sequence_parallel_for_model",
]
