# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session memory for AR-diffusion world models (RFC #4480).

Typed ``MemoryObject`` instances owned per session by ``SessionMemoryManager``.
This package is model-agnostic; a model opts in by adapting its own per-session
cache onto this contract (e.g. ``state_dreamzero_adapter.DreamZeroStateAdapter``).
See RFC #4480 for the full design and roadmap.
"""

from vllm_omni.experimental.world_models.memory.attrs import (
    SessionAttr,
    window_reset_survivors,
)
from vllm_omni.experimental.world_models.memory.base import MemoryObject
from vllm_omni.experimental.world_models.memory.manager import (
    DEFAULT_MAX_SESSIONS,
    SessionMemory,
    SessionMemoryManager,
    resolve_session_memory_config,
)
from vllm_omni.experimental.world_models.memory.objects import (
    EncodeOnceKV,
    LatentBuffer,
)

__all__ = [
    "DEFAULT_MAX_SESSIONS",
    "EncodeOnceKV",
    "LatentBuffer",
    "MemoryObject",
    "SessionAttr",
    "SessionMemory",
    "SessionMemoryManager",
    "resolve_session_memory_config",
    "window_reset_survivors",
]
