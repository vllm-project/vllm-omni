# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session memory for AR-diffusion world models (RFC #4480).

Typed ``StateObject`` instances owned per session by ``SessionStateManager``.
This package is model-agnostic; a model opts in by adapting its own per-session
cache onto this contract (e.g. ``state_dreamzero_adapter.DreamZeroStateAdapter``).
See RFC #4480 for the full design and roadmap.
"""

from vllm_omni.experimental.world_models.session_state.accounting import (
    SeenStorages,
    device_bytes,
)
from vllm_omni.experimental.world_models.session_state.attrs import SessionAttr
from vllm_omni.experimental.world_models.session_state.base import StateObject
from vllm_omni.experimental.world_models.session_state.manager import (
    DEFAULT_MAX_SESSIONS,
    SessionState,
    SessionStateManager,
    resolve_session_state_config,
)
from vllm_omni.experimental.world_models.session_state.objects import LatentBuffer

__all__ = [
    "DEFAULT_MAX_SESSIONS",
    "LatentBuffer",
    "SeenStorages",
    "StateObject",
    "SessionAttr",
    "SessionState",
    "SessionStateManager",
    "device_bytes",
    "resolve_session_state_config",
]
