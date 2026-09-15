# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runtime toggles shared by the CosyVoice3 stages."""

import os
from contextlib import nullcontext

import torch

_FALSE_ENV_VALUES = ("0", "false", "False", "")


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in _FALSE_ENV_VALUES


def cosyvoice3_batch_flow_enabled() -> bool:
    """Return whether cross-request Stage-1 flow batching is enabled."""
    return _env_flag("COSYVOICE3_BATCH_FLOW", "0")


def cosyvoice3_batch_flow_debug() -> bool:
    """Return whether Stage-1 batching diagnostics are enabled."""
    return _env_flag("COSYVOICE3_BATCH_FLOW_DEBUG", "0")


def cosyvoice3_batch_flow_profile(name: str):
    """Create a profiler scope only when batching diagnostics are enabled."""
    if cosyvoice3_batch_flow_debug():
        return torch.profiler.record_function(name)
    return nullcontext()
