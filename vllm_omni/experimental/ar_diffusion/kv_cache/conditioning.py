# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility exports for AR users of the shared conditioning lifecycle."""

from vllm_omni.diffusion.attention.conditioning import (
    ConditioningKVBuildTicket,
    ConditioningKVIdentity,
    ConditioningKVRepresentation,
    ConditioningKVScope,
)

__all__ = ["ConditioningKVBuildTicket", "ConditioningKVIdentity", "ConditioningKVRepresentation", "ConditioningKVScope"]
