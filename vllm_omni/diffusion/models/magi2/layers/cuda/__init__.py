# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CUDA-optimized MAGI-2 layer overrides."""

from .attention_ops import varlen_attention_with_sink
from .mh_moe import Magi2MultiHeadMoE, Magi2MultiHeadMoEConfig, triton_mh_moe_forward

__all__ = [
    "Magi2MultiHeadMoE",
    "Magi2MultiHeadMoEConfig",
    "triton_mh_moe_forward",
    "varlen_attention_with_sink",
]
