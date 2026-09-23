# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ascend NPU-optimized MAGI-2 layer overrides."""

from .attention_ops import varlen_attention_with_sink
from .mh_moe import Magi2MultiHeadMoE, Magi2MultiHeadMoEConfig, npu_mh_moe_forward
from .normalization import MultiModalityRMSNorm
from .rotary_embedding import apply_rotary_emb

__all__ = [
    "Magi2MultiHeadMoE",
    "Magi2MultiHeadMoEConfig",
    "MultiModalityRMSNorm",
    "apply_rotary_emb",
    "npu_mh_moe_forward",
    "varlen_attention_with_sink",
]
