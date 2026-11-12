# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MAGI-2 native defaults with explicit hardware-optimized layer overrides."""

from vllm_omni.platforms import current_omni_platform

from .native import (
    Magi2MultiHeadMoE,
    Magi2MultiHeadMoEConfig,
    MHCHandler,
    MultiModalityRMSNorm,
    apply_rotary_emb,
    varlen_attention_with_sink,
)

if current_omni_platform.is_cuda():
    from .cuda import Magi2MultiHeadMoE, varlen_attention_with_sink
elif current_omni_platform.is_musa():
    from .musa import Magi2MultiHeadMoE, MHCHandler, varlen_attention_with_sink
elif current_omni_platform.is_npu():
    from .npu import Magi2MultiHeadMoE, MultiModalityRMSNorm, apply_rotary_emb, varlen_attention_with_sink

__all__ = [
    "MHCHandler",
    "Magi2MultiHeadMoE",
    "Magi2MultiHeadMoEConfig",
    "MultiModalityRMSNorm",
    "apply_rotary_emb",
    "varlen_attention_with_sink",
]
