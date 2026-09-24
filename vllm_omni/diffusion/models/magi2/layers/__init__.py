# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stable MAGI-2 layer API with platform-specific optimized overrides.

The complete Native implementation is imported first. A supported hardware
backend then replaces only the symbols it optimizes, leaving Native as the
fallback for every other layer and unsupported input contract.
"""

from vllm_omni.platforms import current_omni_platform

# Preserve the helper attributes exposed by the former ``layers.py`` module.
from ..parallel import get_magi2_ep_group, get_magi2_tp_group
from .native import (
    ElementWiseFourierEmbed,
    Magi2GroupedLinear,
    Magi2MultiHeadMoE,
    Magi2MultiHeadMoEConfig,
    MHCHandler,
    ModalityDispatcher,
    MultiModalityRMSNorm,
    apply_rotary_emb,
    compute_topk_probs_and_indices,
    correct_out_lse_with_sink,
    global_sort_routes,
    make_grouped_linear,
    rotate_half,
    sinkhorn_knopp,
    swiglu7,
    torch_mh_moe_forward,
    torch_varlen_attention_with_sink,
    varlen_attention_with_sink,
)

if current_omni_platform.is_cuda():
    from .cuda import Magi2MultiHeadMoE, varlen_attention_with_sink
elif current_omni_platform.is_npu():
    from .npu import (
        Magi2MultiHeadMoE,
        MultiModalityRMSNorm,
        apply_rotary_emb,
        varlen_attention_with_sink,
    )

__all__ = [
    "ElementWiseFourierEmbed",
    "MHCHandler",
    "Magi2GroupedLinear",
    "Magi2MultiHeadMoE",
    "Magi2MultiHeadMoEConfig",
    "ModalityDispatcher",
    "MultiModalityRMSNorm",
    "apply_rotary_emb",
    "compute_topk_probs_and_indices",
    "correct_out_lse_with_sink",
    "get_magi2_ep_group",
    "get_magi2_tp_group",
    "global_sort_routes",
    "make_grouped_linear",
    "rotate_half",
    "sinkhorn_knopp",
    "swiglu7",
    "torch_mh_moe_forward",
    "torch_varlen_attention_with_sink",
    "varlen_attention_with_sink",
]
