# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from ...common_layers import (
    ElementWiseFourierEmbed,
    Magi2GroupedLinear,
    MHCHandler,
    MHCTensorTuple,
    ModalityDispatcher,
    MultiModalityRMSNorm,
    make_grouped_linear,
    sinkhorn_knopp,
    swiglu7,
)
from .attention_ops import correct_out_lse_with_sink, torch_varlen_attention_with_sink, varlen_attention_with_sink
from .mh_moe import (
    Magi2MultiHeadMoE,
    Magi2MultiHeadMoEConfig,
    compute_topk_probs_and_indices,
    global_sort_routes,
    torch_mh_moe_forward,
)
from .rope import apply_rotary_emb, rotate_half

__all__ = [
    "ElementWiseFourierEmbed",
    "MHCHandler",
    "MHCTensorTuple",
    "Magi2GroupedLinear",
    "Magi2MultiHeadMoE",
    "Magi2MultiHeadMoEConfig",
    "ModalityDispatcher",
    "MultiModalityRMSNorm",
    "apply_rotary_emb",
    "compute_topk_probs_and_indices",
    "correct_out_lse_with_sink",
    "global_sort_routes",
    "make_grouped_linear",
    "rotate_half",
    "sinkhorn_knopp",
    "swiglu7",
    "torch_mh_moe_forward",
    "torch_varlen_attention_with_sink",
    "varlen_attention_with_sink",
]
