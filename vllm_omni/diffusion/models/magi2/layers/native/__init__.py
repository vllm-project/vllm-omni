# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Complete, portable MAGI-2 layer implementation."""

from .activation import swiglu7
from .attention_ops import (
    correct_out_lse_with_sink,
    torch_varlen_attention_with_sink,
    varlen_attention_with_sink,
)
from .dispatcher import ModalityDispatcher
from .embedding import ElementWiseFourierEmbed
from .linear import Magi2GroupedLinear, make_grouped_linear
from .mh_moe import (
    Magi2MultiHeadMoE,
    Magi2MultiHeadMoEConfig,
    compute_topk_probs_and_indices,
    global_sort_routes,
    torch_mh_moe_forward,
)
from .mhc import MHCHandler, sinkhorn_knopp
from .normalization import MultiModalityRMSNorm
from .rotary_embedding import apply_rotary_emb, rotate_half

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
    "global_sort_routes",
    "make_grouped_linear",
    "rotate_half",
    "sinkhorn_knopp",
    "swiglu7",
    "torch_mh_moe_forward",
    "torch_varlen_attention_with_sink",
    "varlen_attention_with_sink",
]
