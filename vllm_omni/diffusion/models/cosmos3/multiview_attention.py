# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One model-local dispatch path for local and Ulysses multiview attention."""

import torch

from .multiview_flex_attention import MultiviewAttentionContext, padded_multiview_flex_attention
from .multiview_maskless_attention import maskless_attention_op


def multiview_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
    context: MultiviewAttentionContext,
) -> torch.Tensor:
    if context.maskless_plan is not None:
        if q.shape[0] != 1:
            raise ValueError("Maskless multiview attention requires B == 1.")
        plan, scratch = context.maskless_plan
        return maskless_attention_op(q, k, v, k_und, v_und, plan, scratch, context.fa_version)
    if context.layout.backend == "maskless":
        raise RuntimeError("Prepare the model-local maskless plan before entering compiled GEN layers.")
    return padded_multiview_flex_attention(q, k, v, k_und, v_und, context)
