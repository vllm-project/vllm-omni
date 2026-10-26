# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One model-local dispatch path for local and Ulysses multiview attention."""

import torch

from .multiview_flex_attention import (
    MultiviewAttentionContext,
    padded_multiview_triton_attention,
)


def multiview_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
    context: MultiviewAttentionContext,
) -> torch.Tensor:
    if context.layout.backend == "maskless":
        # Imported lazily: the module registers a custom op and is only needed
        # on workers that resolved the maskless backend.
        from .multiview_maskless_attention import maskless_multiview_attention

        return maskless_multiview_attention(q, k, v, k_und, v_und, context)
    if context.layout.backend == "triton":
        return padded_multiview_triton_attention(q, k, v, k_und, v_und, context)
    raise ValueError(f"Unsupported Cosmos3 multiview attention backend {context.layout.backend!r}.")
