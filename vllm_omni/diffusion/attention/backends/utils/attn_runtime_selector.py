# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime kernel-selection helpers for attention backends."""

import torch


def can_sdpa_use_fused_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    causal: bool,
) -> bool:
    """Return whether PyTorch SDPA can use a fused kernel for this GQA call.

    This check is specific to the SDPA backend. Native GQA backends such as
    FlashAttention should keep compressed K/V heads and do not use this helper.
    """
    if not query.is_cuda:
        return False

    try:
        params = torch.backends.cuda.SDPAParams(
            query,
            key,
            value,
            attention_mask,
            0.0,
            causal,
            True,
        )
        can_use_flash = torch.backends.cuda.can_use_flash_attention(params)
        can_use_cudnn = getattr(torch.backends.cuda, "can_use_cudnn_attention", None)
        return can_use_flash or (can_use_cudnn is not None and can_use_cudnn(params))
    except (AttributeError, RuntimeError, TypeError):
        # Older PyTorch versions may not expose GQA-aware SDPA capability checks.
        # Returning False lets the SDPA backend expand K/V and use the mature
        # equal-head fused-kernel path instead.
        return False
