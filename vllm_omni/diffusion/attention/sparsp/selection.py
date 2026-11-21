# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Provider-independent CDF dependency selection over pooled global keys."""

import torch


def cdf_block_mask(
    query: torch.Tensor,
    keys: torch.Tensor,
    similar_query: torch.Tensor,
    similar_keys: torch.Tensor,
    cdf: float,
    scale: float,
) -> torch.Tensor:
    """Return the global pooled-key dependencies for each local query block."""
    score = (query @ keys.transpose(-1, -2)) * scale
    score.masked_fill_(~similar_keys.unsqueeze(-2), -torch.inf)
    probability = score.softmax(-1)
    values, indices = probability.sort(dim=-1, descending=True, stable=True)
    count = (values.cumsum(-1) < cdf).sum(-1, keepdim=True) + 1
    keep = torch.arange(values.shape[-1], device=query.device) < count
    selected = torch.zeros_like(keep).scatter_(-1, indices, keep)
    selected |= ~similar_keys.unsqueeze(-2)
    selected |= ~similar_query.unsqueeze(-1)
    return selected.contiguous()
