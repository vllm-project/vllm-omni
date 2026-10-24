# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared HF-compatible routing for Hunyuan AR and DiT experts."""

import torch


def pack_hunyuan_topk(logits: torch.Tensor, topk: int, dtype: torch.dtype) -> torch.Tensor:
    gates = torch.softmax(logits, dim=-1, dtype=torch.float32)
    weights, indices = torch.topk(gates, topk, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    # HF combines expert outputs with model-dtype routing weights.
    return torch.cat([weights.to(dtype).float(), indices.to(torch.float32)], dim=-1)


def unpack_hunyuan_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_experts: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native FusedMoE callback; weights have already been normalized."""
    return gating_output[:, :topk].contiguous().float(), gating_output[:, topk:].to(torch.int32)
