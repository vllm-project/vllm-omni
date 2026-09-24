# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.

"""Ascend NPU multi-head MoE implementation for MAGI-2 Preview."""

from __future__ import annotations

import logging

import torch

from ..native.mh_moe import (
    Magi2MultiHeadMoE as NativeMagi2MultiHeadMoE,
)
from ..native.mh_moe import (
    Magi2MultiHeadMoEConfig,
    swiglu7_pair,
)

logger = logging.getLogger(__name__)
_NPU_MH_MOE_LOGGED = False


def npu_mh_moe_forward(
    x: torch.Tensor,
    topk_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Ascend grouped-MatMul path for independently routed MAGI-2 MoE heads."""

    global _NPU_MH_MOE_LOGGED
    if x.ndim != 3:
        raise ValueError("multi-head MoE input must be [tokens,heads,head_dim]")
    sequence, heads, head_dim = x.shape
    if topk_probs.shape != topk_indices.shape or topk_indices.ndim != 3:
        raise ValueError("top-k probabilities and indices must have [heads,tokens,k]")
    if topk_indices.shape[:2] != (heads, sequence):
        raise ValueError(
            "top-k routing shape does not match the multi-head MoE input: "
            f"{tuple(topk_indices.shape)} vs {tuple(x.shape)}"
        )
    num_flat_experts = heads * num_experts
    if num_flat_experts > 1024:
        raise ValueError(
            "Ascend grouped MatMul supports at most 1024 local expert groups; "
            f"got {heads} heads x {num_experts} experts. Use SP/TP head sharding."
        )
    if w_gate.shape[0] != num_flat_experts:
        raise ValueError(f"expected {num_flat_experts} flattened experts, got {w_gate.shape[0]}")

    import torch_npu

    top_k = topk_indices.shape[-1]
    flat_x = x.reshape(sequence * heads, head_dim).contiguous()
    head_offsets = torch.arange(heads, device=x.device, dtype=topk_indices.dtype).view(1, heads, 1).mul(num_experts)
    flat_ids = (topk_indices.permute(1, 0, 2).contiguous() + head_offsets).reshape(sequence * heads, top_k)
    flat_ids = flat_ids.to(torch.int32).contiguous()
    flat_probs = topk_probs.permute(1, 0, 2).contiguous().reshape(sequence * heads, top_k)

    routed = torch_npu.npu_moe_init_routing_v2(
        flat_x,
        flat_ids,
        scale=None,
        active_num=flat_x.shape[0] * top_k,
        expert_num=num_flat_experts,
        expert_tokens_num_type=0,
        expert_tokens_num_flag=True,
        active_expert_range=[0, num_flat_experts],
        quant_mode=-1,
        row_idx_type=0,
    )
    sorted_hidden, expanded_row_idx, expert_tokens = routed[:3]
    grouped_kwargs = {
        "split_item": 2,
        "group_list_type": 0,
        "group_type": 0,
        "group_list": expert_tokens.to(torch.int64),
    }
    gate = torch_npu.npu_grouped_matmul(
        x=[sorted_hidden],
        weight=[w_gate],
        **grouped_kwargs,
    )[0]
    up = torch_npu.npu_grouped_matmul(
        x=[sorted_hidden],
        weight=[w_up],
        **grouped_kwargs,
    )[0]
    hidden = swiglu7_pair(gate, up)
    sorted_output = torch_npu.npu_grouped_matmul(
        x=[hidden],
        weight=[w_down],
        **grouped_kwargs,
    )[0]
    output = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=sorted_output,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=flat_probs.to(sorted_output.dtype),
    )
    if not _NPU_MH_MOE_LOGGED:
        logger.info("MAGI-2 selected Ascend multi-head MoE (routing-v2 + grouped-matmul + SwiGLU7)")
        _NPU_MH_MOE_LOGGED = True
    return output.view_as(x)


class Magi2MultiHeadMoE(NativeMagi2MultiHeadMoE):
    """Checkpoint-compatible MAGI-2 MoE with Ascend fused local kernels."""

    def _local_forward(self, x_heads: torch.Tensor) -> torch.Tensor:
        if x_heads.device.type != "npu":
            return super()._local_forward(x_heads)
        probabilities, indices = self._route(x_heads)
        return npu_mh_moe_forward(
            x_heads,
            probabilities,
            indices,
            self.num_experts,
            self.W_gate,
            self.W_up,
            self.W_down,
        )


__all__ = [
    "Magi2MultiHeadMoE",
    "Magi2MultiHeadMoEConfig",
    "npu_mh_moe_forward",
]
