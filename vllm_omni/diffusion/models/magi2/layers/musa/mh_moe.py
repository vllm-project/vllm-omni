# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.

"""MUSA MAGI-2 BF16 grouped-expert dispatch."""

from __future__ import annotations

import os

import torch

from ...fused_moe_kernels import global_sort_routes, triton_mh_moe_forward
from ..cuda.mh_moe import BF16Magi2MultiHeadMoE, _bf16_fused_moe_forward
from ..native.mh_moe import torch_mh_moe_forward

_BF16_MOE_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 16,
    "num_warps": 16,
    "num_stages": 1,
}


class Magi2MultiHeadMoE(BF16Magi2MultiHeadMoE):
    """MUSA experts retain native routing and their original BF16 settings."""

    def _local_forward(self, x_heads: torch.Tensor) -> torch.Tensor:
        probabilities, indices = self._route(x_heads)
        if (
            x_heads.device.type == "musa"
            and x_heads.dtype == torch.bfloat16
            and os.environ.get("MAGI2_DETERMINISTIC", "0") != "1"
            and not torch.compiler.is_compiling()
        ):
            return _bf16_fused_moe_forward(
                x_heads,
                probabilities,
                indices,
                self._get_bf16_packed_w13(),
                self.W_down,
                self._get_bf16_route_buffers(x_heads.shape[0] * self.local_num_heads * self.top_k, x_heads.device),
                config=_BF16_MOE_CONFIG,
            )
        gather_ids, sorted_probs, offsets = global_sort_routes(probabilities, indices, self.num_experts)
        if x_heads.device.type == "musa":
            return triton_mh_moe_forward(
                x_heads,
                gather_ids,
                sorted_probs,
                offsets,
                self.W_gate,
                self.W_up,
                self.W_down,
                deterministic=os.environ.get("MAGI2_DETERMINISTIC", "0") == "1",
            )
        return torch_mh_moe_forward(x_heads, gather_ids, sorted_probs, offsets, self.W_gate, self.W_up, self.W_down)
