# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compiler boundary for the upstream Sol-Attn kernel."""

from __future__ import annotations

import torch

if not hasattr(torch.ops.vllm_omni, "sol_attention"):

    @torch.library.custom_op("vllm_omni::sol_attention", mutates_args=())
    def _sol_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        tau: float,
        kv_splits: int,
        exact_block_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        from sol_attn import sol_attn

        return sol_attn(
            query,
            key,
            value,
            scale=scale,
            tau=tau,
            thresh_type="diag",
            kv_splits=kv_splits,
            exact_block_mask=exact_block_mask,
        )

    @_sol_attention.register_fake
    def _sol_attention_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        tau: float,
        kv_splits: int,
        exact_block_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return torch.empty_like(query)


sol_attention = torch.ops.vllm_omni.sol_attention
