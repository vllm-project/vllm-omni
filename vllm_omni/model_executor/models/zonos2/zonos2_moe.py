# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""vLLM Triton experts with ZONOS2's FP32 SiLU/up multiplication boundary."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts


class Zonos2TritonExperts(TritonExperts):
    """Reuse vLLM routing/GEMMs while rounding the gated activation only once."""

    @classmethod
    def create(cls, x: torch.Tensor, w13: torch.Tensor, topk: int) -> Zonos2TritonExperts:
        parallel = FusedMoEParallelConfig(
            tp_size=1,
            pcp_size=1,
            dp_size=1,
            ep_size=1,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=1,
            use_ep=False,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        )
        config = FusedMoEConfig(
            num_experts=w13.shape[0],
            experts_per_token=topk,
            hidden_dim=x.shape[-1],
            intermediate_size=w13.shape[1] // 2,
            num_local_experts=w13.shape[0],
            num_logical_experts=w13.shape[0],
            activation=MoEActivation.SILU,
            device=x.device,
            routing_method=RoutingMethodType.Unspecified,
            moe_parallel_config=parallel,
            in_dtype=x.dtype,
        )
        return cls(config, FUSED_MOE_UNQUANTIZED_CONFIG)

    def activation(self, activation: MoEActivation, output: torch.Tensor, input: torch.Tensor, **kwargs: Any) -> None:
        if activation != MoEActivation.SILU:
            raise ValueError("ZONOS2 experts require SiLU")
        gate, up = input.chunk(2, dim=-1)
        # The default CUDA silu_and_mul rounds SiLU to the activation dtype
        # before multiplying. Official ZONOS2 keeps both operations in FP32.
        output.copy_(F.silu(gate.float()) * up.float())

    def forward(
        self,
        x: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        weights: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        workspace1_shape, workspace2_shape, output_shape = self.workspace_shapes(
            x.shape[0],
            w13.shape[1],
            x.shape[-1],
            expert_ids.shape[1],
            w13.shape[0],
            w13.shape[0],
            None,
            MoEActivation.SILU,
        )
        workspace13 = torch.empty(workspace1_shape, device=x.device, dtype=x.dtype)
        workspace2 = torch.empty(workspace2_shape, device=x.device, dtype=x.dtype)
        output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        self.apply(
            output,
            x,
            w13,
            w2,
            weights,
            expert_ids,
            MoEActivation.SILU,
            w13.shape[0],
            None,
            None,
            None,
            workspace13,
            workspace2,
            None,
            False,
        )
        return output
