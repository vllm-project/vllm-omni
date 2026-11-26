# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.

"""Ascend NPU multi-head MoE implementation for MAGI-2 Preview."""

from __future__ import annotations

import torch
import torch_npu

from ...swiglu7_npu import fused_swiglu7_enabled
from ..native.mh_moe import (
    Magi2MultiHeadMoE as NativeMagi2MultiHeadMoE,
)
from ..native.mh_moe import swiglu7_pair

# Routing is per token and grouped MatMul is per row, so slicing the token axis bounds the
# intermediates and leaves the values untouched.
MOE_TOKEN_CHUNK = 16384


def npu_mh_moe_forward(
    x: torch.Tensor,
    topk_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    packed_gate_up: torch.Tensor | None = None,
    token_chunk: int = MOE_TOKEN_CHUNK,
) -> torch.Tensor:
    """Ascend grouped-MatMul path for independently routed MAGI-2 MoE heads.

    ``token_chunk`` routes that many tokens at a time; ``0`` routes them in one call.
    ``packed_gate_up`` holds gate and up side by side, collapsing their two MatMuls into one.
    """

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

    if 0 < token_chunk < sequence:
        return torch.cat(
            [
                npu_mh_moe_forward(
                    x[start : start + token_chunk],
                    topk_probs[:, start : start + token_chunk],
                    topk_indices[:, start : start + token_chunk],
                    num_experts,
                    w_gate,
                    w_up,
                    w_down,
                    packed_gate_up=packed_gate_up,
                    token_chunk=0,
                )
                for start in range(0, sequence, token_chunk)
            ],
            dim=0,
        )

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
    if packed_gate_up is None:
        gate = torch_npu.npu_grouped_matmul(x=[sorted_hidden], weight=[w_gate], **grouped_kwargs)[0]
        up = torch_npu.npu_grouped_matmul(x=[sorted_hidden], weight=[w_up], **grouped_kwargs)[0]
        hidden = swiglu7_pair(gate, up)
    else:
        gate_up = torch_npu.npu_grouped_matmul(x=[sorted_hidden], weight=[packed_gate_up], **grouped_kwargs)[0]
        hidden = torch_npu.npu_clipped_swiglu(gate_up, interleaved=False)
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
    return output.view_as(x)


class Magi2MultiHeadMoE(NativeMagi2MultiHeadMoE):
    """Checkpoint-compatible MAGI-2 MoE with Ascend fused local kernels."""

    def _packed_weights(self) -> torch.Tensor | None:
        """Return a [gate|up] bank view, rebinding both parameters into it on first use."""

        gate, up = self.W_gate, self.W_up
        if (
            not fused_swiglu7_enabled()
            or not self.has_real_moe_heads
            or gate.device.type != "npu"
            or gate.dtype != torch.bfloat16
        ):
            return None
        experts, width, intermediate = gate.shape
        stride = (width * 2 * intermediate, 2 * intermediate, 1)
        # Read off the live parameters, so an offload or mmap rebind repacks instead of reusing.
        if (
            gate.stride() == up.stride() == stride
            and up.storage_offset() == gate.storage_offset() + intermediate
            and gate.untyped_storage().data_ptr() == up.untyped_storage().data_ptr()
        ):
            return gate.as_strided((experts, width, 2 * intermediate), stride)
        with torch.no_grad():
            bank = torch.cat((gate, up), dim=-1)
            gate.data, up.data = bank.split(intermediate, dim=-1)
        return bank

    def prepare_bf16_weights(self) -> None:
        """Pack the bank at load time, so the first denoising step does not pay for it."""

        self._packed_weights()

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
            packed_gate_up=self._packed_weights() if not torch.is_grad_enabled() else None,
        )


__all__ = ["Magi2MultiHeadMoE"]
