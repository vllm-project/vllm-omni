# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.

"""CUDA/Triton multi-head MoE implementation for MAGI-2 Preview."""

from __future__ import annotations

import math
import os

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_omni.platforms import current_omni_platform

from ..native.mh_moe import (
    Magi2MultiHeadMoE as NativeMagi2MultiHeadMoE,
)
from ..native.mh_moe import (
    Magi2MultiHeadMoEConfig,
    global_sort_routes,
    torch_mh_moe_forward,
)

# vLLM's no-Triton placeholder exposes ``tl.constexpr`` as ``None``.
_SWIGLU7_ALPHA = tl.constexpr(1.702) if HAS_TRITON else 1.702
_SWIGLU7_LIMIT = tl.constexpr(7.0) if HAS_TRITON else 7.0
_SWIGLU7_BIAS = tl.constexpr(1.0) if HAS_TRITON else 1.0


@triton.jit
def _swiglu7_kernel(gate, up, out_dtype: tl.constexpr):
    gate_clamped = tl.minimum(gate, _SWIGLU7_LIMIT)
    up_clamped = tl.maximum(tl.minimum(up, _SWIGLU7_LIMIT), -_SWIGLU7_LIMIT)
    sigmoid = tl.sigmoid(_SWIGLU7_ALPHA * gate_clamped)
    swish = gate_clamped * sigmoid
    return (swish * (up_clamped + _SWIGLU7_BIAS)).to(out_dtype)


@triton.jit
def _binary_search_expert(
    cumulative_tiles,
    tile_id,
    num_experts: tl.constexpr,
    log2_num_experts: tl.constexpr,
):
    lo = 0
    hi = num_experts
    for _ in tl.static_range(0, log2_num_experts + 1):
        mid = (lo + hi + 1) // 2
        below = tl.load(cumulative_tiles + mid) <= tile_id
        lo = tl.where(below, mid, lo)
        hi = tl.where(below, hi, mid - 1)
    return lo


@triton.jit
def _mh_moe_kernel(
    x_ptr,
    wg_ptr,
    wu_ptr,
    wd_ptr,
    y_ptr,
    gather_ids_ptr,
    probs_ptr,
    expert_offsets_ptr,
    cumulative_tiles_ptr,
    stride_x_s,
    stride_x_h,
    stride_x_dh,
    stride_wg_e,
    stride_wg_dh,
    stride_wg_de,
    stride_wu_e,
    stride_wu_dh,
    stride_wu_de,
    stride_wd_e,
    stride_wd_de,
    stride_wd_dh,
    stride_y_s,
    stride_y_h,
    stride_y_dh,
    d_head: tl.constexpr,
    d_expert: tl.constexpr,
    num_heads: tl.constexpr,
    num_flat_experts: tl.constexpr,
    log2_num_experts: tl.constexpr,
    block_t: tl.constexpr,
    block_dh: tl.constexpr,
    block_de: tl.constexpr,
    acc_dtype: tl.constexpr,
    deterministic: tl.constexpr = False,
):
    tile_id = tl.program_id(0)
    total_tiles = tl.load(cumulative_tiles_ptr + num_flat_experts)
    if tile_id >= total_tiles:
        return

    expert = _binary_search_expert(cumulative_tiles_ptr, tile_id, num_flat_experts, log2_num_experts)
    expert_i64 = expert.to(tl.int64)
    head = expert // (num_flat_experts // num_heads)
    tile_in_expert = tile_id - tl.load(cumulative_tiles_ptr + expert)
    token_start = tl.load(expert_offsets_ptr + expert) + tile_in_expert * block_t
    expert_end = tl.load(expert_offsets_ptr + expert + 1)
    count = tl.minimum(token_start + block_t, expert_end) - token_start

    dh_block_offsets = tl.arange(0, block_dh)
    de_block_offsets = tl.arange(0, block_de)
    token_offsets = tl.arange(0, block_t)
    dh_offsets = tl.arange(0, d_head)

    token_positions = token_start + token_offsets
    token_mask = token_offsets < count
    gather_ids = tl.load(gather_ids_ptr + token_positions, mask=token_mask, other=0).to(tl.int64)
    probabilities = tl.load(probs_ptr + token_positions, mask=token_mask, other=0.0)
    x_base = gather_ids * stride_x_s + head * stride_x_h
    output_acc = tl.zeros([block_t, d_head], dtype=acc_dtype)

    for de_start in tl.range(0, d_expert, block_de):
        de_offsets = de_start + de_block_offsets
        gate_acc = tl.zeros([block_t, block_de], dtype=acc_dtype)
        up_acc = tl.zeros([block_t, block_de], dtype=acc_dtype)
        for dh_start in tl.static_range(0, d_head, block_dh):
            local_dh = dh_start + dh_block_offsets
            x_block = tl.load(
                x_ptr + x_base[:, None] + local_dh[None, :] * stride_x_dh,
                mask=token_mask[:, None],
                other=0.0,
            )
            wg = tl.load(
                wg_ptr
                + expert_i64 * stride_wg_e
                + local_dh[:, None] * stride_wg_dh
                + de_offsets[None, :] * stride_wg_de
            )
            wu = tl.load(
                wu_ptr
                + expert_i64 * stride_wu_e
                + local_dh[:, None] * stride_wu_dh
                + de_offsets[None, :] * stride_wu_de
            )
            gate_acc += tl.dot(x_block, wg)
            up_acc += tl.dot(x_block, wu)
        hidden = _swiglu7_kernel(gate_acc, up_acc, wd_ptr.dtype.element_ty)
        down = tl.load(
            wd_ptr + expert_i64 * stride_wd_e + de_offsets[:, None] * stride_wd_de + dh_offsets[None, :] * stride_wd_dh
        )
        output_acc += tl.dot(hidden, down)
    output_acc = output_acc * probabilities[:, None]

    if deterministic:
        output_ptrs = y_ptr + token_positions[:, None] * stride_y_s + dh_offsets[None, :] * stride_y_dh
        tl.store(output_ptrs, output_acc.to(y_ptr.dtype.element_ty), mask=token_mask[:, None])
    else:
        output_base = gather_ids * stride_y_s + head * stride_y_h
        output_ptrs = y_ptr + output_base[:, None] + dh_offsets[None, :] * stride_y_dh
        tl.atomic_add(output_ptrs, output_acc.to(y_ptr.dtype.element_ty), mask=token_mask[:, None])


def _deterministic_scatter(
    sorted_output: torch.Tensor,
    reference: torch.Tensor,
    gather_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    num_flat_experts = expert_offsets.numel() - 1
    experts_per_head = num_flat_experts // reference.shape[1]
    expert_lengths = torch.diff(expert_offsets)
    head_values = torch.arange(num_flat_experts, device=gather_ids.device) // experts_per_head
    head_ids = torch.repeat_interleave(head_values, expert_lengths)
    scatter_ids = gather_ids.long() * reference.shape[1] + head_ids.long()
    output = torch.zeros_like(reference).view(-1, reference.shape[-1])
    output.scatter_add_(0, scatter_ids[:, None].expand_as(sorted_output), sorted_output.to(output.dtype))
    return output.view_as(reference)


def _select_block_config() -> tuple[int, int, int, int, int]:
    """Return the reference kernel config, capped for pre-Blackwell GPUs."""

    capability = current_omni_platform.get_device_capability()
    if capability is not None and capability.major >= 10:
        return (128, 64, 32, 2, 8)
    return (64, 64, 32, 2, 4)


def triton_mh_moe_forward(
    x: torch.Tensor,
    gather_ids: torch.Tensor,
    probs: torch.Tensor,
    expert_offsets: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    deterministic: bool = False,
) -> torch.Tensor:
    """Fused gather/expert/scatter kernel for released MAGI dimensions."""

    routed_tokens = gather_ids.numel()
    if routed_tokens == 0:
        return torch.zeros_like(x)
    d_head, d_expert = x.shape[-1], w_down.shape[1]
    block_t, block_dh, block_de, num_stages, num_warps = _select_block_config()
    if d_head % block_dh or d_expert % block_de:
        return torch_mh_moe_forward(x, gather_ids, probs, expert_offsets, w_gate, w_up, w_down)

    if deterministic:
        output = torch.empty((routed_tokens, 1, d_head), device=x.device, dtype=x.dtype)
    else:
        output = torch.zeros_like(x)
    num_flat_experts = expert_offsets.numel() - 1
    expert_tiles = (torch.diff(expert_offsets) + block_t - 1) // block_t
    cumulative_tiles = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=x.device), expert_tiles.cumsum(0, dtype=torch.int32))
    )
    grid = ((routed_tokens + block_t - 1) // block_t + num_flat_experts,)
    log2_experts = max(1, math.ceil(math.log2(max(num_flat_experts, 1) + 1)))
    _mh_moe_kernel[grid](
        x,
        w_gate,
        w_up,
        w_down,
        output,
        gather_ids,
        probs,
        expert_offsets,
        cumulative_tiles,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w_gate.stride(0),
        w_gate.stride(1),
        w_gate.stride(2),
        w_up.stride(0),
        w_up.stride(1),
        w_up.stride(2),
        w_down.stride(0),
        w_down.stride(1),
        w_down.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        d_head,
        d_expert,
        x.shape[1],
        num_flat_experts,
        log2_experts,
        block_t,
        block_dh,
        block_de,
        tl.float32,
        deterministic,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    if deterministic:
        return _deterministic_scatter(output.view(routed_tokens, d_head), x, gather_ids, expert_offsets)
    return output


class Magi2MultiHeadMoE(NativeMagi2MultiHeadMoE):
    """Checkpoint-compatible MAGI-2 MoE with a CUDA/Triton local kernel."""

    def _local_forward(self, x_heads: torch.Tensor) -> torch.Tensor:
        if not x_heads.is_cuda:
            return super()._local_forward(x_heads)
        probabilities, indices = self._route(x_heads)
        gather_ids, sorted_probs, offsets = global_sort_routes(probabilities, indices, self.num_experts)
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


__all__ = [
    "Magi2MultiHeadMoE",
    "Magi2MultiHeadMoEConfig",
    "triton_mh_moe_forward",
]
