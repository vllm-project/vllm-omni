# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.

"""Native multi-head MoE used by MAGI-2 Preview.

Adapted from SandAI's Apache-2.0 ``flash_mh_moe`` implementation and modified
to use vLLM's existing expert-parallel group.  MAGI's routing is unusual: each
of twelve 256-wide hidden-state heads independently selects experts from its
own 256-expert bank.  It is therefore not representable by vLLM's conventional
whole-token :class:`FusedMoE` primitive.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.triton_utils import tl, triton

from vllm_omni.platforms import current_omni_platform

from .parallel import Magi2ParallelGroup, ep_dispatch, ep_undispatch, get_magi2_ep_group

RoutingScore = Literal["softmax", "sigmoid"]


def swiglu7_pair(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Released clamped SwiGLU7 expert activation, evaluated in fp32."""

    dtype = gate.dtype
    gate = gate.float().clamp(max=7.0)
    up = up.float().clamp(min=-7.0, max=7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(dtype)


def compute_topk_probs_and_indices(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    score_func: RoutingScore = "sigmoid",
    expert_bias: torch.Tensor | None = None,
    route_norm: bool = True,
    norm_eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route independently for every ``[head, token]`` pair.

    The auxiliary-free bias affects expert selection but deliberately does not
    affect the returned routing probability, matching the training recipe.
    """

    if router_logits.ndim != 3:
        raise ValueError("router_logits must be [heads,tokens,experts]")
    if not 0 < top_k <= router_logits.shape[-1]:
        raise ValueError("top_k must be in [1, num_experts]")
    if score_func == "sigmoid":
        router_scores = torch.sigmoid(router_logits)
    elif score_func == "softmax":
        router_scores = torch.softmax(router_logits, dim=-1)
    else:
        raise ValueError(f"unsupported routing score function {score_func!r}")
    selection_scores = router_scores
    if expert_bias is not None:
        selection_scores = selection_scores + expert_bias.view(router_logits.shape[0], 1, -1)
    # Keep the reference's default sorted=True behavior.  Besides defining the
    # route order for ties, this also fixes the reduction order used by the
    # following L1 normalization.
    topk_indices = torch.topk(selection_scores, top_k, dim=-1).indices
    topk_probs = router_scores.gather(-1, topk_indices)
    if route_norm:
        topk_probs = F.normalize(topk_probs, p=1, dim=-1, eps=norm_eps)
    return topk_probs, topk_indices


def global_sort_routes(
    topk_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert per-head routes into a stable flattened-expert CSR layout."""

    if topk_probs.shape != topk_indices.shape or topk_indices.ndim != 3:
        raise ValueError("top-k probabilities and indices must have the same [H,S,K] shape")
    heads, sequence, top_k = topk_indices.shape
    device = topk_indices.device
    head_offset = torch.arange(heads, device=device).view(heads, 1, 1) * num_experts
    flattened_experts = (topk_indices + head_offset).reshape(-1)
    flat_probs = topk_probs.reshape(-1)
    flat_tokens = torch.arange(sequence, device=device).view(1, sequence, 1).expand(heads, sequence, top_k).reshape(-1)
    order = flattened_experts.argsort(stable=True)
    gather_ids = flat_tokens[order].to(torch.int32)
    sorted_probs = flat_probs[order].float()
    counts = torch.bincount(flattened_experts, minlength=heads * num_experts)
    offsets = torch.zeros(heads * num_experts + 1, device=device, dtype=torch.long)
    offsets[1:] = counts.cumsum(0)
    return gather_ids, sorted_probs, offsets


def torch_mh_moe_forward(
    x: torch.Tensor,
    gather_ids: torch.Tensor,
    probs: torch.Tensor,
    expert_offsets: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Small-shape correctness oracle for the fused expert kernel."""

    if x.ndim != 3:
        raise ValueError("multi-head MoE input must be [tokens,heads,head_dim]")
    output = torch.zeros_like(x)
    experts_per_head = (expert_offsets.numel() - 1) // x.shape[1]
    for flat_expert in range(expert_offsets.numel() - 1):
        begin = int(expert_offsets[flat_expert].item())
        end = int(expert_offsets[flat_expert + 1].item())
        if begin == end:
            continue
        head = flat_expert // experts_per_head
        token_ids = gather_ids[begin:end].long()
        expert_input = x.index_select(0, token_ids)[:, head]
        gate = expert_input @ w_gate[flat_expert]
        up = expert_input @ w_up[flat_expert]
        hidden = swiglu7_pair(gate, up)
        expert_output = hidden @ w_down[flat_expert]
        expert_output = expert_output * probs[begin:end, None].to(expert_output.dtype)
        output[:, head].index_add_(0, token_ids, expert_output)
    return output


_SWIGLU7_ALPHA = tl.constexpr(1.702)
_SWIGLU7_LIMIT = tl.constexpr(7.0)
_SWIGLU7_BIAS = tl.constexpr(1.0)


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
    acc_dtype: tl.constexpr = tl.float32,
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
    # Token indices fit in int32, but multiplying a large packed-batch index by
    # the hidden-width stride does not. Promote before computing element offsets.
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
    if capability is not None and capability.major >= 10:  # Blackwell
        return (128, 64, 32, 2, 8)
    # BLOCK_T=128 needs 122,880 bytes of shared memory and is not safe on the
    # qualified L20X path.  This is the reference kernel's portable config.
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
    # Match the reference launch bound.  Empty/excess programs return after
    # comparing against ``cumulative_tiles[-1]`` inside the kernel.
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


@dataclass(frozen=True)
class Magi2MultiHeadMoEConfig:
    hidden_size: int
    num_heads: int
    num_experts: int
    top_k: int
    expert_intermediate_size: int
    params_dtype: torch.dtype
    score_func: RoutingScore = "sigmoid"
    route_norm: bool = True
    route_scale: float = 1.0


class Magi2MultiHeadMoE(nn.Module):
    """Checkpoint-compatible MAGI-2 head-routed expert layer."""

    _EP_SHARDED_PARAMETER_NAMES = frozenset(
        {"gate", "W_gate", "W_up", "W_down", "router.expert_bias", "router.expert_bias_ema"}
    )

    def __init__(
        self,
        config: Magi2MultiHeadMoEConfig,
        *,
        ep_group: Magi2ParallelGroup | None = None,
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden_size must be divisible by the number of MoE heads")
        self.config = config
        self.num_heads = config.num_heads
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.d_head = config.hidden_size // config.num_heads
        self.d_expert = config.expert_intermediate_size
        self.ep_group = ep_group or get_magi2_ep_group()
        self.padded_num_heads = math.ceil(self.num_heads / self.ep_group.world_size) * self.ep_group.world_size
        self.local_num_heads = self.padded_num_heads // self.ep_group.world_size
        self.local_flatten_num_experts = self.local_num_heads * self.num_experts
        self.ep_pad_heads = self.padded_num_heads - self.num_heads
        self.local_head_start = self.ep_group.rank * self.local_num_heads
        self.has_real_moe_heads = self.local_head_start < self.num_heads

        self.gate = nn.Parameter(torch.empty(self.local_flatten_num_experts, self.d_head, dtype=torch.float32))
        self.W_gate = nn.Parameter(
            torch.empty(self.local_flatten_num_experts, self.d_head, self.d_expert, dtype=config.params_dtype)
        )
        self.W_up = nn.Parameter(
            torch.empty(self.local_flatten_num_experts, self.d_head, self.d_expert, dtype=config.params_dtype)
        )
        self.W_down = nn.Parameter(
            torch.empty(self.local_flatten_num_experts, self.d_expert, self.d_head, dtype=config.params_dtype)
        )
        self.router = nn.Module()
        # Both tensors are released checkpoint entries.  Non-trainable
        # Parameters let the DLO mmap path bind them on a meta-constructed
        # model; persistent buffers are intentionally not mmap-loaded by the
        # generic backend.
        self.router.expert_bias = nn.Parameter(
            torch.zeros(self.local_flatten_num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.router.expert_bias_ema = nn.Parameter(
            torch.zeros(self.local_flatten_num_experts, dtype=torch.float32),
            requires_grad=False,
        )

        for name in self._EP_SHARDED_PARAMETER_NAMES:
            target: nn.Module | Magi2MultiHeadMoE = self
            parts = name.split(".")
            for part in parts[:-1]:
                target = getattr(target, part)
            parameter = getattr(target, parts[-1])
            parameter.mmap_weight_transform = self.ep_slice

    def ep_slice(self, checkpoint_tensor: torch.Tensor) -> torch.Tensor:
        """Slice flattened ``(head,expert)`` checkpoint rows for this rank."""

        if checkpoint_tensor.shape[0] == self.local_flatten_num_experts:
            return checkpoint_tensor
        start = self.local_head_start * self.num_experts
        end = min(start + self.local_flatten_num_experts, checkpoint_tensor.shape[0])
        if start >= checkpoint_tensor.shape[0]:
            return torch.zeros(
                (self.local_flatten_num_experts, *checkpoint_tensor.shape[1:]),
                dtype=checkpoint_tensor.dtype,
                device=checkpoint_tensor.device,
            )
        local = checkpoint_tensor[start:end]
        if local.shape[0] < self.local_flatten_num_experts:
            # Uneven EP/head partitions require materialized zero padding;
            # divisible production layouts keep the mmap-backed slice above.
            padding = torch.zeros(
                (self.local_flatten_num_experts - local.shape[0], *local.shape[1:]),
                dtype=local.dtype,
                device=local.device,
            )
            local = torch.cat((local, padding), dim=0)
        return local

    def _route(self, x_heads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate.view(self.local_num_heads, self.num_experts, self.d_head).float()
        logits = torch.einsum("shd,hed->hse", x_heads.float(), gate)
        bias_source = (os.environ.get("MAGI2_ROUTER_BIAS_SOURCE") or "ema").strip().lower()
        bias_tensor = self.router.expert_bias if bias_source == "main" else self.router.expert_bias_ema
        bias = bias_tensor.view(self.local_num_heads, self.num_experts)
        probs, indices = compute_topk_probs_and_indices(
            logits,
            self.top_k,
            score_func=self.config.score_func,
            expert_bias=bias,
            route_norm=self.config.route_norm,
        )
        return probs * self.config.route_scale, indices

    def _local_forward(self, x_heads: torch.Tensor) -> torch.Tensor:
        probabilities, indices = self._route(x_heads)
        gather_ids, sorted_probs, offsets = global_sort_routes(probabilities, indices, self.num_experts)
        if x_heads.is_cuda:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ep_group.world_size > 1 and self.ep_group.replicated_sequence:
            # TP column-parallel ``split_linear`` already emits exactly this
            # rank's contiguous MoE-head slice.  Compute it once and leave it
            # sharded for the row-parallel ``merge_linear``; no token dispatch
            # or head all-gather belongs on the true TP path.
            local_hidden_size = self.local_num_heads * self.d_head
            if x.shape[-1] != local_hidden_size:
                raise ValueError(f"TP-local MAGI MoE input has width {x.shape[-1]}, expected {local_hidden_size}")
            local = x.view(-1, self.local_num_heads, self.d_head)
            output = self._local_forward(local) if self.has_real_moe_heads else torch.zeros_like(local)
            return output.reshape(-1, local_hidden_size)

        x_heads = x.view(-1, self.num_heads, self.d_head)
        if self.ep_pad_heads:
            padding = x_heads.new_zeros((x_heads.shape[0], self.ep_pad_heads, self.d_head))
            x_heads = torch.cat((x_heads, padding), dim=1)
        sequence_split_sizes: list[int] | None = None
        if self.ep_group.world_size > 1:
            local_size = torch.tensor([x_heads.shape[0]], dtype=torch.int64, device=x_heads.device)
            gathered_sizes = [torch.empty_like(local_size) for _ in range(self.ep_group.world_size)]
            torch.distributed.all_gather(gathered_sizes, local_size, group=self.ep_group.group)
            sequence_split_sizes = [int(size.item()) for size in gathered_sizes]
            x_heads = ep_dispatch(x_heads, self.ep_group, sequence_split_sizes)
        output = self._local_forward(x_heads) if self.has_real_moe_heads else torch.zeros_like(x_heads)
        if self.ep_group.world_size > 1:
            output = ep_undispatch(output, self.ep_group, sequence_split_sizes)
        if self.ep_pad_heads:
            output = output[:, : self.num_heads]
        return output.reshape(-1, self.num_heads * self.d_head)


__all__ = [
    "Magi2MultiHeadMoE",
    "Magi2MultiHeadMoEConfig",
    "compute_topk_probs_and_indices",
    "global_sort_routes",
    "torch_mh_moe_forward",
    "triton_mh_moe_forward",
]
