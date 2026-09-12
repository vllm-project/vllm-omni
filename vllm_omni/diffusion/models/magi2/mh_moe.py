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

from vllm_omni.platforms import current_omni_platform

from .fused_moe_kernels import (
    global_sort_routes,
    invoke_fused_moe_bf16,
    torch_mh_moe_forward,
    triton_mh_moe_forward,
)
from .parallel import Magi2ParallelGroup, ep_dispatch, ep_undispatch, get_magi2_ep_group

RoutingScore = Literal["softmax", "sigmoid"]


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


def _align_bf16_routes(
    route_ids: torch.Tensor, num_experts: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded expert-block metadata for the BF16 routed GEMM."""

    flat_ids = route_ids.reshape(-1).to(torch.int32)
    route_count = flat_ids.numel()
    order = torch.argsort(flat_ids)
    sorted_experts = flat_ids[order]
    counts = torch.bincount(flat_ids, minlength=num_experts)
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    starts = torch.cumsum(padded_counts, 0) - padded_counts
    ends = torch.cumsum(counts, 0)
    begins = ends - counts
    positions = torch.arange(route_count, device=flat_ids.device, dtype=torch.int64)
    destinations = starts[sorted_experts.long()] + positions - begins[sorted_experts.long()]
    total = int(padded_counts.sum().item())
    sorted_ids = torch.full((total,), route_count, device=flat_ids.device, dtype=torch.int32)
    sorted_ids[destinations] = order.to(torch.int32)
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=flat_ids.device, dtype=torch.int32),
        padded_counts // block_size,
    )
    num_padded = torch.tensor((total,), device=flat_ids.device, dtype=torch.int32)
    return sorted_ids, expert_ids, num_padded


def _bf16_fused_moe_forward(
    x_heads: torch.Tensor,
    probabilities: torch.Tensor,
    indices: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Run the opt-in BF16 grouped GEMM path with head-local routing."""

    num_tokens, num_heads, hidden_size = x_heads.shape
    top_k = probabilities.shape[-1]
    num_experts = w_gate.shape[0]
    if num_experts % num_heads:
        raise ValueError("expert bank must be divisible by local MoE heads")
    experts_per_head = num_experts // num_heads
    if num_tokens == 0:
        return torch.zeros_like(x_heads)
    head_offsets = (
        torch.arange(num_heads, device=x_heads.device, dtype=torch.int32).view(num_heads, 1, 1) * experts_per_head
    )
    route_ids = (indices.to(torch.int32) + head_offsets).reshape(num_heads * num_tokens, top_k)
    route_weights = probabilities.reshape(num_heads * num_tokens, top_k).contiguous()
    sorted_ids, expert_ids, num_padded = _align_bf16_routes(route_ids, num_experts, 128)

    hidden = x_heads.permute(1, 0, 2).contiguous().reshape(num_heads * num_tokens, hidden_size)
    intermediate_size = w_gate.shape[-1]
    # [E, I, 2, D] -> [E, 2I, D]: adjacent rows are gate/up pairs.
    packed_w13 = torch.stack((w_gate.transpose(1, 2), w_up.transpose(1, 2)), dim=2).reshape(
        num_experts, 2 * intermediate_size, hidden_size
    )
    intermediate = torch.empty(
        (num_heads * num_tokens * top_k, intermediate_size), device=x_heads.device, dtype=x_heads.dtype
    )
    # Keep the qualified MUSA point unchanged.  CUDA/H20 benefits from the
    # pre-Blackwell tile found by the MAGI-2 BF16 sweep (smaller K/warp count
    # and deeper pipelining reduce register pressure).  The launch contract
    # remains the same; only legal, device-specific values are selected.
    if current_omni_platform.is_musa():
        config = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 16,
            "num_warps": 16,
            "num_stages": 1,
        }
    else:
        config = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 16,
            "num_warps": 4,
            "num_stages": 3,
        }
    invoke_fused_moe_bf16(
        hidden,
        packed_w13,
        intermediate,
        route_weights,
        sorted_ids,
        expert_ids,
        num_padded,
        top_k=top_k,
        config=config,
        fuse_swiglu=True,
    )
    route_output = torch.empty((num_heads * num_tokens, top_k, hidden_size), device=x_heads.device, dtype=x_heads.dtype)
    invoke_fused_moe_bf16(
        intermediate,
        w_down.transpose(1, 2),
        route_output,
        route_weights,
        sorted_ids,
        expert_ids,
        num_padded,
        top_k=1,
        config=config,
        fuse_swiglu=False,
    )
    return route_output.sum(dim=1).reshape(num_heads, num_tokens, hidden_size).permute(1, 0, 2)


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
        if (
            os.environ.get("MAGI2_USE_BF16_MOE_KERNEL", "0") == "1"
            and x_heads.device.type != "cpu"
            and x_heads.dtype == torch.bfloat16
            and os.environ.get("MAGI2_DETERMINISTIC", "0") != "1"
        ):
            return _bf16_fused_moe_forward(x_heads, probabilities, indices, self.W_gate, self.W_up, self.W_down)
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
