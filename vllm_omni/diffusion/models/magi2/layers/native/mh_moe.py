# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.

"Native MAGI-2 multi-head MoE: routing, checkpoint ownership and EP dispatch."

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...parallel import Magi2ParallelGroup, ep_dispatch, ep_undispatch, get_magi2_ep_group

RoutingScore = Literal["softmax", "sigmoid"]


def swiglu7_pair(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference SwiGLU7 activation used by the CPU correctness path."""

    dtype = gate.dtype
    gate = gate.float().clamp(max=7.0)
    up = up.float().clamp(min=-7.0, max=7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(dtype)


def _reference_topk_probs_and_indices(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    score_func: RoutingScore = "sigmoid",
    expert_bias: torch.Tensor | None = None,
    route_norm: bool = True,
    norm_eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unfused reference routing.  Also the oracle the fused path is tested against."""

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
    return _reference_topk_probs_and_indices(
        router_logits,
        top_k,
        score_func=score_func,
        expert_bias=expert_bias,
        route_norm=route_norm,
        norm_eps=norm_eps,
    )


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
    """Checkpoint-compatible head-routed experts with a pure-Torch fallback."""

    _compute_topk_probs_and_indices = staticmethod(compute_topk_probs_and_indices)

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

    def prepare_bf16_weights(self) -> None:
        """No derived weight layout is required by the native or NPU backend."""

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
        # The router bmm reads a contiguous head-major operand; cast and permute in one copy.
        x_route = x_heads.transpose(0, 1).to(torch.float32, memory_format=torch.contiguous_format)
        logits = torch.einsum("hsd,hed->hse", x_route, gate)
        bias_source = (os.environ.get("MAGI2_ROUTER_BIAS_SOURCE") or "ema").strip().lower()
        bias_tensor = self.router.expert_bias if bias_source == "main" else self.router.expert_bias_ema
        bias = bias_tensor.view(self.local_num_heads, self.num_experts)
        probs, indices = self._compute_topk_probs_and_indices(
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
    "RoutingScore",
    "swiglu7_pair",
    "torch_mh_moe_forward",
]
