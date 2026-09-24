# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.utils.fa import (
    flash_attn_varlen_func as flash_attn_varlen_func_v3,
)
from vllm_omni.diffusion.attention.layer import Attention

LINGBOT_VIDEO_FP32_MODULES = (
    "time_embedder",
    "time_modulation",
    "scale_shift_table",
    "norm",
    "norm1",
    "norm2",
    "norm_q",
    "norm_k",
    "norm_post_attn",
    "norm_post_ffn",
    "norm_out",
    "norm_out_modulation",
    "router",
)


def should_keep_in_fp32(name: str) -> bool:
    return any(module_name in name.split(".") for module_name in LINGBOT_VIDEO_FP32_MODULES)


def _all_to_all_split_cat(
    local_input: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    input_list = [tensor.contiguous() for tensor in torch.tensor_split(local_input, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class LingBotVideoRMSNorm(nn.Module):
    """RMSNorm with fp32 accumulation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply complex RoPE to `(B, S, H, D)` attention tensors."""
    with torch.amp.autocast("cuda", enabled=False):
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        out = torch.view_as_real(x_c * freqs_cis.unsqueeze(2)).flatten(3)
        return out.type_as(x)


class LingBotVideoRotaryEmbedding(nn.Module):
    """Complex64 RoPE table indexed by position ids."""

    def __init__(self, axes_dims: tuple[int, ...], axes_lens: tuple[int, ...], theta: float):
        super().__init__()
        self.axes_dims = tuple(axes_dims)
        self.axes_lens = list(axes_lens)
        self.theta = theta
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: tuple[int, ...], end: tuple[int, ...], theta: float):
        freqs_cis = []
        for d, e in zip(dim, end):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
            timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return freqs_cis

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        # position_ids: (S, 3) int -> (S, head_dim/2) complex64
        device = position_ids.device
        max_vals = position_ids.max(dim=0).values.tolist()
        needs_rebuild = self.freqs_cis is None or any(
            max_val >= axis_len for max_val, axis_len in zip(max_vals, self.axes_lens)
        )
        if needs_rebuild:
            for i in range(len(self.axes_lens)):
                if max_vals[i] >= self.axes_lens[i]:
                    self.axes_lens[i] = int(max_vals[i] * 1.5) + 1
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, tuple(self.axes_lens), theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        return torch.cat([self.freqs_cis[i][position_ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


def make_joint_position_ids(text_len: int, grid_t: int, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
    """Return ``(t, h, w)`` positions with video rows before text rows.

    Text t-axis positions are 1..text_len, and video t-axis positions start at
    text_len + 1. This matches the token order produced by ``_cat_interleave``.
    """
    tt = torch.arange(grid_t, device=device, dtype=torch.int32) + (text_len + 1)
    hh = torch.arange(grid_h, device=device, dtype=torch.int32)
    ww = torch.arange(grid_w, device=device, dtype=torch.int32)
    grid = torch.stack(torch.meshgrid(tt, hh, ww, indexing="ij"), dim=-1).flatten(0, 2)
    text_t = torch.arange(text_len, device=device, dtype=torch.int32) + 1
    text_pos = torch.stack([text_t, torch.zeros_like(text_t), torch.zeros_like(text_t)], dim=-1)
    return torch.cat([grid, text_pos], dim=0)  # (Nx + L, 3)


def _cat_interleave(
    a: torch.Tensor,
    len_a: list[int],
    b: torch.Tensor,
    len_b: list[int],
) -> torch.Tensor:
    a_split = torch.split(a, len_a, dim=1)
    b_split = torch.split(b, len_b, dim=1)
    blocks: list[torch.Tensor] = []
    for x_part, text_part in zip(a_split, b_split):
        blocks.extend([x_part, text_part])
    return torch.cat(blocks, dim=1)


def _packed_block_attention_mask(sample_seq_lens: list[int], device: torch.device) -> torch.Tensor:
    total_seq_len = sum(sample_seq_lens)
    mask = torch.zeros(
        1,
        1,
        total_seq_len,
        total_seq_len,
        dtype=torch.bool,
        device=device,
    )
    start = 0
    for seq_len in sample_seq_lens:
        end = start + seq_len
        mask[:, :, start:end, start:end] = True
        start = end
    return mask


class LingBotVideoTextEmbedder(nn.Module):
    """Matches CondProjection: RMSNorm(text_dim, eps=1e-6 fixed) -> Linear-SiLU-Linear."""

    def __init__(self, text_dim: int, hidden_size: int):
        super().__init__()
        self.norm = LingBotVideoRMSNorm(text_dim, eps=1e-6)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.linear_2(F.silu(self.linear_1(x)))


class LingBotVideoAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, norm_eps, qkv_bias, out_bias):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.norm_q = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.norm_k = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=out_bias)
        self.attn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=False,
            num_kv_heads=num_heads,
        )

    def forward(
        self,
        x,
        rotary_emb,
        attention_mask=None,
        packed_indices: dict[str, torch.Tensor] | None = None,
        parallel_config=None,
    ):
        B, S, _ = x.shape
        q = self.to_q(x).unflatten(2, (self.num_heads, self.head_dim))
        k = self.to_k(x).unflatten(2, (self.num_heads, self.head_dim))
        v = self.to_v(x).unflatten(2, (self.num_heads, self.head_dim))
        q = apply_rotary_emb(self.norm_q(q), rotary_emb)
        k = apply_rotary_emb(self.norm_k(k), rotary_emb)
        if packed_indices is None:
            out = self.attn(q, k, v, attn_metadata=AttentionMetadata(attn_mask=attention_mask))
        else:
            packed_attention_mask = packed_indices.get("attention_mask")
            if packed_attention_mask is not None and parallel_config is None:
                out = self.attn.sdpa_fallback.forward(
                    q,
                    k,
                    v,
                    AttentionMetadata(attn_mask=packed_attention_mask),
                )
                return self.to_out(out.flatten(2, 3).type_as(x))
            if flash_attn_varlen_func_v3 is None:
                raise RuntimeError(
                    "A flash attention varlen function is required for packed context parallel attention."
                )
            if parallel_config is None:
                result = flash_attn_varlen_func_v3(
                    q=q.reshape(-1, self.num_heads, self.head_dim),
                    k=k.reshape(-1, self.num_heads, self.head_dim),
                    v=v.reshape(-1, self.num_heads, self.head_dim),
                    cu_seqlens_q=packed_indices["cu_seqlens_kv"],
                    cu_seqlens_k=packed_indices["cu_seqlens_kv"],
                    max_seqlen_q=packed_indices["max_seqlen_in_batch_kv"],
                    max_seqlen_k=packed_indices["max_seqlen_in_batch_kv"],
                    causal=False,
                )
                out = result[0] if isinstance(result, tuple) else result
                out = out.reshape(B, S, self.num_heads, self.head_dim)
            else:
                group = parallel_config.context_parallel_config._ulysses_mesh.get_group()
                world_size = dist.get_world_size(group)
                local_heads = self.num_heads // world_size
                q_global = _all_to_all_split_cat(
                    q.reshape(B, S, self.num_heads * self.head_dim),
                    scatter_dim=2,
                    gather_dim=1,
                    group=group,
                ).view(B, S * world_size, local_heads, self.head_dim)
                k_global = _all_to_all_split_cat(
                    k.reshape(B, S, self.num_heads * self.head_dim),
                    scatter_dim=2,
                    gather_dim=1,
                    group=group,
                ).view(B, S * world_size, local_heads, self.head_dim)
                v_global = _all_to_all_split_cat(
                    v.reshape(B, S, self.num_heads * self.head_dim),
                    scatter_dim=2,
                    gather_dim=1,
                    group=group,
                ).view(B, S * world_size, local_heads, self.head_dim)
                q_flat = q_global.reshape(-1, local_heads, self.head_dim)
                k_flat = k_global.reshape(-1, local_heads, self.head_dim)
                v_flat = v_global.reshape(-1, local_heads, self.head_dim)
                result = flash_attn_varlen_func_v3(
                    q=q_flat,
                    k=k_flat,
                    v=v_flat,
                    cu_seqlens_q=packed_indices["cu_seqlens_kv"],
                    cu_seqlens_k=packed_indices["cu_seqlens_kv"],
                    max_seqlen_q=packed_indices["max_seqlen_in_batch_kv"],
                    max_seqlen_k=packed_indices["max_seqlen_in_batch_kv"],
                    causal=False,
                )
                out_global = result[0] if isinstance(result, tuple) else result
                out_global = out_global.reshape(B, S * world_size, local_heads * self.head_dim)
                out = _all_to_all_split_cat(
                    out_global,
                    scatter_dim=1,
                    gather_dim=2,
                    group=group,
                ).view(B, S, self.num_heads, self.head_dim)
        return self.to_out(out.flatten(2, 3).type_as(x))


class LingBotVideoMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LingBotVideoRouter(nn.Module):
    """Token-choice top-k router used by the LingBot MoE checkpoints.

    Selection uses the bias-corrected score, while the gating weights use the
    original score. This asymmetry matches the reference LingBot inference path.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        score_func: str,
        norm_topk_prob: bool,
        n_group: int | None,
        topk_group: int | None,
        route_scale: float,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(num_experts),
            persistent=True,
        )

    def _group_limited_topk(self, scores_for_choice: torch.Tensor) -> torch.Tensor:
        if self.n_group is None or self.topk_group is None:
            raise ValueError("group-limited top-k requires n_group and topk_group.")
        seq_len = scores_for_choice.shape[0]
        experts_per_group = self.num_experts // self.n_group
        grouped = scores_for_choice.view(seq_len, self.n_group, experts_per_group)
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(seq_len, self.n_group, experts_per_group).reshape(seq_len, -1)
        masked = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        return torch.topk(masked, k=self.top_k, dim=-1, sorted=False)[1]

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(tokens.device.type, enabled=False):
            logits = F.linear(tokens.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = F.softmax(logits, dim=-1)
        elif self.score_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise ValueError(f"Unsupported LingBot router score_func: {self.score_func!r}.")

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.n_group is not None and self.n_group > 1:
            top_indices = self._group_limited_topk(scores_for_choice)
        else:
            top_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        top_scores = scores.gather(1, top_indices)
        if self.top_k > 1 and self.norm_topk_prob:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale
        return top_indices, top_scores.to(tokens.dtype)


class LingBotVideoGroupedExperts(nn.Module):
    """Grouped expert weights.

    Weight layout matches the reference checkpoint:
    w1 [E, I, H], w2 [E, H, I], w3 [E, I, H].
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class LingBotVideoSparseMoeBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        moe_intermediate_size: int,
        score_func: str,
        norm_topk_prob: bool,
        n_group: int | None,
        topk_group: int | None,
        routed_scaling_factor: float,
        n_shared_experts: int | None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.router = LingBotVideoRouter(
            hidden_size,
            num_experts,
            top_k,
            score_func,
            norm_topk_prob,
            n_group,
            topk_group,
            routed_scaling_factor,
        )
        self.experts = LingBotVideoGroupedExperts(num_experts, hidden_size, moe_intermediate_size)
        self.shared_experts = None
        if n_shared_experts is not None and n_shared_experts > 0:
            self.shared_experts = LingBotVideoMLP(
                hidden_size,
                moe_intermediate_size * n_shared_experts,
            )

    @staticmethod
    def _reorder_tokens(
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        num_tokens = tokens.shape[0]
        top_k = top_indices.shape[1]
        flat_scores = top_scores.reshape(-1)
        flat_indices = top_indices.reshape(-1)
        active_positions = torch.where(flat_scores != 0)[0]
        active_experts = flat_indices[active_positions]

        counts = torch.zeros(num_experts, device=tokens.device, dtype=torch.int64)
        counts.scatter_add_(
            0,
            active_experts,
            torch.ones_like(active_experts, dtype=torch.int64),
        )

        sort_order = torch.argsort(active_experts, stable=True)
        sorted_positions = active_positions[sort_order]
        sorted_scores = flat_scores[sorted_positions]
        original_token_idx = sorted_positions // top_k
        permuted_tokens = tokens[original_token_idx]
        return permuted_tokens, counts, sorted_positions, sorted_scores, num_tokens, top_k

    @staticmethod
    def _pad_grouped_tokens(
        tokens: torch.Tensor,
        counts: torch.Tensor,
        align: int = 8,
    ) -> tuple[torch.Size, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = tokens.shape[0]
        num_experts = int(counts.shape[0])
        max_len = _round_up_to_multiple(num_tokens + num_experts * align, align)
        counts_i64 = counts.to(torch.int64)
        total_per_expert = torch.clamp_min(counts_i64, align)
        aligned_counts_i64 = (total_per_expert + align - 1) // align * align
        write_offsets = torch.cumsum(aligned_counts_i64, dim=0) - aligned_counts_i64
        end_offsets = torch.cumsum(aligned_counts_i64, dim=0)
        start_indices = torch.cumsum(counts_i64, dim=0) - counts_i64

        slots = torch.arange(max_len, dtype=torch.int64, device=tokens.device)
        expert_idx = torch.bucketize(slots, end_offsets, right=True)
        valid_expert = expert_idx < num_experts
        safe_expert_idx = expert_idx.clamp(max=num_experts - 1)
        local_idx = slots - write_offsets[safe_expert_idx]
        source_idx = start_indices[safe_expert_idx] + local_idx
        valid = valid_expert & (local_idx < counts_i64[safe_expert_idx])
        fill = torch.full_like(source_idx, num_tokens)
        permuted_indices = torch.where(valid, source_idx, fill)

        tokens_with_pad = torch.vstack((tokens, tokens.new_zeros((tokens.shape[-1],))))
        input_shape = tokens_with_pad.shape
        return (
            input_shape,
            tokens_with_pad[permuted_indices],
            permuted_indices,
            aligned_counts_i64.to(torch.int32),
        )

    @staticmethod
    def _unpad_grouped_tokens(
        output: torch.Tensor,
        input_shape: torch.Size,
        permuted_indices: torch.Tensor,
    ) -> torch.Tensor:
        unpermuted = output.new_empty(input_shape)
        unpermuted[permuted_indices, :] = output
        return unpermuted[:-1]

    def _run_experts_for_loop(self, tokens: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        count_list = counts.tolist()
        splits = torch.split(tokens, count_list, dim=0)
        outputs = []
        for expert_idx, expert_tokens in enumerate(splits):
            if expert_tokens.numel() == 0:
                continue
            h = F.silu(expert_tokens @ self.experts.w1[expert_idx].transpose(-2, -1))
            h = h * (expert_tokens @ self.experts.w3[expert_idx].transpose(-2, -1))
            h = h @ self.experts.w2[expert_idx].transpose(-2, -1)
            outputs.append(h)
        if not outputs:
            return tokens.new_zeros(tokens.shape)
        return torch.cat(outputs, dim=0)

    def _run_grouped_experts(self, tokens: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        if not hasattr(torch, "_grouped_mm") or tokens.device.type != "cuda":
            return self._run_experts_for_loop(tokens, counts)
        input_shape, padded_tokens, permuted_indices, aligned_counts = self._pad_grouped_tokens(tokens, counts)
        offsets = torch.cumsum(aligned_counts, dim=0, dtype=torch.int32)
        h = F.silu(
            torch._grouped_mm(
                padded_tokens.bfloat16(),
                self.experts.w1.bfloat16().transpose(-2, -1),
                offs=offsets,
            )
        )
        h = h * torch._grouped_mm(
            padded_tokens.bfloat16(),
            self.experts.w3.bfloat16().transpose(-2, -1),
            offs=offsets,
        )
        out = torch._grouped_mm(
            h,
            self.experts.w2.bfloat16().transpose(-2, -1),
            offs=offsets,
        ).type_as(padded_tokens)
        return self._unpad_grouped_tokens(out, input_shape, permuted_indices)

    @staticmethod
    def _restore_tokens(
        expert_output: torch.Tensor,
        sorted_positions: torch.Tensor,
        sorted_scores: torch.Tensor,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        hidden_size = expert_output.shape[-1]
        unsorted = torch.zeros(
            (num_tokens * top_k, hidden_size),
            dtype=expert_output.dtype,
            device=expert_output.device,
        )
        unsorted[sorted_positions] = expert_output
        unsorted = unsorted.reshape(num_tokens, top_k, hidden_size)

        scores_unsorted = torch.zeros(
            num_tokens * top_k,
            dtype=sorted_scores.dtype,
            device=sorted_scores.device,
        )
        scores_unsorted[sorted_positions] = sorted_scores
        scores_unsorted = scores_unsorted.reshape(num_tokens, top_k, 1)
        return (unsorted.float() * scores_unsorted).sum(dim=1).to(expert_output.dtype)

    def _run_selected_experts(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        (
            permuted_tokens,
            counts,
            sorted_positions,
            sorted_scores,
            num_tokens,
            top_k,
        ) = self._reorder_tokens(tokens, top_scores, top_indices, self.router.num_experts)
        expert_output = self._run_grouped_experts(permuted_tokens, counts)
        return self._restore_tokens(
            expert_output,
            sorted_positions,
            sorted_scores,
            num_tokens,
            top_k,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        tokens = hidden_states.reshape(-1, self.hidden_size)
        top_indices, top_scores = self.router(tokens)
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).to(top_scores.dtype)
            top_scores = top_scores * mask
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-9)
            top_scores = top_scores * self.router.route_scale

        out = self._run_selected_experts(tokens, top_scores, top_indices)
        out = out.reshape(batch_size, -1, self.hidden_size)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out


class LingBotVideoBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        norm_eps,
        qkv_bias,
        out_bias,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        decoder_sparse_step,
        mlp_only_layers,
        n_shared_experts,
        score_func,
        norm_topk_prob,
        n_group,
        topk_group,
        routed_scaling_factor,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        h = hidden_size
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6 * h))
        self.norm1 = LingBotVideoRMSNorm(h, norm_eps)
        self.attn = LingBotVideoAttention(h, num_attention_heads, norm_eps, qkv_bias, out_bias)
        self.norm_post_attn = LingBotVideoRMSNorm(h, norm_eps)
        self.norm2 = LingBotVideoRMSNorm(h, norm_eps)
        use_sparse_moe = (
            num_experts > 0 and layer_idx not in mlp_only_layers and (layer_idx + 1) % decoder_sparse_step == 0
        )
        if use_sparse_moe:
            self.ffn = LingBotVideoSparseMoeBlock(
                hidden_size=h,
                num_experts=num_experts,
                top_k=num_experts_per_tok,
                moe_intermediate_size=moe_intermediate_size,
                score_func=score_func,
                norm_topk_prob=norm_topk_prob,
                n_group=n_group,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
                n_shared_experts=n_shared_experts,
            )
        else:
            self.ffn = LingBotVideoMLP(h, intermediate_size)
        self.norm_post_ffn = LingBotVideoRMSNorm(h, norm_eps)

    def forward(
        self,
        x,
        temb6,
        rotary_emb,
        attention_mask=None,
        moe_padding_mask=None,
        packed_indices: dict[str, torch.Tensor] | None = None,
        parallel_config=None,
    ):
        expected_tokens = x.shape[0] * x.shape[1]
        if temb6.ndim != 2 or temb6.shape[0] != expected_tokens:
            raise ValueError(
                "LingBotVideoBlock expects token-level temb6 with shape "
                f"(B*S, 6D); got {tuple(temb6.shape)} for hidden states {tuple(x.shape)}."
            )
        # AdaLN modulation and normalization stay in fp32 for the sensitive path.
        mod = temb6.view(x.shape[0], x.shape[1], -1) + self.scale_shift_table.unsqueeze(0)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        # AdaLN modulation and norms stay in fp32; cast to the transformer
        # compute dtype only at Linear boundaries.
        bulk_dtype = self.attn.to_q.weight.dtype
        attn_in = (self.norm1(x) * scale_msa + shift_msa).to(bulk_dtype)
        attn_out = self.attn(
            attn_in,
            rotary_emb,
            attention_mask,
            packed_indices=packed_indices,
            parallel_config=parallel_config,
        )
        x = x + (gate_msa * self.norm_post_attn(attn_out)).to(x.dtype)

        ffn_in = (self.norm2(x) * scale_mlp + shift_mlp).to(bulk_dtype)
        if isinstance(self.ffn, LingBotVideoSparseMoeBlock):
            ffn_out = self.ffn(ffn_in, padding_mask=moe_padding_mask)
        else:
            ffn_out = self.ffn(ffn_in)
        ffn_normed = self.norm_post_ffn(ffn_out)
        x = x + (gate_mlp * ffn_normed).to(x.dtype)
        return x


class LingBotVideoTransformer3DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False
    _repeated_blocks = ["LingBotVideoBlock"]
    _layerwise_offload_blocks_attr = "blocks"
    _no_split_modules = ["LingBotVideoBlock"]
    _keep_in_fp32_modules = list(LINGBOT_VIDEO_FP32_MODULES)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is None or dtype == torch.float32:
            return super().to(*args, **kwargs)

        dtype_is_floating = torch.is_floating_point(torch.empty((), dtype=dtype))
        if not dtype_is_floating:
            return super().to(*args, **kwargs)

        if device is not None:
            super().to(device=device, non_blocking=non_blocking)

        for name, param in self.named_parameters():
            if not torch.is_floating_point(param):
                continue
            target_dtype = torch.float32 if should_keep_in_fp32(name) else dtype
            param.data = param.data.to(dtype=target_dtype, non_blocking=non_blocking)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype=target_dtype, non_blocking=non_blocking)

        for name, buffer in self.named_buffers():
            if not torch.is_floating_point(buffer):
                continue
            target_dtype = torch.float32 if should_keep_in_fp32(name) else dtype
            buffer.data = buffer.data.to(dtype=target_dtype, non_blocking=non_blocking)

        return self

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        depth: int = 24,
        intermediate_size: int = 6144,
        text_dim: int = 2560,
        freq_dim: int = 256,
        norm_eps: float = 1e-6,
        rope_theta: float = 256.0,
        axes_dims: tuple[int, int, int] = (32, 48, 48),
        axes_lens: tuple[int, int, int] = (8192, 1024, 1024),
        qkv_bias: bool = False,
        out_bias: bool = True,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
        num_experts: int = 0,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 512,
        decoder_sparse_step: int = 1,
        mlp_only_layers: tuple[int, ...] = (),
        n_shared_experts: int | None = None,
        score_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        n_group: int | None = None,
        topk_group: int | None = None,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        head_dim = hidden_size // num_attention_heads
        assert head_dim == sum(axes_dims), f"head_dim {head_dim} != sum(axes_dims) {sum(axes_dims)}"
        mlp_only_layers = tuple(mlp_only_layers)

        self.patch_embedder = nn.Linear(in_channels * math.prod(patch_size), hidden_size, bias=patch_embed_bias)
        self.time_proj = Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(freq_dim, hidden_size, act_fn="silu", sample_proj_bias=timestep_mlp_bias)
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.text_embedder = LingBotVideoTextEmbedder(text_dim, hidden_size)
        self.rope = LingBotVideoRotaryEmbedding(axes_dims, axes_lens, rope_theta)
        self.blocks = nn.ModuleList(
            [
                LingBotVideoBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    norm_eps=norm_eps,
                    qkv_bias=qkv_bias,
                    out_bias=out_bias,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    moe_intermediate_size=moe_intermediate_size,
                    decoder_sparse_step=decoder_sparse_step,
                    mlp_only_layers=mlp_only_layers,
                    n_shared_experts=n_shared_experts,
                    score_func=score_func,
                    norm_topk_prob=norm_topk_prob,
                    n_group=n_group,
                    topk_group=topk_group,
                    routed_scaling_factor=routed_scaling_factor,
                    layer_idx=i,
                )
                for i in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.norm_out_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.proj_out = nn.Linear(hidden_size, math.prod(patch_size) * out_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, C, T, H, W)
        timestep: torch.Tensor,  # (B,) in [0, 1000](= sigma*1000)
        encoder_hidden_states: torch.Tensor,  # (B, L, text_dim)
        encoder_attention_mask: torch.Tensor | None = None,  # (B, L) 1=valid
        return_dict: bool = True,
    ):
        B, C, T, H, W = hidden_states.shape
        pF, pH, pW = self.config.patch_size
        gt, gh, gw = T // pF, H // pH, W // pW
        n_video = gt * gh * gw
        L = encoder_hidden_states.shape[1]
        device = hidden_states.device
        if encoder_attention_mask is not None:
            text_lens = encoder_attention_mask.sum(dim=-1).long()
        else:
            text_lens = torch.full((B,), L, dtype=torch.long, device=device)
        text_lens_list = [int(v) for v in text_lens.detach().cpu().tolist()]
        packed_batch = B > 1

        # patchify: token order (f h w), feature order (pf ph pw c) -- matches patchify_and_embed
        patch_tokens = hidden_states.reshape(B, C, gt, pF, gh, pH, gw, pW)
        patch_tokens = patch_tokens.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
            B,
            n_video,
            pF * pH * pW * C,
        )
        if packed_batch:
            x = torch.cat(
                [self.patch_embedder(patch_tokens[i : i + 1]) for i in range(B)],
                dim=1,
            )
        else:
            x = self.patch_embedder(patch_tokens)

        if packed_batch:
            text_parts = [
                self.text_embedder(encoder_hidden_states[i : i + 1, : text_lens_list[i], :]) for i in range(B)
            ]
            text = torch.cat(text_parts, dim=1)
            joint = _cat_interleave(
                x,
                [n_video] * B,
                text,
                text_lens_list,
            )
        else:
            text = self.text_embedder(encoder_hidden_states)
            joint = torch.cat([x, text], dim=1)  # [video; text]
        joint_seq_len = joint.shape[1]

        # Per-sample RoPE: video t-axis start = real text length of this sample + 1
        rotary_parts = [self.rope(make_joint_position_ids(text_lens_list[i], gt, gh, gw, device)) for i in range(B)]
        if packed_batch:
            rotary = torch.cat(rotary_parts, dim=0).unsqueeze(0)
        else:
            rotary = torch.stack(rotary_parts, dim=0)  # (B, S, head_dim/2) complex64

        parallel_config = getattr(self, "_parallel_config", None)
        use_packed_attention = parallel_config is not None

        attention_mask = None
        moe_padding_mask = None
        packed_indices = None
        has_padding = encoder_attention_mask is not None and bool((text_lens < L).any())
        if packed_batch or use_packed_attention:
            sample_seq_lens = [n_video + text_len for text_len in text_lens_list]
            cu_seqlens = torch.zeros(B + 1, device=device, dtype=torch.int32)
            cu_seqlens[1:] = torch.cumsum(
                torch.tensor(sample_seq_lens, device=device, dtype=torch.int32),
                dim=0,
            )
            packed_indices = {
                "cu_seqlens_kv": cu_seqlens,
                "max_seqlen_in_batch_kv": max(sample_seq_lens),
            }
            if packed_batch and not use_packed_attention:
                packed_indices["attention_mask"] = _packed_block_attention_mask(sample_seq_lens, device)
            has_padding = False
        if has_padding:
            key_mask = torch.cat(
                [torch.ones(B, n_video, dtype=torch.bool, device=device), encoder_attention_mask.bool()],
                dim=1,
            )
            attention_mask = key_mask[:, None, None, :]  # (B,1,1,S) -> SDPA broadcast
            moe_padding_mask = key_mask.reshape(-1).float()  # (B*S,)
        packed_cp = packed_indices is not None and parallel_config is not None
        padding_size = 0
        if packed_cp:
            cp_config = parallel_config.context_parallel_config
            cp_world_size = int(getattr(cp_config, "ulysses_degree", getattr(cp_config, "_world_size", 1)))
            padding_size = (cp_world_size - (joint_seq_len % cp_world_size)) % cp_world_size
            if padding_size:
                joint = torch.cat(
                    [
                        joint,
                        torch.zeros(
                            joint.shape[0],
                            padding_size,
                            joint.shape[2],
                            device=joint.device,
                            dtype=joint.dtype,
                        ),
                    ],
                    dim=1,
                )
                rotary = torch.cat(
                    [
                        rotary,
                        torch.zeros(
                            rotary.shape[0],
                            padding_size,
                            rotary.shape[2],
                            device=rotary.device,
                            dtype=rotary.dtype,
                        ),
                    ],
                    dim=1,
                )
                if packed_indices is None:
                    raise RuntimeError("packed_indices must be initialized for packed context parallel.")
                packed_indices["cu_seqlens_kv"] = torch.cat(
                    [
                        packed_indices["cu_seqlens_kv"],
                        packed_indices["cu_seqlens_kv"][-1:] + padding_size,
                    ],
                    dim=0,
                )
                packed_indices["max_seqlen_in_batch_kv"] = max(
                    int(packed_indices["max_seqlen_in_batch_kv"]),
                    int(padding_size),
                )
                joint_seq_len = joint.shape[1]

        timestep_for_embed = timestep.float()
        timestep_proj = self.time_proj(timestep_for_embed)
        t_emb = self.time_embedder(timestep_proj)  # (B, D)
        if packed_batch:
            temb_input = torch.cat(
                [t_emb[i : i + 1].unsqueeze(1).expand(1, n_video + text_lens_list[i], -1) for i in range(B)],
                dim=1,
            )
            if padding_size:
                temb_input = torch.cat(
                    [
                        temb_input,
                        torch.zeros(
                            temb_input.shape[0],
                            padding_size,
                            temb_input.shape[2],
                            device=temb_input.device,
                            dtype=temb_input.dtype,
                        ),
                    ],
                    dim=1,
                )
            temb6 = self.time_modulation(temb_input.reshape(joint_seq_len, -1))
            temb6 = temb6.reshape(1, joint_seq_len, -1)
        else:
            temb_input = t_emb.unsqueeze(1).expand(B, joint_seq_len, -1)  # (B, S, D)
            temb6 = self.time_modulation(temb_input.reshape(B * joint_seq_len, -1))
            temb6 = temb6.reshape(B, joint_seq_len, -1)  # (B, S, 6D)

        temb6 = temb6.reshape(temb6.shape[0] * temb6.shape[1], -1)

        for block in self.blocks:
            joint = block(
                joint,
                temb6,
                rotary,
                attention_mask,
                moe_padding_mask,
                packed_indices=packed_indices,
                parallel_config=parallel_config,
            )
        final_mod = self.norm_out_modulation(temb_input.reshape(joint.shape[0] * joint.shape[1], -1))
        shift, scale = final_mod.reshape(joint.shape[0], joint.shape[1], -1).chunk(2, dim=-1)
        final_hidden = self.norm_out(joint) * (1.0 + scale) + shift
        projected = self.proj_out(final_hidden.to(self.proj_out.weight.dtype))
        if packed_cp:
            if padding_size:
                projected = projected[:, :-padding_size, :]
        if packed_batch:
            split_lengths: list[int] = []
            for text_len in text_lens_list:
                split_lengths.extend([n_video, text_len])
            parts = torch.split(projected, split_lengths, dim=1)
            x = torch.cat(parts[::2], dim=1).reshape(B, n_video, -1)
        else:
            x = projected[:, :n_video]

        # unpatchify (matches the rearrange in postprocess)
        Cout = self.config.out_channels
        x = x.reshape(B, gt, gh, gw, pF, pH, pW, Cout)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, Cout, T, H, W)

        if not return_dict:
            return (x,)
        return Transformer2DModelOutput(sample=x)
