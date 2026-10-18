# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/huggingface/diffusers.

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)
from vllm_omni.diffusion.forward_context import get_forward_context

logger = init_logger(__name__)


def apply_rotary_emb_nucleus(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


class NucleusMoETimestepProjEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        use_additional_t_cond: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=embedding_dim, time_embed_dim=4 * embedding_dim, out_dim=embedding_dim
        )
        self.timestep_embedder.linear_1 = ReplicatedLinear(
            embedding_dim,
            4 * embedding_dim,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.timestep_embedder.linear_1",
        )
        self.timestep_embedder.linear_2 = ReplicatedLinear(
            4 * embedding_dim,
            embedding_dim,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.timestep_embedder.linear_2",
        )
        self.norm = RMSNorm(embedding_dim, eps=1e-6)
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
        addition_t_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))

        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("When additional_t_cond is True, addition_t_cond must be provided.")
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            conditioning = conditioning + addition_t_emb

        return self.norm(conditioning)


class NucleusMoEEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self._rope_params(pos_index, self.axes_dim[0], self.theta),
                self._rope_params(pos_index, self.axes_dim[1], self.theta),
                self._rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self._rope_params(neg_index, self.axes_dim[0], self.theta),
                self._rope_params(neg_index, self.axes_dim[1], self.theta),
                self._rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.scale_rope = scale_rope
        self._freq_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}

    @staticmethod
    def _rope_params(index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(
        self,
        video_fhw: tuple[int, int, int] | list[tuple[int, int, int]],
        txt_seq_lens: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list) and len(video_fhw) > 1:
            first_fhw = video_fhw[0]
            if not all(fhw == first_fhw for fhw in video_fhw):
                logger.warning(
                    "Batch inference with variable-sized images is not currently supported in NucleusMoEEmbedRope. "
                    "All images in the batch should have the same dimensions (frame, height, width). "
                    f"Detected sizes: {video_fhw}. Using the first image's dimensions {first_fhw} "
                    "for RoPE computation, which may lead to incorrect results for other images in the batch."
                )

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        cache_key = (frame, height, width, idx)
        cached = self._freq_cache.get(cache_key)
        if cached is not None:
            return cached

        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        result = freqs.clone().contiguous()
        self._freq_cache[cache_key] = result
        return result


class NucleusMoEImageRopePrepare(nn.Module):
    def __init__(self, img_in: nn.Module, pos_embed: NucleusMoEEmbedRope):
        super().__init__()
        self.img_in = img_in
        self.pos_embed = pos_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        img_shapes: tuple[int, int, int] | list[tuple[int, int, int]],
        txt_seq_lens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.img_in(hidden_states)
        vid_freqs, txt_freqs = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
        return hidden_states, vid_freqs, txt_freqs


class NucleusMoECrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
        joint_attention_dim: int = 3584,
        out_bias: bool = True,
        qk_norm: bool = True,
        eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.head_dim = head_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads or num_heads
        self.eps = eps

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_qkv",
        )
        self.num_heads = self.to_qkv.num_heads
        self.num_kv_heads = self.to_qkv.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)

        self.add_k_proj = ColumnParallelLinear(
            joint_attention_dim,
            self.total_num_kv_heads * head_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.add_k_proj",
        )
        self.add_v_proj = ColumnParallelLinear(
            joint_attention_dim,
            self.total_num_kv_heads * head_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.add_v_proj",
        )
        self.norm_added_k = RMSNorm(head_dim, eps=eps)

        self.inner_dim = head_dim * self.num_heads

        self.to_out = RowParallelLinear(
            head_dim * self.total_num_heads,
            self.dim,
            bias=out_bias,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out",
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.num_kv_heads,
        )
        try:
            config = get_forward_context().omni_diffusion_config
            self.parallel_config = config.parallel_config
        except Exception:
            self.parallel_config = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        hidden_states_mask: torch.Tensor | None = None,
        cached_txt_key: torch.Tensor | None = None,
        cached_txt_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        img_qkv, _ = self.to_qkv(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        img_query, img_key, img_value = img_qkv.split([q_size, kv_size, kv_size], dim=-1)

        img_query = img_query.unflatten(-1, (self.num_heads, self.head_dim))
        img_key = img_key.unflatten(-1, (self.num_kv_heads, self.head_dim))
        img_value = img_value.unflatten(-1, (self.num_kv_heads, self.head_dim))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)

        txt_key = None
        txt_value = None
        txt_key_needs_rope = False
        if cached_txt_key is not None and cached_txt_value is not None:
            txt_key = cached_txt_key
            txt_value = cached_txt_value
        elif encoder_hidden_states is not None:
            txt_key = self.add_k_proj(encoder_hidden_states)
            txt_value = self.add_v_proj(encoder_hidden_states)
            txt_key = txt_key.unflatten(-1, (self.num_kv_heads, self.head_dim))
            txt_value = txt_value.unflatten(-1, (self.num_kv_heads, self.head_dim))
            txt_key = self.norm_added_k(txt_key)
            txt_key_needs_rope = True

        if image_rotary_emb is not None:
            vid_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_nucleus(img_query, vid_freqs, use_real=False)
            img_key = apply_rotary_emb_nucleus(img_key, vid_freqs, use_real=False)
            if txt_key is not None and txt_key_needs_rope:
                txt_key = apply_rotary_emb_nucleus(txt_key, txt_freqs, use_real=False)

        attn_metadata = None
        image_seq_len = img_query.shape[1]
        use_joint_attention_sp = (
            txt_key is not None
            and txt_value is not None
            and self.parallel_config is not None
            and self.parallel_config.sequence_parallel_size > 1
        )

        if use_joint_attention_sp:
            if self.num_kv_groups > 1:
                img_key = img_key.repeat_interleave(self.num_kv_groups, dim=2)
                img_value = img_value.repeat_interleave(self.num_kv_groups, dim=2)
                txt_key = txt_key.repeat_interleave(self.num_kv_groups, dim=2)
                txt_value = txt_value.repeat_interleave(self.num_kv_groups, dim=2)
            txt_query = torch.zeros(
                (img_query.shape[0], txt_key.shape[1], img_query.shape[2], img_query.shape[3]),
                dtype=img_query.dtype,
                device=img_query.device,
            )
            attn_metadata = AttentionMetadata(
                joint_query=txt_query,
                joint_key=txt_key,
                joint_value=txt_value,
                joint_attn_mask=encoder_hidden_states_mask,
                joint_strategy="rear",
            )
            if hidden_states_mask is not None:
                attn_metadata.attn_mask = hidden_states_mask
            hidden_states = self.attn(img_query, img_key, img_value, attn_metadata)
            hidden_states = hidden_states[:, :image_seq_len, ...]
        else:
            if txt_key is not None and txt_value is not None:
                joint_key = torch.cat([img_key, txt_key], dim=1)
                joint_value = torch.cat([img_value, txt_value], dim=1)
            else:
                joint_key = img_key
                joint_value = img_value

            if self.num_kv_groups > 1:
                joint_key = joint_key.repeat_interleave(self.num_kv_groups, dim=2)
                joint_value = joint_value.repeat_interleave(self.num_kv_groups, dim=2)

            if encoder_hidden_states_mask is not None and txt_key is not None and txt_value is not None:
                image_mask = hidden_states_mask
                if image_mask is None:
                    image_mask = torch.ones(
                        (hidden_states.shape[0], hidden_states.shape[1]),
                        dtype=torch.bool,
                        device=hidden_states.device,
                    )
                joint_mask = torch.cat([image_mask, encoder_hidden_states_mask], dim=1)
                attn_metadata = AttentionMetadata(attn_mask=joint_mask)

            hidden_states = self.attn(img_query, joint_key, joint_value, attn_metadata)

        hidden_states = hidden_states.flatten(2, 3).to(img_query.dtype).contiguous()
        hidden_states = self.to_out(hidden_states)

        return hidden_states


class SwiGLUExperts(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_dim: int,
        num_experts: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.moe_intermediate_dim = moe_intermediate_dim
        self.hidden_size = hidden_size

        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, hidden_size, 2 * moe_intermediate_dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, moe_intermediate_dim, hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens_per_expert_list = num_tokens_per_expert.tolist()

        num_real_tokens = sum(num_tokens_per_expert_list)
        num_padding = x.shape[0] - num_real_tokens

        x_per_expert = torch.split(
            x[:num_real_tokens],
            split_size_or_sections=num_tokens_per_expert_list,
            dim=0,
        )

        expert_outputs = []
        for expert_idx, x_expert in enumerate(x_per_expert):
            gate_up = torch.matmul(x_expert, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            out_expert = torch.matmul(F.silu(gate) * up, self.down_proj[expert_idx])
            expert_outputs.append(out_expert)

        out = torch.cat(expert_outputs, dim=0)
        out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
        return out


class NucleusMoELayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_dim: int,
        num_experts: int,
        capacity_factor: float,
        use_sigmoid: bool,
        route_scale: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.moe_intermediate_dim = moe_intermediate_dim
        self.hidden_size = hidden_size
        self.capacity_factor = capacity_factor
        self.use_sigmoid = use_sigmoid
        self.route_scale = route_scale

        self.gate = ReplicatedLinear(
            hidden_size * 2,
            num_experts,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        self.experts = SwiGLUExperts(
            hidden_size=hidden_size,
            moe_intermediate_dim=moe_intermediate_dim,
            num_experts=num_experts,
        )

        self.shared_expert = FeedForward(
            dim=hidden_size,
            dim_out=hidden_size,
            inner_dim=moe_intermediate_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_expert",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_unmodulated: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, slen, dim = hidden_states.shape

        if timestep is not None:
            timestep_expanded = timestep.unsqueeze(1).expand(-1, slen, -1)
            router_input = torch.cat([timestep_expanded, hidden_states_unmodulated], dim=-1)
        else:
            router_input = hidden_states_unmodulated

        logits = self.gate(router_input)

        if self.use_sigmoid:
            scores = torch.sigmoid(logits.float()).to(logits.dtype)
        else:
            scores = F.softmax(logits.float(), dim=-1).to(logits.dtype)

        affinity = scores.transpose(1, 2)
        capacity = max(1, math.ceil(self.capacity_factor * slen / self.num_experts))

        topk = torch.topk(affinity, k=capacity, dim=-1)
        top_indices = topk.indices
        gating = affinity.gather(dim=-1, index=top_indices)

        batch_offsets = torch.arange(bs, device=hidden_states.device, dtype=torch.long).view(bs, 1, 1) * slen
        global_token_indices = (batch_offsets + top_indices).transpose(0, 1).reshape(self.num_experts, -1).reshape(-1)
        gating_flat = gating.transpose(0, 1).reshape(self.num_experts, -1).reshape(-1)

        token_score_sums = torch.zeros(bs * slen, device=hidden_states.device, dtype=gating_flat.dtype)
        token_score_sums.scatter_add_(0, global_token_indices, gating_flat)
        gating_flat = gating_flat / (token_score_sums[global_token_indices] + 1e-12)
        gating_flat = gating_flat * self.route_scale

        x_flat = hidden_states.reshape(bs * slen, dim)
        routed_input = x_flat[global_token_indices]

        tokens_per_expert = bs * capacity
        num_tokens_per_expert = torch.full(
            (self.num_experts,),
            tokens_per_expert,
            device=hidden_states.device,
            dtype=torch.long,
        )
        routed_output = self.experts(routed_input, num_tokens_per_expert)
        routed_output = (routed_output.float() * gating_flat.unsqueeze(-1)).to(hidden_states.dtype)

        shared_out = self.shared_expert(hidden_states)
        out = shared_out.reshape(bs * slen, dim)

        scatter_idx = global_token_indices.reshape(-1, 1).expand(-1, dim)
        out = out.scatter_add(dim=0, index=scatter_idx, src=routed_output)
        out = out.reshape(bs, slen, dim)

        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        inner_dim: int | None = None,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        inner_dim = inner_dim or int(dim * mult * 2 / 3) // 128 * 128
        dim_out = dim_out or dim

        self.net = nn.ModuleList(
            [
                SwiGLUProj(dim, inner_dim, bias, quant_config, f"{prefix}.net.0"),
                nn.Identity(),
                RowParallelLinear(
                    inner_dim,
                    dim_out,
                    bias=bias,
                    input_is_parallel=True,
                    return_bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.net.2",
                ),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class SwiGLUProj(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool, quant_config, prefix: str):
        super().__init__()
        self.proj = MergedColumnParallelLinear(
            dim_in,
            [dim_out, dim_out],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.proj(x)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * F.silu(gate)


def _is_moe_layer(strategy: str, layer_idx: int, num_layers: int) -> bool:
    if strategy == "leave_first_three_and_last_block_dense":
        return layer_idx >= 3 and layer_idx < num_layers - 1
    elif strategy == "leave_first_three_blocks_dense":
        return layer_idx >= 3
    elif strategy == "leave_first_block_dense":
        return layer_idx >= 1
    elif strategy == "all_moe":
        return True
    elif strategy == "all_dense":
        return False
    return True


class NucleusMoEImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_key_value_heads: int | None = None,
        joint_attention_dim: int = 3584,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        mlp_ratio: float = 4.0,
        moe_enabled: bool = False,
        num_experts: int = 128,
        moe_intermediate_dim: int = 1344,
        capacity_factor: float = 8.0,
        use_sigmoid: bool = False,
        route_scale: float = 2.5,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dim = dim
        self.moe_enabled = moe_enabled

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(dim, 4 * dim, bias=True, return_bias=False, quant_config=None, prefix=f"{prefix}.img_mod"),
        )

        self.encoder_proj = ReplicatedLinear(
            joint_attention_dim,
            dim,
            bias=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder_proj",
        )

        self.pre_attn_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
        self.attn = NucleusMoECrossAttention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            num_kv_heads=num_key_value_heads,
            joint_attention_dim=dim,
            out_bias=False,
            qk_norm=qk_norm == "rms_norm",
            eps=eps,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.pre_mlp_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
        self._text_kv_cache: dict[
            tuple[int, tuple[int, ...], str, str, int | None],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

        if moe_enabled:
            self.img_mlp = NucleusMoELayer(
                hidden_size=dim,
                moe_intermediate_dim=moe_intermediate_dim,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
                use_sigmoid=use_sigmoid,
                route_scale=route_scale,
                quant_config=quant_config,
                prefix=f"{prefix}.img_mlp",
            )
        else:
            mlp_inner_dim = int(dim * mlp_ratio * 2 / 3) // 128 * 128
            self.img_mlp = FeedForward(
                dim=dim,
                dim_out=dim,
                inner_dim=mlp_inner_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.img_mlp",
            )

    def _get_cached_text_kv(
        self,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "_text_kv_cache"):
            self._text_kv_cache = {}

        txt_freqs_ptr = None
        if image_rotary_emb is not None:
            txt_freqs_ptr = image_rotary_emb[1].data_ptr()

        cache_key = (
            encoder_hidden_states.data_ptr(),
            tuple(encoder_hidden_states.shape),
            str(encoder_hidden_states.dtype),
            str(encoder_hidden_states.device),
            txt_freqs_ptr,
        )
        cached = self._text_kv_cache.get(cache_key)
        if cached is not None:
            return cached

        context = self.encoder_proj(encoder_hidden_states)
        txt_key = self.attn.add_k_proj(context)
        txt_value = self.attn.add_v_proj(context)
        txt_key = txt_key.unflatten(-1, (self.attn.num_kv_heads, self.attn.head_dim))
        txt_value = txt_value.unflatten(-1, (self.attn.num_kv_heads, self.attn.head_dim))
        txt_key = self.attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            txt_freqs = image_rotary_emb[1]
            txt_key = apply_rotary_emb_nucleus(txt_key, txt_freqs, use_real=False)

        self._text_kv_cache[cache_key] = (txt_key, txt_value)
        return txt_key, txt_value

    def reset_text_kv_cache(self) -> None:
        self._text_kv_cache = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        hidden_states_mask: torch.Tensor | None = None,
        cached_txt_key: torch.Tensor | None = None,
        cached_txt_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mod_out = self.img_mod(temb)
        if isinstance(mod_out, tuple):
            mod_out = mod_out[0]
        scale1, gate1, scale2, gate2 = mod_out.unsqueeze(1).chunk(4, dim=-1)

        gate1 = gate1.clamp(min=-2.0, max=2.0)
        gate2 = gate2.clamp(min=-2.0, max=2.0)

        context = None if cached_txt_key is not None else self.encoder_proj(encoder_hidden_states)

        img_normed = self.pre_attn_norm(hidden_states)
        img_modulated = img_normed * (1 + scale1)

        img_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=context,
            image_rotary_emb=image_rotary_emb,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            hidden_states_mask=hidden_states_mask,
            cached_txt_key=cached_txt_key,
            cached_txt_value=cached_txt_value,
        )

        hidden_states = hidden_states + gate1.tanh() * img_attn_output

        img_normed2 = self.pre_mlp_norm(hidden_states)
        img_modulated2 = img_normed2 * (1 + scale2)

        if self.moe_enabled:
            img_mlp_output = self.img_mlp(img_modulated2, img_normed2, timestep=temb)
        else:
            img_mlp_output = self.img_mlp(img_modulated2)

        hidden_states = hidden_states + gate2.tanh() * img_mlp_output

        if hidden_states.dtype == torch.float16:
            fp16_finfo = torch.finfo(torch.float16)
            hidden_states = hidden_states.clip(fp16_finfo.min, fp16_finfo.max)

        return hidden_states


class NucleusMoEImageTransformer2DModel(nn.Module):
    _no_split_modules = ["NucleusMoEImageTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    _sp_plan = {
        "image_rope_prepare": {
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
            1: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True, auto_pad=True),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 24,
        attention_head_dim: int = 128,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        joint_attention_dim: int = 3584,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        mlp_ratio: float = 4.0,
        moe_enabled: bool = True,
        dense_moe_strategy: str = "leave_first_three_and_last_block_dense",
        num_experts: int = 128,
        moe_intermediate_dim: int = 1344,
        capacity_factors: float | list[float] = 8.0,
        use_sigmoid: bool = False,
        route_scale: float = 2.5,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        capacity_factors = capacity_factors if isinstance(capacity_factors, list) else [capacity_factors] * num_layers

        self.pos_embed = NucleusMoEEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = NucleusMoETimestepProjEmbeddings(
            embedding_dim=self.inner_dim, quant_config=quant_config, prefix=f"{prefix}.time_text_embed"
        )

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.img_in = ReplicatedLinear(
            in_channels,
            self.inner_dim,
            bias=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.img_in",
        )

        self.transformer_blocks = nn.ModuleList(
            [
                NucleusMoEImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    num_key_value_heads=num_key_value_heads,
                    joint_attention_dim=joint_attention_dim,
                    mlp_ratio=mlp_ratio,
                    moe_enabled=moe_enabled and _is_moe_layer(dense_moe_strategy, idx, num_layers),
                    num_experts=num_experts,
                    moe_intermediate_dim=moe_intermediate_dim,
                    capacity_factor=capacity_factors[idx],
                    use_sigmoid=use_sigmoid,
                    route_scale=route_scale,
                    quant_config=quant_config,
                    prefix=f"{prefix}.transformer_blocks.{idx}",
                )
                for idx in range(num_layers)
            ]
        )
        self.image_rope_prepare = NucleusMoEImageRopePrepare(self.img_in, self.pos_embed)

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = ReplicatedLinear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj_out",
        )

        self.gradient_checkpointing = False

    def reset_text_kv_cache(self) -> None:
        for block in self.transformer_blocks:
            block.reset_text_kv_cache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        img_shapes: tuple[int, int, int] | list[tuple[int, int, int]],
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        timestep: torch.LongTensor = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            attention_kwargs.pop("scale", 1.0)

        txt_seq_len = encoder_hidden_states.shape[1]
        hidden_states, vid_freqs, txt_freqs = self.image_rope_prepare(hidden_states, img_shapes, [txt_seq_len])
        if encoder_hidden_states_mask is not None:
            encoder_hidden_states_mask = encoder_hidden_states_mask.to(device=hidden_states.device, dtype=torch.bool)

        hidden_states_mask = None
        try:
            ctx = get_forward_context()
            if ctx.sp_original_seq_len is not None and ctx.sp_padding_size > 0:
                img_padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size
                hidden_states_mask = torch.ones(
                    hidden_states.shape[0],
                    img_padded_seq_len,
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                hidden_states_mask[:, ctx.sp_original_seq_len :] = False
                if hidden_states_mask.all():
                    hidden_states_mask = None
        except Exception:
            hidden_states_mask = None

        image_rotary_emb = (vid_freqs, txt_freqs)

        timestep = timestep.to(hidden_states.dtype)
        temb = self.time_text_embed(timestep, hidden_states)

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)

        for block in self.transformer_blocks:
            cached_txt_key = None
            cached_txt_value = None
            if encoder_hidden_states is not None:
                cached_txt_key, cached_txt_value = block._get_cached_text_kv(
                    encoder_hidden_states,
                    image_rotary_emb,
                )
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                hidden_states_mask=hidden_states_mask,
                cached_txt_key=cached_txt_key,
                cached_txt_value=cached_txt_value,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        stacked_params_mapping = [
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
        ]

        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name

            prefix = ""
            if lookup_name.startswith("transformer."):
                prefix = "transformer."
                lookup_name = lookup_name[len("transformer.") :]

            if "pos_embed.pos_freqs" in lookup_name or "pos_embed.neg_freqs" in lookup_name:
                continue

            if "norm_added_q" in lookup_name:
                loaded_params.add(original_name)
                continue

            if ".gate_up_proj.weight" in lookup_name or ".down_proj.weight" in lookup_name:
                param_name = lookup_name.replace(".weight", "")
                param = params_dict.get(param_name)
                if param is not None:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(original_name)
                    loaded_params.add(prefix + param_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in lookup_name or param_name in lookup_name:
                    continue
                lookup_name = lookup_name.replace(weight_name, param_name)
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(original_name)
                loaded_params.add(prefix + lookup_name)
                break
            else:
                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                param = params_dict.get(lookup_name)
                if param is not None:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(original_name)
                    loaded_params.add(prefix + lookup_name)

        return loaded_params
