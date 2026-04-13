# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CausalWanModel — 40-layer DiT with causal attention and KV cache.

Adapted from: dreamzero/groot/vla/model/dreamzero/modules/
              wan_video_dit_action_casual_chunk.py L1218-2200

Key differences from WanTransformer3DModel (wan2_2_transformer.py):
- Causal self-attention (new frames only see history)
- KV cache for streaming inference
- Action/state token support (appended after video tokens)
- Extended RoPE with action/state-specific frequencies
- Inference-only forward with KV cache
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.models.dreamzero.modeling.action_encoder import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)

# ── RoPE utilities ──────────────────────────────────────────────────
# Source: wan2_1_submodule.py rope_params / rope_action_apply
#         wan_video_dit_action_casual_chunk.py L93-185 causal_rope_action_apply


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal positional embedding for timesteps.
    Source: wan2_1_submodule.py L16-26
    """
    assert dim % 2 == 0  # L18
    half = dim // 2  # L19
    position = position.type(torch.float64)  # L20
    sinusoid = torch.outer(  # L23-24
        position,
        torch.pow(10000, -torch.arange(half, dtype=position.dtype, device=position.device).div(half)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)  # L25
    return x


def rope_params(max_seq_len: int, dim: int) -> torch.Tensor:
    """Precompute complex-valued RoPE frequencies (polar form).
    Source: wan2_1_submodule.py L37-44 (rope_params_polar)
    Returns: complex tensor [max_seq_len, dim // 2]
    """
    assert dim % 2 == 0  # L38
    freqs = torch.outer(  # L39-42
        torch.arange(max_seq_len),
        1.0 / torch.pow(10000, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)  # L43
    return freqs


def rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x using precomputed complex freqs.
    Source: wan2_1_submodule.py L64-75 (rope_apply_polar)
    """
    B, seq_len, n, _ = x.shape  # L65
    x = torch.view_as_complex(  # L68-70
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )
    freqs = freqs.unsqueeze(0)  # L73
    x = torch.view_as_real(x * freqs).flatten(3)  # L74
    return x


def rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int = 32,
    num_state_per_block: int = 1,
) -> torch.Tensor:
    """RoPE with action/state frequency tables for multi-step sequences.
    Source: wan2_1_submodule.py L130-159 (rope_action_apply_polar)
    """
    B, seq_len, n, _ = x.shape  # L139
    x = torch.view_as_complex(  # L142-144
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )
    if action_register_length is not None:  # L146
        assert num_action_per_block is not None  # L147
        assert num_state_per_block is not None  # L148
        chunk_size = action_register_length // (num_action_per_block + num_state_per_block)  # L150
        freqs_1d_action = freqs_action[: chunk_size * num_action_per_block].view(  # L152
            chunk_size * num_action_per_block, 1, -1
        )
        freqs_1d_state = freqs_state[: chunk_size * num_state_per_block].view(  # L153
            chunk_size * num_state_per_block, 1, -1
        )
        freqs = torch.cat([freqs, freqs_1d_action, freqs_1d_state], dim=0)  # L154
    freqs = freqs.unsqueeze(0)  # L157
    x = torch.view_as_real(x * freqs).flatten(3)  # L158
    return x


def causal_rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
) -> torch.Tensor:
    """RoPE for single inference step (causal / KV-cache mode).
    Source: wan_video_dit_action_casual_chunk.py L153-185 (causal_rope_action_apply_polar)
    """
    B, seq_len, n, _ = x.shape  # L163
    x = torch.view_as_complex(  # L166-168
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )
    if action_register_length is not None:  # L170
        assert action_register_length == (num_action_per_block + num_state_per_block)  # L171
        freqs_action = freqs_action[  # L172-174
            action_state_index * num_action_per_block : (action_state_index + 1) * num_action_per_block
        ]
        freqs_state = freqs_state[  # L175-177
            action_state_index * num_state_per_block : (action_state_index + 1) * num_state_per_block
        ]
        freqs_1d = torch.cat([freqs_action, freqs_state], dim=0).view(  # L178
            action_register_length, 1, -1
        )
        freqs = torch.cat([freqs, freqs_1d], dim=0)  # L179
    freqs = freqs.unsqueeze(0)  # L182
    x = torch.view_as_real(x * freqs).flatten(3)  # L183
    return x


# ── Normalization ───────────────────────────────────────────────────
# Source: wan2_1_submodule.py L162-178 (WanRMSNorm)
#         wan2_2_transformer.py L65-95 (DistributedRMSNorm — TP-aware version)


class WanLayerNorm(nn.LayerNorm):
    """Source: wan2_1_submodule.py L181-184"""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)


class DistributedRMSNorm(nn.Module):
    """RMSNorm that computes global RMS across tensor parallel ranks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if param.shape == loaded_weight.shape:
            param.data.copy_(loaded_weight)
            return

        tp_size = get_tensor_model_parallel_world_size()
        if loaded_weight.shape[0] % tp_size != 0:
            raise ValueError(
                f"Cannot shard RMSNorm weight of shape {tuple(loaded_weight.shape)} across tp_size={tp_size}."
            )

        shard_size = loaded_weight.shape[0] // tp_size
        start_idx = get_tensor_model_parallel_rank() * shard_size
        shard = loaded_weight.narrow(0, start_idx, shard_size)
        if param.shape != shard.shape:
            raise ValueError(f"RMSNorm shard shape mismatch: param={tuple(param.shape)}, shard={tuple(shard.shape)}.")
        param.data.copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        x_float = x.float()
        local_sum_sq = x_float.pow(2).sum(dim=-1, keepdim=True)
        local_count = x.shape[-1]

        if tp_size > 1:
            global_sum_sq = local_sum_sq.clone()
            torch.distributed.all_reduce(global_sum_sq, group=get_tp_group().device_group)
            global_count = local_count * tp_size
        else:
            global_sum_sq = local_sum_sq
            global_count = local_count

        mean_sq = global_sum_sq / global_count
        # Keep the same numerical form as upstream `WanRMSNorm._norm()`:
        #   x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        # For TP>1, extend the mean to the global hidden dimension first, then
        # apply the same `rsqrt` formulation on each local shard.
        return (x_float * torch.rsqrt(mean_sq + self.eps)).type_as(x) * self.weight


# ── Projections ─────────────────────────────────────────────────────


class MLPProj(nn.Module):
    """CLIP feature projection for i2v.
    Source: wan2_1_submodule.py L565-577
    Uses ColumnParallelLinear + RowParallelLinear (Qwen3_VisionMLP pattern).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)  # L571
        self.fc1 = ColumnParallelLinear(  # L571 nn.Linear(in_dim, in_dim)
            in_dim,
            in_dim,
            bias=True,
            return_bias=False,
        )
        self.act = nn.GELU()  # L572
        self.fc2 = RowParallelLinear(  # L572 nn.Linear(in_dim, out_dim)
            in_dim,
            out_dim,
            bias=True,
            return_bias=False,
        )
        self.norm2 = nn.LayerNorm(out_dim)  # L573

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        x = self.norm1(image_embeds)  # L576
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return x


# ── Cross-Attention ─────────────────────────────────────────────────
# Source: wan_video_dit_action_casual_chunk.py L1087-1190 (referenced)
# T2V and I2V cross-attention variants


class WanT2VCrossAttention(nn.Module):
    """Text-to-video cross-attention.
    Source: wan2_1_submodule.py L243-278
    Uses vllm-omni Attention for FlashAttn backend.
    """

    def __init__(self, dim: int, num_heads: int, window_size=(-1, -1), qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        assert dim % num_heads == 0  # L195
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by tp_size={tp_size}.")
        self.tp_num_heads = num_heads // tp_size
        self.tp_inner_dim = self.tp_num_heads * self.head_dim
        self.q = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)  # L205
        self.k = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)  # L206
        self.v = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)  # L207
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True, return_bias=False)  # L208
        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()  # L209
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()  # L210
        self.attn = Attention(
            self.tp_num_heads,
            self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor | None = None,
        crossattn_cache: dict | None = None,
    ) -> torch.Tensor:
        """Source: wan2_1_submodule.py L245-278"""
        del context_lens
        n, d = self.tp_num_heads, self.head_dim  # L253
        q = self.norm_q(self.q(x)).unflatten(2, (n, d))  # L256
        if crossattn_cache is not None:  # L258
            if not crossattn_cache["is_init"]:  # L259
                crossattn_cache["is_init"] = True  # L260
                k = self.norm_k(self.k(context)).unflatten(2, (n, d))  # L261
                v = self.v(context).unflatten(2, (n, d))  # L262
                crossattn_cache["k"] = k  # L263
                crossattn_cache["v"] = v  # L264
            else:
                k = crossattn_cache["k"]  # L266
                v = crossattn_cache["v"]  # L267
        else:
            k = self.norm_k(self.k(context)).unflatten(2, (n, d))  # L269
            v = self.v(context).unflatten(2, (n, d))  # L270
        x = self.attn(q, k, v)  # L273
        x = x.flatten(2)  # L276
        x = self.o(x)  # L277
        return x


class WanI2VCrossAttention(nn.Module):
    """Image-to-video cross-attention (splits first 257 image tokens).
    Source: wan2_1_submodule.py L308-362
    Uses vllm-omni Attention for FlashAttn backend.
    """

    def __init__(self, dim: int, num_heads: int, window_size=(-1, -1), qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by tp_size={tp_size}.")
        self.tp_num_heads = num_heads // tp_size
        self.tp_inner_dim = self.tp_num_heads * self.head_dim
        self.q = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)  # L205
        self.k = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)  # L206
        self.v = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)  # L207
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True, return_bias=False)  # L208
        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_img = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)  # L318
        self.v_img = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)  # L319
        self.norm_k_img = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()  # L321
        self.attn = Attention(
            self.tp_num_heads,
            self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor | None = None,
        crossattn_cache: dict | None = None,
    ) -> torch.Tensor:
        """Source: wan2_1_submodule.py L324-361"""
        del context_lens
        context_img = context[:, :257]  # L330
        context = context[:, 257:]  # L331
        n, d = self.tp_num_heads, self.head_dim  # L332
        q = self.norm_q(self.q(x)).unflatten(2, (n, d))  # L334
        if crossattn_cache is not None:  # L336
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).unflatten(2, (n, d))
                v = self.v(context).unflatten(2, (n, d))
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).unflatten(2, (n, d))  # L348
            v = self.v(context).unflatten(2, (n, d))  # L349
        x = self.attn(q, k, v)  # L350
        k_img = self.norm_k_img(self.k_img(context_img)).unflatten(2, (n, d))  # L352
        v_img = self.v_img(context_img).unflatten(2, (n, d))  # L353
        img_x = self.attn(q, k_img, v_img)  # L354
        x = x.flatten(2)  # L357
        img_x = img_x.flatten(2)  # L358
        x = x + img_x  # L359
        x = self.o(x)  # L360
        return x


WAN_CROSSATTENTION_CLASSES = {  # L364-366
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


# ── Self-Attention with causal masking + KV cache ───────────────────
# Source: wan_video_dit_action_casual_chunk.py L188-1085


class CausalWanSelfAttention(nn.Module):
    """Causal self-attention with KV cache + action/state tokens.
    Source: wan_video_dit_action_casual_chunk.py L188-1085
    Inference-only implementation (KV cache path, L1008-1084).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_seqlen: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
        qk_norm: bool = True,
        eps: float = 1e-6,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
    ) -> None:
        assert dim % num_heads == 0  # L201
        super().__init__()
        self.dim = dim  # L203
        self.num_heads = num_heads  # L204
        self.head_dim = dim // num_heads  # L205
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by tp_size={tp_size}.")
        self.tp_num_heads = num_heads // tp_size
        self.tp_inner_dim = self.tp_num_heads * self.head_dim
        self.local_attn_size = local_attn_size  # L206
        self.num_frame_per_block = num_frame_per_block  # L208
        self.frame_seqlen = frame_seqlen  # L212
        self.num_action_per_block = num_action_per_block  # L213
        self.num_state_per_block = num_state_per_block  # L214
        self.max_attention_size = (  # L211
            21 * frame_seqlen if local_attn_size == -1 else local_attn_size * frame_seqlen
        )
        # layers                                                     # L216-223
        self.q = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.k = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.v = ColumnParallelLinear(dim, dim, bias=True, gather_output=False, return_bias=False)
        self.o = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True, return_bias=False)
        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = Attention(
            self.tp_num_heads,
            self.head_dim,
            causal=False,
            softmax_scale=self.head_dim**-0.5,
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Inference-only forward (KV cache path).
        Source: wan_video_dit_action_casual_chunk.py L786-1084 (kv_cache branch L1008-1084)
        """
        n, d = self.tp_num_heads, self.head_dim  # L803

        # QKV                                                        # L806-812
        q = self.norm_q(self.q(x)).unflatten(2, (n, d))
        k = self.norm_k(self.k(x)).unflatten(2, (n, d))
        v = self.v(x).unflatten(2, (n, d))

        updated_kv_cache: torch.Tensor | None = None

        assert kv_cache is not None, "Inference only — kv_cache required."
        if True:
            # ── Inference path with KV cache ── L1008-1084
            action_state_index = max(0, (current_start_frame - 1) // self.num_frame_per_block)  # L1009

            roped_query = causal_rope_action_apply(  # L1011-1020
                q,
                freqs,
                freqs_action,
                freqs_state,
                action_register_length,
                self.num_action_per_block,
                self.num_state_per_block,
                action_state_index,
            ).type_as(v)
            roped_key = causal_rope_action_apply(  # L1021-1030
                k,
                freqs,
                freqs_action,
                freqs_state,
                action_register_length,
                self.num_action_per_block,
                self.num_state_per_block,
                action_state_index,
            ).type_as(v)

            # Split action/state tokens from video tokens           # L1032-1046
            roped_action_query = None
            roped_action_key = None
            action_v = None

            if action_register_length is not None:  # L1037
                roped_action_query = roped_query[:, -action_register_length:]  # L1038
                roped_query = roped_query[:, :-action_register_length]  # L1039
                roped_action_key = roped_key[:, -action_register_length:]  # L1040
                roped_key = roped_key[:, :-action_register_length]  # L1041
                action_v = v[:, -action_register_length:]  # L1042
                v = v[:, :-action_register_length]  # L1043

            # KV cache update                                        # L1055-1064
            updated_k = kv_cache[0]
            updated_v = kv_cache[1]
            new_k = torch.cat([updated_k, roped_key], dim=1)  # L1059
            new_v = torch.cat([updated_v, v], dim=1)  # L1060
            new_k = new_k[:, -self.max_attention_size :]  # L1063
            new_v = new_v[:, -self.max_attention_size :]  # L1064

            # Attention                                               # L1066-1077
            if action_register_length is not None:  # L1066
                q_cat = torch.cat([roped_query, roped_action_query], dim=1)
                k_cat = torch.cat([new_k, roped_action_key], dim=1)
                v_cat = torch.cat([new_v, action_v], dim=1)
            else:  # L1072
                q_cat = roped_query
                k_cat = new_k
                v_cat = new_v

            x = self.attn(q_cat, k_cat, v_cat)  # L1067-1073
            updated_kv_cache = torch.stack([new_k, new_v], dim=0)  # L1078

        # output                                                     # L1082-1083
        x = x.flatten(2)
        x = self.o(x)
        return x, updated_kv_cache


# ── Attention Block ─────────────────────────────────────────────────
# Source: wan_video_dit_action_casual_chunk.py L1087-1190


class CausalWanAttentionBlock(nn.Module):
    """Transformer block: self-attn + cross-attn + FFN with 6-param modulation.
    Source: wan_video_dit_action_casual_chunk.py L1087-1190
    """

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        frame_seqlen: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)  # L1113
        self.self_attn = CausalWanSelfAttention(  # L1114-1124
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
        )
        self.norm3 = (  # L1126-1128
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](  # L1129-1133
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)  # L1134
        self.ffn = nn.Sequential(  # L1135-1137
            ColumnParallelLinear(dim, ffn_dim, bias=True, gather_output=False, return_bias=False),
            nn.GELU(approximate="tanh"),
            RowParallelLinear(ffn_dim, dim, bias=True, input_is_parallel=True, return_bias=False),
        )
        self.modulation = nn.Parameter(  # L1140
            torch.randn(1, 6, dim) / dim**0.5
        )

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        context: torch.Tensor,
        action_register_length: int | None = None,
        kv_cache: torch.Tensor | None = None,
        crossattn_cache: dict | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Source: wan_video_dit_action_casual_chunk.py L1142-1187"""
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)  # L1162

        # self-attention                                              # L1164-1174
        y, updated_kv_cache = self.self_attn(
            x=(self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)),  # L1166
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            is_tf=is_tf,
            current_start_frame=current_start_frame,
        )
        x = x + (y * e[2].squeeze(2))  # L1175

        # cross-attention + FFN                                       # L1178-1186
        x = x + self.cross_attn(self.norm3(x), context, crossattn_cache=crossattn_cache)  # L1179
        y = self.ffn(  # L1180-1181
            self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
        )
        x = x + (y * e[5].squeeze(2))  # L1183
        return x, updated_kv_cache


# ── Output Head ─────────────────────────────────────────────────────
# Source: wan_video_dit_action_casual_chunk.py L1190-1215


class CausalHead(nn.Module):
    """Output norm + linear with 2-param modulation.
    Source: wan_video_dit_action_casual_chunk.py L1190-1215
    Runs once per step (not TP-critical), uses nn.Linear.
    """

    def __init__(self, dim: int, out_dim: int, patch_size: tuple, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim  # L1194
        self.out_dim = out_dim  # L1195
        self.patch_size = patch_size  # L1196
        out_channels = math.prod(patch_size) * out_dim  # L1200
        self.norm = WanLayerNorm(dim, eps)  # L1201
        self.head = nn.Linear(dim, out_channels)  # L1202
        self.modulation = nn.Parameter(  # L1205
            torch.randn(1, 2, dim) / dim**0.5
        )

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L1, C]
            e: [B, F, 1, C]     (time embedding, unsqueezed)
        Source: wan_video_dit_action_casual_chunk.py L1207-1215
        """
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)  # L1213
        x = self.head(  # L1214
            self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
        )
        return x


# ── Main Model ──────────────────────────────────────────────────────
# Source: wan_video_dit_action_casual_chunk.py L1218-2200


class CausalWanModel(nn.Module):
    """Causal video diffusion transformer for DreamZero.

    Source: wan_video_dit_action_casual_chunk.py L1218-2200
    Architecture (14B): 40 layers, dim=5120, heads=40, ffn=13824

    __init__ params match original L1230-1256:
        model_type, patch_size, frame_seqlen, text_len, in_dim, dim,
        ffn_dim, freq_dim, text_dim, out_dim, num_heads, num_layers,
        max_chunk_size, sink_size, qk_norm, cross_attn_norm, eps,
        num_frame_per_block, action_dim, num_registers, max_state_dim,
        max_num_embodiments, hidden_size, diffusion_model_pretrained_path,
        num_action_per_block, num_state_per_block
    """

    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: tuple[int, int, int] = (1, 2, 2),
        frame_seqlen: int = 220,
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        max_chunk_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        num_frame_per_block: int = 1,
        action_dim: int = 32,
        num_registers: int = 8,
        max_state_dim: int = 64,
        max_num_embodiments: int = 32,
        hidden_size: int = 1024,
        diffusion_model_pretrained_path: str | None = None,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
    ) -> None:
        super().__init__()
        assert model_type in ["t2v", "i2v", "ti2v"]  # L1297
        self.model_type = model_type  # L1298
        self.patch_size = patch_size  # L1300
        self.frame_seqlen = frame_seqlen  # L1301
        self.text_len = text_len  # L1302
        self.dim = dim  # L1304
        self.freq_dim = freq_dim  # L1306
        self.out_dim = out_dim  # L1308
        self.num_heads = num_heads  # L1309
        self.num_layers = num_layers  # L1310
        self.local_attn_size = (  # L1311
            max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        )
        self.num_frame_per_block = num_frame_per_block  # L1315
        self.action_dim = action_dim  # L1317
        self.num_action_per_block = num_action_per_block  # L1322
        self.num_state_per_block = num_state_per_block  # L1323

        # Action encoder/decoder                                      # L1327-1343
        max_num_embodiments_local = 1  # L1325
        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments_local,
            input_dim=max_state_dim,
            hidden_dim=hidden_size,
            output_dim=dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=dim,
            num_embodiments=max_num_embodiments_local,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments_local,
            input_dim=dim,
            hidden_dim=hidden_size,
            output_dim=action_dim,
        )

        # Embeddings                                                  # L1346-1355
        # Upstream DreamZero uses a plain nn.Conv3d here
        # (wan_video_dit_action_casual_chunk.py L1386-L1391).
        #
        # vLLM's Conv3dLayer can rewrite non-overlapping strided convs
        # (kernel_size == stride, no padding) into a GEMM fast path. For
        # DreamZero's bf16 i2v prefill, that changes accumulation order and
        # causes the first-frame KV cache to drift from upstream even though
        # the final outputs may still match. Force the native conv path so
        # patch embedding stays numerically identical to upstream.
        self.patch_embedding = Conv3dLayer(
            in_dim,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  # L1346
        self.patch_embedding.enable_linear = False
        self.text_embedding = nn.Sequential(  # L1348-1350
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(  # L1352-1353
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(  # L1354-1355
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        # Transformer blocks                                         # L1358-1364
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    frame_seqlen,
                    self.local_attn_size,
                    sink_size,
                    num_frame_per_block,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    num_action_per_block,
                    num_state_per_block,
                )
                for _ in range(num_layers)
            ]
        )

        # Head                                                        # L1367
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # RoPE buffers                                                # L1370-1379
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs_action = rope_params(1024 * 10, d)  # L1373
        self.freqs_state = rope_params(1024, d)  # L1374
        self.freqs = [  # L1375-1379
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]

        # Image embedding for i2v only                               # L1380-1381
        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        # Initialize weights                                         # L1383-1384
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize parameters with DreamZero's bare-model scheme.
        Source: wan_video_dit_action_casual_chunk.py L2176-2194

        Upstream initializes every `nn.Linear` with Xavier uniform and
        zero bias. This port applies the same rule to vLLM
        `ColumnParallelLinear` / `RowParallelLinear`, using the full
        unsharded fan-in/fan-out so the TP shards follow the same Xavier
        distribution as the original dense weight.
        """

        def _init_linear_like(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):  # L2182-2185
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                return

            if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
                fan_in = module.input_size
                fan_out = module.output_size
                bound = math.sqrt(6.0 / float(fan_in + fan_out))
                nn.init.uniform_(module.weight, -bound, bound)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Basic init                                                 # L2181-2185
        for module in self.modules():
            _init_linear_like(module)

        # Patch embedding follows upstream Conv3d handling: Xavier
        # for weight, Conv3d-style uniform bias.                    # L2187-2191
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        if self.patch_embedding.bias is not None:
            fan_in = self.patch_embedding.in_channels * math.prod(self.patch_embedding.kernel_size)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.patch_embedding.bias, -bound, bound)

        for module in self.text_embedding.modules():  # L2188-2190
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

        for module in self.time_embedding.modules():  # L2190-2191
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

        nn.init.zeros_(self.head.head.weight)  # L2193

    def _create_freqs(self, grid_size: torch.Tensor, start_frame: int) -> torch.Tensor:
        """Create 3D RoPE frequency tensor.
        Source: wan_video_dit_action_casual_chunk.py L2151-2174
        """
        device = self.patch_embedding.weight.device  # L2156
        if any(freq.device != device for freq in self.freqs):  # L2157-2158
            self.freqs = [freq.to(device) for freq in self.freqs]
        if self.freqs_action.device != device:  # L2159-2160
            self.freqs_action = self.freqs_action.to(device)
        if self.freqs_state.device != device:  # L2161-2162
            self.freqs_state = self.freqs_state.to(device)

        f, h, w = grid_size.tolist()  # L2164
        freqs = torch.cat(
            [  # L2165-2172
                self.freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
        return freqs

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor) -> torch.Tensor:
        """Reconstruct video from patch embeddings.
        Source: wan_video_dit_action_casual_chunk.py L2127-2149
        """
        B = x.shape[0]  # L2142
        c = self.out_dim  # L2143
        grid_size = grid_size.tolist()  # L2144
        assert x.shape[1] == math.prod(grid_size)  # L2145
        x = x.view(B, *grid_size, *self.patch_size, c)  # L2146
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)  # L2147
        x = x.reshape(B, c, *[i * j for i, j in zip(grid_size, self.patch_size)])  # L2148
        return x

    def _forward_blocks(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        kv_cache: list[torch.Tensor],
        current_start_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Source: wan_video_dit_action_casual_chunk.py L1691-1779"""
        x = x.flatten(start_dim=2).transpose(1, 2)  # L1709
        B = x.shape[0]  # L1711
        F_t = timestep.shape[1]  # L1712

        # Action/state encoding                                       # L1714-1726
        if action is not None:
            embodiment_id = torch.tensor([0], device=x.device).repeat(B)  # L1715
            action_features = self.action_encoder(action, timestep_action, embodiment_id)  # L1716
            state_features = self.state_encoder(state, embodiment_id)  # L1717
            action_register = torch.cat([action_features, state_features], dim=1)  # L1718
            action_length = action_features.shape[1]  # L1719
            action_register_length = action_register.shape[1]  # L1720
            x = torch.cat([x, action_register], dim=1)  # L1721
        else:
            action_length = 0  # L1725
            action_register_length = None  # L1726

        # Time embeddings                                             # L1728-1742
        timestep = timestep.unsqueeze(-1).expand(B, F_t, seq_len // F_t).reshape(B, -1)  # L1729
        if action is not None:
            assert timestep_action is not None and state is not None
            state_features_t = self.state_encoder(state, embodiment_id)
            stride = timestep_action.shape[1] // state_features_t.shape[1]  # L1734
            timestep_state = timestep_action[:, ::stride]  # L1735
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)  # L1736

        e = self.time_embedding(  # L1738-1739
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x)
        )
        e = e.unflatten(dim=0, sizes=(B, -1))  # L1740
        e0 = self.time_projection(e)  # L1741
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))  # L1742

        # Context embedding                                           # L1744-1749
        context = self.text_embedding(context)  # L1745
        if clip_feature is not None:  # L1747
            clip_embedding = self.img_emb(clip_feature)  # L1748
            context = torch.cat([clip_embedding, context], dim=1)  # L1749

        # Transformer blocks                                          # L1751-1764
        updated_kv_caches: list[torch.Tensor] = []
        for block_index, block in enumerate(self.blocks):
            x, updated_kv_cache = block(
                x=x,
                e=e0,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                context=context,
                action_register_length=action_register_length,
                kv_cache=kv_cache[block_index] if kv_cache else None,
                current_start_frame=current_start_frame,
            )
            updated_kv_caches.append(updated_kv_cache)

        # Action decoding                                              # L1766-1770
        if action is not None:
            action_noise_pred = x[:, seq_len : seq_len + action_length]  # L1767
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)  # L1768
        else:
            action_noise_pred = None  # L1770

        x_video = x[:, :seq_len]  # L1773
        e_video = e[:, :seq_len]  # L1774
        x_video = self.head(x_video, e_video.unsqueeze(2))  # L1777

        return x_video, action_noise_pred, updated_kv_caches

    def _forward_inference(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[torch.Tensor],
        current_start_frame: int,
        y: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        timestep_action: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Source: wan_video_dit_action_casual_chunk.py L1863-1950"""
        if self.model_type == "i2v":  # L1910
            assert clip_feature is not None and y is not None
        assert context.shape[1] == self.text_len  # L1912

        if y is not None:  # L1914
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)  # L1915

        x = self.patch_embedding(x)  # L1918
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)  # L1919
        freqs = self._create_freqs(grid_size, current_start_frame)  # L1921-1924

        x_video, action_noise_pred, updated_kv_caches = self._forward_blocks(  # L1926-1939
            x=x,
            seq_len=seq_len,
            freqs=freqs,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            embodiment_id=embodiment_id,
            action=action,
            timestep_action=timestep_action,
            state=state,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
        )

        x_video = x_video.clone()  # L1942
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()  # L1944

        video_noise_pred = self.unpatchify(x_video, grid_size)  # L1948
        return video_noise_pred, action_noise_pred, updated_kv_caches

    def forward(self, *args: Any, **kwargs: Any):
        """Inference only. Requires kv_cache."""
        return self._forward_inference(*args, **kwargs)
