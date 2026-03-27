# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import is_torch_npu_available
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput, SequenceParallelOutput

logger = init_logger(__name__)


def _get_span_length(pos_1d: torch.Tensor) -> int:
    return int(pos_1d.max().item() - pos_1d.min().item() + 1)


def _find_correction_factor(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _find_correction_range(low_ratio, high_ratio, dim, base, ori_max_pe_len):
    low = np.floor(_find_correction_factor(low_ratio, dim, base, ori_max_pe_len))
    high = np.ceil(_find_correction_factor(high_ratio, dim, base, ori_max_pe_len))
    return max(low, 0), min(high, dim - 1)


def _linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _find_newbase_ntk(dim, base, scale):
    return base * (scale ** (dim / (dim - 2)))


def _get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,
    yarn=True,
    max_pe_len=None,
    ori_max_pe_len=64,
    current_timestep=1.0,
    resonance: bool = True,
    resonance_min_rot: float = 1.0,
):
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    device = pos.device
    dtype = freqs_dtype

    idx = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim

    if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
        if not isinstance(max_pe_len, torch.Tensor):
            max_pe_len = torch.tensor(max_pe_len与其他dtype=dtype, device=device)
        scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)

        if resonance:
            L = torch.tensor(float(ori_max_pe_len), dtype=dtype, device=device)
            theta_t = torch.tensor(theta, dtype=dtype, device=device)
            inv_freq = torch.exp(-idx * torch.log(theta_t))
            rot = (L * inv_freq) / (2.0 * math.pi)
            use_resonance = rot >= 1.0
            rot_rounded = torch.round(rot)
            k = torch.clamp(rot_rounded, min=resonance_min_rot)
            target_inv_freq = (2.0 * math.pi * k) / L
            alpha_res = -torch.log(target_inv_freq) / torch.log(theta_t)
            idx_eff = torch.where(use_resonance, alpha_res, idx)
        else:
            idx_eff = idx

        beta_0, beta_1 = 1.25, 0.75
        gamma_0, gamma_1 = 16, 2

        freqs_base = 1.0 / torch.exp(idx_eff * torch.log(torch.tensor(theta, dtype=dtype, device=device)))
        freqs_linear = freqs_base / scale

        new_base = _find_newbase_ntk(dim, theta, scale)
        if isinstance(new_base, torch.Tensor) and new_base.dim() > 0:
            new_base = new_base.view(-1, 1)
        new_base = torch.tensor(float(new_base), dtype=dtype, device=device)
        freqs_ntk = 1.0 / torch.exp(idx_eff * torch.log(new_base))

        low, high = _find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
        low = max(0, low)
        high = min(dim // 2, high)
        mask_beta = 1 - _linear_ramp_mask(low, high, dim // 2).to(device).to(dtype)
        freqs = freqs_linear * (1 - mask_beta) + freqs_ntk * mask_beta

        low, high = _find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
        low = max(0, low)
        high = min(dim // 2, high)
        mask_gamma = 1 - _linear_ramp_mask(low, high, dim // 2).to(device).to(dtype)
        freqs = freqs * (1 - mask_gamma) + freqs_base * mask_gamma

    else:
        theta_ntk = theta * ntk_factor
        idx0 = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim
        freqs = 1.0 / torch.exp(idx0 * torch.log(torch.tensor(theta_ntk, dtype=dtype, device=device)))
        freqs = freqs / linear_factor

    freqs = torch.outer(pos if isinstance(pos, torch.Tensor) else torch.tensor(pos, device=device), freqs)

    if freqs.device.type == "npu":
        freqs = freqs.float()

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
            mscale = torch.where(
                scale <= 1.0, torch.tensor(1.0, device=scale.device, dtype=scale.dtype), 0.1 * torch.log(scale) + 1.0
            )
            freqs_cos = freqs_cos * mscale
            freqs_sin = freqs_sin * mscale
        return freqs_cos, freqs_sin
    elif use_real:
        return torch.cat([freqs.cos(), freqs.cos()], dim=-1).float(), torch.cat(
            [freqs.sin(), freqs.sin()], dim=-1
        ).float()
    else:
        return torch.polar(torch.ones_like(freqs), freqs)


class UltraFluxPosEmbed(nn.Module):
    def __init__(
        self,
        theta: int,
        axes_dim: list[int],
        method: str = "yarn",
        base_resolution_hw: tuple[int, int] | None = None,
        axis_order: str = "t-h-w",
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.method = method

        self.base_resolution = 2048
        self.patch_size = 32
        self.base_patches = self.base_resolution // self.patch_size

        if base_resolution_hw is None:
            H_train_px, W_train_px = self.base_resolution, self.base_resolution
        else:
            H_train_px, W_train_px = base_resolution_hw

        self.base_patches_hw = (
            max(1, H_train_px // self.patch_size),
            max(1, W_train_px // self.patch_size),
        )

        axis_order = axis_order.lower().replace("_", "").replace("-", "")
        if axis_order not in ("thw", "twh"):
            raise ValueError("axis_order must be 't-h-w' or 't-w-h'")
        self.axis_order = axis_order

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        assert n_axes == len(self.axes_dim), "axes_dim must match the number of axes in ids"

        cos_out, sin_out = [], []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64

        if self.axis_order == "thw":
            axis_to_base_len = {0: None, 1: self.base_patches_hw[0], 2: self.base_patches_hw[1]}
        else:
            axis_to_base_len = {0: None, 1: self.base_patches_hw[1], 2: self.base_patches_hw[0]}

        for i in range(n_axes):
            common_kwargs = {
                "dim": self.axes_dim[i],
                "pos": pos[:, i],
                "theta": self.theta,
                "repeat_interleave_real": True,
                "use_real": True,
                "freqs_dtype": freqs_dtype,
            }

            base_len = axis_to_base_len.get(i, None)

            if self.method == "yarn" and base_len is not None:
                current_len = _get_span_length(pos[:, i])
                if current_len > base_len:
                    max_pe_len = torch.tensor(current_len, dtype=freqs_dtype, device=pos.device)
                    cos, sin = _get_1d_rotary_pos_embed(
                        **common_kwargs,
                        yarn=True,
                        max_pe_len=max_pe_len,
                        ori_max_pe_len=base_len,
                    )
                else:
                    cos, sin = _get_1d_rotary_pos_embed(**common_kwargs)
            else:
                cos, sin = _get_1d_rotary_pos_embed(**common_kwargs)

            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class ColumnParallelApproxGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, approximate: str, bias: bool = True, quant_config=None):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim_in,
            dim_out,
            bias=bias,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
        )
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return F.gelu(x, approximate=self.approximate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        activation_fn: str = "gelu-approximate",
        inner_dim: int | None = None,
        bias: bool = True,
        quant_config=None,
    ) -> None:
        super().__init__()

        assert activation_fn == "gelu-approximate", "Only gelu-approximate is supported."

        inner_dim = inner_dim or int(dim * mult)
        dim_out = dim_out or dim

        layers: list[nn.Module] = [
            ColumnParallelApproxGELU(dim, inner_dim, approximate="tanh", bias=bias, quant_config=quant_config),
            nn.Identity(),
            RowParallelLinear(
                inner_dim,
                dim_out,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
            ),
        ]

        self.net = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class UltraFluxAttention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
        quant_config=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            bias=bias,
            quant_config=quant_config,
        )

        if not self.pre_only:
            self.to_out = nn.ModuleList(
                [
                    RowParallelLinear(
                        self.inner_dim,
                        self.out_dim,
                        bias=out_bias,
                        input_is_parallel=True,
                        return_bias=False,
                        quant_config=quant_config,
                    ),
                    nn.Dropout(dropout),
                ]
            )

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                bias=added_proj_bias,
                quant_config=quant_config,
            )

            self.to_add_out = RowParallelLinear(
                self.inner_dim,
                query_dim,
                bias=out_bias,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
            )

        if not self.pre_only:
            self.to_out = nn.ModuleList(
                [
                    RowParallelLinear(
                        self.inner_dim,
                        self.out_dim,
                        bias=out_bias,
                        input_is_parallel=True,
                        return_bias=False,
                    ),
                    nn.Dropout(dropout),
                ]
            )

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                bias=added_proj_bias,
            )

            self.to_add_out = RowParallelLinear(
                self.inner_dim,
                query_dim,
                bias=out_bias,
                input_is_parallel=True,
                return_bias=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        query, key, value = self.to_qkv(hidden_states).chunk(3, dim=-1)

        encoder_query = encoder_key = encoder_value = None
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_query, encoder_key, encoder_value = self.add_kv_proj(encoder_hidden_states).chunk(3, dim=-1)

        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (self.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = Attention.apply_attention(query, key, value, attn_mask=attention_mask)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class UltraFluxSingleTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0, quant_config=None
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        self.attn = UltraFluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


class UltraFluxTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        quant_config=None,
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = UltraFluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
            quant_config=quant_config,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", quant_config=quant_config)

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", quant_config=quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class UltraFluxTransformer2DModel(nn.Module):
    _repeated_blocks = ["UltraFluxTransformerBlock"]

    _sp_plan = {
        "pos_embed": {
            0: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True),
            1: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = True,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        method: str = "yarn",
    ):
        super().__init__()
        model_config = od_config.tf_model_config
        num_layers = model_config.num_layers
        self.parallel_config = od_config.parallel_config
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.guidance_embeds = guidance_embeds

        quant_config = od_config.quantization_config.get_vllm_quant_config() if od_config.quantization_config else None

        self.pos_embed = UltraFluxPosEmbed(theta=10000, axes_dim=axes_dims_rope, method=method)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                UltraFluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    quant_config=quant_config,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                UltraFluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    quant_config=quant_config,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000

        if guidance is not None:
            guidance = guidance.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        if is_torch_npu_available():
            freqs_cos, freqs_sin = self.pos_embed(ids.cpu())
            image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
        else:
            image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)
        return loaded_params
