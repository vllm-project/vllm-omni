# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Krea 2 single-stream MMDiT transformer for vLLM-Omni."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rope_to_qk

logger = init_logger(__name__)


class Krea2RMSNorm(nn.Module):
    """Zero-centered RMSNorm: effective multiplier is ``1 + weight``."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = F.rms_norm(
            hidden_states.float(), (self.dim,), weight=self.weight + 1.0, eps=self.eps
        )
        return hidden_states.to(dtype)


class Krea2Attention(nn.Module):
    """GQA self-attention with RMSNorm on Q/K, optional RoPE, and sigmoid output gate."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads

        self.to_q = nn.Linear(hidden_size, self.head_dim * self.num_heads, bias=False)
        self.to_k = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_v = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = Krea2RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Krea2RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False)])

        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        gate = self.to_gate(hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)

        query, key = apply_rope_to_qk(self.rope, query, key, image_rotary_emb)

        attn_metadata = None
        if attention_mask is not None:
            attn_metadata = AttentionMetadata(attn_mask=attention_mask)

        hidden_states = self.attn(query, key, value, attn_metadata)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return self.to_out[0](hidden_states)


class Krea2SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class Krea2TextFusionBlock(nn.Module):
    """Pre-norm transformer block for text fusion (no RoPE, no timestep modulation)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = Krea2RMSNorm(dim, eps=eps)
        self.norm2 = Krea2RMSNorm(dim, eps=eps)
        self.attn = Krea2Attention(dim, num_heads, num_kv_heads, eps=eps)
        self.ff = Krea2SwiGLU(dim, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), attention_mask=attention_mask
        )
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class Krea2TextFusion(nn.Module):
    """Fuses stacked text-encoder hidden states into a single sequence.

    Two layerwise blocks attend across the tapped-layer axis per token,
    a linear projector collapses that axis, then two refiner blocks attend
    across the token sequence.
    """

    def __init__(
        self,
        num_text_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        num_layerwise_blocks: int,
        num_refiner_blocks: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_layerwise_blocks)
            ]
        )
        self.projector = nn.Linear(num_text_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_refiner_blocks)
            ]
        )

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape

        hidden_states = encoder_hidden_states.reshape(batch_size * seq_len, num_text_layers, dim)
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states.contiguous())

        hidden_states = hidden_states.reshape(batch_size, seq_len, num_text_layers, dim).permute(
            0, 1, 3, 2
        )
        hidden_states = self.projector(hidden_states).squeeze(-1)

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        return hidden_states


class Krea2TransformerBlock(nn.Module):
    """Main DiT block with 6-channel AdaLN modulation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))
        self.norm1 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Krea2Attention(hidden_size, num_heads, num_kv_heads, eps=norm_eps)
        self.ff = Krea2SwiGLU(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        modulation = temb.unflatten(-1, (6, -1)) + self.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)

        attn_out = self.attn(
            (1.0 + prescale) * self.norm1(hidden_states) + preshift,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + pregate * attn_out
        ff_out = self.ff((1.0 + postscale) * self.norm2(hidden_states) + postshift)
        hidden_states = hidden_states + postgate * ff_out
        return hidden_states


class Krea2TimestepEmbedding(nn.Module):
    """Sinusoidal flow-time embedding (scale 1000) followed by a two-layer MLP."""

    def __init__(self, embed_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = nn.Linear(embed_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(1e4) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half
        )
        args = (timestep.float() * 1e3)[:, None, None] * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        return self.linear_2(F.gelu(self.linear_1(emb), approximate="tanh"))


class Krea2TextProjection(nn.Module):
    """Projects fused text features into the transformer width."""

    def __init__(self, text_dim: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = Krea2RMSNorm(text_dim, eps=eps)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(self.norm(hidden_states))
        return self.linear_2(F.gelu(hidden_states, approximate="tanh"))


class Krea2FinalLayer(nn.Module):
    """Final adaptive RMSNorm and output projection."""

    def __init__(self, hidden_size: int, out_channels: int, eps: float) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))
        self.norm = Krea2RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        modulation = temb + self.scale_shift_table
        scale, shift = modulation.chunk(2, dim=1)
        hidden_states = (1.0 + scale) * self.norm(hidden_states) + shift
        return self.linear(hidden_states)


class Krea2PosEmbed(nn.Module):
    """3-axis RoPE position embedding for the combined [text, image] sequence."""

    def __init__(self, theta: float, axes_dim: list[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = torch.float32 if ids.device.type == "mps" else torch.float64
        for i in range(n_axes):
            freqs_cis = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                use_real=False,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(freqs_cis.real)
            sin_out.append(freqs_cis.imag)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class Krea2Transformer2DModel(nn.Module):
    """Krea 2 single-stream MMDiT flow-matching backbone.

    Ported from the diffusers ``Krea2Transformer2DModel`` to vLLM-Omni
    conventions: plain ``nn.Module``, vLLM-Omni attention layer, and
    ``load_weights`` for checkpoint loading.
    """

    _repeated_blocks = ["Krea2TransformerBlock"]

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        in_channels: int = 64,
        num_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 12,
        intermediate_size: int = 16384,
        timestep_embed_dim: int = 256,
        text_hidden_dim: int = 2560,
        num_text_layers: int = 12,
        text_num_attention_heads: int = 20,
        text_num_key_value_heads: int = 20,
        text_intermediate_size: int = 6912,
        num_layerwise_text_blocks: int = 2,
        num_refiner_text_blocks: int = 2,
        axes_dims_rope: tuple[int, int, int] = (32, 48, 48),
        rope_theta: float = 1000.0,
        norm_eps: float = 1e-5,
        quant_config: object | None = None,
    ) -> None:
        super().__init__()

        if od_config is not None:
            model_config = od_config.tf_model_config
            num_layers = getattr(model_config, "num_layers", num_layers)

        hidden_size = attention_head_dim * num_attention_heads

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size

        self.img_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.time_embed = Krea2TimestepEmbedding(timestep_embed_dim, hidden_size)
        self.time_mod_proj = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

        self.text_fusion = Krea2TextFusion(
            num_text_layers=num_text_layers,
            dim=text_hidden_dim,
            num_heads=text_num_attention_heads,
            num_kv_heads=text_num_key_value_heads,
            intermediate_size=text_intermediate_size,
            num_layerwise_blocks=num_layerwise_text_blocks,
            num_refiner_blocks=num_refiner_text_blocks,
            eps=norm_eps,
        )
        self.txt_in = Krea2TextProjection(text_hidden_dim, hidden_size, eps=norm_eps)
        self.rotary_emb = Krea2PosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))

        self.transformer_blocks = nn.ModuleList(
            [
                Krea2TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_key_value_heads,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer = Krea2FinalLayer(hidden_size, out_channels=in_channels, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor]:
        batch_size, image_seq_len, _ = hidden_states.shape
        text_seq_len = encoder_hidden_states.shape[1]

        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))

        text_attention_mask = None
        attention_mask = None
        if encoder_attention_mask is not None:
            text_attention_mask = encoder_attention_mask
            image_mask = encoder_attention_mask.new_ones((batch_size, image_seq_len))
            attention_mask = torch.cat([encoder_attention_mask, image_mask], dim=1)

        encoder_hidden_states = self.text_fusion(
            encoder_hidden_states, attention_mask=text_attention_mask
        )
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        hidden_states = self.img_in(hidden_states)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        image_rotary_emb = self.rotary_emb(position_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, temb_mod, image_rotary_emb, attention_mask)

        hidden_states = hidden_states[:, text_seq_len:]
        output = self.final_layer(hidden_states, temb)

        return (output,)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name not in params_dict and ".to_out.0." in name:
                name = name.replace(".to_out.0.", ".to_out.")
            if name not in params_dict:
                logger.warning("Skipping unknown weight: %s", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
