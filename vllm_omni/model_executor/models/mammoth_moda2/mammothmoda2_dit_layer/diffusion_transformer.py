# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. on 2025-09-30.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://www.apache.org/licenses/LICENSE-2.0
#
# This modified file is released under the same license.
#
# --- Upstream header preserved below ---
#
# Copyright 2025 BAAI, The Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from .attention_processor import AttnProcessor
from .block_lumina2 import (
    Lumina2CombinedTimestepCaptionEmbedding,
    LuminaFeedForward,
    LuminaLayerNormContinuous,
    LuminaRMSNormZero,
)
from .rope_real import RotaryPosEmbedReal


class TransformerBlock(nn.Module):
    """MammothModa2 DiT transformer block（推理使用）。"""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        """Initialize the transformer block."""
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.modulation = modulation

        processor = AttnProcessor()

        # Initialize attention layer
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm=None,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=processor,
        )
        # 显式使用 transformers 的 Qwen2RMSNorm，避免依赖 diffusers 内部创建的 `RMSNorm` 再做递归替换。
        self.attn.norm_q = Qwen2RMSNorm(self.head_dim, eps=1e-5)
        self.attn.norm_k = Qwen2RMSNorm(self.head_dim, eps=1e-5)

        # Initialize feed-forward network
        self.feed_forward = LuminaFeedForward(
            dim=dim, inner_dim=4 * dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier
        )

        # Initialize normalization layers
        if modulation:
            self.norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True)
        else:
            self.norm1 = Qwen2RMSNorm(dim, eps=norm_eps)

        self.ffn_norm1 = Qwen2RMSNorm(dim, eps=norm_eps)
        self.norm2 = Qwen2RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = Qwen2RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            if temb is None:
                raise ValueError("temb must be provided when modulation is enabled")

            norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
            hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
        else:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = hidden_states + self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
            hidden_states = hidden_states + self.ffn_norm2(mlp_output)

        return hidden_states


class Transformer2DModel(ModelMixin, ConfigMixin):
    """MammothModa2 DiT transformer（推理使用）。"""

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: int | None = None,
        hidden_size: int = 2304,
        num_layers: int = 26,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 24,
        num_kv_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        norm_eps: float = 1e-5,
        axes_dim_rope: tuple[int, int, int] = (32, 32, 32),
        axes_lens: tuple[int, int, int] = (300, 512, 512),
        text_feat_dim: int = 1024,
        timestep_scale: float = 1.0,
    ) -> None:
        """Initialize the  transformer model."""
        super().__init__()
        self.hidden_size = hidden_size

        # Validate configuration
        if (hidden_size // num_attention_heads) != sum(axes_dim_rope):
            raise ValueError(
                f"hidden_size // num_attention_heads ({hidden_size // num_attention_heads}) "
                f"must equal sum(axes_dim_rope) ({sum(axes_dim_rope)})"
            )

        self.out_channels = out_channels or in_channels
        # Initialize embeddings
        self.rope_embedder = RotaryPosEmbedReal(
            theta=10000,
            axes_dim=axes_dim_rope,
            axes_lens=axes_lens,
            patch_size=patch_size,
        )

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.ref_image_patch_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size,
            text_feat_dim=text_feat_dim,
            norm_eps=norm_eps,
            timestep_scale=timestep_scale,
        )

        # Initialize transformer blocks
        self.noise_refiner = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.ref_image_refiner = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=False,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        # 3. Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )

        # Add learnable embeddings to distinguish different images
        self.image_index_embedding = nn.Parameter(torch.randn(5, hidden_size))  # support max 5 ref images

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        text_attention_mask: torch.Tensor,
        ref_image_hidden_states: list[list[torch.Tensor]] | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        if return_dict:
            raise ValueError("return_dict=True is not supported in vLLM inference.")
        if ref_image_hidden_states is not None:
            raise ValueError("ref_image_hidden_states is not supported in vLLM inference.")
        if hidden_states.ndim != 4:
            raise ValueError(f"Expected hidden_states to be 4D [B,C,H,W], got shape={tuple(hidden_states.shape)}")

        batch_size, _channels, height, width = hidden_states.shape
        if batch_size != text_hidden_states.shape[0] or batch_size != text_attention_mask.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"hidden_states={batch_size}, text_hidden_states={text_hidden_states.shape[0]}, "
                f"text_attention_mask={text_attention_mask.shape[0]}"
            )

        p = self.config.patch_size
        if height % p != 0 or width % p != 0:
            raise ValueError(f"Input latent H/W must be divisible by patch_size={p}, got {height}x{width}")

        device = hidden_states.device

        temb, text_hidden_states = self.time_caption_embed(timestep, text_hidden_states, hidden_states.dtype)

        img_tokens = rearrange(hidden_states, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        img_tokens = self.x_embedder(img_tokens)

        img_len = (height // p) * (width // p)
        img_mask = torch.ones((batch_size, img_len), dtype=torch.bool, device=device)
        l_effective_img_len = [img_len for _ in range(batch_size)]
        img_sizes = [(height, width) for _ in range(batch_size)]

        l_effective_ref_img_len = [[] for _ in range(batch_size)]
        ref_img_sizes = [None for _ in range(batch_size)]

        (
            context_rotary_emb,
            _ref_img_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        ) = self.rope_embedder(
            freqs_cis,
            text_attention_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
            device,
        )

        for layer in self.context_refiner:
            text_hidden_states = layer(text_hidden_states, text_attention_mask, context_rotary_emb)

        for layer in self.noise_refiner:
            img_tokens = layer(img_tokens, img_mask, noise_rotary_emb, temb)

        max_seq_len = max(seq_lengths)
        attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        joint_hidden_states = hidden_states.new_zeros(batch_size, max_seq_len, self.config.hidden_size)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            attention_mask[i, :seq_len] = True
            joint_hidden_states[i, :encoder_seq_len] = text_hidden_states[i, :encoder_seq_len]
            joint_hidden_states[i, encoder_seq_len : encoder_seq_len + img_len] = img_tokens[i, :img_len]

        hidden_states = joint_hidden_states
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)

        hidden_states = self.norm_out(hidden_states, temb)

        img_hidden_states = torch.stack(
            [
                hidden_states[i, encoder_seq_len : encoder_seq_len + img_len]
                for i, encoder_seq_len in enumerate(encoder_seq_lengths)
            ],
            dim=0,
        )
        output = rearrange(
            img_hidden_states,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=height // p,
            w=width // p,
            p1=p,
            p2=p,
        )
        return output
