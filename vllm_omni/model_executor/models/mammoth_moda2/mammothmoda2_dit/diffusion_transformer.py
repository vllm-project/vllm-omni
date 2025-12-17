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

import itertools
from typing import Any

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from einops import rearrange
from loguru import logger
from torch import nn

from mammothmoda2.model.mammothmoda2_qwen2_5_vl.modeling_mammothmoda2_qwen2_5_vl import Qwen2RMSNorm

from .attention_processor import AttnProcessor, AttnProcessorFlash2Varlen
from .block_lumina2 import (
    Lumina2CombinedTimestepCaptionEmbedding,
    LuminaFeedForward,
    LuminaLayerNormContinuous,
    LuminaRMSNormZero,
)
from .rope_real import RotaryPosEmbedReal


def replace_rmsnorm_with_custom(module):
    """Replace RMSNorm with custom implementation recursively."""
    for name, child in module.named_children():
        # Import RMSNorm here to avoid circular import issues
        if child.__class__.__name__ == "RMSNorm":
            # 获取构造RMSNorm所需的参数
            eps = getattr(child, "eps", 1e-5)
            normalized_shape = getattr(child, "normalized_shape", None)
            if normalized_shape is None and hasattr(child, "weight"):
                normalized_shape = child.weight.shape

            new_norm = Qwen2RMSNorm(normalized_shape, eps=eps)
            # Copy weights and biases if they exist
            if hasattr(child, "weight") and hasattr(new_norm, "weight"):
                new_norm.weight.data.copy_(child.weight.data)
            if hasattr(child, "bias") and hasattr(new_norm, "bias"):
                new_norm.bias.data.copy_(child.bias.data)

            setattr(module, name, new_norm)
        else:
            replace_rmsnorm_with_custom(child)


class TransformerBlock(nn.Module):
    """
    Transformer block for  model.

    This block implements a transformer layer with:
    - Multi-head attention with flash attention
    - Feed-forward network with SwiGLU activation
    - RMS normalization
    - Optional modulation for conditional generation

    Args:
        dim: Dimension of the input and output tensors
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key-value heads
        multiple_of: Multiple of which the hidden dimension should be
        ffn_dim_multiplier: Multiplier for the feed-forward network dimension
        norm_eps: Epsilon value for normalization layers
        modulation: Whether to use modulation for conditional generation
        use_fused_rms_norm: Whether to use fused RMS normalization
        use_fused_swiglu: Whether to use fused SwiGLU activation
    """

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

        try:
            processor = AttnProcessorFlash2Varlen()
        except ImportError:
            processor = AttnProcessor()

        # Initialize attention layer
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm="rms_norm",
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=processor,
        )

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

        self.initialize_weights()
        replace_rmsnorm_with_custom(self)

    def initialize_weights(self) -> None:
        """
        Initialize the weights of the transformer block.

        Uses Xavier uniform initialization for linear layers and zero initialization for biases.
        """
        nn.init.xavier_uniform_(self.attn.to_q.weight)
        nn.init.xavier_uniform_(self.attn.to_k.weight)
        nn.init.xavier_uniform_(self.attn.to_v.weight)
        nn.init.xavier_uniform_(self.attn.to_out[0].weight)

        nn.init.xavier_uniform_(self.feed_forward.linear_1.weight)
        nn.init.xavier_uniform_(self.feed_forward.linear_2.weight)
        nn.init.xavier_uniform_(self.feed_forward.linear_3.weight)

        if self.modulation:
            nn.init.zeros_(self.norm1.linear.weight)
            nn.init.zeros_(self.norm1.linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Args:
            hidden_states: Input hidden states tensor
            attention_mask: Attention mask tensor
            image_rotary_emb: Rotary embeddings for image tokens
            temb: Optional timestep embedding tensor

        Returns:
            torch.Tensor: Output hidden states after transformer block processing
        """
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


class Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
     Transformer 2D Model.

    A transformer-based diffusion model for image generation with:
    - Patch-based image processing
    - Rotary position embeddings
    - Multi-head attention
    - Conditional generation support

    Args:
        patch_size: Size of image patches
        in_channels: Number of input channels
        out_channels: Number of output channels (defaults to in_channels)
        hidden_size: Size of hidden layers
        num_layers: Number of transformer layers
        num_refiner_layers: Number of refiner layers
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key-value heads
        multiple_of: Multiple of which the hidden dimension should be
        ffn_dim_multiplier: Multiplier for feed-forward network dimension
        norm_eps: Epsilon value for normalization layers
        axes_dim_rope: Dimensions for rotary position embeddings
        axes_lens: Lengths for rotary position embeddings
        text_feat_dim: Dimension of text features
        timestep_scale: Scale factor for timestep embeddings
        use_fused_rms_norm: Whether to use fused RMS normalization
        use_fused_swiglu: Whether to use fused SwiGLU activation
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    _skip_layerwise_casting_patterns = ["x_embedder", "norm"]

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

        self.gradient_checkpointing = True

        self.initialize_weights()

        # TeaCache settings
        self.enable_teacache = False
        self.teacache_rel_l1_thresh = 0.05

        coefficients = [-5.48259225, 11.48772289, -4.47407401, 2.47730926, -0.03316487]
        self.rescale_func = np.poly1d(coefficients)

    def initialize_weights(self) -> None:
        """
        Initialize the weights of the model.

        Uses Xavier uniform initialization for linear layers.
        """
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0.0)

        nn.init.xavier_uniform_(self.ref_image_patch_embedder.weight)
        nn.init.constant_(self.ref_image_patch_embedder.bias, 0.0)

        nn.init.zeros_(self.norm_out.linear_1.weight)
        nn.init.zeros_(self.norm_out.linear_1.bias)
        nn.init.zeros_(self.norm_out.linear_2.weight)
        nn.init.zeros_(self.norm_out.linear_2.bias)

        nn.init.normal_(self.image_index_embedding, std=0.02)

    def img_patch_embed_and_refine(
        self,
        hidden_states,
        ref_image_hidden_states,
        padded_img_mask,
        padded_ref_img_mask,
        noise_rotary_emb,
        ref_img_rotary_emb,
        l_effective_ref_img_len,
        l_effective_img_len,
        temb,
    ):
        batch_size = len(hidden_states)
        max_combined_img_len = max(
            [img_len + sum(ref_img_len) for img_len, ref_img_len in zip(l_effective_img_len, l_effective_ref_img_len)]
        )

        hidden_states = self.x_embedder(hidden_states)
        ref_image_hidden_states = self.ref_image_patch_embedder(ref_image_hidden_states)

        for i in range(batch_size):
            shift = 0
            for j, ref_img_len in enumerate(l_effective_ref_img_len[i]):
                ref_image_hidden_states[i, shift : shift + ref_img_len, :] = (
                    ref_image_hidden_states[i, shift : shift + ref_img_len, :] + self.image_index_embedding[j]
                )
                shift += ref_img_len

        for layer in self.noise_refiner:
            hidden_states = layer(hidden_states, padded_img_mask, noise_rotary_emb, temb)

        flat_l_effective_ref_img_len = list(itertools.chain(*l_effective_ref_img_len))
        num_ref_images = len(flat_l_effective_ref_img_len)
        max_ref_img_len = max(flat_l_effective_ref_img_len)

        batch_ref_img_mask = ref_image_hidden_states.new_zeros(num_ref_images, max_ref_img_len, dtype=torch.bool)
        batch_ref_image_hidden_states = ref_image_hidden_states.new_zeros(
            num_ref_images, max_ref_img_len, self.config.hidden_size
        )
        # ref_img_rotary_emb is a tuple (cos, sin)
        ref_img_rotary_emb_cos, ref_img_rotary_emb_sin = ref_img_rotary_emb
        batch_ref_img_rotary_emb_cos = hidden_states.new_zeros(
            num_ref_images, max_ref_img_len, ref_img_rotary_emb_cos.shape[-1], dtype=ref_img_rotary_emb_cos.dtype
        )
        batch_ref_img_rotary_emb_sin = hidden_states.new_zeros(
            num_ref_images, max_ref_img_len, ref_img_rotary_emb_sin.shape[-1], dtype=ref_img_rotary_emb_sin.dtype
        )
        batch_temb = temb.new_zeros(num_ref_images, *temb.shape[1:], dtype=temb.dtype)
        # sequence of ref imgs to batch
        idx = 0
        for i in range(batch_size):
            shift = 0
            for ref_img_len in l_effective_ref_img_len[i]:
                batch_ref_img_mask[idx, :ref_img_len] = True
                batch_ref_image_hidden_states[idx, :ref_img_len] = ref_image_hidden_states[
                    i, shift : shift + ref_img_len
                ]
                batch_ref_img_rotary_emb_cos[idx, :ref_img_len] = ref_img_rotary_emb_cos[i, shift : shift + ref_img_len]
                batch_ref_img_rotary_emb_sin[idx, :ref_img_len] = ref_img_rotary_emb_sin[i, shift : shift + ref_img_len]
                batch_temb[idx] = temb[i]
                shift += ref_img_len
                idx += 1
        batch_ref_img_rotary_emb = (batch_ref_img_rotary_emb_cos, batch_ref_img_rotary_emb_sin)

        # refine ref imgs separately
        for layer in self.ref_image_refiner:
            batch_ref_image_hidden_states = layer(
                batch_ref_image_hidden_states, batch_ref_img_mask, batch_ref_img_rotary_emb, batch_temb
            )

        # batch of ref imgs to sequence
        idx = 0
        for i in range(batch_size):
            shift = 0
            for ref_img_len in l_effective_ref_img_len[i]:
                ref_image_hidden_states[i, shift : shift + ref_img_len] = batch_ref_image_hidden_states[
                    idx, :ref_img_len
                ]
                shift += ref_img_len
                idx += 1

        combined_img_hidden_states = hidden_states.new_zeros(batch_size, max_combined_img_len, self.config.hidden_size)
        for i, (ref_img_len, img_len) in enumerate(zip(l_effective_ref_img_len, l_effective_img_len)):
            combined_img_hidden_states[i, : sum(ref_img_len)] = ref_image_hidden_states[i, : sum(ref_img_len)]
            combined_img_hidden_states[i, sum(ref_img_len) : sum(ref_img_len) + img_len] = hidden_states[i, :img_len]

        return combined_img_hidden_states

    def flat_and_pad_to_seq(self, hidden_states, ref_image_hidden_states):
        batch_size = len(hidden_states)
        p = self.config.patch_size
        device = hidden_states[0].device

        img_sizes = [(img.size(1), img.size(2)) for img in hidden_states]
        l_effective_img_len = [(H // p) * (W // p) for (H, W) in img_sizes]

        if ref_image_hidden_states is not None:
            ref_img_sizes = [
                [(img.size(1), img.size(2)) for img in imgs] if imgs is not None else None
                for imgs in ref_image_hidden_states
            ]
            l_effective_ref_img_len = [
                [(ref_img_size[0] // p) * (ref_img_size[1] // p) for ref_img_size in _ref_img_sizes]
                if _ref_img_sizes is not None
                else [0]
                for _ref_img_sizes in ref_img_sizes
            ]
        else:
            ref_img_sizes = [None for _ in range(batch_size)]
            l_effective_ref_img_len = [[0] for _ in range(batch_size)]

        max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
        max_img_len = max(l_effective_img_len)

        # ref image patch embeddings
        flat_ref_img_hidden_states = []
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                imgs = []
                for ref_img in ref_image_hidden_states[i]:
                    C, H, W = ref_img.size()
                    ref_img = rearrange(ref_img, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=p, p2=p)
                    imgs.append(ref_img)

                img = torch.cat(imgs, dim=0)
                flat_ref_img_hidden_states.append(img)
            else:
                flat_ref_img_hidden_states.append(None)

        # image patch embeddings
        flat_hidden_states = []
        for i in range(batch_size):
            img = hidden_states[i]
            C, H, W = img.size()

            img = rearrange(img, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=p, p2=p)
            flat_hidden_states.append(img)

        padded_ref_img_hidden_states = torch.zeros(
            batch_size,
            max_ref_img_len,
            flat_hidden_states[0].shape[-1],
            device=device,
            dtype=flat_hidden_states[0].dtype,
        )
        padded_ref_img_mask = torch.zeros(batch_size, max_ref_img_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                padded_ref_img_hidden_states[i, : sum(l_effective_ref_img_len[i])] = flat_ref_img_hidden_states[i]
                padded_ref_img_mask[i, : sum(l_effective_ref_img_len[i])] = True

        padded_hidden_states = torch.zeros(
            batch_size, max_img_len, flat_hidden_states[0].shape[-1], device=device, dtype=flat_hidden_states[0].dtype
        )
        padded_img_mask = torch.zeros(batch_size, max_img_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            padded_hidden_states[i, : l_effective_img_len[i]] = flat_hidden_states[i]
            padded_img_mask[i, : l_effective_img_len[i]] = True

        return (
            padded_hidden_states,
            padded_ref_img_hidden_states,
            padded_img_mask,
            padded_ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        )

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.Tensor,
        text_hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        text_attention_mask: torch.Tensor,
        ref_image_hidden_states: list[list[torch.Tensor]] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | Transformer2DModelOutput:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        # 1. Condition, positional & patch embedding
        batch_size = len(hidden_states)
        is_hidden_states_tensor = isinstance(hidden_states, torch.Tensor)

        if is_hidden_states_tensor:
            assert hidden_states.ndim == 4
            hidden_states = [_hidden_states for _hidden_states in hidden_states]

        device = hidden_states[0].device

        temb, text_hidden_states = self.time_caption_embed(
            timestep, text_hidden_states, hidden_states[0].dtype
        )

        (
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

        (
            context_rotary_emb,
            ref_img_rotary_emb,
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

        # 2. Context refinement
        for layer in self.context_refiner:
            text_hidden_states = layer(text_hidden_states, text_attention_mask, context_rotary_emb)

        combined_img_hidden_states = self.img_patch_embed_and_refine(
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            noise_rotary_emb,
            ref_img_rotary_emb,
            l_effective_ref_img_len,
            l_effective_img_len,
            temb,
        )

        # 3. Joint Transformer blocks
        max_seq_len = max(seq_lengths)

        attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        joint_hidden_states = hidden_states.new_zeros(batch_size, max_seq_len, self.config.hidden_size)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            attention_mask[i, :seq_len] = True
            joint_hidden_states[i, :encoder_seq_len] = text_hidden_states[i, :encoder_seq_len]
            joint_hidden_states[i, encoder_seq_len:seq_len] = combined_img_hidden_states[i, : seq_len - encoder_seq_len]

        hidden_states = joint_hidden_states

        if self.enable_teacache:
            teacache_hidden_states = hidden_states.clone()
            teacache_temb = temb.clone()
            modulated_inp, _, _, _ = self.layers[0].norm1(teacache_hidden_states, teacache_temb)
            if self.teacache_params.is_first_or_last_step:
                should_calc = True
                self.teacache_params.accumulated_rel_l1_distance = 0
            else:
                self.teacache_params.accumulated_rel_l1_distance += self.rescale_func(
                    (
                        (modulated_inp - self.teacache_params.previous_modulated_inp).abs().mean()
                        / self.teacache_params.previous_modulated_inp.abs().mean()
                    )
                    .cpu()
                    .item()
                )
                if self.teacache_params.accumulated_rel_l1_distance < self.teacache_rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.teacache_params.accumulated_rel_l1_distance = 0
            self.teacache_params.previous_modulated_inp = modulated_inp

        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.teacache_params.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()
                for layer_idx, layer in enumerate(self.layers):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        hidden_states = self._gradient_checkpointing_func(
                            layer, hidden_states, attention_mask, rotary_emb, temb
                        )
                    else:
                        hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)
                self.teacache_params.previous_residual = hidden_states - ori_hidden_states
        else:
            for layer_idx, layer in enumerate(self.layers):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        layer, hidden_states, attention_mask, rotary_emb, temb
                    )
                else:
                    hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)

        # 4. Output norm & projection
        # print(f"hidden_states.shape: {hidden_states.shape}")
        # print(f"temb.shape: {temb.shape}")
        hidden_states = self.norm_out(hidden_states, temb)

        p = self.config.patch_size
        output = []
        for i, (img_size, img_len, seq_len) in enumerate(zip(img_sizes, l_effective_img_len, seq_lengths)):
            height, width = img_size
            output.append(
                rearrange(
                    hidden_states[i][seq_len - img_len : seq_len],
                    "(h w) (p1 p2 c) -> c (h p1) (w p2)",
                    h=height // p,
                    w=width // p,
                    p1=p,
                    p2=p,
                )
            )
        if is_hidden_states_tensor:
            output = torch.stack(output, dim=0)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output
        return Transformer2DModelOutput(sample=output)
