# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DiT stack for GR00T-N1.7.

Ported from ``Isaac-GR00T/gr00t/model/modules/dit.py``.  We do **not** inherit
from ``diffusers.{ModelMixin,ConfigMixin}`` — those pull in
``diffusers.models.modeling_utils`` which transitively touches an unrelated
torchao path that has been flaky on recent torchao builds.  At inference time
the DiT is constructed from explicit kwargs and weights are loaded via
``load_state_dict`` from the GR00T-N1.7 root checkpoint, so the diffusers
mixins buy us nothing here.

The inner attention is :class:`diffusers.models.attention.Attention`, matching
upstream exactly so submodule names (``to_q``, ``to_k``, ``to_v``,
``to_out.0``) match the upstream checkpoint without remapping.

Training-only knobs (``upcast_attention``, ``num_embeds_ada_norm``,
``activation_fn`` variants beyond ``geglu``/``gelu-approximate``) are kept as
constructor args so weight shapes stay identical; their inference behaviour
matches upstream when defaults match the checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class TimestepEncoder(nn.Module):
    """Diffusion timestep → embedding via diffusers' ``Timesteps`` +
    ``TimestepEmbedding``."""

    def __init__(self, embedding_dim: int, compute_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(proj)


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm: scale/shift conditioned on a temb."""

    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    """One DiT block: ada-norm → attention (self or cross) → FFN.

    Inner attention is :class:`diffusers.models.attention.Attention` —
    submodule names match upstream checkpoints (``to_q``, ``to_k``, ``to_v``,
    ``to_out.0``).
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: str | None = None,
        num_positional_embeddings: int | None = None,
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and num_positional_embeddings is None:
            raise ValueError(
                "`num_positional_embeddings` is required when "
                "`positional_embeddings` is set."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=(
                encoder_attention_mask
                if encoder_hidden_states is not None
                else attention_mask
            ),
        )
        if self.final_dropout is not None:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(nn.Module):
    """Standard cross-attention DiT: alternates cross-attn / self-attn blocks
    when ``interleave_self_attention=True``."""

    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: int | None = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype: torch.dtype = torch.float32,
        final_dropout: bool = True,
        positional_embeddings: str | None = "sinusoidal",
        interleave_self_attention: bool = False,
        cross_attention_dim: int | None = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_bias = attention_bias
        self.activation_fn = activation_fn
        self.num_embeds_ada_norm = num_embeds_ada_norm
        self.upcast_attention = upcast_attention
        self.norm_type = norm_type
        self.norm_elementwise_affine = norm_elementwise_affine
        self.norm_eps = norm_eps
        self.max_num_positional_embeddings = max_num_positional_embeddings
        self.compute_dtype = compute_dtype
        self.final_dropout = final_dropout
        self.positional_embeddings = positional_embeddings
        self.interleave_self_attention = interleave_self_attention
        self.cross_attention_dim = cross_attention_dim

        self.inner_dim = num_attention_heads * attention_head_dim

        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=compute_dtype
        )

        blocks = []
        for idx in range(num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = (
                cross_attention_dim if not use_self_attn else None
            )
            blocks.append(
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
    ):
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)


class AlternateVLDiT(DiT):
    """DiT that alternates cross-attention between image and non-image VL
    tokens every ``attend_text_every_n_blocks`` cross-attention layers."""

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
        image_mask: torch.Tensor | None = None,
        backbone_attention_mask: torch.Tensor | None = None,
    ):
        if image_mask is None:
            raise ValueError("AlternateVLDiT requires an image_mask.")
        if not self.interleave_self_attention:
            raise ValueError(
                "AlternateVLDiT requires interleave_self_attention=True."
            )

        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask

        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    curr_enc_attn_mask = non_image_attention_mask
                else:
                    curr_enc_attn_mask = image_attention_mask
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_enc_attn_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)


class SelfAttentionTransformer(nn.Module):
    """Stack of self-attention BasicTransformerBlocks; used by GR00T as the
    optional ``vl_self_attention`` head before cross-attention into the DiT."""

    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: int | None = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype: torch.dtype = torch.float32,
        final_dropout: bool = True,
        positional_embeddings: str | None = "sinusoidal",
        interleave_self_attention: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_bias = attention_bias
        self.activation_fn = activation_fn
        self.num_embeds_ada_norm = num_embeds_ada_norm
        self.upcast_attention = upcast_attention
        self.max_num_positional_embeddings = max_num_positional_embeddings
        self.compute_dtype = compute_dtype
        self.final_dropout = final_dropout
        self.positional_embeddings = positional_embeddings
        self.interleave_self_attention = interleave_self_attention

        self.inner_dim = num_attention_heads * attention_head_dim

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_all_hidden_states: bool = False,
    ):
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)
        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        return hidden_states
