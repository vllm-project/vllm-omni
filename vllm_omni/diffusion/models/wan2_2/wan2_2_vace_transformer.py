# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""VACE variant of WanTransformer3DModel for conditional video generation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput
from vllm_omni.diffusion.distributed.sp_sharding import sp_shard
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    Transformer2DModelOutput,
    WanTransformer3DModel,
    WanTransformerBlock,
)


class VaceWanTransformerBlock(WanTransformerBlock):
    """VACE variant of WanTransformerBlock with proj_in/proj_out for skip connections."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        cross_attn_norm: bool = False,
        block_id: int = 0,
    ):
        super().__init__(dim, ffn_dim, num_heads, eps, added_kv_proj_dim, cross_attn_norm)
        self.proj_in = nn.Linear(dim, dim) if block_id == 0 else None
        self.proj_out = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        hidden_states_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proj_in is not None:
            control_hidden_states = self.proj_in(control_hidden_states)
            control_hidden_states = control_hidden_states + hidden_states

        control_hidden_states = super().forward(
            control_hidden_states,
            encoder_hidden_states,
            temb,
            rotary_emb,
            hidden_states_mask,
        )

        conditioning_states = self.proj_out(control_hidden_states)
        return conditioning_states, control_hidden_states


class WanVACETransformer3DModel(WanTransformer3DModel):
    """VACE-extended WAN Transformer with conditioning blocks for video editing."""

    # TODO: `vace_blocks` are not layerwise-offloaded yet. The current offloader only
    # supports a single block group (`blocks`); extend it to support both
    # `vace_blocks` and `blocks`.

    # Shard hidden_states before VACE blocks (replaces parent's blocks.0)
    _sp_plan = {
        **{k: v for k, v in WanTransformer3DModel._sp_plan.items() if k != "blocks.0"},
        "_sp_shard_point": {
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
        },
    }

    def __init__(
        self,
        *,
        vace_layers: list[int] | None = None,
        vace_in_channels: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vace_blocks = None
        self.vace_patch_embedding = None
        self.vace_layers = None
        self.vace_layers_mapping = None

        # SP shard point: Identity module that _sp_plan hooks into to shard
        # hidden_states before VACE processing (instead of at blocks.0)
        self._sp_shard_point = nn.Identity()

        if vace_layers is not None:
            inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
            self.vace_layers = list(vace_layers)
            self.vace_layers_mapping = {layer_idx: vace_idx for vace_idx, layer_idx in enumerate(vace_layers)}

            vace_in_channels = vace_in_channels or self.config.in_channels
            self.vace_patch_embedding = nn.Conv3d(
                vace_in_channels,
                inner_dim,
                kernel_size=self.config.patch_size,
                stride=self.config.patch_size,
            )
            self.vace_blocks = nn.ModuleList(
                [
                    VaceWanTransformerBlock(
                        inner_dim,
                        self.config.ffn_dim,
                        self.config.num_attention_heads,
                        self.config.eps,
                        self.config.added_kv_proj_dim,
                        self.config.cross_attn_norm,
                        block_id=i,
                    )
                    for i in range(len(vace_layers))
                ]
            )

    def embed_vace_context(
        self,
        vace_context: torch.Tensor,
        seq_len: int,
        sp_size: int = 1,
    ) -> torch.Tensor:
        """Compute VACE patch embeddings, aligned and sharded for SP.

        Args:
            vace_context: Raw conditioning tensor [B, C, T, H, W].
            seq_len: Target full (padded) sequence length to align to.
            sp_size: Sequence parallel world size.
        """
        vace_embeds = self.vace_patch_embedding(vace_context)
        vace_embeds = vace_embeds.flatten(2).transpose(1, 2)

        # Align to target seq_len (may include SP padding)
        if vace_embeds.size(1) < seq_len:
            vace_embeds = F.pad(vace_embeds, (0, 0, 0, seq_len - vace_embeds.size(1)))

        if sp_size > 1:
            vace_embeds = sp_shard(vace_embeds, dim=1)
        return vace_embeds

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        vace_context: torch.Tensor | None = None,
        vace_context_scale: float | list[float] = 1.0,
    ) -> torch.Tensor | Transformer2DModelOutput:
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Compute RoPE embeddings (sharded by _sp_plan via split_output=True)
        rotary_emb = self.rope(hidden_states)

        # Patch embedding and flatten to sequence
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        timestep_proj = self.timestep_proj_prepare(timestep_proj, ts_seq_len)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Shard hidden_states via _sp_plan hook (before VACE, not at blocks.0)
        hidden_states = self._sp_shard_point(hidden_states)

        # SP state and attention mask for padding
        hidden_states_mask = None
        ctx = get_forward_context()
        parallel_config = ctx.omni_diffusion_config.parallel_config
        sp_size = parallel_config.sequence_parallel_size if parallel_config is not None else 1
        if ctx.sp_original_seq_len is not None and ctx.sp_padding_size > 0:
            padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size
            hidden_states_mask = torch.ones(
                batch_size,
                padded_seq_len,
                dtype=torch.bool,
                device=hidden_states.device,
            )
            hidden_states_mask[:, ctx.sp_original_seq_len :] = False

        # VACE: embed context and run conditioning blocks
        vace_hints = None
        if vace_context is not None and self.vace_blocks is not None:
            full_seq_len = hidden_states.shape[1] * sp_size
            control_hidden_states = self.embed_vace_context(vace_context.to(hidden_states.dtype), full_seq_len, sp_size)
            vace_hints = []
            for block in self.vace_blocks:
                conditioning_states, control_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    control_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    hidden_states_mask,
                )
                vace_hints.append(conditioning_states)

        # Normalize scale to per-layer list
        if vace_hints is not None and isinstance(vace_context_scale, (int, float)):
            vace_context_scale = [vace_context_scale] * len(vace_hints)

        # Transformer blocks with VACE hint application
        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, hidden_states_mask)
            if vace_hints is not None and self.vace_layers_mapping is not None and i in self.vace_layers_mapping:
                vace_idx = self.vace_layers_mapping[i]
                hidden_states = hidden_states + vace_hints[vace_idx] * vace_context_scale[vace_idx]

        # Output norm, projection & unpatchify
        shift, scale = self.output_scale_shift_prepare(temb)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        if shift.ndim == 2:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
