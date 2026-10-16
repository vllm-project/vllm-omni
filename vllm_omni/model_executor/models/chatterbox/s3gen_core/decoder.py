# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
"""S3Gen's flow estimator: Matcha-TTS's causal U-Net with one level and diffusers' transformer blocks."""

import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import TimestepEmbedding

from vllm_omni.model_executor.models.chatterbox.s3gen_core.matcha_decoder import (
    Block1D,
    ResnetBlock1D,
    SinusoidalPosEmb,
)
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import CausalConv1d
from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import mask_to_bias


class Transpose(torch.nn.Module):
    """Swap two dimensions, so a LayerNorm can sit inside a channels-first ``Sequential``.

    Args:
        dim0: The first dimension.
        dim1: The second dimension.
    """

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class CausalBlock1D(Block1D):
    """Matcha's convolution block with a causal convolution and LayerNorm in place of GroupNorm.

    Args:
        dim: Input channels.
        dim_out: Output channels.
    """

    def __init__(self, dim: int, dim_out: int):
        super().__init__(dim, dim_out)
        self.block = torch.nn.Sequential(
            CausalConv1d(dim, dim_out, 3),
            Transpose(1, 2),
            nn.LayerNorm(dim_out),
            Transpose(1, 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        output = self.block(x * mask)
        return output * mask


class CausalResnetBlock1D(ResnetBlock1D):
    """Matcha's residual block built from two causal blocks.

    Args:
        dim: Input channels.
        dim_out: Output channels.
        time_emb_dim: Width of the time embedding added between the blocks.
        groups: Unused by the causal blocks; the parent's signature.
    """

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__(dim, dim_out, time_emb_dim, groups)
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)


class ConditionalDecoder(nn.Module):
    """The velocity estimator: a causal U-Net with one channel level, as the S3Gen checkpoint has it.

    The input and the target have the same number of frames.

    Args:
        in_channels: Channels of the concatenated input: noisy mel, encoder
            output, speaker embedding and conditioning mel.
        out_channels: Mel bins of the predicted velocity.
        channels: The U-Net's channel levels; the checkpoint has one.
        dropout: Dropout of the transformer blocks.
        attention_head_dim: Width of each attention head.
        n_blocks: Transformer blocks per level.
        num_mid_blocks: Middle levels.
        num_heads: Attention heads.
        act_fn: Feed-forward activation of the transformer blocks.
        meanflow: Whether the checkpoint is a meanflow model, which also
            embeds the step's end time.

    Raises:
        ValueError: If ``channels`` has more than one level: the checkpoint's
            U-Net does not down- or upsample, and the layers that would are
            not ported.
    """

    def __init__(
        self,
        in_channels: int = 320,
        out_channels: int = 80,
        channels: tuple[int, ...] = (256,),
        dropout: float = 0.0,
        attention_head_dim: int = 64,
        n_blocks: int = 4,
        num_mid_blocks: int = 12,
        num_heads: int = 8,
        act_fn: str = "gelu",
        meanflow: bool = False,
    ):
        super().__init__()
        if len(channels) != 1:
            raise ValueError(f"ConditionalDecoder supports one channel level, got {len(channels)}")
        channels = tuple(channels)
        self.meanflow = meanflow
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        for channel in channels:
            input_channel = output_channel
            output_channel = channel
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = CausalConv1d(output_channel, output_channel, 3)
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for _ in range(num_mid_blocks):
            input_channel = channels[-1]
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = CausalConv1d(output_channel, output_channel, 3)
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))
        self.final_block = CausalBlock1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)
        # Mixes the start- and end-time embeddings of a meanflow step.
        self.time_embed_mixer = nn.Linear(time_embed_dim * 2, time_embed_dim, bias=False) if self.meanflow else None

    @property
    def dtype(self) -> torch.dtype:
        """The estimator's parameter dtype, which the solvers cast their inputs to."""
        return self.final_proj.weight.dtype

    def forward(self, x, mask, mu, t, spks=None, cond=None, r=None):
        """Predict the velocity of the noisy mel ``x`` at time ``t``.

        Args:
            x (torch.Tensor): Noisy mel sample, shape (B, 80, T).
            mask (torch.Tensor): Frame mask, shape (B, 1, T).
            mu (torch.Tensor): Encoder output the sample is conditioned on, shape (B, 80, T).
            t (torch.Tensor): Diffusion time, shape (B,).
            spks (torch.Tensor, optional): Speaker embedding, shape (B, spk_emb_dim). Defaults to None.
            cond (torch.Tensor, optional): Conditioning mel, shape (B, 80, T). Defaults to None.
            r (torch.Tensor, optional): End time for meanflow mode, shape (1,). Defaults to None.

        Returns:
            output (torch.Tensor): The predicted velocity, shape (B, 80, T), masked by `mask`.
        """
        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        if self.meanflow:
            r = self.time_embeddings(r).to(t.dtype)
            r = self.time_mlp(r)
            concat_embed = torch.cat([t, r], dim=1)
            t = self.time_embed_mixer(concat_embed)

        x = torch.cat([x, mu], dim=1)

        if spks is not None:
            spks = spks.unsqueeze(-1).tile(1, 1, x.shape[-1])
            x = torch.cat([x, spks], dim=1)
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = x.transpose(1, 2).contiguous()
            # A row with no valid frame attends to every frame rather than to none.
            attn_mask = mask_down.bool() | ~mask_down.bool().any(dim=-1, keepdim=True)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = x.transpose(1, 2).contiguous()
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = x.transpose(1, 2).contiguous()
            # A row with no valid frame attends to every frame rather than to none.
            attn_mask = mask_mid.bool() | ~mask_mid.bool().any(dim=-1, keepdim=True)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = x.transpose(1, 2).contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = torch.cat([x[:, :, : skip.shape[-1]], skip], dim=1)
            x = resnet(x, mask_up, t)
            x = x.transpose(1, 2).contiguous()
            # A row with no valid frame attends to every frame rather than to none.
            attn_mask = mask_up.bool() | ~mask_up.bool().any(dim=-1, keepdim=True)
            attn_mask = mask_to_bias(attn_mask, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = x.transpose(1, 2).contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask
