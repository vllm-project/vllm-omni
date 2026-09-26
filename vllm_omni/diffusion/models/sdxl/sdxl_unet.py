# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)

logger = init_logger(__name__)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    half_dim = embedding_dim // 2
    exponent = (
        -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    )
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SDXLTimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class SDXLAddTimestepEmbedding(nn.Module):
    def __init__(self, addition_time_embed_dim: int, text_embed_dim: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(text_embed_dim + addition_time_embed_dim * 6, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, text_embeds: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        time_embeds = get_timestep_embedding(time_ids.flatten(), 256)
        time_embeds = time_embeds.reshape(text_embeds.shape[0], -1)
        add_embeds = torch.cat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(dtype=self.linear_1.weight.dtype)
        add_embeds = self.linear_1(add_embeds)
        add_embeds = self.act(add_embeds)
        add_embeds = self.linear_2(add_embeds)
        return add_embeds


class SDXLResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        groups: int = 32,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb_proj = nn.Linear(time_embed_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-5)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.nonlinearity = nn.SiLU()

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)
        hidden_states = hidden_states + temb[:, :, None, None]

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual


class SDXLGEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = MergedColumnParallelLinear(dim_in, [dim_out, dim_out], bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class SDXLFeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.geglu = SDXLGEGLU(dim, inner_dim)
        self.out_proj = RowParallelLinear(inner_dim, dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.geglu(hidden_states)
        hidden_states, _ = self.out_proj(hidden_states)
        return hidden_states


class SDXLSelfAttention(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, prefix: str = ""):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=attention_head_dim,
            total_num_heads=num_attention_heads,
            bias=False,
        )
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(inner_dim, dim, bias=True),
            ]
        )
        self.attention = Attention(
            num_heads=num_attention_heads,
            head_size=attention_head_dim,
            causal=False,
            softmax_scale=1.0 / math.sqrt(attention_head_dim),
            role="self",
            prefix=prefix,
        )


class SDXLCrossAttention(nn.Module):
    def __init__(
        self, dim: int, cross_attention_dim: int, num_attention_heads: int, attention_head_dim: int, prefix: str = ""
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.to_q = ColumnParallelLinear(dim, inner_dim, bias=False)
        self.to_k = ColumnParallelLinear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = ColumnParallelLinear(cross_attention_dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(inner_dim, dim, bias=True),
            ]
        )
        self.attention = Attention(
            num_heads=num_attention_heads,
            head_size=attention_head_dim,
            causal=False,
            softmax_scale=1.0 / math.sqrt(attention_head_dim),
            role="cross",
            prefix=prefix,
            skip_sequence_parallel=True,
        )


class SDXLBasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        prefix: str = "",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=True)

        self.attn1 = SDXLSelfAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            prefix=f"{prefix}.attn1",
        )
        self.attn2 = SDXLCrossAttention(
            dim=dim,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            prefix=f"{prefix}.attn2",
        )

        # Feed-forward
        self.ff = SDXLFeedForward(dim=dim, inner_dim=dim * 4)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: None = None,
    ) -> torch.Tensor:
        local_num_heads = self.attn1.to_qkv.num_heads
        head_dim = self.attention_head_dim

        # Self-attention
        norm_hidden = self.norm1(hidden_states)
        qkv, _ = self.attn1.to_qkv(norm_hidden)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.unflatten(-1, (local_num_heads, head_dim))
        k = k.unflatten(-1, (local_num_heads, head_dim))
        v = v.unflatten(-1, (local_num_heads, head_dim))
        attn_out = self.attn1.attention(q, k, v)
        attn_out = attn_out.flatten(-2)
        attn_out, _ = self.attn1.to_out[0](attn_out)
        hidden_states = hidden_states + attn_out

        # Cross-attention
        norm_hidden = self.norm2(hidden_states)
        q, _ = self.attn2.to_q(norm_hidden)
        k, _ = self.attn2.to_k(encoder_hidden_states)
        v, _ = self.attn2.to_v(encoder_hidden_states)
        q = q.unflatten(-1, (local_num_heads, head_dim))
        k = k.unflatten(-1, (local_num_heads, head_dim))
        v = v.unflatten(-1, (local_num_heads, head_dim))
        attn_out = self.attn2.attention(q, k, v)
        attn_out = attn_out.flatten(-2)
        attn_out, _ = self.attn2.to_out[0](attn_out)
        hidden_states = hidden_states + attn_out

        # Feed-forward
        norm_hidden = self.norm3(hidden_states)
        ff_out = self.ff(norm_hidden)
        hidden_states = hidden_states + ff_out

        return hidden_states


class SDXLTransformer2DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        num_layers: int,
        cross_attention_dim: int,
        prefix: str = "",
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                SDXLBasicTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    prefix=f"{prefix}.transformer_blocks.{i}",
                )
                for i in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

        return hidden_states + residual


class SDXLDownsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states)


class SDXLUpsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_states)


class SDXLDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_layers: int = 2,
        add_downsample: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            res_in = in_channels if i == 0 else out_channels
            self.resnets.append(SDXLResnetBlock2D(res_in, out_channels, time_embed_dim))

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([SDXLDownsample2D(out_channels)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class SDXLCrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_layers: int = 2,
        num_attention_heads: int = 10,
        cross_attention_dim: int = 2048,
        transformer_layers_per_block: int = 1,
        add_downsample: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        attention_head_dim = out_channels // num_attention_heads

        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for i in range(num_layers):
            res_in = in_channels if i == 0 else out_channels
            self.resnets.append(SDXLResnetBlock2D(res_in, out_channels, time_embed_dim))
            self.attentions.append(
                SDXLTransformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    prefix=f"{prefix}.attentions.{i}",
                )
            )

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([SDXLDownsample2D(out_channels)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class SDXLUNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        num_attention_heads: int = 20,
        cross_attention_dim: int = 2048,
        transformer_layers_per_block: int = 10,
        prefix: str = "",
    ):
        super().__init__()
        attention_head_dim = in_channels // num_attention_heads

        self.resnets = nn.ModuleList(
            [
                SDXLResnetBlock2D(in_channels, in_channels, time_embed_dim),
                SDXLResnetBlock2D(in_channels, in_channels, time_embed_dim),
            ]
        )
        self.attentions = nn.ModuleList(
            [
                SDXLTransformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    prefix=f"{prefix}.attentions.0",
                )
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states


class SDXLUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        time_embed_dim: int,
        num_layers: int = 3,
        add_upsample: bool = True,
        resnet_in_channels_list: list[int] | None = None,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            if resnet_in_channels_list is not None:
                res_in = resnet_in_channels_list[i]
            else:
                skip_channels = prev_output_channel if i == 0 else out_channels
                res_in = (in_channels if i == 0 else out_channels) + skip_channels
            self.resnets.append(SDXLResnetBlock2D(res_in, out_channels, time_embed_dim))

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([SDXLUpsample2D(out_channels)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        res_hidden_states_tuple: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        for resnet in self.resnets:
            res_hidden = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states


class SDXLCrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        time_embed_dim: int,
        num_layers: int = 3,
        num_attention_heads: int = 10,
        cross_attention_dim: int = 2048,
        transformer_layers_per_block: int = 1,
        add_upsample: bool = True,
        prefix: str = "",
        resnet_in_channels_list: list[int] | None = None,
    ):
        super().__init__()
        attention_head_dim = out_channels // num_attention_heads

        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for i in range(num_layers):
            if resnet_in_channels_list is not None:
                res_in = resnet_in_channels_list[i]
            else:
                skip_channels = prev_output_channel if i == 0 else out_channels
                res_in = (in_channels if i == 0 else out_channels) + skip_channels
            self.resnets.append(SDXLResnetBlock2D(res_in, out_channels, time_embed_dim))
            self.attentions.append(
                SDXLTransformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                    prefix=f"{prefix}.attentions.{i}",
                )
            )

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([SDXLUpsample2D(out_channels)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        res_hidden_states_tuple: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)

        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states


def _build_sdxl_sp_plan() -> dict:
    """Build _sp_plan for all attention modules in the SDXL UNet.

    Shards hidden_states at the first transformer block input and gathers
    at proj_out for each SDXLTransformer2DModel instance.
    """
    plan: dict = {}
    attn_paths = [
        # down_blocks[1]: 2 attention modules
        "down_blocks.1.attentions.0",
        "down_blocks.1.attentions.1",
        # down_blocks[2]: 2 attention modules
        "down_blocks.2.attentions.0",
        "down_blocks.2.attentions.1",
        # mid_block: 1 attention module
        "mid_block.attentions.0",
        # up_blocks[0]: 3 attention modules
        "up_blocks.0.attentions.0",
        "up_blocks.0.attentions.1",
        "up_blocks.0.attentions.2",
        # up_blocks[1]: 3 attention modules
        "up_blocks.1.attentions.0",
        "up_blocks.1.attentions.1",
        "up_blocks.1.attentions.2",
    ]
    for path in attn_paths:
        plan[f"{path}.transformer_blocks.0"] = {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True),
        }
        plan[f"{path}.proj_out"] = SequenceParallelOutput(gather_dim=1, expected_dims=3)
    return plan


class SDXLUNet2DConditionModel(nn.Module):
    _repeated_blocks = ["SDXLBasicTransformerBlock", "SDXLResnetBlock2D"]
    _layerwise_offload_blocks_attrs = ["down_blocks", "up_blocks"]
    _sp_plan = _build_sdxl_sp_plan()

    @staticmethod
    def _is_shardable_block(name: str, module) -> bool:
        """Match transformer blocks and ResNet blocks for HSDP sharding."""
        if not name.split(".")[-1].isdigit():
            return False
        return "transformer_blocks" in name or "resnets" in name

    _hsdp_shard_conditions = [_is_shardable_block]

    def __init__(self, *, od_config: OmniDiffusionConfig):
        super().__init__()
        self.od_config = od_config

        # SDXL Base configuration
        model_channels = 320
        time_embed_dim = model_channels * 4  # 1280
        block_out_channels = [320, 640, 1280]
        layers_per_block = 2
        transformer_layers_per_block = [1, 2, 10]
        cross_attention_dim = 2048
        addition_time_embed_dim = 256
        num_head_channels = 64

        self.in_channels = 4
        self.out_channels = 4

        # Input convolution
        self.conv_in = nn.Conv2d(4, model_channels, kernel_size=3, padding=1)

        # Time embedding
        self.time_proj_dim = model_channels
        self.time_embedding = SDXLTimestepEmbedding(model_channels, time_embed_dim)

        # Addition embedding (for pooled text + crop/size conditioning)
        self.add_embedding = SDXLAddTimestepEmbedding(
            addition_time_embed_dim=addition_time_embed_dim,
            text_embed_dim=1280,
            time_embed_dim=time_embed_dim,
        )

        # Down blocks
        self.down_blocks = nn.ModuleList()

        # Level 0: DownBlock2D (no attention)
        self.down_blocks.append(
            SDXLDownBlock2D(
                in_channels=model_channels,
                out_channels=block_out_channels[0],
                time_embed_dim=time_embed_dim,
                num_layers=layers_per_block,
                add_downsample=True,
            )
        )
        # Level 1: CrossAttnDownBlock2D
        self.down_blocks.append(
            SDXLCrossAttnDownBlock2D(
                in_channels=block_out_channels[0],
                out_channels=block_out_channels[1],
                time_embed_dim=time_embed_dim,
                num_layers=layers_per_block,
                num_attention_heads=block_out_channels[1] // num_head_channels,
                cross_attention_dim=cross_attention_dim,
                transformer_layers_per_block=transformer_layers_per_block[1],
                add_downsample=True,
                prefix="down_blocks.1",
            )
        )
        # Level 2: CrossAttnDownBlock2D (no downsampler)
        self.down_blocks.append(
            SDXLCrossAttnDownBlock2D(
                in_channels=block_out_channels[1],
                out_channels=block_out_channels[2],
                time_embed_dim=time_embed_dim,
                num_layers=layers_per_block,
                num_attention_heads=block_out_channels[2] // num_head_channels,
                cross_attention_dim=cross_attention_dim,
                transformer_layers_per_block=transformer_layers_per_block[2],
                add_downsample=False,
                prefix="down_blocks.2",
            )
        )

        # Mid block
        self.mid_block = SDXLUNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[2],
            time_embed_dim=time_embed_dim,
            num_attention_heads=block_out_channels[2] // num_head_channels,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block[2],
            prefix="mid_block",
        )

        # Up blocks (reversed, with skip connections)
        # Skip connection channels consumed from down_block_res_samples (right to left):
        # down_block_res_samples channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280]
        # up_blocks[0] consumes last 3: skips are [1280, 1280, 640] (consumed from right)
        # up_blocks[1] consumes next 3: skips are [640, 640, 320] (consumed from right)
        # up_blocks[2] consumes next 3: skips are [320, 320, 320] (consumed from right)
        self.up_blocks = nn.ModuleList()

        # Level 0: CrossAttnUpBlock2D (mirrors down_blocks[2])
        # Input from mid: 1280, skips (consumed right-to-left): 1280, 1280, 640
        self.up_blocks.append(
            SDXLCrossAttnUpBlock2D(
                in_channels=block_out_channels[2],
                prev_output_channel=block_out_channels[2],
                out_channels=block_out_channels[2],
                time_embed_dim=time_embed_dim,
                num_layers=layers_per_block + 1,
                num_attention_heads=block_out_channels[2] // num_head_channels,
                cross_attention_dim=cross_attention_dim,
                transformer_layers_per_block=transformer_layers_per_block[2],
                add_upsample=True,
                prefix="up_blocks.0",
                resnet_in_channels_list=[1280 + 1280, 1280 + 1280, 1280 + 640],
            )
        )
        # Level 1: CrossAttnUpBlock2D (mirrors down_blocks[1])
        # Input from up_blocks[0]: 1280, skips (consumed right-to-left): 640, 640, 320
        self.up_blocks.append(
            SDXLCrossAttnUpBlock2D(
                in_channels=block_out_channels[2],
                prev_output_channel=block_out_channels[1],
                out_channels=block_out_channels[1],
                time_embed_dim=time_embed_dim,
                num_layers=layers_per_block + 1,
                num_attention_heads=block_out_channels[1] // num_head_channels,
                cross_attention_dim=cross_attention_dim,
                transformer_layers_per_block=transformer_layers_per_block[1],
                add_upsample=True,
                prefix="up_blocks.1",
                resnet_in_channels_list=[1280 + 640, 640 + 640, 640 + 320],
            )
        )
        # Level 2: UpBlock2D (mirrors down_blocks[0], no attention, no upsample)
        # Input from up_blocks[1]: 640, skips (consumed right-to-left): 320, 320, 320
        self.up_blocks.append(
            SDXLUpBlock2D(
                in_channels=block_out_channels[1],
                prev_output_channel=block_out_channels[0],
                out_channels=block_out_channels[0],
                time_embed_dim=time_embed_dim,
                num_layers=layers_per_block + 1,
                add_upsample=False,
                resnet_in_channels_list=[640 + 320, 320 + 320, 320 + 320],
            )
        )

        # Output
        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[0], eps=1e-5)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], 4, kernel_size=3, padding=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: dict,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        # Time embedding
        dtype = hidden_states.dtype
        t_emb = get_timestep_embedding(timestep, self.time_proj_dim).to(dtype=dtype)
        emb = self.time_embedding(t_emb)

        # Addition embedding (pooled text + size/crop)
        text_embeds = added_cond_kwargs["text_embeds"].to(dtype=dtype)
        time_ids = added_cond_kwargs["time_ids"]
        add_emb = self.add_embedding(text_embeds, time_ids)
        emb = emb + add_emb

        # Input conv
        hidden_states = self.conv_in(hidden_states)

        # Down blocks
        down_block_res_samples = [hidden_states]
        for down_block in self.down_blocks:
            if isinstance(down_block, SDXLCrossAttnDownBlock2D):
                hidden_states, res_samples = down_block(hidden_states, emb, encoder_hidden_states)
            else:
                hidden_states, res_samples = down_block(hidden_states, emb)
            down_block_res_samples.extend(res_samples)

        # Mid block
        hidden_states = self.mid_block(hidden_states, emb, encoder_hidden_states)

        # Up blocks
        for up_block in self.up_blocks:
            # Determine how many skip connections this block needs
            n_resnets = len(up_block.resnets)
            res_samples = tuple(down_block_res_samples[-n_resnets:])
            down_block_res_samples = down_block_res_samples[:-n_resnets]

            if isinstance(up_block, SDXLCrossAttnUpBlock2D):
                hidden_states = up_block(hidden_states, emb, encoder_hidden_states, res_samples)
            else:
                hidden_states = up_block(hidden_states, emb, res_samples)

        # Output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return (hidden_states,)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # Self-attention QKV fusion
            (".attn1.to_qkv", ".attn1.to_q", "q"),
            (".attn1.to_qkv", ".attn1.to_k", "k"),
            (".attn1.to_qkv", ".attn1.to_v", "v"),
        ]
        self.stacked_params_mapping = stacked_params_mapping

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Handle GEGLU: diffusers stores as ff.net.0.proj.weight/bias
            # MergedColumnParallelLinear needs the two halves loaded separately
            if ".ff.net.0.proj." in name:
                name = name.replace(".ff.net.0.proj.", ".ff.geglu.proj.")
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    half = loaded_weight.shape[0] // 2
                    weight_loader(param, loaded_weight[:half], 0)
                    weight_loader(param, loaded_weight[half:], 1)
                    loaded_params.add(name)
                continue
            elif ".ff.net.2." in name:
                name = name.replace(".ff.net.2.", ".ff.out_proj.")

            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                handled = True
                break

            if not handled:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params
