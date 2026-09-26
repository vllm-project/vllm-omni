# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from SeedVR, Copyright 2025 ByteDance Ltd., Apache-2.0.
"""SeedVR2's causal s8/c16/t4 VAE, with released checkpoint names.

This implementation processes a whole clip. Group normalization and attention
operate independently on each frame; only the causal convolutions mix time.
"""

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from diffusers.models.attention_processor import Attention
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.models.seedvr2.vae_spatial import SpatialContext, tiled_convolution


@dataclass
class TemporalContext:
    """Past convolution inputs owned by one encode or decode call."""

    first: bool = True
    cache_temporal: bool = False
    tile_spatial: bool = False
    spatial: SpatialContext | None = None
    history: dict[nn.Module, torch.Tensor] = field(default_factory=dict)


def _frame_norm_silu(norm: nn.GroupNorm, x: torch.Tensor, context: TemporalContext | None = None) -> torch.Tensor:
    if context is not None and context.spatial is not None:
        return F.silu(context.spatial.normalize(norm, x))
    if x.is_cuda and x.dtype == torch.float16:
        from vllm_omni.diffusion.models.seedvr2.frame_norm import frame_norm_silu

        return frame_norm_silu(norm, x)
    return F.silu(_frame_norm(norm, x))


def _frame_norm(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    frames = x.shape[2]
    x = norm(rearrange(x, "b c t h w -> (b t) c h w"))
    return rearrange(x, "(b t) c h w -> b c t h w", t=frames)


class CausalConv3d(nn.Conv3d):
    """Replicate the first frame to pad only the past temporal context."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, (0, padding[1], padding[2]))
        self.temporal_padding = 2 * padding[0]

    def forward(self, x: torch.Tensor, context: TemporalContext | None = None) -> torch.Tensor:
        if self.temporal_padding:
            if context is None or not context.cache_temporal or context.first:
                head = x[:, :, :1].expand(-1, -1, self.temporal_padding, -1, -1)
            else:
                head = context.history[self]
            x = torch.cat((head, x), dim=2)
            if context is not None and context.cache_temporal:
                # Copy the halo so it does not retain a whole chunk's storage.
                context.history[self] = x[:, :, -self.temporal_padding :].clone()
                if not context.first:
                    x = x[:, :, self.stride[0] - 1 :]
        if context is not None:
            if context.spatial is not None:
                return context.spatial.convolution(self, x)
            if context.tile_spatial and x.shape[-2] > 256:
                return tiled_convolution(self, x)
        return super().forward(x)


class ResnetBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-6)
        self.conv1 = CausalConv3d(in_channels, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6)
        self.conv2 = CausalConv3d(out_channels, out_channels)
        self.conv_shortcut = (
            CausalConv3d(in_channels, out_channels, (1, 1, 1), padding=(0, 0, 0))
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor, context: TemporalContext | None = None) -> torch.Tensor:
        h = self.conv1(_frame_norm_silu(self.norm1, x, context), context)
        h = self.conv2(_frame_norm_silu(self.norm2, h, context), context)
        return (self.conv_shortcut(x, context) if self.conv_shortcut is not None else x) + h


class Downsample3d(nn.Module):
    def __init__(self, channels: int, temporal: bool) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            channels,
            channels,
            (3 if temporal else 1, 3, 3),
            (2 if temporal else 1, 2, 2),
            (1 if temporal else 0, 0, 0),
        )

    def forward(self, x: torch.Tensor, context: TemporalContext | None = None) -> torch.Tensor:
        bottom = 0 if context is not None and context.spatial is not None else 1
        return self.conv(F.pad(x, (0, 1, 0, bottom)), context)


class Upsample3d(nn.Module):
    def __init__(self, channels: int, temporal: bool) -> None:
        super().__init__()
        self.temporal_ratio = 2 if temporal else 1
        self.upscale_conv = nn.Conv3d(channels, channels * 4 * self.temporal_ratio, kernel_size=1)
        self.conv = CausalConv3d(channels, channels)

    def forward(self, x: torch.Tensor, context: TemporalContext | None = None) -> torch.Tensor:
        x = rearrange(
            self.upscale_conv(x),
            "b (x y z c) t h w -> b c (t z) (h x) (w y)",
            x=2,
            y=2,
            z=self.temporal_ratio,
        )
        if self.temporal_ratio == 2 and (context is None or context.first):
            x = torch.cat((x[:, :, :1], x[:, :, 2:]), dim=2)
        return self.conv(x, context)


class EncoderBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, index: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock3d(in_channels, out_channels, groups), ResnetBlock3d(out_channels, out_channels, groups)]
        )
        self.downsamplers = nn.ModuleList([Downsample3d(out_channels, temporal=index >= 1)] if index < 3 else [])

    def forward(self, x: torch.Tensor, context: TemporalContext | None = None) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, context)
        for downsampler in self.downsamplers:
            x = downsampler(x, context)
        return x


class DecoderBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, index: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock3d(in_channels, out_channels, groups)]
            + [ResnetBlock3d(out_channels, out_channels, groups) for _ in range(2)]
        )
        self.upsamplers = nn.ModuleList([Upsample3d(out_channels, temporal=index < 2)] if index < 3 else [])

    def forward(self, x: torch.Tensor, context: TemporalContext | None = None) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, context)
        for upsampler in self.upsamplers:
            x = upsampler(x, context)
        return x


class MidBlock3d(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock3d(channels, channels, groups) for _ in range(2)])
        self.attentions = nn.ModuleList(
            [
                Attention(
                    channels,
                    heads=1,
                    dim_head=channels,
                    eps=1e-6,
                    norm_num_groups=groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            ]
        )

    def forward(self, x: torch.Tensor, context: TemporalContext | None = None) -> torch.Tensor:
        frames = x.shape[2]
        x = self.resnets[0](x, context)
        spatial = context.spatial if context is not None else None
        if spatial is not None:
            x = spatial.gather(x)
        x = self.attentions[0](rearrange(x, "b c t h w -> (b t) c h w"))
        x = rearrange(x, "(b t) c h w -> b c t h w", t=frames)
        if spatial is not None:
            x = spatial.split(x)
        return self.resnets[1](x, context)


class Encoder3d(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int], groups: int, latent_channels: int) -> None:
        super().__init__()
        self.conv_in = CausalConv3d(3, channels[0])
        self.down_blocks = nn.ModuleList(
            [EncoderBlock3d(channels[max(0, i - 1)], width, groups, i) for i, width in enumerate(channels)]
        )
        self.mid_block = MidBlock3d(channels[-1], groups)
        self.conv_norm_out = nn.GroupNorm(groups, channels[-1], eps=1e-6)
        self.conv_out = CausalConv3d(channels[-1], 2 * latent_channels)

    def forward(self, x: torch.Tensor, context: TemporalContext | None = None) -> torch.Tensor:
        x = self.conv_in(x, context)
        for block in self.down_blocks:
            x = block(x, context)
        x = self.mid_block(x, context)
        return self.conv_out(_frame_norm_silu(self.conv_norm_out, x, context), context)


class Decoder3d(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int], groups: int, latent_channels: int) -> None:
        super().__init__()
        self.conv_in = CausalConv3d(latent_channels, channels[-1])
        self.mid_block = MidBlock3d(channels[-1], groups)
        widths = tuple(reversed(channels))
        self.up_blocks = nn.ModuleList(
            [DecoderBlock3d(widths[max(0, i - 1)], width, groups, i) for i, width in enumerate(widths)]
        )
        self.conv_norm_out = nn.GroupNorm(groups, channels[0], eps=1e-6)
        self.conv_out = CausalConv3d(channels[0], 3)

    def forward(self, x: torch.Tensor, context: TemporalContext | None = None) -> torch.Tensor:
        x = self.mid_block(self.conv_in(x, context), context)
        for block in self.up_blocks:
            x = block(x, context)
        return self.conv_out(_frame_norm_silu(self.conv_norm_out, x, context), context)


class SeedVR2VAE(nn.Module, DistributedVaeMixin):
    """Whole-clip VAE; sampling uses the caller's request-local generator."""

    spatial_scale_factor = 8
    temporal_scale_factor = 4

    def __init__(
        self,
        channels: tuple[int, int, int, int] = (128, 256, 512, 512),
        groups: int = 32,
        latent_channels: int = 16,
    ) -> None:
        super().__init__()
        self.use_tiling = False
        self._spatial_group: dist.ProcessGroup | None = None
        self.encoder = Encoder3d(channels, groups, latent_channels)
        self.decoder = Decoder3d(channels, groups, latent_channels)

    def set_parallel_size(self, parallel_size: int, mode: str = "tile") -> None:
        if mode == "spatial_shard_width":
            raise ValueError("SeedVR2 VAE supports height sharding; select spatial_shard_height or tile")
        if parallel_size > 1:
            group = get_sp_group().device_group
            if dist.get_world_size(group) != parallel_size:
                raise ValueError("SeedVR2 VAE patch parallel size must match the SP group")
            self._spatial_group = group
        else:
            self._spatial_group = None

    def _context(self, latent_height: int) -> TemporalContext:
        spatial = None
        if self.use_tiling and self._spatial_group is not None:
            group = self._spatial_group
            size = dist.get_world_size(group)
            if latent_height >= size:
                units = tuple((i + 1) * latent_height // size - i * latent_height // size for i in range(size))
                spatial = SpatialContext(group, dist.get_rank(group), units)
        return TemporalContext(tile_spatial=self.use_tiling, spatial=spatial)

    def encode(self, x: torch.Tensor, *, chunk_size: int | None = None) -> DiagonalGaussianDistribution:
        if chunk_size is None and self.use_tiling:
            chunk_size = 8
        if chunk_size is not None and (chunk_size < 4 or chunk_size % 4):
            raise ValueError("SeedVR2 encode chunk_size must be a positive multiple of four")
        context = self._context(x.shape[-2] // 8)
        if context.spatial is not None:
            x = context.spatial.split(x)
        if chunk_size is None or x.shape[2] <= chunk_size + 1:
            encoded = self.encoder(x, context)
        else:
            context.cache_temporal = True
            chunks = [self.encoder(x[:, :, : chunk_size + 1], context)]
            context.first = False
            for start in range(chunk_size + 1, (x.shape[2] - 1) // 4 * 4 + 1, chunk_size):
                chunks.append(self.encoder(x[:, :, start : start + chunk_size], context))
            encoded = torch.cat(chunks, dim=2)
        if context.spatial is not None:
            encoded = context.spatial.gather(encoded)
        return DiagonalGaussianDistribution(encoded)

    def decode(self, z: torch.Tensor, *, chunk_size: int | None = None) -> torch.Tensor:
        if chunk_size is None and self.use_tiling:
            chunk_size = 2
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("SeedVR2 decode chunk_size must be positive")
        context = self._context(z.shape[-2])
        if context.spatial is not None:
            z = context.spatial.split(z)
        if chunk_size is None or z.shape[2] <= chunk_size:
            decoded = self.decoder(z, context)
        else:
            context.cache_temporal = True
            chunks = []
            for start in range(0, z.shape[2], chunk_size):
                chunks.append(self.decoder(z[:, :, start : start + chunk_size], context))
                context.first = False
            decoded = torch.cat(chunks, dim=2)
        if context.spatial is not None:
            decoded = context.spatial.gather(decoded)
        return decoded
