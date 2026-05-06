from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.activations import get_activation
from einops import rearrange

from .utils import DiagonalGaussianDistribution, cast_tuple, ceildiv, is_odd, pad_at_dim


def get_conv_builder(conv_type: str) -> Callable[..., nn.Module]:
    """Get a convolution builder function for the given convolution type."""

    if conv_type == "std":
        return StdConv3d
    # TODO(chrisburgess): add a causal convolution type
    else:
        raise ValueError(f"invalid convolution type `{conv_type}`")


class StdConv3d(nn.Module):
    """Standard 3D convolution block."""

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: int | tuple[int, int, int],
        strides: int | tuple[int, int, int] = 1,
        pad_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        padding: int | tuple[int, int, int] | Literal["same", "valid"] | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        if padding is None:
            # If not specified, we compute the padding that retains the same feature sizes
            #  (accounting for strides). Kernel sizes must be odd in this case.
            assert (
                is_odd(time_kernel_size)
                and is_odd(height_kernel_size)
                and is_odd(width_kernel_size)
            )

            time_pad = time_kernel_size // 2
            height_pad = height_kernel_size // 2
            width_pad = width_kernel_size // 2
            padding = (time_pad, height_pad, width_pad)

        self.conv = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=strides,
            padding_mode=pad_mode,
            padding=padding,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        conv_fn: Callable[..., nn.Module],
        main_kernel_size: int | tuple[int, int, int],
        activation: nn.Module,
        use_conv_shortcut: bool = False,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.activate = activation
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = conv_fn(
            in_channels, self.filters, kernel_size=main_kernel_size, bias=False
        )
        self.norm2 = nn.GroupNorm(num_groups, self.filters)
        self.conv2 = conv_fn(
            self.filters, self.filters, kernel_size=main_kernel_size, bias=False
        )
        if in_channels != filters:
            if self.use_conv_shortcut:
                self.conv3 = conv_fn(
                    in_channels, self.filters, kernel_size=main_kernel_size, bias=False
                )
            else:
                self.conv3 = conv_fn(
                    in_channels, self.filters, kernel_size=(1, 1, 1), bias=False
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.in_channels != self.filters:
            residual = self.conv3(residual)
        return x + residual


class Encoder(nn.Module):
    """Video encoder module."""

    def __init__(
        self,
        in_out_channels: int = 4,
        latent_embed_dim: int = 512,
        base_channels: int = 128,
        main_kernel_size: int | tuple[int, int, int] = (3, 3, 3),
        num_res_blocks: int = 4,
        conv_type: str = "std",
        channel_multipliers: tuple[int, ...] = (1, 2, 2, 4),
        temporal_downsample: tuple[bool, ...] = (False, True, True),
        spatial_downsample: tuple[bool, ...] = (False, False, False),
        num_groups: int = 32,
        activation_fn: str = "silu",
    ):
        super().__init__()
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim

        self.activate = get_activation(activation_fn)
        self.conv_fn = get_conv_builder(conv_type)
        self.block_args = dict(
            main_kernel_size=main_kernel_size,
            conv_fn=self.conv_fn,
            activation=self.activate,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        # first layer conv
        self.conv_in = self.conv_fn(
            in_out_channels,
            base_channels,
            kernel_size=main_kernel_size,
            bias=False,
        )

        # ResBlocks and conv downsample
        self.block_res_blocks = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        base_channels = self.base_channels
        prev_base_channels = base_channels  # record for in_channels
        for i in range(self.num_blocks):
            base_channels = self.base_channels * self.channel_multipliers[i]
            block_items = nn.ModuleList([])
            for _ in range(self.num_res_blocks):
                block_items.append(
                    ResBlock(prev_base_channels, base_channels, **self.block_args)  # type: ignore[arg-type]
                )  # type: ignore[arg-type]
                prev_base_channels = base_channels  # update in_channels
            self.block_res_blocks.append(block_items)

            if i < self.num_blocks - 1:
                if self.temporal_downsample[i] or self.spatial_downsample[i]:
                    t_stride = 2 if self.temporal_downsample[i] else 1
                    s_stride = 2 if self.spatial_downsample[i] else 1
                    strides = (t_stride, s_stride, s_stride)
                    self.conv_blocks.append(
                        self.conv_fn(
                            prev_base_channels,
                            base_channels,
                            kernel_size=main_kernel_size,
                            strides=strides,
                        )
                    )
                    prev_base_channels = base_channels  # update in_channels
                else:
                    # no temporal downsample, add an identity instead of a downsampling layer
                    self.conv_blocks.append(nn.Identity())
                    prev_base_channels = base_channels  # update in_channels

        # last layer res block
        self.res_blocks = nn.ModuleList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(
                ResBlock(prev_base_channels, base_channels, **self.block_args)  # type: ignore[arg-type]
            )  # type: ignore[arg-type]
            prev_base_channels = base_channels  # update in_channels

        self.norm1 = nn.GroupNorm(self.num_groups, prev_base_channels)
        self.conv2 = self.conv_fn(
            prev_base_channels,
            self.embedding_dim,
            strides=(1, 1, 1),
            kernel_size=(1, 1, 1),
            padding="same",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)

        for i in range(self.num_blocks):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)  # type: ignore[index]
            if i < self.num_blocks - 1:
                x = self.conv_blocks[i](x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    """Video decoder module."""

    def __init__(
        self,
        in_out_channels: int = 4,
        latent_embed_dim: int = 512,
        base_channels: int = 128,
        main_kernel_size: int | tuple[int, int, int] = (3, 3, 3),
        num_res_blocks: int = 4,
        conv_type: str = "std",
        channel_multipliers: tuple[int, ...] = (1, 2, 2, 4),
        temporal_downsample: tuple[bool, ...] = (False, True, True),
        spatial_downsample: tuple[bool, ...] = (False, False, False),
        temporal_upsample_mode: Literal[
            "depth-to-time", "interpolate"
        ] = "depth-to-time",
        upsample_interp_mode: str = "nearest-exact",
        num_groups: int = 32,
        activation_fn: str = "silu",
    ):
        super().__init__()
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim
        self.temporal_upsample_mode = temporal_upsample_mode
        self.upsample_interp_mode = upsample_interp_mode

        self.activate = get_activation(activation_fn)
        self.conv_fn = get_conv_builder(conv_type)
        self.block_args = dict(
            main_kernel_size=main_kernel_size,
            conv_fn=self.conv_fn,
            activation=self.activate,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        base_channels = self.base_channels * self.channel_multipliers[-1]
        prev_base_channels = base_channels

        # last conv
        self.conv1 = self.conv_fn(
            self.embedding_dim, base_channels, kernel_size=main_kernel_size, bias=True
        )

        # last layer res block
        self.res_blocks = nn.ModuleList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(
                ResBlock(base_channels, base_channels, **self.block_args)  # type: ignore[arg-type]
            )

        # ResBlocks and conv upsample
        self.block_res_blocks = nn.ModuleList([])
        self.num_blocks = len(self.channel_multipliers)
        self.conv_blocks = nn.ModuleList([])
        # reverse to keep track of the in_channels, but append also in a reverse direction
        for i in reversed(range(self.num_blocks)):
            base_channels = self.base_channels * self.channel_multipliers[i]
            # resblock handling
            block_items = nn.ModuleList([])
            for _ in range(self.num_res_blocks):
                block_items.append(
                    ResBlock(prev_base_channels, base_channels, **self.block_args)  # type: ignore[arg-type]
                )
                prev_base_channels = base_channels  # update in_channels
            self.block_res_blocks.insert(0, block_items)  # SCH: append in front

            # conv blocks with upsampling
            if i > 0:
                if self.temporal_downsample[i - 1]:
                    # t_stride is depth-to-time stride
                    t_stride, _ = self.temporal_upsample(i)

                    self.conv_blocks.insert(
                        0,
                        self.conv_fn(
                            prev_base_channels,
                            prev_base_channels * t_stride,
                            kernel_size=main_kernel_size,
                        ),
                    )
                else:
                    self.conv_blocks.insert(0, nn.Identity())
        self.norm1 = nn.GroupNorm(self.num_groups, prev_base_channels)
        self.conv_out = self.conv_fn(base_channels, in_out_channels, 3)

    def temporal_upsample(self, block_i: int) -> tuple[int, int]:
        """
        Get the temporal upsampling stride and interpolation scale for the given block.

        Parameters:
            block_i (int): The index of the block.

        Returns:
            tuple[int, int]: The temporal upsampling stride and interpolation scale.
        """
        if self.temporal_upsample_mode == "depth-to-time":
            t_stride = 2 if self.temporal_downsample[block_i - 1] else 1
            t_interp = 1
        elif self.temporal_upsample_mode == "interpolate":
            t_stride = 1
            t_interp = 2 if self.temporal_downsample[block_i - 1] else 1
        else:
            raise ValueError(
                f"invalid temporal_upsample_mode {self.temporal_upsample_mode}"
            )
        return t_stride, t_interp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)
        for i in reversed(range(self.num_blocks)):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)  # type: ignore[index]
            if i > 0:
                t_stride, t_interp = self.temporal_upsample(i)
                x = self.conv_blocks[i - 1](x)
                if t_stride > 1:
                    x = rearrange(
                        x,
                        "B (C ts) T H W -> B C (T ts) H W",
                        ts=t_stride,
                    )
                spatial_interp = (2, 2) if self.spatial_downsample[i - 1] else (1, 1)
                temporal_interp = (t_interp,)
                interp_scales = temporal_interp + spatial_interp

                if interp_scales != (1, 1, 1):
                    x = F.interpolate(
                        x,
                        scale_factor=interp_scales,
                        mode=self.upsample_interp_mode,
                    )

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv_out(x)
        return x


class VideoVAE(nn.Module):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations back to videos.

    Parameters:
        cfg (VideoVAEConfig): The configuration for the VAE.
    """

    def __init__(self, cfg: VideoVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.time_downsample_factor: int = 2 ** sum(cfg.temporal_downsample)
        self.spatial_downsample_factor: int = 2 ** sum(cfg.spatial_downsample)
        self.downsample_factors: tuple[int, int, int] = (
            self.time_downsample_factor,
            self.spatial_downsample_factor,
            self.spatial_downsample_factor,
        )

        self.encoder = Encoder(
            in_out_channels=cfg.in_out_channels,
            latent_embed_dim=cfg.latent_embed_dim * 2,
            base_channels=cfg.base_channels,
            main_kernel_size=cfg.main_kernel_size,
            num_res_blocks=cfg.num_res_blocks,
            conv_type=cfg.conv_type,
            channel_multipliers=cfg.channel_multipliers,
            temporal_downsample=cfg.temporal_downsample,
            spatial_downsample=cfg.spatial_downsample,
            num_groups=cfg.num_groups,
            activation_fn=cfg.activation_fn,
        )

        self.decoder = Decoder(
            in_out_channels=cfg.in_out_channels,
            latent_embed_dim=cfg.latent_embed_dim,
            base_channels=cfg.base_channels,
            main_kernel_size=cfg.main_kernel_size,
            num_res_blocks=cfg.num_res_blocks,
            conv_type=cfg.conv_type,
            channel_multipliers=cfg.channel_multipliers,
            temporal_downsample=cfg.temporal_downsample,
            spatial_downsample=cfg.spatial_downsample,
            num_groups=cfg.num_groups,
            activation_fn=cfg.activation_fn,
            temporal_upsample_mode=cfg.temporal_upsample_mode,  # type: ignore[arg-type] # validated by TwoStageVAEConfig
            upsample_interp_mode=cfg.upsample_interp_mode,
        )

    def get_time_padding(self, num_frames: int) -> int:
        """
        Get the time padding for the given number of frames.

        Parameters:
            num_frames (int): The number of frames.

        Returns:
            int: The time padding.
        """
        return (
            0
            if (num_frames % self.time_downsample_factor == 0)
            else self.downsample_factors[0] - num_frames % self.downsample_factors[0]
        )

    def input_to_latent_size(
        self, input_size: tuple[int | None, int | None, int | None]
    ) -> tuple[int | None, int | None, int | None]:
        """
        Compute the encoded latent size for the given input video size (Tv, Hv, Wv).
        Any None elements are ignored and returned as None.

        Parameters:
            input_size (tuple[int | None, int | None, int | None]): The input size (Tv, Hv, Wv).

        Returns:
            tuple[int | None, int | None, int | None]: The encoded latent size (Tl, Hl, Wl).
        """

        def latent_size_i(i: int) -> int | None:
            if input_size[i] is not None:
                size_i: int = input_size[i]  # type: ignore[assignment]
                if i == 0:  # handle time special case with time padding
                    padded_size = size_i + self.get_time_padding(size_i)
                    return padded_size // self.downsample_factors[i]
                else:
                    return ceildiv(size_i, self.downsample_factors[i])
            return None

        return (latent_size_i(0), latent_size_i(1), latent_size_i(2))

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        time_padding = self.get_time_padding(x.shape[2])
        x = pad_at_dim(x, (time_padding, 0), dim=2)
        moments = self.encoder(x)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(
        self,
        z: torch.Tensor,
        num_frames: int | None = None,
        spatial_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:

        z = self.decoder(z)

        if spatial_size is not None:
            h, w = spatial_size
        else:
            h, w = z.shape[-2:]
        frame_offset = 0 if num_frames is None else self.get_time_padding(num_frames)
        z = z[:, :, frame_offset:, :h, :w]
        return z

    def forward(
        self, x: torch.Tensor, sample_posterior: bool = True
    ) -> tuple[DiagonalGaussianDistribution, torch.Tensor]:
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        recon_video = self.decode(z, num_frames=x.shape[-3], spatial_size=x.shape[-2:])  # type: ignore[arg-type]
        return posterior, recon_video


@dataclass
class VideoVAEConfig:
    r"""
    Configuration for :class:`VideoVAE`.

    Parameters:
        in_out_channels (int): Number of channels in the input and output.
        latent_embed_dim (int): Dimension of the latent embedding.
        base_channels (int): Base number of channels. This is multiplied by the channel_multipliers to get the number of
          filters for each block.
        main_kernel_size (tuple[int, int, int]): Default kernel size for the convolution blocks.
        num_res_blocks (int): Number of residual layers in each block.
        conv_type (str): Type of convolution to use in the encoder and decoder.
        channel_multipliers (tuple[int, ...]): Multipliers for the number of channels in the encoder and decoder.
        temporal_downsample (tuple[bool, ...]): For each block except one, whether to apply temporal downsampling.
        spatial_downsample (tuple[bool, ...]): For each block except one, whether to apply spatial downsampling.
        num_groups (int): Number of groups for each group normalization layer.
        activation_fn (str): Activation function to use throughout the VAE.
        temporal_upsample_mode (str): Mode for temporal upsampling. Either "depth-to-time" or "interpolate".
        upsample_interp_mode (str): If interpolation is used, this is the mode for upsampling interpolation.
    """

    in_out_channels: int = 3
    latent_embed_dim: int = 16
    base_channels: int = 128
    main_kernel_size: tuple[int, int, int] = (3, 3, 3)
    num_res_blocks: int = 4
    conv_type: str = "std"
    channel_multipliers: tuple[int, ...] = (1, 2, 2, 4)
    temporal_downsample: tuple[bool, ...] = (True, True, False)
    spatial_downsample: tuple[bool, ...] = (False, False, False)
    num_groups: int = 32
    activation_fn: str = "silu"
    temporal_upsample_mode: str = "depth-to-time"
    upsample_interp_mode: str = "nearest-exact"

    def __post_init__(self) -> None:
        assert self.in_out_channels > 0, "in_out_channels must be positive"
        assert self.latent_embed_dim > 0, "latent_embed_dim must be positive"
        assert self.base_channels > 0, "base_channels must be positive"
        assert self.main_kernel_size is not None, "main_kernel_size must be provided"
        assert all(k > 0 for k in self.main_kernel_size), (
            "main_kernel_size must be positive"
        )
        assert self.num_res_blocks > 0, "num_res_blocks must be positive"
        get_conv_builder(
            self.conv_type
        )  # conv_builder ensures that the conv_type is valid
        assert len(self.channel_multipliers) > 0, (
            "channel_multipliers must be non-empty"
        )
        assert all(c > 0 for c in self.channel_multipliers), (
            "channel_multipliers must be positive"
        )
        # temporal_downsample and spatial_downsample must have length of channel_multipliers minus 1
        assert len(self.temporal_downsample) == len(self.channel_multipliers) - 1, (
            "temporal_downsample must be the same length as channel_multipliers minus 1"
        )
        assert len(self.spatial_downsample) == len(self.channel_multipliers) - 1, (
            "spatial_downsample must be the same length as channel_multipliers minus 1"
        )
        assert self.base_channels % self.num_groups == 0, (
            "base_channels must be divisible by num_groups"
        )
        assert self.temporal_upsample_mode in {"depth-to-time", "interpolate"}
        # ensure that activation_fn is valid by calling get_activation on it
        get_activation(self.activation_fn)
        # ensure that `upsample_interp_mode` is valid by calling it with a dummy tensor
        F.interpolate(
            torch.ones(1, 2, 1, 1, 1),
            scale_factor=(1, 1, 1),
            mode=self.upsample_interp_mode,
        )

    def make(self) -> VideoVAE:
        return VideoVAE(self)
