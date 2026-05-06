from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers.models.activations import get_activation
from diffusers.models.unets.unet_2d_blocks import (
    UNetMidBlock2D,
    get_down_block,
    get_up_block,
)
from torch.utils.checkpoint import checkpoint

from .utils import DiagonalGaussianDistribution, ceildiv


class ImageVAE(nn.Module):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`tuple[int, ...]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        decoder_block_out_channels (`tuple[int, ...]`, *optional*, defaults to `None`):
            Tuple of block output channels for the decoder. If not provided, it will be set to `block_out_channels`.
        layers_per_block (`int`, *optional*, defaults to 1): Number of layers per block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        norm_num_groups (`int`, *optional*, defaults to 32): Number of groups for normalization.
        mid_block_add_attention (`bool`, *optional*, default to `True`):
            If enabled, the mid_block of the Encoder and Decoder will have attention blocks. If set to false, the
            mid_block will only have resnet blocks
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",),
        up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        decoder_block_out_channels: tuple[int, ...] | None = None,
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        mid_block_add_attention: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.encoder = EncoderConfig(
            in_channels=in_channels,
            out_channels=2 * latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            mid_block_add_attention=mid_block_add_attention,
        ).make()
        self.encoder.gradient_checkpointing = gradient_checkpointing

        if decoder_block_out_channels is None:
            decoder_block_out_channels = block_out_channels

        self.decoder = DecoderConfig(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        ).make()
        self.decoder.gradient_checkpointing = gradient_checkpointing

    def encode(
        self,
        x: torch.Tensor,
        skip_first_n_down_blocks: int = 0,
    ) -> DiagonalGaussianDistribution:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.

        Returns:
            `DiagonalGaussianDistribution`: The latent posterior of the encoded images.
        """

        moments = self.encoder(x, skip_first_n_down_blocks=skip_first_n_down_blocks)
        return DiagonalGaussianDistribution(moments)

    def decode(
        self,
        z: torch.Tensor,
        skip_last_n_up_blocks: int = 0,
    ) -> torch.Tensor:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.

        Returns:
            `torch.FloatTensor`: The decoded images.

        """
        decoded: torch.Tensor = self.decoder(
            z, skip_last_n_up_blocks=skip_last_n_up_blocks
        )
        return decoded

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = True,
        skip_n_blocks: int = 0,
        generator: torch.Generator | None = None,
    ) -> tuple[DiagonalGaussianDistribution, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `True`):
                Whether to sample from the posterior.
            skip_n_blocks (`int`, defaults to `0`):
                Number of encoder down blocks, and decoder up blocks to skip during forward pass.

        Returns:
            `DiagonalGaussianDistribution`: The latent posterior of the encoded images.
            `torch.Tensor`: The decoded images.
        """
        x = sample
        posterior = self.encode(x, skip_first_n_down_blocks=skip_n_blocks)
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        return posterior, self.decode(z, skip_last_n_up_blocks=skip_n_blocks)


class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        cfg (EncoderConfig): The configuration for the encoder.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        cfg: EncoderConfig,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.cfg = cfg

        self.conv_in = nn.Conv2d(
            cfg.in_channels,
            cfg.block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = cfg.block_out_channels[0]
        for i, down_block_type in enumerate(cfg.down_block_types):
            input_channel = output_channel
            output_channel = cfg.block_out_channels[i]
            is_final_block = i == len(cfg.block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=cfg.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=cfg.act_fn,
                resnet_groups=cfg.norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,  # type: ignore[arg-type] # diffusers allows None despite annotation
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=cfg.block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=cfg.act_fn,
            output_scale_factor=1,
            attention_head_dim=cfg.block_out_channels[-1],
            resnet_groups=cfg.norm_num_groups,
            temb_channels=None,  # type: ignore[arg-type] # diffusers allows None despite annotation
            add_attention=cfg.mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=cfg.block_out_channels[-1],
            num_groups=cfg.norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = get_activation(cfg.act_fn)
        self.conv_out = nn.Conv2d(
            cfg.block_out_channels[-1], cfg.out_channels, 3, padding=1
        )

        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self, x: torch.Tensor, skip_first_n_down_blocks: int = 0
    ) -> torch.Tensor:
        r"""The forward method of the `Encoder` class.

        Args:
            sample (`torch.Tensor`): Input image to be encoded.
            skip_first_n_down_blocks (`int`, *optional*, defaults to `0`):
                Number of down blocks to skip during forward pass.

        Returns:
            `torch.Tensor`: The encoded latent embeddings. The shape is (batch_size, latent_channels, height, width)
             where height and width are determined by the input sample and the downsample factor of the encoder.
        """

        x = self.conv_in(x)

        if self.training and self.gradient_checkpointing:
            # down
            for down_block in self.down_blocks[skip_first_n_down_blocks:]:
                x = checkpoint(down_block, x, use_reentrant=False)
            # middle
            x = checkpoint(self.mid_block, x, use_reentrant=False)

        else:
            # down
            for down_block in self.down_blocks[skip_first_n_down_blocks:]:
                x = down_block(x)
            # middle
            x = self.mid_block(x)

        # post-process
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x

    def get_downsample_factors(
        self, skip_first_n_down_blocks: int = 0
    ) -> tuple[int, int]:
        # all but one down_blocks are used for downsampling, and subtract skip_first_n_down_blocks:
        used_downsample_blocks = len(self.down_blocks) - 1 - skip_first_n_down_blocks
        assert used_downsample_blocks > 0, "final_downsample_blocks must be positive"
        # each downsample has a factor of 2, so the total downsample factor is 2^used_downsample_blocks
        downsample_factor: int = 2**used_downsample_blocks
        return (
            downsample_factor,
            downsample_factor,
        )  # downsample factor for height and width are the same

    def input_to_latent_size(
        self,
        input_size: tuple[int | None, int | None],
        skip_first_n_down_blocks: int = 0,
    ) -> tuple[int | None, int | None]:
        """
        Compute the encoded latent size for the given input size (Hi, Wi).
        Any None elements are ignored and returned as None.

        Parameters:
            input_size (tuple[int | None, int | None]): The input size (Hi, Wi).

        Returns:
            tuple[int | None, int | None]: The encoded latent size (Hl, Wl).
        """
        downsample_factors = self.get_downsample_factors(skip_first_n_down_blocks)
        return (
            ceildiv(input_size[0], downsample_factors[0])
            if input_size[0] is not None
            else None,
            ceildiv(input_size[1], downsample_factors[1])
            if input_size[1] is not None
            else None,
        )


@dataclass
class EncoderConfig:
    r"""
    Configuration for :class:`Encoder`.

    Parameters:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 32):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        mid_block_add_attention (`bool`, *optional*, defaults to `True`):
            Whether to add attention to the middle block.
    """

    in_channels: int = 3
    out_channels: int = 32
    down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",)
    block_out_channels: tuple[int, ...] = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    mid_block_add_attention: bool = True

    def __post_init__(self) -> None:
        assert self.in_channels > 0, "in_channels must be positive"
        assert self.out_channels > 0, "out_channels must be positive"
        assert len(self.down_block_types) == len(self.block_out_channels), (
            "down_block_types and block_out_channels must have the same length"
        )
        assert self.layers_per_block > 0, "layers_per_block must be positive"
        assert self.norm_num_groups > 0, "norm_num_groups must be positive"
        # ensure all block_out_channels are positive
        assert all(c > 0 for c in self.block_out_channels), (
            "block_out_channels must be positive"
        )
        # ensure all block_out_channels are divisible by norm_num_groups
        assert all(c % self.norm_num_groups == 0 for c in self.block_out_channels), (
            "block_out_channels must be divisible by norm_num_groups"
        )
        # ensure that activation_fn is valid by calling get_activation on it
        get_activation(self.act_fn)

    def make(self) -> Encoder:
        return Encoder(self)


class Decoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        cfg (DecoderConfig): The configuration for the decoder.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        cfg: DecoderConfig,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.cfg = cfg

        self.conv_in = nn.Conv2d(
            cfg.in_channels,
            cfg.block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=cfg.block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=cfg.act_fn,
            output_scale_factor=1,
            attention_head_dim=cfg.block_out_channels[-1],
            resnet_groups=cfg.norm_num_groups,
            temb_channels=cfg.conditioning_channels,  # type: ignore[arg-type] # diffusers allows None despite annotation
            add_attention=cfg.mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(cfg.block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(cfg.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(cfg.block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=cfg.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,  # type: ignore[arg-type] # diffusers allows None despite annotation
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=cfg.act_fn,
                resnet_groups=cfg.norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=cfg.conditioning_channels,  # type: ignore[arg-type] # diffusers allows None despite annotation
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=cfg.block_out_channels[0],
            num_groups=cfg.norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = get_activation(cfg.act_fn)
        self.conv_out = nn.Conv2d(
            cfg.block_out_channels[0], cfg.out_channels, 3, padding=1
        )
        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        skip_last_n_up_blocks: int = 0,
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class.

        Args:
            sample (`torch.Tensor`): Input latent embeddings to be decoded, shape (batch_size, latent_channels, height, width)
            conditioning (`torch.Tensor`, *optional*): Conditioning tensor, shape (batch_size, conditioning_channels)
            skip_last_n_up_blocks (`int`, *optional*, defaults to `0`):
                Number of up blocks to skip during forward pass.

        Returns:
            `torch.Tensor`: The decoded images. The shape is (batch_size, out_channels, height, width)
             where height and width are determined by the input sample and the upsample factor of the decoder.
        """
        x = self.conv_in(x)

        # when skipping n final blocks, we skip from the penultimate block back, and *always* use the final block
        if skip_last_n_up_blocks > 0:
            rest_up_blocks = self.up_blocks[:-1]
            final_up_blocks = (
                rest_up_blocks[:-skip_last_n_up_blocks] + self.up_blocks[-1:]
            )
        else:
            final_up_blocks = self.up_blocks

        if self.training and self.gradient_checkpointing:
            # middle
            x = checkpoint(self.mid_block, x, conditioning, use_reentrant=False)

            # up
            for up_block in final_up_blocks:
                x = checkpoint(up_block, x, conditioning, use_reentrant=False)
        else:
            # middle
            x = self.mid_block(x, conditioning)

            # up
            for up_block in final_up_blocks:
                x = up_block(x, conditioning)

        # post-process
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x

    def get_upsample_factors(self, skip_last_n_up_blocks: int = 0) -> tuple[int, int]:
        # all but one up_blocks are used for upsampling, and subtract skip_last_n_up_blocks:
        used_upsample_blocks = len(self.up_blocks) - 1 - skip_last_n_up_blocks
        assert used_upsample_blocks > 0, "used_upsample_blocks must be positive"
        # each upsample has a factor of 2, so the total upsample factor is 2^used_upsample_blocks
        upsample_factor: int = 2**used_upsample_blocks
        return (
            upsample_factor,
            upsample_factor,
        )  # upsample factor for height and width are the same


@dataclass
class DecoderConfig:
    r"""
    Configuration for :class:`Decoder`.

    Parameters:
        in_channels (`int`, *optional*, defaults to 16):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        conditioning_channels (`int`, *optional*, defaults to `None`):
            The number of conditioning channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        mid_block_add_attention (`bool`, *optional*, defaults to `False`):
            Whether to add attention to the middle block.
    """

    in_channels: int = 16
    out_channels: int = 3
    conditioning_channels: int | None = None
    up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",)
    block_out_channels: tuple[int, ...] = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    mid_block_add_attention: bool = False

    def __post_init__(self) -> None:
        assert len(self.up_block_types) == len(self.block_out_channels), (
            "up_block_types and block_out_channels must have the same length"
        )
        assert self.in_channels > 0, "in_channels must be positive"
        assert self.out_channels > 0, "out_channels must be positive"
        assert self.layers_per_block > 0, "layers_per_block must be positive"
        assert self.norm_num_groups > 0, "norm_num_groups must be positive"
        # ensure all block_out_channels are positive
        assert all(c > 0 for c in self.block_out_channels), (
            "block_out_channels must be positive"
        )
        # ensure all block_out_channels are divisible by norm_num_groups
        assert all(c % self.norm_num_groups == 0 for c in self.block_out_channels), (
            "block_out_channels must be divisible by norm_num_groups"
        )
        # ensure that activation_fn is valid by calling get_activation on it
        get_activation(self.act_fn)

    def make(self) -> Decoder:
        return Decoder(self)
