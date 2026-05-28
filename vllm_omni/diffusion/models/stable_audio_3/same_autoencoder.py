# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SAME (Semantic-Acoustic Music Encoder) autoencoder for Stable Audio 3.

PORT_FROM: stable-audio-3
  - models/autoencoders.py (638 lines)
  - models/bottleneck.py (71 lines, ported in full)
  - models/pretransforms.py (84 lines, ported in full)
  - models/blocks.py parts: WNConv1d, ResidualUnit

The autoencoder is a 4-layer composition:

  AutoencoderPretransform        ← outer wrapper, exposes chunked decode
    └─ AudioAutoencoder           ← top-level facade
         ├─ encoder = SAMEEncoder ← transformer-based downsampling
         ├─ bottleneck = SoftNormBottleneck
         ├─ decoder = SAMEDecoder ← transformer-based upsampling
         └─ pretransform = PatchedPretransform

Chunked decode defaults (per upstream):
  chunk_size = 128 latents, overlap = 32 latents
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from torch.nn.utils import weight_norm
from torchaudio.transforms import Resample


# ---------------------------------------------------------------------------
# Small helpers (PORT_FROM: blocks.py + autoencoders.py top)
# ---------------------------------------------------------------------------


def WNConv1d(*args, **kwargs):
    """PORT_FROM: blocks.py:16-17"""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def get_activation(activation: str, channels: int | None = None) -> nn.Module:
    """PORT_FROM: blocks.py:6-14"""
    if activation == "elu":
        return nn.ELU()
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {activation}")


def checkpoint(function, *args, **kwargs):
    """PORT_FROM: autoencoders.py:14-16"""
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


def _zero_pad_modulo_sequence(x: torch.Tensor, size: int, dim: int = -2) -> torch.Tensor:
    """Pad x along `dim` to be a multiple of `size`. PORT_FROM: autoencoders.py:25-32"""
    input_len = x.shape[dim]
    pad_len = (size - input_len % size) % size
    if pad_len > 0:
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        x = torch.cat([x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=dim)
    return x


class Transpose(nn.Module):
    """PORT_FROM: autoencoders.py:19-23"""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return rearrange(x, "... a b -> ... b a")


class ResidualUnit(nn.Module):
    """PORT_FROM: blocks.py:19-39"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        depthwise: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.dilation = dilation
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            get_activation("elu"),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding,
                groups=1 if not depthwise else out_channels,
                bias=bias,
            ),
            get_activation("elu"),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


# ---------------------------------------------------------------------------
# Bottleneck (PORT_FROM: bottleneck.py — full file, verbatim)
# ---------------------------------------------------------------------------


class SoftNormBottleneck(nn.Module):
    """Learnable scaling/bias of the latent space.

    PORT_FROM: bottleneck.py SoftNormBottleneck (lines 4-71).
    """

    def __init__(
        self,
        dim: int = 32,
        noise_augment_dim: int = 0,
        noise_regularize: bool = False,
        auto_scale: bool = False,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.noise_augment_dim = noise_augment_dim
        self.noise_regularize = noise_regularize
        self.freeze = freeze

        self.scaling_factor = nn.Parameter(torch.ones(1, dim, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1))
        self.noise_scaling_factor = nn.Parameter(torch.ones(1, noise_augment_dim, 1))

        if freeze:
            self.scaling_factor.requires_grad = False
            self.bias.requires_grad = False
            self.noise_scaling_factor.requires_grad = False
        if auto_scale:
            self.register_parameter("running_std", nn.Parameter(torch.ones(1), requires_grad=False))

    def encode(self, x: torch.Tensor, return_info: bool = False, **kwargs):
        info = {}
        x = x * self.scaling_factor + self.bias

        if self.training and hasattr(self, "running_std") and not self.freeze:
            self.running_std.data = (
                self.running_std.data * 0.999 + x.std().detach() * 0.001
            ).clamp(min=1e-4)
        if hasattr(self, "running_std"):
            x = x / self.running_std

        if self.training and return_info:
            var = (x.std(dim=-1) ** 2).clip(min=1e-4)
            logvar = torch.log(var)
            mean = x.mean(dim=-1)
            loss = (mean * mean + var - logvar - 1).mean()
            var = (x.std(dim=-2) ** 2).clip(min=1e-4)
            logvar = torch.log(var)
            mean = x.mean(dim=-2)
            loss = loss + 0.4 * (mean * mean + var - logvar - 1).mean()
            info["softnorm_loss"] = loss

        if return_info:
            return x, info
        return x

    def decode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if hasattr(self, "running_std"):
            x = x * self.running_std

        if self.noise_regularize:
            scaling = self.running_std if hasattr(self, "running_std") else x.std(dim=-1).unsqueeze(-1)
            scale = 5e-2 if self.training else 1e-3
            x = x + torch.randn_like(x) * scaling * scale

        if self.noise_augment_dim > 0:
            noise = self.noise_scaling_factor * torch.randn(
                x.shape[0], self.noise_augment_dim, x.shape[-1],
            ).type_as(x)
            x = torch.cat([x, noise], dim=1)
        return x


# ---------------------------------------------------------------------------
# Pretransforms (PORT_FROM: pretransforms.py — full file, verbatim)
# ---------------------------------------------------------------------------


class AutoencoderPretransform(nn.Module):
    """Outer wrapper around AudioAutoencoder, adds chunked-decode + scale.

    PORT_FROM: pretransforms.py AutoencoderPretransform (lines 7-25).
    """

    def __init__(
        self,
        model: nn.Module,
        scale: float = 1.0,
        iterate_batch: bool = False,
        chunked: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale = scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.enable_grad = False
        self.iterate_batch = iterate_batch
        self.chunked = chunked

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.encode_audio(
            x, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs,
        ) / self.scale

    def decode(self, z: torch.Tensor, chunked: bool | None = None, **kwargs) -> torch.Tensor:
        chunked = self.chunked if chunked is None else chunked
        return self.model.decode_audio(
            z * self.scale, chunked=chunked, iterate_batch=self.iterate_batch, **kwargs,
        )


class PatchedPretransform(nn.Module):
    """Channel↔time folding: [B,C,L*H] ↔ [B,C*H,L].

    PORT_FROM: pretransforms.py PatchedPretransform (lines 37-84).
    """

    def __init__(
        self,
        channels: int,
        patch_size: int,
        oversampling: int = 1,
        postfilter_channels: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.oversampling = oversampling
        self.downsampling_ratio = patch_size
        self.io_channels = channels
        self.encoded_channels = channels * patch_size
        self.enable_grad = False

        if oversampling > 1:
            self.input_upsampler = Resample(1, oversampling)
            self.output_downsampler = Resample(oversampling, 1)

        if postfilter_channels > 0:
            # PORT_FROM: pretransforms.py:51-65
            self.postfilter = nn.Sequential(
                WNConv1d(
                    in_channels=channels,
                    out_channels=postfilter_channels,
                    kernel_size=7,
                    padding=3,
                    bias=True,
                ),
                ResidualUnit(postfilter_channels, postfilter_channels, dilation=1, bias=True),
                ResidualUnit(postfilter_channels, postfilter_channels, dilation=3, bias=True),
                ResidualUnit(postfilter_channels, postfilter_channels, dilation=9, bias=True),
                WNConv1d(
                    in_channels=postfilter_channels,
                    out_channels=channels,
                    kernel_size=7,
                    padding=3,
                    bias=False,
                ),
            )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-1]
        pad_len = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.cat([x, torch.zeros_like(x[:, :, :pad_len])], dim=-1)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.oversampling > 1:
            x = self.input_upsampler(x)
        x = self._pad(x)
        return rearrange(x, "b c (l h) -> b (c h) l", h=self.patch_size)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b (c h) l -> b c (l h)", h=self.patch_size)
        if hasattr(self, "postfilter"):
            x = self.postfilter(x)
        if self.oversampling > 1:
            x = self.output_downsampler(x)
        return x


# ---------------------------------------------------------------------------
# TransformerResamplingBlock (PORT_FROM: autoencoders.py:34-223 — large class)
# Used inside SAMEEncoder / SAMEDecoder for learned up/downsampling.
# ---------------------------------------------------------------------------


class TransformerResamplingBlock(nn.Module):
    """Learned resampling with transformer blocks + new-token injection.

    PORT_FROM: autoencoders.py:34-223 (verbatim, large).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        sliding_window: int | None = None,
        chunk_size: int = 128,
        chunk_midpoint_shift: bool = False,
        type: str = "encoder",
        transformer_depth: int = 3,
        checkpointing: bool = False,
        conformer: bool = False,
        layer_scale: bool = False,
        dim_heads: int = 128,
        differential: bool = True,
        variable_stride: bool = False,
        feat_scale: bool = False,
        sinusoidal_blocks: int = 0,
        mask_noise: float = 0,
        ff_mult: int = 3,
        mapping_bias: bool = True,
        cross_attn: bool = False,
        dyt: bool = True,
        conv_mapping: bool = False,
        freeze_backbone: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if type not in ("encoder", "decoder"):
            raise ValueError(f"Unknown type {type}. Must be 'encoder' or 'decoder'")

        # Import here to avoid circular import with transformer.py
        from vllm_omni.diffusion.models.stable_audio_3.stable_audio_3_transformer import (
            TransformerBlock,
        )

        self.checkpointing = checkpointing
        transformer_dim = out_channels if type == "encoder" else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.variable_stride = variable_stride
        self.stride = stride
        self.mapping = (
            WNConv1d(in_channels, out_channels, 3 if conv_mapping else 1, padding="same", bias=mapping_bias)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.chunk_size = chunk_size
        self.chunk_midpoint_shift = chunk_midpoint_shift
        self.type = type
        self.mask_noise = mask_noise
        self.sliding_window_latents = sliding_window

        self.sliding_window_seq = self._get_sliding_window_size(sliding_window, stride)
        self.input_seg_size, self.output_seg_size, self.sub_chunk_size = self._get_seg_sizes(stride)
        self.transformer_depth = transformer_depth

        transformers = []
        for i in range(transformer_depth):
            sinusoidal = (transformer_depth - i) < sinusoidal_blocks
            transformers.append(
                TransformerBlock(
                    transformer_dim,
                    dim_heads=dim_heads,
                    causal=False,
                    zero_init_branch_outputs=not layer_scale,
                    norm_type="dyt" if dyt else "rms_norm",
                    conformer=conformer,
                    layer_scale=layer_scale,
                    add_rope=True,
                    attn_kwargs={
                        "qk_norm": "dyt" if dyt else "rms",
                        "qk_norm_eps": 1e-3,
                        "differential": differential,
                        "feat_scale": feat_scale,
                    },
                    ff_kwargs={"mult": ff_mult, "no_bias": False, "sinusoidal": sinusoidal},
                    norm_kwargs={"eps": 1e-3},
                    cross_attend=cross_attn,
                ),
            )

        self.new_tokens = nn.Parameter(
            1e-5 * torch.randn(
                1,
                self.output_seg_size if not self.variable_stride else 1,
                out_channels if type == "encoder" else in_channels,
            ),
        )
        self.transformers = nn.ModuleList(transformers)

        if freeze_backbone:
            for p in self.transformers.parameters():
                p.requires_grad = False
            self.new_tokens.requires_grad = False

    def _get_sliding_window_size(self, window, stride, prepend_cond_length=0):
        if window is None:
            return None
        return [(win * (stride + 1 + prepend_cond_length)) for win in window]

    def _get_seg_sizes(self, stride: int, prepend_cond_length: int = 0):
        sub_chunk_size = stride + 1 + prepend_cond_length
        if self.sliding_window_latents is None:
            assert (self.chunk_size % stride) == 0, f"Stride must fit evenly into chunk_size:{self.chunk_size}"
        input_seg_size = stride if self.type == "encoder" else 1
        output_seg_size = 1 if self.type == "encoder" else stride
        return input_seg_size, output_seg_size, sub_chunk_size

    def forward(
        self,
        x: torch.Tensor,
        stride: int | None = None,
        return_features: bool = False,
        override_new_tokens: torch.Tensor | None = None,
        prepend_cond: torch.Tensor | None = None,
        cross_attn_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # PORT_FROM: autoencoders.py:131-223 (verbatim).
        batch_size = x.shape[0]
        if return_features:
            features = []

        if stride is None:
            input_seg_size = self.input_seg_size
            output_seg_size = self.output_seg_size
            sub_chunk_size = self.sub_chunk_size
            sliding_window = self.sliding_window_seq
        else:
            if not self.variable_stride:
                print("cannot override stride if variable_stride is not set")
            prepend_cond_length = prepend_cond.shape[-2] if prepend_cond is not None else 0
            input_seg_size, output_seg_size, sub_chunk_size = self._get_seg_sizes(stride, prepend_cond_length)
            sliding_window = self._get_sliding_window_size(self.sliding_window_latents, stride, prepend_cond_length)

        if self.type == "encoder":
            if self.transformer_depth > 0:
                pad_modulo = self.chunk_size if sliding_window is None else input_seg_size
                x = _zero_pad_modulo_sequence(x, pad_modulo, dim=-1)
            x = self.mapping(x)

        if self.transformer_depth > 0:
            x = rearrange(x, "... a b -> ... b a")
            if return_features:
                features.append(x)

            if self.type != "encoder":
                if sliding_window is None:
                    active_stride = stride if stride is not None else self.stride
                    pad_modulo = self.chunk_size // active_stride
                    x = _zero_pad_modulo_sequence(x, pad_modulo)
                else:
                    x = _zero_pad_modulo_sequence(x, input_seg_size)

            x = rearrange(x, "b (n c) d -> (b n) c d", c=input_seg_size)
            new_token_seq_dim = -1 if not self.variable_stride else output_seg_size
            new_tokens = self.new_tokens.expand([x.shape[0], new_token_seq_dim, -1])
            if override_new_tokens is not None:
                override_new_tokens = rearrange(
                    override_new_tokens, "b (n c) d -> (b n) c d", c=output_seg_size,
                )
                new_tokens = new_tokens + override_new_tokens
            elif self.mask_noise > 0:
                new_tokens = new_tokens + torch.randn_like(new_tokens) * self.mask_noise
            x = torch.cat([x, new_tokens], dim=-2)

            if prepend_cond is not None:
                n = x.shape[0] // batch_size
                cond_folded = (
                    prepend_cond.unsqueeze(1)
                    .expand(batch_size, n, prepend_cond.shape[-2], x.shape[-1])
                    .reshape(n * batch_size, prepend_cond.shape[-2], x.shape[-1])
                )
                x = torch.cat([cond_folded, x], dim=-2)

            x = rearrange(x, "(b n) c d -> b (n c) d", b=batch_size)

            if sliding_window is None:
                prepend_cond_length = prepend_cond.shape[-2] if prepend_cond is not None else 0
                effective_chunk_size = self.chunk_size + self.chunk_size * (1 + prepend_cond_length) // (
                    stride if stride is not None else self.stride
                )

            if sliding_window is None and self.chunk_midpoint_shift:
                split = self.transformer_depth // 2
                shift = effective_chunk_size // 2

                nc = x.shape[1] // effective_chunk_size
                x = rearrange(x, "b (nc cc) d -> (b nc) cc d", cc=effective_chunk_size)
                cross_attn_first = cross_attn_cond.repeat_interleave(nc, dim=0) if cross_attn_cond is not None else None
                for layer in self.transformers[:split]:
                    x = (
                        checkpoint(layer, x, context=cross_attn_first, self_attention_flash_sliding_window=None)
                        if self.checkpointing
                        else layer(x, context=cross_attn_first)
                    )
                    if return_features:
                        features.append(rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size))
                x = rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size)

                x = torch.cat([x[:, :shift, :], x, x[:, -shift:, :]], dim=1)
                nc_shifted = x.shape[1] // effective_chunk_size
                x = rearrange(x, "b (nc cc) d -> (b nc) cc d", cc=effective_chunk_size)
                cross_attn_second = (
                    cross_attn_cond.repeat_interleave(nc_shifted, dim=0) if cross_attn_cond is not None else None
                )
                for layer in self.transformers[split:]:
                    x = (
                        checkpoint(layer, x, context=cross_attn_second, self_attention_flash_sliding_window=None)
                        if self.checkpointing
                        else layer(x, context=cross_attn_second)
                    )
                    if return_features:
                        feat = rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size)
                        features.append(feat[:, shift:-shift, :])
                x = rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size)
                x = x[:, shift:-shift, :]
            else:
                if sliding_window is None:
                    x = rearrange(x, "b (nc cc) d -> (b nc) cc d", cc=effective_chunk_size)

                for layer in self.transformers:
                    x = (
                        checkpoint(layer, x, context=cross_attn_cond, self_attention_flash_sliding_window=sliding_window)
                        if self.checkpointing
                        else layer(x, context=cross_attn_cond, self_attention_flash_sliding_window=sliding_window)
                    )
                    if return_features:
                        features.append(x)

                if sliding_window is None:
                    x = rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size)

            x = rearrange(x, "b (n c) d -> (b n) c d", c=sub_chunk_size)
            x = x[:, -output_seg_size:, :]
            x = rearrange(x, "(b n) c d -> b d (n c)", b=batch_size)

        if self.type == "decoder":
            x = self.mapping(x)

        if return_features:
            return x, features
        return x


# ---------------------------------------------------------------------------
# SAMEEncoder / SAMEDecoder (PORT_FROM: autoencoders.py:225-349)
# ---------------------------------------------------------------------------


class SAMEEncoder(nn.Module):
    """Multi-stage transformer-based audio encoder.

    PORT_FROM: autoencoders.py:225-288 (verbatim).
    """

    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: list[int] | None = None,
        strides: list[int] | None = None,
        transformer_depths: list[int] | None = None,
        sliding_window: int | None = None,
        checkpointing: bool = False,
        conformer: bool = False,
        layer_scale: bool = False,
        causal: bool = False,
        differential: bool = True,
        variable_stride: bool = False,
        mask_noise: float = 0.0,
        conv_mapping: bool = False,
        freeze_backbone: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        c_mults = c_mults or [1, 2, 4, 8]
        strides = strides or [2, 4, 8, 8]
        transformer_depths = transformer_depths or [3, 3, 3, 3]

        self.in_channels = in_channels
        self.strides = strides

        channel_dims = [c * channels for c in c_mults]
        channel_dims = [in_channels] + channel_dims
        self.depth = len(c_mults)

        layers = []
        for i in range(self.depth):
            layers.append(
                TransformerResamplingBlock(
                    in_channels=channel_dims[i],
                    out_channels=channel_dims[i + 1],
                    stride=strides[i],
                    transformer_depth=transformer_depths[i],
                    sliding_window=sliding_window,
                    checkpointing=checkpointing,
                    conformer=conformer,
                    layer_scale=layer_scale,
                    causal=causal,
                    differential=differential,
                    variable_stride=variable_stride,
                    mask_noise=mask_noise,
                    conv_mapping=conv_mapping,
                    freeze_backbone=freeze_backbone,
                    **kwargs,
                ),
            )

        layers += [Transpose(), nn.Linear(channel_dims[-1], latent_dim), Transpose()]
        self.layers = nn.ModuleList(layers)

        if freeze_backbone:
            for p in self.layers[-2].parameters():
                p.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        override_stride: list[int] | None = None,
        return_features: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if override_stride is not None:
            assert isinstance(override_stride, list)
            assert len(override_stride) == self.depth

        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerResamplingBlock):
                stride = override_stride[i] if override_stride is not None else None
                if return_features:
                    x, features = layer(x, stride=stride, return_features=True)
                else:
                    x = layer(x, stride=stride)
            else:
                x = layer(x)

        if return_features:
            return x, features
        return x


class SAMEDecoder(nn.Module):
    """Multi-stage transformer-based audio decoder (mirror of encoder).

    PORT_FROM: autoencoders.py:290-349 (verbatim).
    """

    def __init__(
        self,
        out_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: list[int] | None = None,
        strides: list[int] | None = None,
        transformer_depths: list[int] | None = None,
        sliding_window: int | None = None,
        checkpointing: bool = False,
        conformer: bool = False,
        layer_scale: bool = False,
        causal: bool = False,
        differential: bool = True,
        variable_stride: bool = False,
        sinusoidal_blocks: list[int] | None = None,
        mask_noise: float = 0.0,
        conv_mapping: bool = False,
        freeze_backbone: bool = False,
        soft_clip: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        c_mults = c_mults or [1, 2, 4, 8]
        strides = strides or [2, 4, 8, 8]
        transformer_depths = transformer_depths or [3, 3, 3, 3]
        sinusoidal_blocks = sinusoidal_blocks or [0, 0, 0, 0]

        channel_dims = [c * channels for c in c_mults]
        channel_dims = [out_channels] + channel_dims
        self.depth = len(c_mults)

        layers: list[nn.Module] = [Transpose(), nn.Linear(latent_dim, channel_dims[-1]), Transpose()]
        for i in range(self.depth, 0, -1):
            layers.append(
                TransformerResamplingBlock(
                    in_channels=channel_dims[i],
                    out_channels=channel_dims[i - 1],
                    stride=strides[i - 1],
                    type="decoder",
                    transformer_depth=transformer_depths[i - 1],
                    sliding_window=sliding_window,
                    checkpointing=checkpointing,
                    conformer=conformer,
                    layer_scale=layer_scale,
                    causal=causal,
                    differential=differential,
                    variable_stride=variable_stride,
                    sinusoidal_blocks=sinusoidal_blocks[i - 1],
                    mask_noise=mask_noise,
                    conv_mapping=conv_mapping,
                    freeze_backbone=freeze_backbone,
                    **kwargs,
                ),
            )
        self.layers = nn.ModuleList(layers)

        if freeze_backbone:
            for p in self.layers[1].parameters():
                p.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        override_stride: list[int] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if override_stride is not None:
            assert isinstance(override_stride, list)
            assert len(override_stride) == self.depth

        transformer_layer_index = 0
        for layer in self.layers:
            if isinstance(layer, TransformerResamplingBlock):
                stride = (
                    override_stride[transformer_layer_index] if override_stride is not None else None
                )
                x = layer(x, stride=stride)
                transformer_layer_index += 1
            else:
                x = layer(x)
        return x


# ---------------------------------------------------------------------------
# AudioAutoencoder (PORT_FROM: autoencoders.py:351-638)
# ---------------------------------------------------------------------------


class AudioAutoencoder(nn.Module):
    """SAME autoencoder facade. Defaults from upstream chunked decode: chunk_size=128, overlap=32."""

    def __init__(
        self,
        encoder: SAMEEncoder,
        decoder: SAMEDecoder,
        latent_dim: int,
        downsampling_ratio: int,
        sample_rate: int = 44100,
        io_channels: int = 2,
        bottleneck: SoftNormBottleneck | None = None,
        pretransform: PatchedPretransform | None = None,
        in_channels: int | None = None,
        out_channels: int | None = None,
        soft_clip: bool = False,
        freeze_pretransform: bool = False,
    ) -> None:
        super().__init__()
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = in_channels if in_channels is not None else io_channels
        self.out_channels = out_channels if out_channels is not None else io_channels
        self.min_length = downsampling_ratio
        self.bottleneck = bottleneck
        self.encoder = encoder
        self.decoder = decoder
        self.pretransform = pretransform
        self.freeze_pretransform = freeze_pretransform
        self.soft_clip = soft_clip
        self.is_discrete = False

        if self.pretransform is not None:
            requires_grad = not freeze_pretransform
            for p in self.pretransform.parameters():
                p.requires_grad = requires_grad

        # vllm-omni accessor: pipeline reads `self.vae.config.sampling_rate` etc.
        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.sampling_rate = sample_rate
        self.config.latent_channels = latent_dim
        self.config.downsampling_ratio = downsampling_ratio

    # ----- encode/decode  (PORT_FROM: autoencoders.py:411-495) -----

    def encode(
        self,
        audio: torch.Tensor,
        return_info: bool = False,
        skip_pretransform: bool = False,
        iterate_batch: bool = False,
        return_pretransform: bool = False,
        **kwargs,
    ):
        info: dict = {}
        if self.pretransform is not None and not skip_pretransform:
            if self.pretransform.enable_grad:
                audio = (
                    torch.cat([self.pretransform.encode(audio[i:i + 1]) for i in range(audio.shape[0])], dim=0)
                    if iterate_batch
                    else self.pretransform.encode(audio)
                )
            else:
                with torch.no_grad():
                    audio = (
                        torch.cat([self.pretransform.encode(audio[i:i + 1]) for i in range(audio.shape[0])], dim=0)
                        if iterate_batch
                        else self.pretransform.encode(audio)
                    )

        if self.encoder is not None:
            if iterate_batch:
                latents = torch.cat(
                    [self.encoder(audio[i:i + 1], **kwargs) for i in range(audio.shape[0])], dim=0,
                )
            else:
                latents = self.encoder(audio, **kwargs)
        else:
            latents = audio

        if self.bottleneck is not None:
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)
            info.update(bottleneck_info)

        if return_info and return_pretransform:
            return latents, info, audio
        if return_info:
            return latents, info
        if return_pretransform:
            return latents, audio
        return latents

    def decode(
        self,
        latents: torch.Tensor,
        iterate_batch: bool = False,
        return_loss: bool = False,
        **kwargs,
    ):
        if self.bottleneck is not None:
            latents = (
                torch.cat([self.bottleneck.decode(latents[i:i + 1]) for i in range(latents.shape[0])], dim=0)
                if iterate_batch
                else self.bottleneck.decode(latents)
            )

        if iterate_batch:
            decoded = torch.cat(
                [self.decoder(latents[i:i + 1], **kwargs) for i in range(latents.shape[0])], dim=0,
            )
        else:
            if return_loss:
                decoded, loss = self.decoder(latents, **kwargs)
            else:
                decoded = self.decoder(latents, **kwargs)

        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                decoded = (
                    torch.cat([self.pretransform.decode(decoded[i:i + 1]) for i in range(decoded.shape[0])], dim=0)
                    if iterate_batch
                    else self.pretransform.decode(decoded)
                )
            else:
                with torch.no_grad():
                    decoded = (
                        torch.cat([self.pretransform.decode(decoded[i:i + 1]) for i in range(decoded.shape[0])], dim=0)
                        if iterate_batch
                        else self.pretransform.decode(decoded)
                    )

        if self.soft_clip:
            decoded = torch.tanh(decoded)

        return (decoded, loss) if return_loss else decoded

    # ----- chunked entry points (PORT_FROM: autoencoders.py:551-638) -----

    def encode_audio(
        self,
        audio: torch.Tensor,
        chunked: bool = False,
        overlap: int = 32,
        chunk_size: int = 128,
        **kwargs,
    ) -> torch.Tensor:
        """Encode audio in (optionally) overlapping chunks. PORT_FROM: autoencoders.py:551-595."""
        samples_per_latent = int(self.downsampling_ratio)
        if not chunked or audio.shape[-1] < chunk_size * samples_per_latent:
            return self.encode(audio, **kwargs)

        chunk_size_samples = chunk_size * samples_per_latent
        hop_samples = (chunk_size - overlap) * samples_per_latent
        total_samples = audio.shape[-1]

        chunk_starts = list(range(0, total_samples - chunk_size_samples + 1, hop_samples))
        if chunk_starts[-1] != total_samples - chunk_size_samples:
            chunk_starts.append(total_samples - chunk_size_samples)

        encoded_chunks = [self.encode(audio[..., s:s + chunk_size_samples]) for s in chunk_starts]
        total_latents = total_samples // samples_per_latent
        half_overlap_latents = overlap // 2
        output = audio.new_zeros(*encoded_chunks[0].shape[:-1], total_latents)
        num_chunks = len(chunk_starts)

        for i, (start_sample, chunk) in enumerate(zip(chunk_starts, encoded_chunks)):
            is_first = i == 0
            is_last = i == num_chunks - 1
            out_start = (total_latents - chunk_size) if is_last else (start_sample // samples_per_latent)
            left = 0 if is_first else half_overlap_latents
            right = chunk_size if is_last else chunk_size - half_overlap_latents
            output[..., out_start + left:out_start + right] = chunk[..., left:right]

        return output

    def decode_audio(
        self,
        latents: torch.Tensor,
        chunked: bool = False,
        overlap: int = 32,
        chunk_size: int = 128,
        **kwargs,
    ) -> torch.Tensor:
        """Decode latents → waveform. Chunked decode caps VRAM. PORT_FROM: autoencoders.py:596-638."""
        if not chunked or latents.shape[-1] < chunk_size:
            return self.decode(latents, **kwargs)

        samples_per_latent = int(self.downsampling_ratio)
        hop_latents = chunk_size - overlap
        total_latents = latents.shape[-1]

        chunk_starts = list(range(0, total_latents - chunk_size + 1, hop_latents))
        if chunk_starts[-1] != total_latents - chunk_size:
            chunk_starts.append(total_latents - chunk_size)

        decoded_chunks = [self.decode(latents[..., s:s + chunk_size]) for s in chunk_starts]
        total_samples = total_latents * samples_per_latent
        chunk_size_samples = chunk_size * samples_per_latent
        half_overlap_samples = (overlap // 2) * samples_per_latent
        output = latents.new_zeros(*decoded_chunks[0].shape[:-1], total_samples)
        num_chunks = len(chunk_starts)

        for i, (start_latent, chunk) in enumerate(zip(chunk_starts, decoded_chunks)):
            is_first = i == 0
            is_last = i == num_chunks - 1
            out_start = (total_samples - chunk_size_samples) if is_last else (start_latent * samples_per_latent)
            left = 0 if is_first else half_overlap_samples
            right = chunk_size_samples if is_last else chunk_size_samples - half_overlap_samples
            output[..., out_start + left:out_start + right] = chunk[..., left:right]

        return output
