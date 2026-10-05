# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DiT ``ResBlock`` for HunyuanImage3 -- default implementation.

Plain PyTorch: ``GroupNorm -> SiLU`` runs as two ops and the adaptive
GroupNorm as ``norm(h) * (1 + scale) + shift``. Hardware-specific variants
live in sibling packages (e.g. ``nvidia/``) and are picked by the dispatch in
``hunyuan_image3/__init__.py``.

"""

import torch
from torch import nn

from vllm_omni.diffusion.models.hunyuan_image3.layers.common import conv_nd, linear, normalization, zero_module


class ResBlock(nn.Module):
    r"""
    A residual block that can optionally change the number of channels.
    Args:
        in_channels (`int`):
            The number of input channels.
        emb_channels (`int`):
            The number of timestep embedding channels.
        dropout (`float`):
            The rate of dropout.
        out_channels (`int`, *optional*):
            If specified, the number of output channels.
        use_conv (`bool`, *optional*):
            If True and out_channels is specified, use a spatial convolution instead of a
            smaller 1x1 convolution to change the channels in the skip connection.
        dims (`int`, *optional*):
            Determines if the signal is 1D, 2D, or 3D.
        up (`bool`, *optional*):
            If True, use this block for upsampling.
        down (`bool`, *optional*):
            If True, use this block for downsampling.
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels=None,
        dropout=0.0,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or self.in_channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(self.in_channels, **factory_kwargs),
            nn.SiLU(),
            conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs),  # noqa: N802
        )

        self.updown = up or down
        self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channels, 2 * self.out_channels, **factory_kwargs))

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, **factory_kwargs)),  # noqa: N802
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs)  # noqa: N802
        else:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 1, **factory_kwargs)  # noqa: N802

    def forward(self, x, emb) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # Adaptive Group Normalization
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1.0 + scale) + shift
        h = out_rest(h)

        return self.skip_connection(x) + h


__all__ = ["ResBlock"]
