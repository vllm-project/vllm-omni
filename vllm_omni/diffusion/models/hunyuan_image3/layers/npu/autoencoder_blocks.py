# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""NPU VAE layers, selected once by the HunyuanImage3 layer dispatcher."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard import all_gather_along_dim, halo_exchange
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.autoencoder_blocks import (
    AttnBlock as NativeAttnBlock,
)
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.autoencoder_blocks import (
    ResnetBlock as NativeResnetBlock,
)
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.vae_context import VAE_SPATIAL_CONTEXT
from vllm_omni.diffusion.models.hunyuan_image3.layers.npu.normalization import (
    global_group_norm,
    vae_group_norm_for_conv,
    vae_group_norm_silu,
)


class VaeConv3d(nn.Conv3d):
    """Normal Conv3d outside a spatial call; symmetric halo exchange inside it."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | str = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=dtype,
        )
        self._halo_recv_top: torch.Tensor | None = None
        self._halo_recv_bottom: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = VAE_SPATIAL_CONTEXT.get()
        if context is None:
            return super().forward(x)
        pad_t, pad_h, pad_w = self.padding
        if pad_h:
            x, self._halo_recv_top, self._halo_recv_bottom = halo_exchange(
                x,
                group=context.group,
                halo_size=pad_h,
                recv_top_buf=self._halo_recv_top,
                recv_bottom_buf=self._halo_recv_bottom,
            )
        return F.conv3d(x, self.weight, self.bias, self.stride, (pad_t, 0, pad_w), self.dilation, self.groups)


class ResnetBlock(NativeResnetBlock):
    conv_cls = VaeConv3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(vae_group_norm_silu(x, self.norm1))
        h = self.conv2(vae_group_norm_silu(h, self.norm2))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(NativeAttnBlock):
    conv_cls = VaeConv3d

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return vae_group_norm_for_conv(x, self.norm, activation="identity")

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        context = VAE_SPATIAL_CONTEXT.get()
        if context is None:
            return super().attention(x)
        x = global_group_norm(x, self.norm, activation="identity")
        q = self.q(x)
        k = all_gather_along_dim(self.k(x), group=context.group, dim=3)
        v = all_gather_along_dim(self.v(x), group=context.group, dim=3)
        batch, channels, frames, height, width = q.shape
        q = q.flatten(2).transpose(1, 2).unsqueeze(1)
        k = k.flatten(2).transpose(1, 2).unsqueeze(1)
        v = v.flatten(2).transpose(1, 2).unsqueeze(1)
        y = F.scaled_dot_product_attention(q, k, v)
        return y.squeeze(1).transpose(1, 2).reshape(batch, channels, frames, height, width)
