# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ResnetBlock for the HunyuanImage3 autoencoder — NVIDIA CUDA + Triton implementation.

Split out from autoencoder.py because its ``GroupNorm -> SiLU`` pairs are served
by :func:`fused_group_norm_silu`, a single Triton kernel that falls back to
native ``F.silu(F.group_norm(...))`` when Triton is unavailable.
"""

from torch import nn

from vllm_omni.diffusion.models.hunyuan_image3.layers.nvidia._cudnn import cudnn_settings
from vllm_omni.model_executor.models.common.ops import fused_group_norm_silu


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # cuDNN autotune is scoped to this block rather than set at import time.
        # These ResnetBlocks carry the bulk of the VAE's conv work: ~29 instances
        # x 2 convs across every resolution (up to the 1024x1024 decode), versus
        # the scattered convs left in autoencoder.py (AttnBlock's 1x1 at the
        # lowest-res mid, down/up-samplers, conv_in/out) whose activations are
        # negligible. A process-wide flag would instead silently change every
        # other model's conv behaviour in the same process.
        with cudnn_settings(benchmark=True, deterministic=True):
            h = x
            h = fused_group_norm_silu(
                h, self.norm1.weight, self.norm1.bias, num_groups=self.norm1.num_groups, eps=self.norm1.eps
            )
            h = self.conv1(h)

            h = fused_group_norm_silu(
                h, self.norm2.weight, self.norm2.bias, num_groups=self.norm2.num_groups, eps=self.norm2.eps
            )
            h = self.conv2(h)

            if self.in_channels != self.out_channels:
                x = self.nin_shortcut(x)
            return x + h


__all__ = ["ResnetBlock"]
