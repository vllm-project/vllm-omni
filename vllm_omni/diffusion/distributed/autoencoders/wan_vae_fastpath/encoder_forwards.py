# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replacement forwards for Cosmos3's residual Wan encoder."""

from __future__ import annotations

from typing import Any

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    AvgDown3D,
    WanCausalConv3d,
    WanEncoder3d,
    WanResample,
    WanResidualDownBlock,
)
from torch import nn

from . import forwards as fp
from . import triton_data_movement as dm
from . import triton_downsample as down


def is_spatial_downsample(module: WanResample) -> bool:
    """Recognize the exact pad/conv pair whose input assembly can be fused."""
    resample = module.resample
    return (
        module.mode in ("downsample2d", "downsample3d")
        and type(resample) is nn.Sequential
        and len(resample) == 2
        and type(resample[0]) is nn.ZeroPad2d
        and resample[0].padding == (0, 1, 0, 1)
        and type(resample[1]) is nn.Conv2d
    )


def downsample_forward(
    self: WanResample, x: torch.Tensor, feat_cache: list[Any] | None = None, feat_idx: list[int] | None = None
) -> torch.Tensor:
    if feat_idx is None:
        feat_idx = [0]
    if not fp._kernels_allowed(x) or not is_spatial_downsample(self):
        return WanResample.forward(self, x, feat_cache=feat_cache, feat_idx=feat_idx)
    cfg = getattr(self, fp.CFG_ATTR, None)
    if x.stride(1) == 1 and (cfg is None or not cfg.channels_last):
        # With singleton batch/time, upstream reshape can stop suggesting NHWC
        # to ZeroPad2d even for channels-last inputs. Canonicalizing that layout
        # is a channels_last optimization; lossless preserves the original call.
        return WanResample.forward(self, x, feat_cache=feat_cache, feat_idx=feat_idx)

    batch, _, frames, _, _ = x.shape
    padded = down.spatial_downsample_input(x)
    if padded is None:
        spatial = self.resample(fp._merge_batch_and_frames(x))
    else:
        # Call the convolution module normally: its hooks/wrappers are retained.
        spatial = self.resample[1](padded)
    x = fp._split_batch_and_frames(spatial, batch, frames)

    if self.mode == "downsample3d" and feat_cache is not None:
        index = feat_idx[0]
        cached = feat_cache[index]
        if cached is None:
            # First chunk is passed through without the temporal stride-2 conv.
            feat_cache[index] = x.clone()
        else:
            pair = None
            if isinstance(cached, torch.Tensor) and type(self.time_conv) is WanCausalConv3d:
                # Exactly one previous frame, no temporal zeros; unlike CACHE_T=2
                # causal-conv assembly. Refresh the last *input* frame, not output.
                pair = dm.cat_time_5d(x, cached[:, :, -1:], pad_front=1, keep_cache_frames=1)
            if pair is None:
                next_cache = x[:, :, -1:].clone()
                x = self.time_conv(torch.cat([cached[:, :, -1:], x], dim=2))
            else:
                assembled, next_cache = pair
                x = self.time_conv(assembled)
            feat_cache[index] = next_cache
        feat_idx[0] += 1
    return x


def residual_down_block_forward(
    self: WanResidualDownBlock,
    x: torch.Tensor,
    feat_cache: list[Any] | None = None,
    feat_idx: list[int] | None = None,
) -> torch.Tensor:
    if feat_idx is None:
        feat_idx = [0]
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        return WanResidualDownBlock.forward(self, x, feat_cache=feat_cache, feat_idx=feat_idx)
    cfg = getattr(self, fp.CFG_ATTR, None)
    source = x.clone() if cfg is None or cfg.clone_encoder_shortcuts else x
    for resnet in self.resnets:
        x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
    if self.downsampler is not None:
        x = self.downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)

    shortcut = self.avg_shortcut
    if cfg is not None and cfg.channels_last and type(shortcut) is AvgDown3D:
        out = down.avg_down3d_add(x, source, shortcut.factor_t, shortcut.factor_s, shortcut.group_size)
        if out is not None:
            return out
    # Lossless retains ATen's reduction and its intermediate dtype rounding.
    return x + shortcut(source)


def encoder_forward(
    self: WanEncoder3d,
    x: torch.Tensor,
    feat_cache: list[Any] | None = None,
    feat_idx: list[int] | None = None,
) -> torch.Tensor:
    if feat_idx is None:
        feat_idx = [0]
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        return WanEncoder3d.forward(self, x, feat_cache=feat_cache, feat_idx=feat_idx)
    if feat_cache is None:
        x = self.conv_in(x)
    else:
        x = fp._run_cached_causal_conv(self.conv_in, x, feat_cache, feat_idx[0])
        feat_idx[0] += 1
    for block in self.down_blocks:
        x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)
    x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
    if feat_cache is None:
        return self.conv_out(fp._norm_act(self.norm_out, self.nonlinearity, x))
    x = fp._run_norm_act_cached_conv(self.norm_out, self.nonlinearity, self.conv_out, x, feat_cache, feat_idx[0])
    feat_idx[0] += 1
    return x
