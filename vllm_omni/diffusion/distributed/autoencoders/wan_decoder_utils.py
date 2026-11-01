# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Utility functions for the streaming Wan decoder fast path."""

from __future__ import annotations

import torch
from torch import nn

from vllm_omni.diffusion.distributed.autoencoders.wan_decoder_kernels import (
    can_use_nearest_upsample_nhwc,
    canonical_nhwc,
    nearest_upsample_nhwc,
)

_CACHE_T = 2  # diffusers.models.autoencoders.autoencoder_kl_wan.CACHE_T


def _upsample_forward(self: nn.Upsample, x: torch.Tensor) -> torch.Tensor:
    # ``WanUpsample.forward`` is ``super().forward(x.float()).type_as(x)``; a nearest gather never changes a
    # value, so running it on ``x`` itself is the same tensor without the two casts.
    return nn.Upsample.forward(self, x)


def _is_nearest_upsample(module: nn.Module) -> bool:
    return (
        module.__class__.__name__ == "WanUpsample"
        and isinstance(module, nn.Upsample)
        and str(getattr(module, "mode", "")) in ("nearest", "nearest-exact")
    )


def _upsample_forward_channels_last(self: nn.Upsample, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and self.size is None and can_use_nearest_upsample_nhwc(x, self.scale_factor, self.mode):
        return nearest_upsample_nhwc(x, self.scale_factor)
    return nn.Upsample.forward(self, x)


def _dense_channels_last_3d(x: torch.Tensor) -> bool:
    if x.dim() != 5 or x.shape[1] <= 1:
        return False
    b, c, t, h, w = x.shape
    return x.stride() == (t * h * w * c, 1, h * w * c, w * c, c)


def _interleave_time_pairs(x: torch.Tensor, b: int, c: int, t: int, h: int, w: int) -> torch.Tensor:
    """``[B, 2C, T, H, W] -> [B, C, 2T, H, W]``: the time-conv's two channel halves interleaved along time.

    Same values as the eager ``reshape / stack / reshape``; on a dense channels_last_3d input the result is
    written channels_last_3d with one copy instead of materialising it channels-first.
    """
    if _dense_channels_last_3d(x):
        out = torch.empty((b, c, t * 2, h, w), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
        out.view(b, c, t, 2, h, w).copy_(x.view(b, 2, c, t, h, w).permute(0, 2, 3, 1, 4, 5))
        return out
    x = x.reshape(b, 2, c, t, h, w)
    x = torch.stack((x[:, 0], x[:, 1]), 3)
    return x.reshape(b, c, t * 2, h, w)


def _frames_nhwc(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, T, H, W] -> [B*T, C, H, W]`` for the 2D resample, with canonical NHWC strides when the data is
    laid out that way (a size-1 batch leaves a degenerate stride that aten's layout test rejects)."""
    b, c, t, h, w = x.shape
    frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    if frames.is_contiguous(memory_format=torch.channels_last) and not canonical_nhwc(frames) and frames.stride(1) == 1:
        frames = frames.as_strided((b * t, c, h, w), (h * w * c, 1, w * c, c))
    return frames


def _resample_forward_channels_last(self: nn.Module, x: torch.Tensor, feat_cache=None, feat_idx=[0]) -> torch.Tensor:  # noqa: B006
    """``WanResample.forward`` for the upsample modes with the layout-preserving interleave and frame view."""
    b, c, t, h, w = x.size()
    if self.mode == "upsample3d" and feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
            feat_cache[idx] = "Rep"
            feat_idx[0] += 1
        else:
            cache_x = x[:, :, -_CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
            if feat_cache[idx] == "Rep":
                x = self.time_conv(x)
            else:
                x = self.time_conv(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
            x = _interleave_time_pairs(x, b, c, t, h, w)
    t = x.shape[2]
    frames = self.resample(_frames_nhwc(x))
    return frames.view(b, t, frames.size(1), frames.size(2), frames.size(3)).permute(0, 2, 1, 3, 4)


def _persistent_input_buffer(module: nn.Module, shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    """One zero-initialised input buffer per conv, reallocated only when the shape, dtype or device changes.

    The callers write only the activation interior and the halo slots, so the padding rows and columns stay
    zero for the buffer's lifetime and are never filled again.
    """
    memory_format = getattr(module, "input_memory_format", torch.contiguous_format)
    buf = module._input_buf
    if (
        buf is None
        or buf.shape != shape
        or buf.dtype != reference.dtype
        or buf.device != reference.device
        or not buf.is_contiguous(memory_format=memory_format)
    ):
        buf = torch.empty(shape, dtype=reference.dtype, device=reference.device, memory_format=memory_format).zero_()
        module._input_buf = buf
    return buf
