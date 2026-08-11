# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guarded Wan VAE decoder data-movement fast paths.

This module adapts the lossless Wan decoder optimizations from SGLang while
keeping Diffusers' parameter names and module topology unchanged. Every
optimized method falls back to the Diffusers 0.38.0 operation order when the
kernel, device, layout, dtype, autograd state, or module type is unsupported.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    CACHE_T,
    DupUp3D,
    WanCausalConv3d,
    WanDecoder3d,
    WanResample,
    WanResidualBlock,
    WanResidualUpBlock,
)

from vllm_omni.diffusion.distributed.autoencoders import wan_spatial_shard

try:
    from vllm_omni.diffusion.kernels.wan_vae_data_movement import (
        cat_pad_5d,
        dup_up3d_add,
    )
except ImportError:  # pragma: no cover - Triton is optional on non-GPU installs.
    cat_pad_5d = None
    dup_up3d_add = None

_INSTALL_ATTR = "_vllm_omni_wan_data_movement_installed"
_DYNAMIC_SPATIAL_CONV_ATTR = "_vllm_omni_dynamic_spatial_shard_conv"


def _cache_payload(cache: Any) -> torch.Tensor | None:
    return cache if isinstance(cache, torch.Tensor) else None


def _is_dynamic_spatial_conv(conv: nn.Module) -> bool:
    return bool(getattr(conv, _DYNAMIC_SPATIAL_CONV_ATTR, False))


def _spatial_shard_is_active() -> bool:
    return wan_spatial_shard._SPATIAL_SHARD_CONTEXT.get() is not None


def _causal_padding(conv: nn.Module) -> tuple[int, ...]:
    if _is_dynamic_spatial_conv(conv):
        return tuple(conv._source_padding)
    return tuple(conv._padding)


def _can_use_fused_cache_path(conv: nn.Module, x: torch.Tensor) -> bool:
    return bool(
        cat_pad_5d is not None
        and (type(conv) is WanCausalConv3d or _is_dynamic_spatial_conv(conv))
        and x.dim() == 5
        and x.is_cuda
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and not _spatial_shard_is_active()
    )


def _run_cached_causal_conv(
    conv: nn.Module,
    x: torch.Tensor,
    cache_list: list[Any],
    index: int,
) -> torch.Tensor:
    """Run a causal conv and refresh its feature-cache slot."""
    cache = cache_list[index]
    is_repeat_marker = isinstance(cache, str)
    payload = None if is_repeat_marker else _cache_payload(cache)
    if _can_use_fused_cache_path(conv, x) and (
        payload is None or (payload.device == x.device and payload.dtype == x.dtype)
    ):
        pair = cat_pad_5d(x, payload, _causal_padding(conv), keep_cache_frames=CACHE_T)
        if pair is not None:
            conv_input, cache_list[index] = pair
            return nn.Conv3d.forward(conv, conv_input)

    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    if cache_x.shape[2] < CACHE_T and payload is not None:
        cache_x = torch.cat(
            [payload[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
            dim=2,
        )
    elif cache_x.shape[2] < CACHE_T and is_repeat_marker:
        cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)

    out = conv(x) if payload is None else conv(x, payload)
    cache_list[index] = cache_x
    return out


def _causal_conv_forward(self: WanCausalConv3d, x: torch.Tensor, cache_x: torch.Tensor | None = None):
    padding = list(self._padding)
    if (
        any(padding)
        and _can_use_fused_cache_path(self, x)
        and (cache_x is None or (cache_x.device == x.device and cache_x.dtype == x.dtype))
    ):
        conv_input = cat_pad_5d(x, cache_x, padding)
        if conv_input is not None:
            return nn.Conv3d.forward(self, conv_input)

    if cache_x is not None and padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    x = F.pad(x, padding)
    return nn.Conv3d.forward(self, x)


def _resample_forward(
    self: WanResample,
    x: torch.Tensor,
    feat_cache: list[Any] | None = None,
    feat_idx: list[int] | None = None,
):
    if feat_idx is None:
        feat_idx = [0]
    batch, channels, frames, height, width = x.size()
    if self.mode == "upsample3d" and feat_cache is not None:
        index = feat_idx[0]
        if feat_cache[index] is None:
            feat_cache[index] = "Rep"
            feat_idx[0] += 1
        else:
            x = _run_cached_causal_conv(self.time_conv, x, feat_cache, index)
            feat_idx[0] += 1
            x = x.reshape(batch, 2, channels, frames, height, width)
            x = torch.stack((x[:, 0], x[:, 1]), 3)
            x = x.reshape(batch, channels, frames * 2, height, width)

    frames = x.shape[2]
    x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    x = self.resample(x)
    x = x.view(batch, frames, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

    if self.mode == "downsample3d" and feat_cache is not None:
        index = feat_idx[0]
        if feat_cache[index] is None:
            feat_cache[index] = x.clone()
            feat_idx[0] += 1
        else:
            cache_x = x[:, :, -1:, :, :].clone()
            x = self.time_conv(torch.cat([feat_cache[index][:, :, -1:, :, :], x], dim=2))
            feat_cache[index] = cache_x
            feat_idx[0] += 1
    return x


def _residual_block_forward(
    self: WanResidualBlock,
    x: torch.Tensor,
    feat_cache: list[Any] | None = None,
    feat_idx: list[int] | None = None,
):
    if feat_idx is None:
        feat_idx = [0]
    residual = self.conv_shortcut(x)
    x = self.nonlinearity(self.norm1(x))
    if feat_cache is not None:
        index = feat_idx[0]
        x = _run_cached_causal_conv(self.conv1, x, feat_cache, index)
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    x = self.dropout(self.nonlinearity(self.norm2(x)))
    if feat_cache is not None:
        index = feat_idx[0]
        x = _run_cached_causal_conv(self.conv2, x, feat_cache, index)
        feat_idx[0] += 1
    else:
        x = self.conv2(x)
    return x + residual


def _residual_up_block_forward(
    self: WanResidualUpBlock,
    x: torch.Tensor,
    feat_cache: list[Any] | None = None,
    feat_idx: list[int] | None = None,
    first_chunk: bool = False,
):
    if feat_idx is None:
        feat_idx = [0]
    shortcut_source = x.clone()
    for resnet in self.resnets:
        if feat_cache is None:
            x = resnet(x)
        else:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)

    if self.upsampler is not None:
        if feat_cache is None:
            x = self.upsampler(x)
        else:
            x = self.upsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)

    shortcut = self.avg_shortcut
    if shortcut is None:
        return x
    if (
        dup_up3d_add is not None
        and type(shortcut) is DupUp3D
        and x.is_cuda
        and shortcut_source.is_cuda
        and x.dtype == shortcut_source.dtype
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and not _spatial_shard_is_active()
    ):
        fused = dup_up3d_add(
            x,
            shortcut_source,
            shortcut.factor_t,
            shortcut.factor_s,
            shortcut.repeats,
            first_chunk,
        )
        if fused is not None:
            return fused
    return x + shortcut(shortcut_source, first_chunk=first_chunk)


def _decoder_forward(
    self: WanDecoder3d,
    x: torch.Tensor,
    feat_cache: list[Any] | None = None,
    feat_idx: list[int] | None = None,
    first_chunk: bool = False,
):
    if feat_idx is None:
        feat_idx = [0]
    if feat_cache is not None:
        index = feat_idx[0]
        x = _run_cached_causal_conv(self.conv_in, x, feat_cache, index)
        feat_idx[0] += 1
    else:
        x = self.conv_in(x)

    x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
    for up_block in self.up_blocks:
        x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

    x = self.nonlinearity(self.norm_out(x))
    if feat_cache is not None:
        index = feat_idx[0]
        x = _run_cached_causal_conv(self.conv_out, x, feat_cache, index)
        feat_idx[0] += 1
    else:
        x = self.conv_out(x)
    return x


def install_wan_vae_data_movement(vae: nn.Module) -> bool:
    """Install guarded fast paths on a Diffusers Wan decoder once."""
    if getattr(vae, _INSTALL_ATTR, False):
        return True
    if cat_pad_5d is None or dup_up3d_add is None:
        return False

    decoder = getattr(vae, "decoder", None)
    if type(decoder) is not WanDecoder3d:
        return False

    decoder.forward = MethodType(_decoder_forward, decoder)
    patched = 0
    for module in decoder.modules():
        module_type = type(module)
        if module_type is WanCausalConv3d:
            module.forward = MethodType(_causal_conv_forward, module)
            patched += 1
        elif _is_dynamic_spatial_conv(module):
            # The dynamic wrapper owns halo exchange while a spatial request is
            # active. Its direct path can still use the lossless fused cache
            # movement through ``_run_cached_causal_conv``.
            patched += 1
        elif module_type is WanResidualBlock:
            module.forward = MethodType(_residual_block_forward, module)
        elif module_type is WanResidualUpBlock:
            module.forward = MethodType(_residual_up_block_forward, module)
        elif module_type is WanResample:
            module.forward = MethodType(_resample_forward, module)

    if patched == 0:
        return False
    setattr(vae, _INSTALL_ATTR, True)
    return True
