# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact fast path for the Wan VAE decoder on a streaming (frame-at-a-time) decode.

A kernel ledger of the LingBot-World served path (4xH200, 480x832, width-sharded bf16 decode) put the
decoder at 91 ms of GPU time per chunk per rank, half of it in ~2,300 launches of glue around the
convolutions: autocast re-casting every conv weight and bias on every call (the autocast weight cache only
serves leaves that require grad, so under ``torch.no_grad`` it never hits), the nearest upsample's fp32
round trip, and a freshly allocated, zero-filled, concatenated input tensor per conv call. None of that
touches a value that reaches the convolution, so it can go without changing a single output bit:

- ``conv_dtype``: the decoder's convolution parameters are cast once to the dtype the decode already runs
  under (``decode_autocast_dtype``), the same cast autocast applied per call. Norm parameters stay in their
  own dtype, so the normalisation arithmetic and its fp32 promotion are untouched.
- ``WanUpsample`` runs its nearest-exact gather on the activation dtype directly instead of via
  ``x.float()`` and ``type_as``: a gather moves values, so the result is identical.
- The spatially sharded conv wrappers keep one input buffer per conv across calls
  (``reuse_input_buffer``), writing only the activation interior and the halo slots into it.

Numerics: bit-identical to the plain path (``tests/diffusion/distributed/test_wan_decoder_fast_path.py``).
Memory: one persistent input buffer per sharded conv (about the size of that conv's input).

Level ``"fused"`` adds the rest of what the ledger showed, at the cost of exactness (it is gated with the bf16
decode option, whose quality gate it shares): the decoder runs channels_last_3d end to end so cuDNN uses its
NHWC kernels without the NCHW<->NHWC transposes it otherwise inserts around every convolution, every
``WanRMS_norm -> SiLU`` chain becomes one Triton kernel that reads and writes the activation dtype (the eager
chain promotes to fp32 at ``* gamma`` and leaves the cast to the next convolution), the nearest upsample is a
channels_last gather, and the ``upsample3d`` time-pair interleave writes channels_last_3d directly. Values
differ from eager only through the norm's fp32 reduction order and the single rounding at the SiLU output.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.wan_decoder_kernels import (
    can_use_nearest_upsample_nhwc,
    can_use_wan_rmsnorm_silu,
    canonical_nhwc,
    nearest_upsample_nhwc,
    wan_rmsnorm_silu,
    wan_rmsnorm_silu_reference,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard import WanDistCausalConv3d, WanDistConv2d

logger = init_logger(__name__)

_INSTALLED_ATTR = "_vllm_omni_wan_decoder_fast_path"
LEVELS = ("exact", "fused")
# diffusers' WanRMS_norm, and vLLM-Omni's RMSNormVAE that patch_wan_rms_norm swaps in for every Wan-family VAE.
_NORM_CLASSES = ("WanRMS_norm", "RMSNormVAE")
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


class FusedWanRMSNormSiLU(nn.Module):
    """``WanRMS_norm`` followed by SiLU as one kernel on channels_last_3d CUDA tensors.

    Keeps the norm's parameters registered under their own names (``...norm1.gamma``), so state dicts and
    weight loading are unchanged. Off the kernel's domain (CPU, other layouts, grad) it runs the eager op
    chain, and a channels-first CUDA input is converted to channels_last_3d once (the attention block's
    gathered frame is the only such producer).
    """

    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        self.gamma = norm.gamma
        bias = norm.bias
        if isinstance(bias, torch.Tensor):
            self.bias = bias
            self._bias_value = 0.0
        else:
            self.bias = None
            self._bias_value = float(bias) if bias is not None else 0.0
        self.scale = float(norm.scale)
        # diffusers' WanRMS_norm normalises x.float() with F.normalize's default eps; vLLM-Omni's RMSNormVAE
        # (patched in for every Wan-family VAE it loads) normalises x itself with its own epsilon.
        self.upcast = norm.__class__.__name__ == "WanRMS_norm"
        self.eps = 1e-12 if self.upcast else float(getattr(norm, "epsilon", 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5 and x.is_cuda and not (torch.is_grad_enabled() and x.requires_grad):
            if not _dense_channels_last_3d(x):
                x = x.contiguous(memory_format=torch.channels_last_3d)
            if can_use_wan_rmsnorm_silu(x, self.gamma, self.bias):
                return wan_rmsnorm_silu(x, self.gamma, self.bias, self.scale, self.eps, self.upcast)
        bias = self.bias if self.bias is not None else self._bias_value
        return wan_rmsnorm_silu_reference(x, self.gamma, bias, self.scale, self.eps, self.upcast)


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


def _install_fused(decoder: nn.Module, post_quant_conv: nn.Module | None, counts: dict[str, int]) -> None:
    """Channels_last weights and buffers, fused norm+SiLU, layout-preserving resample. Fails closed."""
    blocks = [m for m in decoder.modules() if m.__class__.__name__ == "WanResidualBlock"]
    for m in blocks:
        for name in ("norm1", "norm2"):
            norm = getattr(m, name, None)
            if norm.__class__.__name__ not in _NORM_CLASSES or not getattr(norm, "channel_first", False):
                raise ValueError(
                    f"fused fast path needs a channel-first Wan RMSNorm at {name}; got {type(norm).__name__}"
                )
        if not isinstance(m.nonlinearity, nn.SiLU) or m.nonlinearity.inplace:
            raise ValueError("fused fast path needs a plain SiLU residual block nonlinearity")
    head_norm = getattr(decoder, "norm_out", None)
    if head_norm.__class__.__name__ not in _NORM_CLASSES or not isinstance(
        getattr(decoder, "nonlinearity", None), nn.SiLU
    ):
        raise ValueError("fused fast path needs the standard WanRMS_norm + SiLU decoder head")
    for m in blocks:
        m.norm1 = FusedWanRMSNormSiLU(m.norm1)
        m.norm2 = FusedWanRMSNormSiLU(m.norm2)
        m.nonlinearity = nn.Identity()
        counts["fused_norm_silu"] += 2
    decoder.norm_out = FusedWanRMSNormSiLU(decoder.norm_out)
    decoder.nonlinearity = nn.Identity()
    counts["fused_norm_silu"] += 1
    roots = [decoder] + ([post_quant_conv] if post_quant_conv is not None else [])
    for root in roots:
        for module in root.modules():
            if isinstance(module, nn.Conv3d):
                module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last_3d)
                counts["channels_last_convs"] += 1
            elif isinstance(module, nn.Conv2d):
                module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last)
                counts["channels_last_convs"] += 1
            if isinstance(module, WanDistCausalConv3d):
                module.input_memory_format = torch.channels_last_3d
            elif isinstance(module, WanDistConv2d):
                module.input_memory_format = torch.channels_last
            if _is_nearest_upsample(module):
                module.forward = MethodType(_upsample_forward_channels_last, module)
            if module.__class__.__name__ == "WanResample" and str(getattr(module, "mode", "")).startswith("upsample"):
                module.forward = MethodType(_resample_forward_channels_last, module)
                counts["resample_forwards"] += 1


def install_wan_decoder_fast_path(vae: Any, *, conv_dtype: torch.dtype | None, level: str = "exact") -> dict[str, int]:
    """Install the fast path on ``vae``'s decoder (and ``post_quant_conv``); idempotent.

    ``conv_dtype`` must be the dtype the decode runs under (``decode_autocast_dtype``): with ``None`` the
    parameters are left alone, because casting them without autocast would change the convolution's
    arithmetic rather than remove a cast. ``level`` is ``"exact"`` or ``"fused"`` (see the module docstring).
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}; got {level!r}")
    installed = getattr(vae, _INSTALLED_ATTR, None)
    if installed is not None:
        if installed.get("level") != level:
            raise ValueError(f"Wan decoder fast path already installed at level {installed.get('level')!r}")
        return installed
    decoder = getattr(vae, "decoder", None)
    if decoder is None:
        raise ValueError("Wan decoder fast path requires a decoder module.")
    if conv_dtype is not None and conv_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"conv_dtype must be float16, bfloat16 or None; got {conv_dtype!r}")
    roots = [decoder]
    post_quant_conv = getattr(vae, "post_quant_conv", None)
    if isinstance(post_quant_conv, nn.Module):
        roots.append(post_quant_conv)
    counts: dict[str, Any] = {
        "level": level,
        "conv_params_cast": 0,
        "upsamples": 0,
        "persistent_input_buffers": 0,
        "fused_norm_silu": 0,
        "channels_last_convs": 0,
        "resample_forwards": 0,
    }
    for root in roots:
        for module in root.modules():
            if conv_dtype is not None and isinstance(module, (nn.Conv2d, nn.Conv3d)):
                if any(p.dtype != conv_dtype for p in module.parameters(recurse=False)):
                    module.to(dtype=conv_dtype)
                    counts["conv_params_cast"] += 1
            if _is_nearest_upsample(module) and not getattr(module, "_vllm_omni_fast_path_upsample", False):
                module.forward = MethodType(_upsample_forward, module)
                module._vllm_omni_fast_path_upsample = True
                counts["upsamples"] += 1
            if isinstance(module, (WanDistCausalConv3d, WanDistConv2d)):
                module.reuse_input_buffer = True
                counts["persistent_input_buffers"] += 1
    if level == "fused":
        _install_fused(decoder, post_quant_conv if isinstance(post_quant_conv, nn.Module) else None, counts)
    setattr(vae, _INSTALLED_ATTR, counts)
    logger.info("Installed the %s Wan decoder fast path (conv dtype %s): %s", level, conv_dtype, counts)
    return counts
