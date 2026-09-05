# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exact replacement forwards for the diffusers Wan VAE decoder modules.

Each function below is bound per module instance (``types.MethodType``) by
:mod:`.install`; the diffusers classes themselves are never modified. Every
function reproduces the diffusers 0.40 operation order exactly and only
differs in how the bytes move:

* ``WanRMS_norm``: fp32 ``vector_norm`` straight from the low-precision tensor
  plus one fused Triton epilogue (see :mod:`.triton_rms_norm`), optionally with
  the following SiLU folded in. At the ``channels_last`` level the single-pass
  kernel also absorbs the bias of the ``conv1`` that feeds ``norm2``.
* ``WanCausalConv3d`` and every cached call site: the ``clone`` + ``cat`` +
  ``F.pad`` triple becomes one layout-preserving kernel that also writes the
  next cache frames; all-zero paddings skip ``F.pad`` entirely.
* ``WanResidualUpBlock``: ``x + DupUp3D(x)`` becomes one gather + add that
  also applies the bias of the upsampler's ``Conv2d`` (which feeds nothing else).
* ``WanResample``: the ``upsample3d`` time interleave becomes one strided copy.
* ``WanUpsample``: the nearest-exact 2x upsample becomes one gather kernel
  that reads the strided frame-major input in place (see
  :mod:`.triton_upsample`); it only replicates values, so the upstream fp32
  round trip is the identity and is skipped.

Whenever a kernel declines an input, the code falls through to the reference
expression, so the result is identical on every path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

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
    WanUpsample,
)
from vllm.logger import init_logger

from . import triton_data_movement as dm
from . import triton_rms_norm as rn
from . import triton_rms_norm_cl as cl
from . import triton_upsample as up

logger = init_logger(__name__)

CFG_ATTR = "_vllm_omni_wan_fastpath_cfg"
_NORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class FastPathConfig:
    """Per-installation switches read by the bound forwards."""

    fused_silu_dtypes: frozenset[torch.dtype] = frozenset()
    channels_last: bool = False


def is_diffusers_rms_norm(module: Any) -> bool:
    """True for a diffusers ``WanRMS_norm`` instance.

    Identified by name because ``vllm_omni.diffusion.models.wan2_2.patch_diffusers``
    rebinds the name ``WanRMS_norm`` to ``RMSNormVAE`` in diffusers' own module
    namespace; instances created before that patch keep the original class.
    ``RMSNormVAE`` has different numerics (eps 1e-6, no fp32 upcast) and is
    deliberately not matched.
    """
    cls = type(module)
    return (
        cls.__name__ == "WanRMS_norm"
        and cls.__module__.startswith("diffusers.")
        and isinstance(getattr(module, "gamma", None), torch.Tensor)
        and hasattr(module, "scale")
        and hasattr(module, "channel_first")
    )


def _kernels_allowed(x: torch.Tensor) -> bool:
    return x.is_cuda and not torch.is_grad_enabled() and not torch.compiler.is_compiling()


# --------------------------------------------------------------------------- #
# WanRMS_norm
# --------------------------------------------------------------------------- #


def _as_rows(
    x: torch.Tensor, denom: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]] | None:
    """View ``x``/``denom`` as contiguous ``(N, C, S)`` / ``(N, 1, S)`` without copying.

    Handles the dense layouts the decoder produces: plain contiguous 4D/5D
    tensors and the frame-major 5D view ``WanResample`` emits
    (``permute(0, 2, 1, 3, 4)`` of a contiguous ``(b, t, c, h, w)`` buffer).
    Returns the two views and a function mapping a kernel output back to
    ``x``'s shape and strides, or ``None`` for any other layout.
    """
    if x.dim() == 5:
        b, c, t, h, w = x.shape
        if x.is_contiguous():
            spatial = t * h * w
            return x.view(b, c, spatial), denom.view(b, 1, spatial), lambda out: out.view(b, c, t, h, w)
        frame_major = x.permute(0, 2, 1, 3, 4)
        denom_frame_major = denom.permute(0, 2, 1, 3, 4)
        if frame_major.is_contiguous() and denom_frame_major.is_contiguous():
            spatial = h * w
            return (
                frame_major.view(b * t, c, spatial),
                denom_frame_major.view(b * t, 1, spatial),
                lambda out: out.view(b, t, c, h, w).permute(0, 2, 1, 3, 4),
            )
        return None
    if x.dim() == 4 and x.is_contiguous():
        n, c, h, w = x.shape
        spatial = h * w
        return x.view(n, c, spatial), denom.view(n, 1, spatial), lambda out: out.view(n, c, h, w)
    return None


def _add_channel_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Out-of-place ``x + bias`` per channel; rounds exactly like ATen's in-place conv bias ``add_``."""
    return x + bias.view(1, -1, *([1] * (x.dim() - 2)))


def rms_norm_fastpath(
    norm: nn.Module, x: torch.Tensor, *, silu: bool = False, bias: torch.Tensor | None = None
) -> torch.Tensor | None:
    """``WanRMS_norm(x + bias)`` (optionally followed by SiLU) with fewer passes, or ``None``.

    ``bias`` is the un-added per-channel bias of the convolution that produced
    ``x``; only the channels-last kernel absorbs it, every other path adds it
    first with ATen's rounding. ``None`` means the caller must run the reference
    forward (tensor bias, channel-last norm, unsupported dtype, gamma/activation
    dtype mismatch) and is responsible for adding ``bias`` itself.
    """
    if not getattr(norm, "channel_first", False):
        return None
    norm_bias = norm.bias
    if not (isinstance(norm_bias, float) and norm_bias == 0.0):
        return None
    gamma = norm.gamma
    if x.dtype not in _NORM_DTYPES or gamma.dtype is not x.dtype:
        return None
    if x.dim() not in (4, 5) or gamma.dim() != x.dim() - 1 or x.numel() == 0:
        return None

    cfg = getattr(norm, CFG_ATTR, None)
    if cfg is not None and cfg.channels_last and _kernels_allowed(x):
        # Tier 2: one read, one write; not bit-exact (different reduction order).
        out = cl.rms_norm_channels_last(x, gamma, norm.scale, silu=silu, bias=bias)
        if out is not None:
            return out
    if bias is not None:
        x = _add_channel_bias(x, bias)

    # Same reduction kernel as ``F.normalize(x.float(), dim=1)`` runs, reading
    # the low-precision tensor once instead of materializing an fp32 copy.
    denom = torch.linalg.vector_norm(x, dim=1, keepdim=True, dtype=torch.float32).clamp_min_(1e-12)

    if _kernels_allowed(x):
        rows = _as_rows(x, denom)
        if rows is not None:
            x_rows, denom_rows, restore = rows
            out = rn.rms_norm_scale(x_rows, denom_rows, gamma.reshape(-1), norm.scale, silu=silu)
            if out is not None:
                return restore(out)

    # Exact PyTorch restructuring: the divide computes in fp32 opmath and rounds
    # once on store, exactly like ``(x.float() / denom).to(x.dtype)``; the
    # epilogue is upstream's own expression.
    normalized = torch.empty_like(x)
    torch.div(x, denom, out=normalized)
    out = normalized * norm.scale * gamma + norm_bias
    return F.silu(out) if silu else out


def rms_norm_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = rms_norm_fastpath(self, x)
    if out is not None:
        return out
    return type(self).forward(self, x)


def _norm_act(
    norm: nn.Module, act: nn.Module, x: torch.Tensor, *, pending_bias: torch.Tensor | None = None
) -> torch.Tensor:
    """``act(norm(x + pending_bias))`` with the SiLU (and the bias) folded into the norm kernel when possible.

    ``pending_bias`` is the per-channel bias the producing convolution left
    un-added (see :func:`_run_cached_causal_conv` with ``return_bias=True``).
    """
    cfg = getattr(norm, CFG_ATTR, None)
    if cfg is not None and is_diffusers_rms_norm(norm):
        # SiLU is folded into the norm kernel when the epilogue was proven exact
        # for this dtype, or always under the (tolerance-based) channels_last level.
        fuse = type(act) is nn.SiLU and not act.inplace and (x.dtype in cfg.fused_silu_dtypes or cfg.channels_last)
        out = rms_norm_fastpath(norm, x, silu=fuse, bias=pending_bias)
        if out is not None:
            return out if fuse else act(out)
    if pending_bias is not None:
        x = _add_channel_bias(x, pending_bias)
    return act(norm(x))


# --------------------------------------------------------------------------- #
# WanCausalConv3d and its cached call sites
# --------------------------------------------------------------------------- #


def causal_conv_forward(self: WanCausalConv3d, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
    padding = self._padding
    if not any(padding):
        # 1x1x1 convolutions: upstream still runs ``F.pad(x, [0] * 6)``, a full
        # fill + copy. Skipping it is the identity.
        return nn.Conv3d.forward(self, x)
    if _kernels_allowed(x) and (cache_x is None or (cache_x.device == x.device and cache_x.dtype == x.dtype)):
        assembled = dm.cat_pad_5d(x, cache_x if padding[4] > 0 else None, padding)
        if assembled is not None:
            return nn.Conv3d.forward(self, assembled)
    return WanCausalConv3d.forward(self, x, cache_x)


def _conv_without_bias(conv: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    """The cuDNN call ``nn.Conv3d.forward`` makes, minus ATen's separate ``add_(bias)``."""
    return F.conv3d(x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)


# Per conv module: {(x.shape, cache_frames, dtype, layout): cuDNN padding is bitwise
# identical to a pre-padded input}. ``False`` routes that shape to the padded path.
_SPATIAL_PAD_VERDICTS: WeakKeyDictionary[nn.Module, dict[tuple, bool]] = WeakKeyDictionary()
# Per conv module: {(x.shape, cache_frames, dtype): the convolution of the channels-last
# assembled input is bitwise identical to the channels-first one}.
_CONV_OUT_LAYOUT_VERDICTS: WeakKeyDictionary[nn.Module, dict[tuple, bool]] = WeakKeyDictionary()


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    int_dtype = torch.int16 if a.element_size() == 2 else torch.int32
    return bool(torch.equal(a.contiguous().view(int_dtype), b.contiguous().view(int_dtype)))


def _conv_with_spatial_padding(
    conv: nn.Conv3d,
    assembled: torch.Tensor,
    bias: torch.Tensor | None,
    verdicts: dict[tuple, bool],
    key: tuple,
) -> torch.Tensor:
    """Convolve the time-assembled input, letting cuDNN apply the spatial zero padding.

    Upstream pads the input tensor explicitly and convolves with ``padding=0``.
    Handing the spatial padding to cuDNN avoids materializing the padded copy,
    but is only bit-identical if cuDNN selects the same kernel for both
    formulations, so the first call for each (conv, shape) runs both and
    compares bitwise; later calls reuse the verdict.
    """
    pad_height, pad_width = conv._padding[2], conv._padding[0]
    fast = F.conv3d(assembled, conv.weight, bias, conv.stride, (0, pad_height, pad_width), conv.dilation, conv.groups)
    if key in verdicts:
        return fast
    padded = F.pad(assembled, (pad_width, pad_width, pad_height, pad_height, 0, 0))
    reference = F.conv3d(padded, conv.weight, bias, conv.stride, 0, conv.dilation, conv.groups)
    verdicts[key] = verdict = _bitwise_equal(fast, reference)
    if not verdict:
        logger.info(
            "cuDNN spatial padding is not bitwise identical to pre-padded input for %s %s; "
            "using the pre-padded path for this shape",
            type(conv).__name__,
            tuple(assembled.shape),
        )
        return reference
    return fast


def _layout_tag(x: torch.Tensor) -> str:
    return "channels_last" if x.shape[1] > 1 and x.stride(1) == 1 else "channels_first"


def _run_cached_causal_conv(
    conv: nn.Module,
    x: torch.Tensor,
    cache_list: list[Any],
    index: int,
    *,
    return_bias: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Run a causal conv against ``cache_list[index]`` and refresh that slot.

    Reproduces the upstream call-site pattern::

        cache_x = x[:, :, -CACHE_T:].clone()
        if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
            cache_x = cat([feat_cache[idx][:, :, -1:], cache_x], dim=2)   # or zeros for "Rep"
        x = conv(x, feat_cache[idx])
        feat_cache[idx] = cache_x

    The fused path stores the last ``CACHE_T`` frames of the assembled conv
    input instead. On the very first chunk that is ``[0, x]`` where upstream
    stores ``[x]``; the next conv input is identical either way because the
    causal padding shrinks by the cache length.

    With ``return_bias=True`` the fused path leaves the convolution bias unadded
    and returns it alongside the output so the consumer can fold it into its own
    kernel; the reference path returns ``(output, None)``.
    """
    cache = cache_list[index]
    is_repeat_marker = isinstance(cache, str)
    payload = cache if isinstance(cache, torch.Tensor) else None
    if (
        type(conv) is WanCausalConv3d
        and _kernels_allowed(x)
        and x.dim() == 5
        and (payload is None or (payload.device == x.device and payload.dtype == x.dtype))
    ):
        fold_bias = return_bias and conv.bias is not None
        bias = None if fold_bias else conv.bias
        returned_bias = conv.bias if fold_bias else None

        # Preferred: temporal concat only (aligned plane copies) + cuDNN spatial
        # padding, once verified bitwise for this (conv, shape).
        verdicts = _SPATIAL_PAD_VERDICTS.setdefault(conv, {})
        key = (tuple(x.shape), 0 if payload is None else payload.shape[2], x.dtype, _layout_tag(x))
        if verdicts.get(key, True):
            pair = dm.cat_time_5d(x, payload, conv._padding[4], keep_cache_frames=CACHE_T)
            if pair is not None:
                assembled, cache_list[index] = pair
                out = _conv_with_spatial_padding(conv, assembled, bias, verdicts, key)
                return (out, returned_bias) if return_bias else out

        pair = dm.cat_pad_5d(x, payload, conv._padding, keep_cache_frames=CACHE_T)
        if pair is not None:
            conv_input, cache_list[index] = pair
            out = F.conv3d(conv_input, conv.weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups)
            return (out, returned_bias) if return_bias else out

    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    if cache_x.shape[2] < CACHE_T and payload is not None:
        cache_x = torch.cat([payload[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
    elif cache_x.shape[2] < CACHE_T and is_repeat_marker:
        cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)

    out = conv(x) if payload is None else conv(x, payload)
    cache_list[index] = cache_x
    return (out, None) if return_bias else out


def _run_conv_out_channels_last(
    conv: nn.Module, x: torch.Tensor, cache_list: list[Any], index: int
) -> torch.Tensor | None:
    """``conv_out`` on a channels-last copy of its padded input, once verified bitwise; ``None`` to decline.

    In the lossless level the decoder runs channels-first and cuDNN transposes
    every convolution input to NDHWC internally. ``conv_out`` is the one
    convolution whose spatial padding cuDNN does not reproduce bitwise, so its
    input is assembled pre-padded anyway; assembling it channels-last directly
    (one fused transpose in the pixels kernel) saves the separate channels-first
    assembly and cuDNN's transpose. The output is made contiguous, so nothing
    downstream sees a layout change. The first call per (conv, shape) also runs
    the channels-first formulation and compares bitwise; a mismatch routes that
    shape back to the standard path forever.
    """
    cache = cache_list[index]
    payload = cache if isinstance(cache, torch.Tensor) else None
    if type(conv) is not WanCausalConv3d or not _kernels_allowed(x) or x.dim() != 5 or x.stride(1) == 1:
        return None
    if payload is not None and (payload.device != x.device or payload.dtype != x.dtype):
        return None
    verdicts = _CONV_OUT_LAYOUT_VERDICTS.setdefault(conv, {})
    key = (tuple(x.shape), 0 if payload is None else payload.shape[2], x.dtype)
    if not verdicts.get(key, True):
        return None
    pair = dm.cat_pad_5d(x, payload, conv._padding, keep_cache_frames=CACHE_T, channels_last_output=True)
    if pair is None:
        return None
    assembled, cache_list[index] = pair
    out = F.conv3d(assembled, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
    out = out.contiguous()
    if key in verdicts:
        return out
    # The channels-first path convolves exactly this tensor in NCDHW layout.
    reference = F.conv3d(
        assembled.contiguous(), conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups
    )
    verdicts[key] = verdict = _bitwise_equal(out, reference)
    if not verdict:
        logger.info(
            "channels-last conv_out input is not bitwise identical to the channels-first one for %s; "
            "using the channels-first path for this shape",
            tuple(assembled.shape),
        )
        return reference
    return out


def _residual_add(
    x: torch.Tensor,
    bias_x: torch.Tensor | None,
    h: torch.Tensor,
    bias_h: torch.Tensor | None,
) -> torch.Tensor:
    """``(x + bias_x) + (h + bias_h)`` where the biases are the un-added conv biases."""
    if bias_x is None and bias_h is None:
        return x + h
    if _kernels_allowed(x):
        out = dm.add_bias_residual(x, bias_x, h, bias_h)
        if out is not None:
            return out
    # Out-of-place ``x + bias`` rounds exactly like ATen's in-place ``add_``.
    if bias_x is not None:
        x = x + bias_x.view(1, -1, 1, 1, 1)
    if bias_h is not None:
        h = h + bias_h.view(1, -1, 1, 1, 1)
    return x + h


# --------------------------------------------------------------------------- #
# WanResample / WanUpsample
# --------------------------------------------------------------------------- #


def _interleave_time(x: torch.Tensor, batch: int, channels: int, frames: int, height: int, width: int) -> torch.Tensor:
    """Upstream ``reshape(b, 2, c, t, h, w)`` + ``stack(..., 3)`` + ``reshape`` as one strided copy.

    ``out[:, :, 2i] = x[:, :c, i]`` and ``out[:, :, 2i + 1] = x[:, c:, i]``; pure
    data movement that also preserves a channels_last_3d layout (``torch.stack``
    on the 6-D view always produced a contiguous tensor).
    """
    memory_format = torch.contiguous_format
    if channels > 1 and x.is_contiguous(memory_format=torch.channels_last_3d):
        memory_format = torch.channels_last_3d
    out = torch.empty(
        (batch, channels, frames * 2, height, width),
        dtype=x.dtype,
        device=x.device,
        memory_format=memory_format,
    )
    out[:, :, 0::2].copy_(x[:, :channels])
    out[:, :, 1::2].copy_(x[:, channels:])
    return out


def _merge_batch_and_frames(x: torch.Tensor) -> torch.Tensor:
    """``(b, c, t, h, w)`` -> ``(b * t, c, h, w)`` view, keeping a channels-last layout recognizable.

    For a channels_last_3d tensor with ``b * t == 1``, ``permute().reshape()`` hands
    the size-1 batch dimension an arbitrary stride that PyTorch's layout heuristic
    (``suggest_memory_format``) does not accept as channels_last, so ops without a
    weight to vote for the layout, such as the nearest upsample, would allocate
    NCHW output and force the next convolution to transpose it. Building the same
    view with canonical strides avoids that; the values are untouched.
    """
    b, c, t, h, w = x.shape
    if c > 1 and x.is_contiguous(memory_format=torch.channels_last_3d):
        return x.as_strided((b * t, c, h, w), (h * w * c, 1, w * c, c))
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


def _split_batch_and_frames(x: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
    """``(b * t, c, h, w)`` -> ``(b, c, t, h, w)`` view; see :func:`_merge_batch_and_frames`."""
    _, c, h, w = x.shape
    if c > 1 and x.is_contiguous(memory_format=torch.channels_last):
        return x.as_strided((batch, c, frames, h, w), (frames * h * w * c, 1, h * w * c, w * c, c))
    return x.view(batch, frames, c, h, w).permute(0, 2, 1, 3, 4)


def _is_upsample_conv_pair(resample: nn.Module) -> bool:
    """``nn.Sequential(WanUpsample, Conv2d)`` with a bias and zero padding, as the up-samplers are built."""
    if type(resample) is not nn.Sequential or len(resample) != 2:
        return False
    upsample, conv = resample[0], resample[1]
    return (
        isinstance(upsample, nn.Upsample)
        and type(conv) is nn.Conv2d
        and conv.bias is not None
        and conv.padding_mode == "zeros"
    )


def resample_forward(
    self: WanResample,
    x: torch.Tensor,
    feat_cache: list[Any] | None = None,
    feat_idx: list[int] | None = None,
    *,
    return_bias: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """``WanResample.forward``; with ``return_bias=True`` the upsampler's ``Conv2d`` bias may be left un-added.

    The caller then receives ``(output, bias)`` and must fold ``bias`` into its
    consumer (or add it); ``(output, None)`` means the bias is already applied.
    """
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
            x = _interleave_time(x, batch, channels, frames, height, width)

    frames = x.shape[2]
    x = _merge_batch_and_frames(x)
    pending_bias = None
    if return_bias and _is_upsample_conv_pair(self.resample) and _kernels_allowed(x):
        upsample, conv = self.resample[0], self.resample[1]
        x = F.conv2d(upsample(x), conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
        pending_bias = conv.bias
    else:
        x = self.resample(x)
    x = _split_batch_and_frames(x, batch, frames)

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
    return (x, pending_bias) if return_bias else x


def _is_nearest_2x(upsample: nn.Upsample) -> bool:
    """True when ``upsample`` doubles height and width by nearest-neighbour selection."""
    if upsample.size is not None or upsample.mode not in ("nearest", "nearest-exact"):
        return False
    scale = upsample.scale_factor
    if isinstance(scale, (tuple, list)):
        return len(scale) == 2 and all(float(factor) == 2.0 for factor in scale)
    return scale is not None and float(scale) == 2.0


def upsample_forward(self: WanUpsample, x: torch.Tensor) -> torch.Tensor:
    # nearest-exact only replicates input values, so ``x.float()`` + ``type_as``
    # is the identity; run the interpolation in the activation dtype.
    if x.dtype in _NORM_DTYPES:
        if _kernels_allowed(x) and _is_nearest_2x(self):
            out = up.upsample_nearest_2x(x)
            if out is not None:
                return out
        return nn.Upsample.forward(self, x)
    return WanUpsample.forward(self, x)


# --------------------------------------------------------------------------- #
# Blocks and decoder
# --------------------------------------------------------------------------- #


def residual_block_forward(
    self: WanResidualBlock,
    x: torch.Tensor,
    feat_cache: list[Any] | None = None,
    feat_idx: list[int] | None = None,
) -> torch.Tensor:
    if feat_idx is None:
        feat_idx = [0]
    shortcut = self.conv_shortcut
    residual_bias = None
    if (
        type(shortcut) is WanCausalConv3d
        and shortcut.bias is not None
        and not any(shortcut._padding)
        and _kernels_allowed(x)
    ):
        residual = _conv_without_bias(shortcut, x)
        residual_bias = shortcut.bias
    else:
        residual = shortcut(x)
    x = _norm_act(self.norm1, self.nonlinearity, x)
    # conv1 feeds only norm2. At the channels_last level the single-pass norm
    # kernel adds the conv bias itself (rounded like ATen's ``add_``), so the
    # convolution runs without it; the lossless level keeps the separate add
    # because ``vector_norm`` must see the same bytes as upstream.
    cfg = getattr(self, CFG_ATTR, None)
    conv1_bias = None
    if feat_cache is not None:
        index = feat_idx[0]
        if cfg is not None and cfg.channels_last:
            x, conv1_bias = _run_cached_causal_conv(self.conv1, x, feat_cache, index, return_bias=True)
        else:
            x = _run_cached_causal_conv(self.conv1, x, feat_cache, index)
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    x = self.dropout(_norm_act(self.norm2, self.nonlinearity, x, pending_bias=conv1_bias))
    conv2_bias = None
    if feat_cache is not None:
        index = feat_idx[0]
        x, conv2_bias = _run_cached_causal_conv(self.conv2, x, feat_cache, index, return_bias=True)
        feat_idx[0] += 1
    else:
        x = self.conv2(x)
    return _residual_add(x, conv2_bias, residual, residual_bias)


def residual_up_block_forward(
    self: WanResidualUpBlock,
    x: torch.Tensor,
    feat_cache: list[Any] | None = None,
    feat_idx: list[int] | None = None,
    first_chunk: bool = False,
) -> torch.Tensor:
    if feat_idx is None:
        feat_idx = [0]
    # Upstream clones ``x`` for the shortcut; nothing below mutates it in place.
    shortcut_source = x
    for resnet in self.resnets:
        if feat_cache is None:
            x = resnet(x)
        else:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)

    shortcut = self.avg_shortcut
    # The upsampler's Conv2d output feeds only ``x + DupUp3D(...)``, so its bias
    # is folded into that kernel (exact: rounded like ATen's ``add_`` first).
    fuse_shortcut = type(shortcut) is DupUp3D and _kernels_allowed(x) and x.dtype == shortcut_source.dtype
    conv_bias = None
    if self.upsampler is not None:
        if feat_cache is None:
            x = self.upsampler(x)
        elif fuse_shortcut and type(self.upsampler) is WanResample and hasattr(self.upsampler, CFG_ATTR):
            x, conv_bias = resample_forward(self.upsampler, x, feat_cache, feat_idx, return_bias=True)
        else:
            x = self.upsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)

    if shortcut is None:
        return x
    if fuse_shortcut:
        fused = dm.dup_up3d_add(
            x,
            shortcut_source,
            shortcut.factor_t,
            shortcut.factor_s,
            shortcut.repeats,
            first_chunk,
            main_bias=conv_bias,
        )
        if fused is not None:
            return fused
    if conv_bias is not None:
        x = _add_channel_bias(x, conv_bias)
    return x + shortcut(shortcut_source, first_chunk=first_chunk)


def decoder_forward(
    self: WanDecoder3d,
    x: torch.Tensor,
    feat_cache: list[Any] | None = None,
    feat_idx: list[int] | None = None,
    first_chunk: bool = False,
) -> torch.Tensor:
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

    x = _norm_act(self.norm_out, self.nonlinearity, x)
    if feat_cache is not None:
        index = feat_idx[0]
        # Channels-first (lossless) only: channels-last activations already take
        # the pixels kernel and transpose-free cuDNN path inside the standard call.
        out = _run_conv_out_channels_last(self.conv_out, x, feat_cache, index)
        if out is None:
            out = _run_cached_causal_conv(self.conv_out, x, feat_cache, index)
        x = out
        feat_idx[0] += 1
    else:
        x = self.conv_out(x)
    return x


__all__ = [
    "CFG_ATTR",
    "FastPathConfig",
    "causal_conv_forward",
    "decoder_forward",
    "is_diffusers_rms_norm",
    "resample_forward",
    "residual_block_forward",
    "residual_up_block_forward",
    "rms_norm_fastpath",
    "rms_norm_forward",
    "upsample_forward",
]
