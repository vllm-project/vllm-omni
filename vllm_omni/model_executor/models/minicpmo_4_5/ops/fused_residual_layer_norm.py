# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# ruff: noqa: N803

"""Fused ``act(layer_norm(residual + gate * y) * weight + bias)`` (optional conv taps)."""

import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON, tl, triton

_ACTIVATIONS = {None: 0, "mish": 1}


def _use_triton(tensor: torch.Tensor) -> bool:
    """Triton on CUDA tensors; everything else runs the native path."""
    return HAS_TRITON and tensor.is_cuda


@triton.jit
def _residual_layer_norm_kernel(
    out_ptr,
    residual_out_ptr,
    x_ptr,
    y_ptr,
    gate_ptr,
    y_bias_ptr,
    weight_ptr,
    bias_ptr,
    frames,
    channels,
    stride_on,
    stride_out_t,
    stride_rn,
    stride_rt,
    stride_xn,
    stride_xt,
    stride_yn,
    stride_yt,
    eps,
    HAS_X: tl.constexpr,
    HAS_Y: tl.constexpr,
    TAPS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    HAS_Y_BIAS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STORE_RESIDUAL: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    n = row // frames
    t = row % frames
    cols = tl.arange(0, BLOCK_C)
    mask = cols < channels

    value = tl.zeros((BLOCK_C,), dtype=tl.float32)
    if HAS_Y:
        y = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for k in tl.static_range(TAPS):
            y += tl.load(
                y_ptr + n * stride_yn + (t + k) * stride_yt + k * channels + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        if HAS_Y_BIAS:
            y += tl.load(y_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_GATE:
            y = tl.load(gate_ptr + cols, mask=mask, other=0.0).to(tl.float32) * y
        value = y
    if HAS_X:
        value += tl.load(x_ptr + n * stride_xn + t * stride_xt + cols, mask=mask, other=0.0).to(tl.float32)
    if STORE_RESIDUAL:
        tl.store(residual_out_ptr + n * stride_rn + t * stride_rt + cols, value, mask=mask)

    mean = tl.sum(value, axis=0) / channels
    centered = tl.where(mask, value - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / channels
    normalized = centered * tl.rsqrt(variance + eps)
    if HAS_WEIGHT:
        normalized = normalized * tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    if HAS_BIAS:
        normalized = normalized + tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    if ACTIVATION == 1:
        e = tl.exp(tl.minimum(normalized, 20.0))
        numerator = e * (e + 2.0)
        factor = tl.where(normalized > 20.0, 1.0, numerator / (numerator + 2.0))
        normalized = normalized * factor
    tl.store(out_ptr + n * stride_on + t * stride_out_t + cols, normalized, mask=mask)


def _check_rows(name: str, tensor: torch.Tensor, rows: int, frames: int, width: int) -> None:
    if tensor.dim() != 3 or int(tensor.shape[0]) != rows or int(tensor.shape[1]) < frames:
        raise ValueError(f"residual_layer_norm: {name} must be (N, >= T, C), got {tuple(tensor.shape)}")
    if int(tensor.shape[2]) != width or (width > 1 and tensor.stride(2) != 1):
        raise ValueError(f"residual_layer_norm: {name} needs {width} contiguous channels, got {tuple(tensor.shape)}")


def _native(
    *,
    residual,
    y,
    gate,
    weight,
    bias,
    eps,
    taps,
    y_bias,
    activation,
    residual_out,
    out,
    frames,
    channels,
):
    value = None
    if y is not None:
        value = sum(y[:, k : k + frames, k * channels : (k + 1) * channels] for k in range(taps))
        if y_bias is not None:
            value = value + y_bias
        if gate is not None:
            value = gate * value
    if residual is not None:
        value = residual[:, :frames] if value is None else residual[:, :frames] + value
    if residual_out is not None:
        residual_out.copy_(value)
    result = F.layer_norm(value, (channels,), None, None, eps)
    if weight is not None:
        result = result * weight
    if bias is not None:
        result = result + bias
    if activation == "mish":
        result = F.mish(result)
    out.copy_(result)
    return out


def residual_layer_norm(
    residual: torch.Tensor | None,
    y: torch.Tensor | None = None,
    *,
    gate: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
    taps: int = 1,
    y_bias: torch.Tensor | None = None,
    activation: str | None = None,
    residual_out: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``act(layer_norm(residual + gate * y) * weight + bias)`` in one pass per row."""
    if activation not in _ACTIVATIONS:
        raise ValueError(f"residual_layer_norm: unsupported activation {activation!r}")
    if residual is None and y is None:
        raise ValueError("residual_layer_norm needs a residual or a y")
    reference = residual if residual is not None else y
    rows = int(reference.shape[0])
    if residual is not None:
        frames, channels = int(residual.shape[1]), int(residual.shape[2])
        _check_rows("residual", residual, rows, frames, channels)
    else:
        channels = int(y.shape[2]) // taps
        frames = int(y.shape[1]) - (taps - 1)
    if y is not None:
        _check_rows("y", y, rows, frames + taps - 1, taps * channels)
    if residual_out is not None:
        _check_rows("residual_out", residual_out, rows, frames, channels)
    if out is None:
        out = torch.empty((rows, frames, channels), device=reference.device, dtype=reference.dtype)
    _check_rows("out", out, rows, frames, channels)

    if not _use_triton(reference):
        return _native(
            residual=residual,
            y=y,
            gate=gate,
            weight=weight,
            bias=bias,
            eps=eps,
            taps=taps,
            y_bias=y_bias,
            activation=activation,
            residual_out=residual_out,
            out=out,
            frames=frames,
            channels=channels,
        )
    if rows * frames == 0:
        return out
    dummy = reference
    _residual_layer_norm_kernel[(rows * frames,)](
        out,
        residual_out if residual_out is not None else dummy,
        residual if residual is not None else dummy,
        y if y is not None else dummy,
        gate if gate is not None else dummy,
        y_bias if y_bias is not None else dummy,
        weight if weight is not None else dummy,
        bias if bias is not None else dummy,
        frames,
        channels,
        out.stride(0),
        out.stride(1),
        residual_out.stride(0) if residual_out is not None else 0,
        residual_out.stride(1) if residual_out is not None else 0,
        residual.stride(0) if residual is not None else 0,
        residual.stride(1) if residual is not None else 0,
        y.stride(0) if y is not None else 0,
        y.stride(1) if y is not None else 0,
        float(eps),
        HAS_X=residual is not None,
        HAS_Y=y is not None,
        TAPS=int(taps),
        HAS_GATE=gate is not None,
        HAS_Y_BIAS=y_bias is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        STORE_RESIDUAL=residual_out is not None,
        ACTIVATION=_ACTIVATIONS[activation],
        BLOCK_C=triton.next_power_of_2(channels),
        num_warps=4 if channels <= 1024 else 8,
    )
    return out
