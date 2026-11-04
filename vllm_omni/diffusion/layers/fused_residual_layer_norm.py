# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused residual add + LayerNorm + scale for MUSA, behind an opt-in env gate.

The reference path relies on the compiler to fuse ``hidden + gate * update`` with
``layer_norm(h) * scale`` into a single kernel. This module provides a hand-written
replacement that measures 1.30x faster on MUSA S5000 (158.8 us against 206.0 us
per call at T=5120, H=4096), but it changes the rounding schedule, so it stays
opt-in.

Numerics are chosen to match the reference path exactly, one rounding at a time:

* ``gate * update`` rounds to bf16, then the add rounds to bf16;
* the normalisation reduces in fp32, then rounds to bf16;
* the scale multiply is a bf16 multiply.

Measured against that reference the result stays within one bf16 ulp on every
element (max difference exactly one ulp) with an error profile against an fp64
truth identical to the reference; the hardware test asserts the looser two-ulp
bound so that it stays meaningful on other shapes. Folding the residual
and the scale multiply into one fp32 expression instead is *more* accurate, but it
is a different answer and deviates from the reference on 11% of elements.
"""

import os

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

#: Opt-in switch. Unset or false keeps the reference path byte for byte.
FUSED_RESIDUAL_NORM_ENV = "VLLM_OMNI_MUSA_FUSED_RESIDUAL_NORM"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: This op exists for MUSA. Other backends keep the compiler fusion, which is
#: already good there and has not been measured against this kernel.
# The kernels are plain Triton and run anywhere, but the arithmetic is qualified
# against one compiler's output: they reproduce MUSA's Inductor normalisation to two
# ulp, and nobody has checked what CUDA's Inductor produces for the same expression.
# Keep the list to the measured device and add cuda only with that evidence.
_QUALIFIED_DEVICE_TYPES = ("musa",)


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def fused_residual_norm_enabled() -> bool:
    """Whether the opt-in fused residual+LayerNorm path is requested."""
    return _enabled(FUSED_RESIDUAL_NORM_ENV)


def eligible(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
) -> bool:
    """Structural check only. Says nothing about whether the env gate is open.

    Safe to call while tracing: it reads shapes and dtypes, never data.
    """
    if not HAS_TRITON or hidden.device.type not in _QUALIFIED_DEVICE_TYPES:
        return False
    if hidden.dtype != torch.bfloat16 or hidden.dim() != 3 or not hidden.is_contiguous():
        return False
    if update.shape != hidden.shape or update.dtype != hidden.dtype:
        return False
    if not (update.is_contiguous() and gate.is_contiguous() and scale.is_contiguous()):
        return False
    if torch.is_grad_enabled():
        return False
    cols = hidden.shape[-1]
    rows = hidden.numel() // cols
    for tensor in (gate, scale):
        if tensor.numel() not in (cols, rows * cols):
            return False
        if tensor.dtype != hidden.dtype or tensor.device != hidden.device:
            return False
    return hidden.numel() > 0 and cols > 0


if HAS_TRITON:

    @triton.jit
    def _fused_residual_norm_kernel(
        hidden_ptr,
        update_ptr,
        gate_ptr,
        scale_ptr,
        residual_ptr,
        output_ptr,
        stride,
        eps,
        n_cols,
        per_token_gate: tl.constexpr,
        per_token_scale: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_size)
        mask = cols < n_cols
        off = row * stride + cols

        hidden = tl.load(hidden_ptr + off, mask=mask, other=0.0)
        update = tl.load(update_ptr + off, mask=mask, other=0.0)
        if per_token_gate:
            gate = tl.load(gate_ptr + off, mask=mask, other=0.0)
        else:
            gate = tl.load(gate_ptr + cols, mask=mask, other=0.0)

        # Two roundings, matching the reference: the product rounds to bf16 and
        # the sum rounds to bf16.
        update = (gate.to(tl.float32) * update.to(tl.float32)).to(tl.bfloat16)
        residual = hidden + update
        tl.store(residual_ptr + off, residual, mask=mask)

        values = residual.to(tl.float32)
        count = n_cols.to(tl.float32)
        mean = tl.sum(values, 0) / count
        centred = values - mean
        variance = tl.sum(centred * centred, 0) / count
        # LayerNorm rounds its output to bf16 before the scale multiply.
        normalised = (centred * tl.rsqrt(variance + eps)).to(tl.bfloat16)

        if per_token_scale:
            scale = tl.load(scale_ptr + off, mask=mask, other=0.0)
        else:
            scale = tl.load(scale_ptr + cols, mask=mask, other=0.0)
        output = (normalised.to(tl.float32) * scale.to(tl.float32)).to(tl.bfloat16)
        tl.store(output_ptr + off, output, mask=mask)


def _launch(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cols = hidden.shape[-1]
    rows = hidden.numel() // cols
    residual = torch.empty_like(hidden)
    output = torch.empty_like(hidden)
    block = triton.next_power_of_2(cols)
    _fused_residual_norm_kernel[(rows,)](
        hidden,
        update,
        gate,
        scale,
        residual,
        output,
        cols,
        eps,
        cols,
        per_token_gate=gate.numel() == rows * cols,
        per_token_scale=scale.numel() == rows * cols,
        block_size=block,
        num_warps=8,
    )
    return residual, output


if HAS_TRITON:
    # Inductor cannot inline a custom op, which is the point: it keeps the fused
    # kernel instead of re-deriving the expression and re-fusing it its own way.
    @torch.library.custom_op("vllm_omni::fused_residual_layer_norm", mutates_args=())
    def _fused_residual_layer_norm_op(
        hidden: torch.Tensor,
        update: torch.Tensor,
        gate: torch.Tensor,
        scale: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _launch(hidden, update, gate, scale, eps)

    @_fused_residual_layer_norm_op.register_fake
    def _(
        hidden: torch.Tensor,
        update: torch.Tensor,
        gate: torch.Tensor,
        scale: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device),
            torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device),
        )


def fused_residual_layer_norm(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
):
    """Return ``(hidden + gate * update, layer_norm(hidden + gate * update) * scale)``.

    Returns ``None`` when the env gate is closed or the inputs are not eligible, so
    callers keep the reference path without branching on platform details.
    """
    if not fused_residual_norm_enabled():
        return None
    if not eligible(hidden, update, gate, scale):
        return None
    # Called from inside a compiled region on purpose: registering this as a
    # custom op keeps it opaque to Inductor, which would otherwise re-derive the
    # expression and re-fuse it. `register_fake` covers tracing.
    return _fused_residual_layer_norm_op(hidden, update, gate, scale, eps)


def _eligible_multiplier(x: torch.Tensor, multiplier: torch.Tensor) -> bool:
    """Shared structural check for a per-channel or per-token broadcast tensor."""
    if multiplier.numel() not in (x.shape[-1], x.numel()):
        return False
    return multiplier.dtype == x.dtype and multiplier.device == x.device and multiplier.is_contiguous()


def eligible_norm_scale(x: torch.Tensor, multiplier: torch.Tensor) -> bool:
    """Structural check for normalise-and-scale. Says nothing about the env gate."""
    if not HAS_TRITON or x.device.type not in _QUALIFIED_DEVICE_TYPES:
        return False
    if x.dtype != torch.bfloat16 or x.dim() != 3 or not x.is_contiguous():
        return False
    if torch.is_grad_enabled():
        return False
    return x.numel() > 0 and x.shape[-1] > 0 and _eligible_multiplier(x, multiplier)


if HAS_TRITON:

    @triton.jit
    def _layer_norm_scale_kernel(
        input_ptr,
        multiplier_ptr,
        output_ptr,
        stride,
        eps,
        n_cols,
        per_token_multiplier: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_size)
        mask = cols < n_cols
        off = row * stride + cols

        values = tl.load(input_ptr + off, mask=mask, other=0.0).to(tl.float32)
        count = n_cols.to(tl.float32)
        mean = tl.sum(values, 0) / count
        centred = values - mean
        variance = tl.sum(centred * centred, 0) / count
        # LayerNorm rounds its output to bf16 before the multiply, as the
        # reference does through two separate kernels.
        normalised = (centred * tl.rsqrt(variance + eps)).to(tl.bfloat16)

        if per_token_multiplier:
            multiplier = tl.load(multiplier_ptr + off, mask=mask, other=0.0)
        else:
            multiplier = tl.load(multiplier_ptr + cols, mask=mask, other=0.0)
        output = (normalised.to(tl.float32) * multiplier.to(tl.float32)).to(tl.bfloat16)
        tl.store(output_ptr + off, output, mask=mask)


def _launch_norm_scale(x: torch.Tensor, multiplier: torch.Tensor, eps: float) -> torch.Tensor:
    cols = x.shape[-1]
    rows = x.numel() // cols
    output = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    _layer_norm_scale_kernel[(rows,)](
        x,
        multiplier,
        output,
        cols,
        eps,
        cols,
        per_token_multiplier=multiplier.numel() == x.numel(),
        block_size=block,
        num_warps=8,
    )
    return output


if HAS_TRITON:

    @torch.library.custom_op("vllm_omni::fused_layer_norm_scale", mutates_args=())
    def _fused_layer_norm_scale_op(x: torch.Tensor, multiplier: torch.Tensor, eps: float) -> torch.Tensor:
        return _launch_norm_scale(x, multiplier, eps)

    @_fused_layer_norm_scale_op.register_fake
    def _(x: torch.Tensor, multiplier: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def fused_layer_norm_scale(x: torch.Tensor, multiplier: torch.Tensor, eps: float):
    """Return ``layer_norm(x) * multiplier`` in one kernel, or ``None`` to fall back."""
    if not fused_residual_norm_enabled():
        return None
    if not eligible_norm_scale(x, multiplier):
        return None
    return _fused_layer_norm_scale_op(x, multiplier, eps)
