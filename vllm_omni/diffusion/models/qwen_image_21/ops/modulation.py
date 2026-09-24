# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fused modulation and gated-residual pointwise pairs for the Qwen-Image 2.1 block.

The block reads one shared ``modulation`` tensor and slices a scale and a gate out of it per
sublayer. Both consumers first expand the compact per-sample parameters over the token axis
and then apply two more pointwise steps::

    scale path: where(mask, scale[b], scale[t=0]) -> (1 + scale) -> x * scale
    gate  path: where(mask, gate[b],  gate[t=0])  -> tanh(gate)  -> * sublayer -> + residual

That is seven full-width kernels and five ``[batch, seq, channels]`` intermediates per block
per step, all of them pure elementwise work around an otherwise untouched native LayerNorm.
These kernels read the compact parameter row directly and emit only the final tensor.

``_select_modulation_rows`` is ported from diffusers'
``transformer_qwenimage21._select_modulation_rows``: with ``causal_condition`` the parameters
carry ``batch_size + 1`` rows, rows ``[0, batch)`` come from the sampled timestep, and the
trailing row is the ``t = 0`` row that text and condition-image (prefix) tokens read. The
mask is one-dimensional over the local sequence and is shared across the batch.

Rounding contract, matching the eager block::

    one_plus = round_bf16(1 + scale)            # BF16 add
    out      = round_bf16(x * one_plus)         # BF16 multiply
    tanh_g   = round_bf16(tanhf(gate))
    out      = round_bf16(residual + round_bf16(tanh_g * sublayer))
"""

from __future__ import annotations

import torch
from torch.library import Library
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.diffusion.layers.numerics import (
    add_rn_f32,
    mul_rn_f32,
    round_bf16_to_fp32,
    tanh_f32,
)
from vllm_omni.platforms import current_omni_platform

_BLOCK = 1024
_SUPPORTED_DTYPE = torch.bfloat16
_MODULATE_OP = "qwen_image_21_modulate"
_GATED_RESIDUAL_OP = "qwen_image_21_gated_residual"


@triton.jit
def _select_row(batch, token, mask_ptr, num_rows, has_mask: tl.constexpr):
    """Row each token reads: its own sample, or the trailing t=0 row for prefix tokens."""
    if has_mask:
        return tl.where(tl.load(mask_ptr + token), batch, num_rows - 1)
    return batch


@triton.jit
def _modulate_kernel(
    x_ptr,
    params_ptr,
    mask_ptr,
    out_ptr,
    x_stride_b,
    x_stride_s,
    params_stride_r,
    out_stride_b,
    out_stride_s,
    num_rows,
    channels: tl.constexpr,
    block: tl.constexpr,
    has_mask: tl.constexpr,
):
    batch = tl.program_id(0)
    token = tl.program_id(1)
    offset = tl.program_id(2) * block + tl.arange(0, block)
    live = offset < channels

    row = _select_row(batch, token, mask_ptr, num_rows, has_mask)
    x = tl.load(x_ptr + batch * x_stride_b + token * x_stride_s + offset, mask=live, other=0.0).to(tl.float32)
    params = tl.load(params_ptr + row * params_stride_r + offset, mask=live, other=0.0).to(tl.float32)
    one_plus = round_bf16_to_fp32(add_rn_f32(1.0, params))
    tl.store(
        out_ptr + batch * out_stride_b + token * out_stride_s + offset,
        mul_rn_f32(x, one_plus),
        mask=live,
    )


@triton.jit
def _gated_residual_kernel(
    residual_ptr,
    sublayer_ptr,
    params_ptr,
    mask_ptr,
    out_ptr,
    stride_b,
    stride_s,
    params_stride_r,
    num_rows,
    channels: tl.constexpr,
    block: tl.constexpr,
    has_mask: tl.constexpr,
):
    batch = tl.program_id(0)
    token = tl.program_id(1)
    offset = tl.program_id(2) * block + tl.arange(0, block)
    live = offset < channels
    element = batch * stride_b + token * stride_s + offset

    row = _select_row(batch, token, mask_ptr, num_rows, has_mask)
    params = tl.load(params_ptr + row * params_stride_r + offset, mask=live, other=0.0).to(tl.float32)
    gate = round_bf16_to_fp32(tanh_f32(params))
    sublayer = tl.load(sublayer_ptr + element, mask=live, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + element, mask=live, other=0.0).to(tl.float32)
    gated = round_bf16_to_fp32(mul_rn_f32(gate, sublayer))
    tl.store(out_ptr + element, add_rn_f32(residual, gated), mask=live)


def select_modulation_rows(params: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    """Broadcast per-sample modulation ``params`` over the token axis (eager reference)."""
    if token_mask is None:
        return params.unsqueeze(1)
    real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
    return torch.where(token_mask.view(1, -1, 1), real, zero)


def _reference_modulate(x: torch.Tensor, params: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    return x * (1 + select_modulation_rows(params, token_mask))


def _reference_gated_residual(
    residual: torch.Tensor,
    sublayer: torch.Tensor,
    params: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    return residual + select_modulation_rows(params, token_mask).tanh() * sublayer


def _supported(x: torch.Tensor, params: torch.Tensor, token_mask: torch.Tensor | None) -> bool:
    channels = x.shape[-1]
    return (
        HAS_TRITON
        and current_omni_platform.is_cuda()
        and x.is_cuda
        and x.dtype is _SUPPORTED_DTYPE
        and x.ndim == 3
        and x.is_contiguous()
        and params.dtype is _SUPPORTED_DTYPE
        and params.ndim == 2
        and params.shape[-1] == channels
        and params.stride(-1) == 1
        # With a mask the trailing row is the t=0 row; without one every sample reads
        # its own row, so the parameter count is exactly the batch size.
        and params.shape[0] == (x.shape[0] + 1 if token_mask is not None else x.shape[0])
        and (token_mask is None or (token_mask.dtype is torch.bool and token_mask.shape == (x.shape[1],)))
    )


def _check_rows(x: torch.Tensor, params: torch.Tensor, token_mask: torch.Tensor | None) -> None:
    """Reject a parameter count that matches neither the masked nor the mask-free mode."""
    required = x.shape[0] + 1 if token_mask is not None else x.shape[0]
    if params.ndim != 2 or params.shape[0] not in (required, 1):
        raise ValueError(
            f"expected {required} modulation rows for batch {x.shape[0]} "
            f"with token_mask {'set' if token_mask is not None else 'unset'}, got {tuple(params.shape)}"
        )


def _grid(x: torch.Tensor) -> tuple[int, int, int]:
    return (x.shape[0], x.shape[1], triton.cdiv(x.shape[-1], _BLOCK))


def _launch_modulate(x: torch.Tensor, params: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    out = torch.empty_like(x)
    if out.numel() == 0:
        return out
    _modulate_kernel[_grid(x)](
        x,
        params,
        token_mask,
        out,
        x.stride(0),
        x.stride(1),
        params.stride(0),
        out.stride(0),
        out.stride(1),
        params.shape[0],
        channels=x.shape[-1],
        block=_BLOCK,
        has_mask=token_mask is not None,
    )
    return out


def _launch_gated_residual(
    residual: torch.Tensor, sublayer: torch.Tensor, params: torch.Tensor, token_mask: torch.Tensor | None
) -> torch.Tensor:
    out = torch.empty_like(residual)
    if out.numel() == 0:
        return out
    _gated_residual_kernel[_grid(residual)](
        residual,
        sublayer,
        params,
        token_mask,
        out,
        residual.stride(0),
        residual.stride(1),
        params.stride(0),
        params.shape[0],
        channels=residual.shape[-1],
        block=_BLOCK,
        has_mask=token_mask is not None,
    )
    return out


def _modulate_fake(x: torch.Tensor, params: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    del params, token_mask
    return torch.empty_like(x)


def _gated_residual_fake(
    residual: torch.Tensor,
    sublayer: torch.Tensor,
    params: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    del sublayer, params, token_mask
    return torch.empty_like(residual)


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, _MODULATE_OP):
    direct_register_custom_op(
        op_name=_MODULATE_OP,
        op_func=_launch_modulate,
        fake_impl=_modulate_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )
if not hasattr(torch.ops.vllm_omni, _GATED_RESIDUAL_OP):
    direct_register_custom_op(
        op_name=_GATED_RESIDUAL_OP,
        op_func=_launch_gated_residual,
        fake_impl=_gated_residual_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def apply_modulation(x: torch.Tensor, params: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    """``x * (1 + select_modulation_rows(params, token_mask))`` in one launch."""
    if x.ndim != 3:
        raise ValueError(f"x must be [batch, seq, channels], got {tuple(x.shape)}")
    _check_rows(x, params, token_mask)
    if not _supported(x, params, token_mask):
        return _reference_modulate(x, params, token_mask)
    return torch.ops.vllm_omni.qwen_image_21_modulate(x, params, token_mask)


def apply_gated_residual(
    residual: torch.Tensor,
    sublayer: torch.Tensor,
    params: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    """``residual + tanh(select_modulation_rows(params, token_mask)) * sublayer`` in one launch."""
    if residual.shape != sublayer.shape:
        raise ValueError(f"residual and sublayer must match, got {tuple(residual.shape)} vs {tuple(sublayer.shape)}")
    _check_rows(residual, params, token_mask)
    if not _supported(residual, params, token_mask):
        return _reference_gated_residual(residual, sublayer, params, token_mask)
    return torch.ops.vllm_omni.qwen_image_21_gated_residual(residual, sublayer, params, token_mask)


__all__ = ["apply_gated_residual", "apply_modulation", "select_modulation_rows"]
