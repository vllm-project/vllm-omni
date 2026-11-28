# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Activation producers fused with dynamic per-token FP8 quantization.

If the next linear layer accepts a pre-quantized dynamic per-token FP8 input,
the activation in front of it is quantized in the same kernel that computes
it, and the layer skips its own input quantization. Two producers:

* ``silu_mul_fp8_per_token(gate, up, linear)``: ``silu(gate) * up`` with
  separate gate and up tensors (MLPs whose gate and up projections are not
  merged);
* ``add_rms_norm_fp8_per_token(x, residual, norm, linears)``:
  ``residual + x``, then RMSNorm of the sum, quantized for ``linears``; the
  sum is returned as the new residual.

Each is one single-pass Triton kernel per row (the row stays in registers),
replacing the separate activation or norm pass and the two-pass quantization
(amax, then scale) of the unfused path. The rounding points of the unfused
path are kept: the sum, SiLU, product and normalized output are rounded to the
activation dtype before quantization. Both return the plain unfused result
when no fused kernel applies, so a forward can call them unconditionally.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.library import Library
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation, get_input_quant_key
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max, kFp8DynamicTokenSym
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# Same floor as vLLM's dynamic per-token FP8 quantization, so scales match the unfused path.
_FP8_MAX = get_fp8_min_max()[1]
_FP8_MIN_SCALE = 1.0 / (_FP8_MAX * 512.0)
_MAX_HIDDEN = 16384


@triton.jit
def _silu_mul_fp8_kernel(g_ptr, u_ptr, q_ptr, s_ptr, n, fp8_max, min_scale, block: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, block)
    m = offs < n
    dt = g_ptr.dtype.element_ty
    g = tl.load(g_ptr + row * n + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + row * n + offs, mask=m, other=0.0).to(tl.float32)
    silu = (g * tl.sigmoid(g)).to(dt).to(tl.float32)
    y = (silu * u).to(dt).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(y), axis=0) / fp8_max, min_scale)
    q = tl.clamp(y / s, -fp8_max, fp8_max)
    tl.store(q_ptr + row * n + offs, q.to(q_ptr.dtype.element_ty), mask=m)
    tl.store(s_ptr + row, s)


@triton.jit
def _add_rms_norm_fp8_kernel(
    x_ptr, r_ptr, w_ptr, r_out_ptr, q_ptr, s_ptr, n, eps, fp8_max, min_scale, block: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, block)
    m = offs < n
    dt = x_ptr.dtype.element_ty
    x = tl.load(x_ptr + row * n + offs, mask=m, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * n + offs, mask=m, other=0.0).to(tl.float32)
    h = (x + r).to(dt)
    tl.store(r_out_ptr + row * n + offs, h, mask=m)
    h = h.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(h * h, axis=0) / n + eps)
    w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
    y = (w * (h * rstd)).to(dt).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(y), axis=0) / fp8_max, min_scale)
    q = tl.clamp(y / s, -fp8_max, fp8_max)
    tl.store(q_ptr + row * n + offs, q.to(q_ptr.dtype.element_ty), mask=m)
    tl.store(s_ptr + row, s)


def _silu_mul_fp8_impl(gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = gate.shape[-1]
    g, u = gate.reshape(-1, n).contiguous(), up.reshape(-1, n).contiguous()
    q = torch.empty(gate.shape, dtype=current_platform.fp8_dtype(), device=gate.device)
    s = torch.empty(g.shape[0], 1, dtype=torch.float32, device=gate.device)
    if g.shape[0]:
        _silu_mul_fp8_kernel[(g.shape[0],)](
            g, u, q, s, n, _FP8_MAX, _FP8_MIN_SCALE, block=triton.next_power_of_2(n), num_warps=32
        )
    return q, s


def _silu_mul_fp8_fake(gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = gate.numel() // gate.shape[-1]
    return (
        torch.empty(gate.shape, dtype=current_platform.fp8_dtype(), device=gate.device),
        torch.empty(rows, 1, dtype=torch.float32, device=gate.device),
    )


def _add_rms_norm_fp8_impl(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = x.shape[-1]
    x2, r2 = x.reshape(-1, n).contiguous(), residual.reshape(-1, n).contiguous()
    r_out = torch.empty_like(x)
    q = torch.empty(x.shape, dtype=current_platform.fp8_dtype(), device=x.device)
    s = torch.empty(x2.shape[0], 1, dtype=torch.float32, device=x.device)
    if x2.shape[0]:
        _add_rms_norm_fp8_kernel[(x2.shape[0],)](
            x2, r2, weight, r_out, q, s, n, eps, _FP8_MAX, _FP8_MIN_SCALE, block=triton.next_power_of_2(n), num_warps=16
        )
    return r_out, q, s


def _add_rms_norm_fp8_fake(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = x.numel() // x.shape[-1]
    return (
        torch.empty_like(x),
        torch.empty(x.shape, dtype=current_platform.fp8_dtype(), device=x.device),
        torch.empty(rows, 1, dtype=torch.float32, device=x.device),
    )


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "silu_mul_fp8_per_token"):
    direct_register_custom_op(
        op_name="silu_mul_fp8_per_token",
        op_func=_silu_mul_fp8_impl,
        fake_impl=_silu_mul_fp8_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )
if not hasattr(torch.ops.vllm_omni, "add_rms_norm_fp8_per_token"):
    direct_register_custom_op(
        op_name="add_rms_norm_fp8_per_token",
        op_func=_add_rms_norm_fp8_impl,
        fake_impl=_add_rms_norm_fp8_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def _fusable(x: torch.Tensor, linears: Sequence[torch.nn.Module]) -> bool:
    return (
        HAS_TRITON
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.shape[-1] <= _MAX_HIDDEN
        and all(get_input_quant_key(linear) == kFp8DynamicTokenSym for linear in linears)
    )


def _as_quantized(q: torch.Tensor, s: torch.Tensor, dtype: torch.dtype) -> QuantizedActivation:
    return QuantizedActivation(data=q, scale=s, orig_dtype=dtype, orig_shape=q.shape, quant_key=kFp8DynamicTokenSym)


def silu_mul_fp8_per_token(
    gate: torch.Tensor, up: torch.Tensor, linear: torch.nn.Module
) -> torch.Tensor | QuantizedActivation:
    """``silu(gate) * up``, quantized for ``linear`` when it consumes dynamic per-token FP8."""
    if not _fusable(gate, [linear]):
        return F.silu(gate) * up
    q, s = torch.ops.vllm_omni.silu_mul_fp8_per_token(gate, up)
    return _as_quantized(q, s, gate.dtype)


def add_rms_norm_fp8_per_token(
    x: torch.Tensor, residual: torch.Tensor, norm: torch.nn.Module, linears: Sequence[torch.nn.Module]
) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
    """Return ``(norm(residual + x), residual + x)``; the first is quantized when all ``linears`` consume it.

    ``norm`` is an RMSNorm with ``weight`` and ``variance_epsilon``; one quantized
    activation is shared by every linear in ``linears``.
    """
    if not _fusable(x, linears):
        return norm(x, residual)
    r_out, q, s = torch.ops.vllm_omni.add_rms_norm_fp8_per_token(x, residual, norm.weight, float(norm.variance_epsilon))
    return _as_quantized(q, s, x.dtype), r_out
