# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gated residual add + LayerNorm (+ modulation) + MXFP8 quantization in one pass.

A DiT block runs ``residual = x + y * gate``, then ``LayerNorm(residual) * w + b``
(``w = 1 + scale``, ``b = shift`` for adaptive norms) into an MXFP8 linear, which
quantizes its input. Unfused that is three passes over the [tokens, hidden]
activation (gated add, normalization, quantization); ``fused_add_layernorm_quant``
does it in one kernel (deepklox ``fused_add_layer_norm_quant``) and hands the
linear a ``QuantizedActivation``.
"""

import torch
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

try:
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
        get_input_quant_key,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
except ImportError:  # older vLLM without the fusion contract
    QuantizedActivation = None  # type: ignore[assignment]
    get_input_quant_key = None  # type: ignore[assignment]
    kMxfp8Dynamic = None  # noqa: N816

_MX_GROUP = 32
_K_ALIGN = 128  # the deepklox kernel's row-width requirement


def _deepklox_kernel():
    if not current_omni_platform.is_xpu():
        return None
    try:
        import deepklox._C as _DK  # noqa: F401
    except ImportError:
        return None
    return getattr(_DK, "fused_add_layer_norm_quant", None)


_KERNEL = _deepklox_kernel()
_FUSED_ADD_LN_MXFP8 = _KERNEL is not None

if _FUSED_ADD_LN_MXFP8:
    # The mutated argument is the caller's tensor itself, not a 2-D view of it:
    # inductor's auto-functionalization of a mutated view fails to decompose
    # under dynamic shapes (the regional compile of the DiT blocks).
    @torch.library.custom_op("vllm_omni::fused_add_layer_norm_mxfp8", mutates_args=("x",), device_types="xpu")
    def _fused_add_layer_norm_mxfp8(
        x: torch.Tensor,
        y: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = x.shape[-1]
        q, s, _ = _KERNEL(x.view(-1, hidden), y.reshape(-1, hidden), gate, weight, bias, eps)
        return q, s

    @_fused_add_layer_norm_mxfp8.register_fake
    def _(x, y, gate, weight, bias, eps):
        rows = x.numel() // x.shape[-1]
        return (
            x.new_empty((rows, x.shape[-1]), dtype=torch.float8_e4m3fn),
            x.new_empty((rows, x.shape[-1] // _MX_GROUP), dtype=torch.float8_e8m0fnu),
        )

    # Two token segments with their own gate / weight / bias rows ([2, hidden] each): rows
    # [0, seg) take row 0, rows [seg, end) row 1 (Wan S2V modulates its noisy tokens and its
    # reference / motion tokens separately). One kernel call per non-empty segment on its
    # contiguous row range of the caller's tensor, outputs concatenated in row order.
    @torch.library.custom_op("vllm_omni::fused_add_layer_norm_mxfp8_2seg", mutates_args=("x",), device_types="xpu")
    def _fused_add_layer_norm_mxfp8_2seg(
        x: torch.Tensor,
        y: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        seg: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = x.shape[-1]
        x2 = x.view(-1, hidden)
        y2 = y.reshape(-1, hidden)
        rows = x2.shape[0]
        qs, ss = [], []
        for i, (lo, hi) in enumerate(((0, seg), (seg, rows))):
            if hi > lo:
                q, s, _ = _KERNEL(x2[lo:hi], y2[lo:hi], gate[i], weight[i], bias[i], eps)
                qs.append(q)
                ss.append(s)
        if len(qs) == 1:
            return qs[0], ss[0]
        return torch.cat(qs), torch.cat(ss)

    @_fused_add_layer_norm_mxfp8_2seg.register_fake
    def _(x, y, gate, weight, bias, seg, eps):
        rows = x.numel() // x.shape[-1]
        return (
            x.new_empty((rows, x.shape[-1]), dtype=torch.float8_e4m3fn),
            x.new_empty((rows, x.shape[-1] // _MX_GROUP), dtype=torch.float8_e8m0fnu),
        )


def _as_row_vector(t: torch.Tensor | None, hidden: int, like: torch.Tensor, fill: float) -> torch.Tensor | None:
    """[hidden] view of a broadcast modulation vector; None if it is not one."""
    if t is None:
        return torch.full((hidden,), fill, dtype=like.dtype, device=like.device)
    if t.numel() != hidden:
        return None
    t = t.reshape(hidden)
    if t.dtype != like.dtype:
        t = t.to(like.dtype)
    return t.contiguous()


def can_fuse_add_layernorm_quant(
    next_linear: torch.nn.Module,
    x: torch.Tensor,
    *modulation: torch.Tensor | None,
) -> bool:
    """Whether ``fused_add_layernorm_quant`` will take the fused path.

    Callers that must copy ``x`` before the in-place residual update check this
    first so the unfused path pays no copy.
    """
    if (
        not _FUSED_ADD_LN_MXFP8
        or get_input_quant_key is None
        or get_input_quant_key(next_linear) != kMxfp8Dynamic
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.dim() < 2
        or x.shape[-1] % _K_ALIGN != 0
        or not x.is_contiguous()
    ):
        return False
    hidden = x.shape[-1]
    # Per-batch modulation (batch > 1 with distinct vectors) is not a [hidden]
    # row vector; the kernel takes one weight/bias/gate vector per call.
    return all(m is None or m.numel() == hidden for m in modulation)


_logged_active = False


def fused_add_layernorm_quant(
    next_linear: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor | None,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> "tuple[torch.Tensor, QuantizedActivation]":
    """``x += y * gate``; return ``(x, LayerNorm(x) * weight + bias)`` quantized for ``next_linear``.

    ``x`` is updated in place and returned as the residual stream, so pass a
    tensor the caller owns. ``gate``/``weight``/``bias`` are [hidden] vectors
    (or broadcastable to one); None means 1, 1 and 0. Only valid when
    ``can_fuse_add_layernorm_quant`` returned True for the same arguments.
    """
    global _logged_active
    hidden = x.shape[-1]
    if not y.is_contiguous():
        y = y.contiguous()
    gate_v = _as_row_vector(gate, hidden, x, 1.0)
    weight_v = _as_row_vector(weight, hidden, x, 1.0)
    bias_v = _as_row_vector(bias, hidden, x, 0.0)
    if not _logged_active:
        _logged_active = True
        logger.info(
            "fused residual add + LayerNorm + MXFP8 activation quantization active for %s",
            type(next_linear).__name__,
        )
    q, scales = torch.ops.vllm_omni.fused_add_layer_norm_mxfp8(x, y, gate_v, weight_v, bias_v, eps)
    if getattr(next_linear, "mxfp8_block_major_a_scale", False):
        # The SYCL-TLA GEMM reads [K/32, M] storage directly and would otherwise
        # transpose-copy the row-major scales itself.
        scales = scales.t().contiguous().t()
    return x, QuantizedActivation(
        data=q,
        scale=scales,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kMxfp8Dynamic,
    )


def _as_segment_rows(t: torch.Tensor | None, hidden: int, like: torch.Tensor, fill: float) -> torch.Tensor | None:
    """[2, hidden] per-segment rows of a modulation tensor; None if it is neither one shared
    [hidden] vector nor one [hidden] vector per segment."""
    if t is None:
        return torch.full((2, hidden), fill, dtype=like.dtype, device=like.device)
    if t.numel() == hidden:
        t = t.reshape(1, hidden).expand(2, hidden)
    elif t.numel() == 2 * hidden:
        t = t.reshape(2, hidden)
    else:
        return None
    if t.dtype != like.dtype:
        t = t.to(like.dtype)
    return t.contiguous()


def can_fuse_add_layernorm_quant_2seg(
    next_linear: torch.nn.Module,
    x: torch.Tensor,
    *modulation: torch.Tensor | None,
) -> bool:
    """Whether ``fused_add_layernorm_quant_2seg`` will take the fused path.

    ``x`` is one sequence ([1, tokens, hidden]); each modulation tensor is one [hidden]
    vector or one per segment ([..., 2, hidden] with 2 * hidden elements).
    """
    if not can_fuse_add_layernorm_quant(next_linear, x) or x.dim() != 3 or x.shape[0] != 1:
        return False
    hidden = x.shape[-1]
    return all(m is None or m.numel() in (hidden, 2 * hidden) for m in modulation)


def fused_add_layernorm_quant_2seg(
    next_linear: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor | None,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    seg: int,
    eps: float,
) -> "tuple[torch.Tensor, QuantizedActivation]":
    """``fused_add_layernorm_quant`` with separate modulation for tokens ``[0, seg)`` and ``[seg, end)``.

    ``gate``/``weight``/``bias`` hold the two segments' vectors ([..., 2, hidden]) or one
    vector shared by both; None means 1, 1 and 0. ``x`` is updated in place. Only valid when
    ``can_fuse_add_layernorm_quant_2seg`` returned True for the same arguments.
    """
    global _logged_active
    hidden = x.shape[-1]
    if not y.is_contiguous():
        y = y.contiguous()
    gate_v = _as_segment_rows(gate, hidden, x, 1.0)
    weight_v = _as_segment_rows(weight, hidden, x, 1.0)
    bias_v = _as_segment_rows(bias, hidden, x, 0.0)
    if not _logged_active:
        _logged_active = True
        logger.info(
            "fused residual add + LayerNorm + MXFP8 activation quantization active for %s",
            type(next_linear).__name__,
        )
    q, scales = torch.ops.vllm_omni.fused_add_layer_norm_mxfp8_2seg(x, y, gate_v, weight_v, bias_v, seg, eps)
    if getattr(next_linear, "mxfp8_block_major_a_scale", False):
        scales = scales.t().contiguous().t()
    return x, QuantizedActivation(
        data=q,
        scale=scales,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kMxfp8Dynamic,
    )


def _xpu_layernorm_quant_available() -> bool:
    if not current_omni_platform.is_xpu():
        return False
    try:
        import vllm_xpu_kernels._C  # noqa: F401
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops._xpu_C, "layer_norm") and hasattr(torch.ops._C, "per_token_group_fp8_quant")


_LN_MXFP8 = _xpu_layernorm_quant_available()
_logged_ln_active = False


def can_fuse_layernorm_quant(
    next_linear: torch.nn.Module,
    x: torch.Tensor,
    *modulation: torch.Tensor | None,
) -> bool:
    """Whether ``fused_layernorm_quant`` will take the kernel path."""
    if (
        not _LN_MXFP8
        or get_input_quant_key is None
        or get_input_quant_key(next_linear) != kMxfp8Dynamic
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.dim() < 2
        or x.shape[-1] % _MX_GROUP != 0
    ):
        return False
    hidden = x.shape[-1]
    return all(m is None or m.numel() == hidden for m in modulation)


def fused_layernorm_quant(
    next_linear: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> "QuantizedActivation":
    """``LayerNorm(x) * weight + bias`` quantized for ``next_linear``, without touching ``x``.

    For the norm at the block entry, whose input is not the block's to update:
    the vllm-xpu-kernels LayerNorm writes the modulated bf16 row and the
    per-token-group quantizer consumes it, in the scale layout the consumer's
    GEMM reads directly. Two kernels instead of the normalize / modulate /
    quantize chain, and no residual copy.
    """
    hidden = x.shape[-1]
    x2 = x.reshape(-1, hidden)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    weight_v = _as_row_vector(weight, hidden, x, 1.0)
    bias_v = _as_row_vector(bias, hidden, x, 0.0)
    global _logged_ln_active
    if not _logged_ln_active:
        _logged_ln_active = True
        logger.info("LayerNorm + MXFP8 activation quantization active for %s", type(next_linear).__name__)
    normed = torch.empty_like(x2)
    torch.ops._xpu_C.layer_norm(normed, x2, weight_v, eps, bias_v)
    num_tokens = x2.shape[0]
    q = torch.empty_like(x2, dtype=torch.float8_e4m3fn)
    if getattr(next_linear, "mxfp8_block_major_a_scale", False):
        scales = torch.empty((hidden // _MX_GROUP, num_tokens), dtype=torch.float8_e8m0fnu, device=x.device).t()
    else:
        scales = torch.empty((num_tokens, hidden // _MX_GROUP), dtype=torch.float8_e8m0fnu, device=x.device)
    finfo = torch.finfo(torch.float8_e4m3fn)
    # The quantizer infers the scale layout from the strides of `scales`.
    torch.ops._C.per_token_group_fp8_quant(
        normed, q, scales, _MX_GROUP, 1e-10, finfo.min, finfo.max, True, False, False
    )
    return QuantizedActivation(
        data=q,
        scale=scales,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kMxfp8Dynamic,
    )
