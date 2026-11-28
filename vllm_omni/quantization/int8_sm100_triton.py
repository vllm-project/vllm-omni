# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Triton kernels for the SM100 INT8 fallback (``int8_sm100.py``).

``torch._int_mm`` is as fast as the tensor cores allow, but the torch epilogue
(int32 -> fp32 copy, two broadcast multiplies, bias add, cast to bf16) costs more
than the GEMM itself. These kernels fuse the pieces:

- ``dynamic_per_token_quant``: absmax, scale, round and clamp in one pass.
- ``int8_scale_epilogue``: ``acc * x_scale[M,1] * w_scale[1,N] (+ bias)`` to the
  output dtype in one pass.
"""

# ruff: noqa: N803

import torch
import triton
import triton.language as tl


@triton.jit
def _dyn_quant_kernel(x_ptr, pq_ptr, q_ptr, s_ptr, K, stride_x, stride_q, HAS_PQ: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    xl = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    if HAS_PQ:
        # SmoothQuant: same as torch ``x * pre_quant_scale.to(x.dtype)``: both operands
        # in the activation dtype, product rounded to that dtype, then quantized.
        pq = tl.load(pq_ptr + cols, mask=mask, other=0.0).to(xl.dtype).to(tl.float32)
        xl = (xl.to(tl.float32) * pq).to(xl.dtype)
    x = xl.to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    amax = tl.maximum(amax, 1.1754943508222875e-38)  # float32 tiny
    scale = tl.math.div_rn(amax, 127.0)
    # Same arithmetic as the CUDA kernel: x * (127 / amax), round half to even.
    q = tl.extra.cuda.libdevice.rint(x * tl.math.div_rn(127.0, amax))
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(q_ptr + row * stride_q + cols, q.to(tl.int8), mask=mask)
    tl.store(s_ptr + row, scale)


def dynamic_per_token_quant(
    x: torch.Tensor, pre_quant_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token INT8 quantization. Returns ``(q int8 [M,K], scale fp32 [M,1])``.

    With ``pre_quant_scale`` ([K]) the SmoothQuant multiply ``x * pre_quant_scale.to(x.dtype)``
    is fused in, bit-identical to the separate torch multiply followed by quantization.
    """
    assert x.ndim == 2 and x.stride(-1) == 1
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    block = triton.next_power_of_2(k)
    num_warps = 4 if block <= 4096 else 8 if block <= 8192 else 16
    pq = None if pre_quant_scale is None else pre_quant_scale.reshape(-1).contiguous()
    _dyn_quant_kernel[(m,)](
        x,
        pq if pq is not None else x,
        q,
        s,
        k,
        x.stride(0),
        q.stride(0),
        HAS_PQ=pq is not None,
        BLOCK=block,
        num_warps=num_warps,
    )
    return q, s


@triton.jit
def _epilogue_kernel(
    acc_ptr,
    xs_ptr,
    ws_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = rm < M
    mn = rn < N
    offs = rm.to(tl.int64)[:, None] * N + rn[None, :]
    acc = tl.load(acc_ptr + offs, mask=mm[:, None] & mn[None, :], other=0).to(tl.float32)
    xs = tl.load(xs_ptr + rm, mask=mm, other=0.0)
    ws = tl.load(ws_ptr + rn, mask=mn, other=0.0)
    out = acc * xs[:, None] * ws[None, :]
    if HAS_BIAS:
        b = tl.load(bias_ptr + rn, mask=mn, other=0.0).to(tl.float32)
        out = out + b[None, :]
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mm[:, None] & mn[None, :])


def int8_scale_epilogue(
    acc: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``acc [M,N] int32`` to ``out_dtype``; ``x_scale`` is [M,1] or [1,1], ``w_scale`` is [N,1] or [1]."""
    m, n = acc.shape
    acc = acc.contiguous()
    x_scale = x_scale.reshape(-1).float()
    if x_scale.numel() == 1:
        x_scale = x_scale.expand(m).contiguous()
    w_scale = w_scale.reshape(-1).float().contiguous()
    if w_scale.numel() == 1:
        w_scale = w_scale.expand(n).contiguous()
    out = torch.empty((m, n), dtype=out_dtype, device=acc.device)
    has_bias = bias is not None
    bias_arg = bias.contiguous() if has_bias else acc  # dummy pointer when unused
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 256))
    _epilogue_kernel[grid](
        acc,
        x_scale,
        w_scale,
        bias_arg,
        out,
        m,
        n,
        HAS_BIAS=has_bias,
        BLOCK_M=32,
        BLOCK_N=256,
        num_warps=4,
        enable_fp_fusion=False,  # no FMA: bit-identical to the torch epilogue
    )
    return out


@triton.jit
def _dequant_weight_kernel(
    q_ptr,
    ws_ptr,
    pq_ptr,
    out_ptr,
    N,
    K,
    stride_qn,
    stride_qk,
    HAS_PQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mn = rn < N
    mk = rk < K
    mask = mn[:, None] & mk[None, :]
    q = tl.load(
        q_ptr + rn.to(tl.int64)[:, None] * stride_qn + rk.to(tl.int64)[None, :] * stride_qk,
        mask=mask,
        other=0,
    ).to(tl.float32)
    ws = tl.load(ws_ptr + rn, mask=mn, other=0.0)
    w = q * ws[:, None]
    if HAS_PQ:
        pq = tl.load(pq_ptr + rk, mask=mk, other=0.0)
        w = w * pq[None, :]
    tl.store(out_ptr + rn.to(tl.int64)[:, None] * K + rk[None, :], w.to(tl.bfloat16), mask=mask)


def dequant_weight_bf16(
    q_kn: torch.Tensor, weight_scale: torch.Tensor, pre_quant_scale: torch.Tensor | None
) -> torch.Tensor:
    """Dense BF16 ``[N, K]`` weight from an INT8 ``[K, N]`` view in one pass.

    ``bf16(float(q) * weight_scale[n] * pre_quant_scale[k])``, same fp32 operation
    order as the torch reference, so the result is bit-identical. Any strides of
    ``q_kn`` work (the stored layout is a ``[K, N]`` view of a contiguous ``[N, K]``).
    """
    k, n = q_kn.shape
    ws = weight_scale.reshape(-1).to(device=q_kn.device, dtype=torch.float32).contiguous()
    pq = None
    if pre_quant_scale is not None:
        pq = pre_quant_scale.reshape(-1).to(device=q_kn.device, dtype=torch.float32).contiguous()
    out = torch.empty((n, k), dtype=torch.bfloat16, device=q_kn.device)
    block_n, block_k = 32, 256
    grid = (triton.cdiv(n, block_n), triton.cdiv(k, block_k))
    _dequant_weight_kernel[grid](
        q_kn,
        ws,
        pq if pq is not None else ws,
        out,
        n,
        k,
        q_kn.stride(1),
        q_kn.stride(0),
        HAS_PQ=pq is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
