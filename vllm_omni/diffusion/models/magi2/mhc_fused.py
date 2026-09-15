# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exact fused Sinkhorn normalization for MAGI-2's mHC residual mixing.

MAGI-2 evaluates ``MHCHandler.compute_post_residual`` in every transformer
block of every denoise step.  Its Sinkhorn-Knopp normalization iterates a
``[tokens, num_streams, num_streams]`` FP32 matrix ``sinkhorn_iterations``
(20) times with two four-element reductions per iteration.  At the released
``num_streams == 4`` the matrices are tiny, so eager evaluation is dominated
by kernel-launch overhead: the exp/amax prelude plus ``2 * iterations``
reductions and divisions issue more than eighty CUDA kernels per call, per
block, per step.

The Triton kernel below runs the whole loop in registers for a block of
tokens while preserving eager numerics exactly:

* the logits scaling ``(alpha * scale) * logits + bias`` stays FP32 in the
  eager association order, with the final add behind an inline-asm barrier
  so the compiler cannot contract it into an FMA (eager runs the multiply
  and the add as separate kernels and can never fuse them);
* ``exp(x - amax)`` uses the same accurate libdevice ``exp`` as eager
  (``tl.exp`` may lower to the approximate ``ex2`` path);
* the amax is NaN-propagating, matching ``torch.amax``;
* the strided ``sum(dim=-2)`` is reproduced as a sequential ascending chain
  of per-element vector adds, and the contiguous ``sum(dim=-1)`` as eager's
  interleaved lane pairing ``(x0 + x2) + (x1 + x3)``;
* every division uses the IEEE ``div_rn`` libdevice call (Triton's ``/``
  does not guarantee round-to-nearest);
* the iteration count, epsilon, FP32 accumulation, and the final
  ``out_dtype`` materialization boundary are preserved.

Unsupported inputs, autograd, and compiled regions retain the eager
formula.  Runtime launch failures are cached per device and output dtype
and fail closed.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

logger = init_logger(__name__)

_SUPPORTED_STREAMS = (2, 4)
_SUPPORTED_OUT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
_MAX_INT32_INDEX = 2**31 - 1
_TOKENS_PER_PROGRAM = 32
_FAILED_RUNTIME_KEYS: set[tuple[int | None, torch.dtype]] = set()

_OUT_DTYPE_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}


@triton.jit
def _asm_add(a, b):
    """Opaque FP32 add that blocks FMA contraction of ``a * b + c``.

    Eager evaluates the scaled logits as two separate CUDA kernels, so the
    multiply and the add can never fuse.  An inline-asm add is invisible to
    the optimizer's FMA contraction and keeps the fused kernel bit-equal.
    """
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mhc_sinkhorn_kernel(
    logits_ptr,
    alpha_ptr,
    bias_ptr,
    out_ptr,
    matmul_scale,
    epsilon,
    num_tokens,
    logits_stride0,
    num_streams: tl.constexpr,
    iterations: tl.constexpr,
    tokens_per_program: tl.constexpr,
    out_dtype_code: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * tokens_per_program + tl.arange(0, tokens_per_program)
    t_mask = offs_t < num_tokens
    alpha = tl.load(alpha_ptr)

    # ``compute_logits()`` hands the kernel a strided view of the
    # [T, N*(N+2)] projection (stride (N*(N+2), N, 1)), so the token row
    # stride is a runtime argument.  64-bit addressing: the row stride can
    # push offsets past INT32 even when numel() fits the guard below.
    offs_row = offs_t.to(tl.int64) * logits_stride0
    offs_out = offs_t.to(tl.int64) * (num_streams * num_streams)

    # m[i][j]: one [TPB] vector per matrix element; m[i][j][t] = logits[t, i, j].
    # Triton's frontend models Python lists as immutable tuples, so updates
    # rebuild the tuples; every index below is a compile-time constant.
    m = ()
    for i in tl.static_range(num_streams):
        row = ()
        for j in tl.static_range(num_streams):
            v = tl.load(
                logits_ptr + offs_row + i * num_streams + j,
                mask=t_mask,
                other=0.0,
            )
            b = tl.load(bias_ptr + i * num_streams + j)
            p = (alpha * matmul_scale) * v
            row = row + (_asm_add(p, b),)
        m = m + (row,)

    # NaN-propagating amax over all N*N elements, matching torch.amax.
    amax = m[0][0]
    any_nan = m[0][0] != m[0][0]
    for i in tl.static_range(num_streams):
        for j in tl.static_range(num_streams):
            any_nan = any_nan | (m[i][j] != m[i][j])
            amax = tl.maximum(amax, m[i][j])
    amax = tl.where(any_nan, float("nan"), amax)

    new_m = ()
    for i in tl.static_range(num_streams):
        row = ()
        for j in tl.static_range(num_streams):
            row = row + (tldevice.exp(m[i][j] - amax),)
        new_m = new_m + (row,)
    m = new_m

    for _ in tl.static_range(iterations):
        # matrix / (matrix.sum(dim=-2, keepdim=True) + epsilon):
        # the strided reduction is a sequential ascending sum over i.
        col_sum = ()
        for j in tl.static_range(num_streams):
            s = m[0][j]
            for i in tl.static_range(1, num_streams):
                s = s + m[i][j]
            col_sum = col_sum + (s,)
        new_m = ()
        for i in tl.static_range(num_streams):
            row = ()
            for j in tl.static_range(num_streams):
                row = row + (tldevice.div_rn(m[i][j], col_sum[j] + epsilon),)
            new_m = new_m + (row,)
        m = new_m
        # matrix / (matrix.sum(dim=-1, keepdim=True) + epsilon):
        # the contiguous reduction pairs interleaved lanes,
        # (x0 + x2) + (x1 + x3), as ascending even and odd chains.
        row_sum = ()
        for i in tl.static_range(num_streams):
            even = m[i][0]
            for j in tl.static_range(2, num_streams, 2):
                even = even + m[i][j]
            odd = m[i][1]
            for j in tl.static_range(3, num_streams, 2):
                odd = odd + m[i][j]
            row_sum = row_sum + (even + odd,)
        new_m = ()
        for i in tl.static_range(num_streams):
            row = ()
            for j in tl.static_range(num_streams):
                row = row + (tldevice.div_rn(m[i][j], row_sum[i] + epsilon),)
            new_m = new_m + (row,)
        m = new_m

    for i in tl.static_range(num_streams):
        for j in tl.static_range(num_streams):
            v = m[i][j]
            if out_dtype_code == 1:
                v = v.to(tl.bfloat16)
            elif out_dtype_code == 2:
                v = v.to(tl.float16)
            tl.store(
                out_ptr + offs_out + i * num_streams + j,
                v,
                mask=t_mask,
            )


def _can_use_fused_mhc_sinkhorn(
    residual_logits: torch.Tensor,
    alpha_residual: torch.Tensor,
    bias_residual: torch.Tensor,
    out_dtype: torch.dtype,
) -> bool:
    """Return whether the bit-exact CUDA fast path supports these tensors."""
    if residual_logits.ndim != 3:
        return False
    num_streams = residual_logits.shape[2]
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and residual_logits.is_cuda
        and residual_logits.dtype == torch.float32
        and alpha_residual.dtype == torch.float32
        and bias_residual.dtype == torch.float32
        and residual_logits.shape[1] == num_streams
        and num_streams in _SUPPORTED_STREAMS
        and alpha_residual.numel() == 1
        and bias_residual.shape == (num_streams, num_streams)
        # ``compute_logits()`` returns the residual slice as a [T, N, N] view
        # of the [T, N*(N+2)] projection with stride (N*(N+2), N, 1); the
        # kernel takes the token stride as a runtime argument, so only the
        # last two dims must be in standard layout.
        and residual_logits.stride(2) == 1
        and residual_logits.stride(1) == num_streams
        and residual_logits.stride(0) >= 1
        and bias_residual.is_contiguous()
        and alpha_residual.is_contiguous()
        and out_dtype in _SUPPORTED_OUT_DTYPES
        # Only autograd that is actually active blocks the fused path.
        # ``mhc_alpha_res_*``/``mhc_bias_res_*`` are nn.Parameters that keep
        # their default requires_grad=True flag through load_weights; under
        # inference_mode()/no_grad() (torch.is_grad_enabled() is False) their
        # eager outputs carry no grad, so the fusion stays eligible.
        and not (
            torch.is_grad_enabled()
            and (residual_logits.requires_grad or alpha_residual.requires_grad or bias_residual.requires_grad)
        )
        and residual_logits.numel() > 0
        and residual_logits.numel() <= _MAX_INT32_INDEX
    )


def _launch_fused_mhc_sinkhorn(
    residual_logits: torch.Tensor,
    alpha_residual: torch.Tensor,
    bias_residual: torch.Tensor,
    *,
    matmul_scale: float,
    iterations: int,
    epsilon: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    tokens, num_streams, _ = residual_logits.shape
    alpha = alpha_residual.reshape(1) if alpha_residual.ndim == 0 else alpha_residual
    output = torch.empty(
        (tokens, num_streams, num_streams),
        dtype=out_dtype,
        device=residual_logits.device,
    )
    grid = (triton.cdiv(tokens, _TOKENS_PER_PROGRAM),)
    with torch.accelerator.device_index(residual_logits.device.index):
        _mhc_sinkhorn_kernel[grid](
            residual_logits,
            alpha,
            bias_residual,
            output,
            matmul_scale,
            epsilon,
            tokens,
            residual_logits.stride(0),
            num_streams=num_streams,
            iterations=iterations,
            tokens_per_program=_TOKENS_PER_PROGRAM,
            out_dtype_code=_OUT_DTYPE_CODE[out_dtype],
            num_warps=1,
        )
    return output


def sinkhorn_knopp(matrix_logits: torch.Tensor, iterations: int, epsilon: float) -> torch.Tensor:
    """Sinkhorn-Knopp normalization in the eager association order.

    Defined here so both the eager fallback and ``layers`` share one
    implementation without a module cycle.
    """
    matrix = torch.exp(matrix_logits - matrix_logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iterations):
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + epsilon)
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + epsilon)
    return matrix


def _eager_sinkhorn_matrix(
    residual_logits: torch.Tensor,
    alpha_residual: torch.Tensor,
    bias_residual: torch.Tensor,
    *,
    matmul_scale: float,
    iterations: int,
    epsilon: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Mirror of ``MHCHandler.compute_post_residual``'s eager expression."""
    return sinkhorn_knopp(
        alpha_residual * matmul_scale * residual_logits.float() + bias_residual.unsqueeze(0).float(),
        iterations,
        epsilon,
    ).to(out_dtype)


def mhc_sinkhorn_matrix(
    residual_logits: torch.Tensor,
    alpha_residual: torch.Tensor,
    bias_residual: torch.Tensor,
    *,
    matmul_scale: float,
    iterations: int,
    epsilon: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Return the mHC residual mixing matrix, using a bit-exact CUDA fusion.

    The direct Triton call intentionally stays outside compiled regions:
    Inductor can already fuse the eager loop there, while an opaque kernel
    boundary would inhibit surrounding graph optimization.
    """
    if torch.compiler.is_compiling():
        return _eager_sinkhorn_matrix(
            residual_logits,
            alpha_residual,
            bias_residual,
            matmul_scale=matmul_scale,
            iterations=iterations,
            epsilon=epsilon,
            out_dtype=out_dtype,
        )

    runtime_key = (
        residual_logits.device.index if residual_logits.is_cuda else None,
        out_dtype,
    )
    if runtime_key not in _FAILED_RUNTIME_KEYS and _can_use_fused_mhc_sinkhorn(
        residual_logits, alpha_residual, bias_residual, out_dtype
    ):
        try:
            return _launch_fused_mhc_sinkhorn(
                residual_logits,
                alpha_residual,
                bias_residual,
                matmul_scale=matmul_scale,
                iterations=iterations,
                epsilon=epsilon,
                out_dtype=out_dtype,
            )
        except torch.OutOfMemoryError:
            # Transient allocation failure: propagate instead of
            # permanently disabling the fusion for this device/dtype.
            raise
        except Exception as exc:
            _FAILED_RUNTIME_KEYS.add(runtime_key)
            logger.warning_once(
                "Disabling fused mHC Sinkhorn on %s/%s after a runtime failure: %s",
                residual_logits.device,
                out_dtype,
                exc,
            )
    return _eager_sinkhorn_matrix(
        residual_logits,
        alpha_residual,
        bias_residual,
        matmul_scale=matmul_scale,
        iterations=iterations,
        epsilon=epsilon,
        out_dtype=out_dtype,
    )


__all__ = ["mhc_sinkhorn_matrix", "sinkhorn_knopp"]
