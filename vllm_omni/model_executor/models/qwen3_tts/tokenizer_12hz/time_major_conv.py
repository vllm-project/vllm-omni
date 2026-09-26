# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Time-major causal 1-D convolution for streaming vocoders.

Activations are ``[B, T, C]`` so that every dense convolution is one GEMM over
``B * T`` rows (bf16 tensor cores through cuBLAS) instead of an NCT cuDNN
convolution, which for these long, thin decoder layers picks non-tensor-core
kernels and adds layout transforms around each call.

- :func:`snake_im2col` gathers each output step's causal, dilated window into a
  ``[B * T, K * C]`` row (tap-major), optionally applying ``bias`` and SnakeBeta
  on the way in; ``K = 1`` is a standalone (bias +) SnakeBeta.
- :func:`causal_conv` is the dense causal (dilated) convolution as an implicit
  GEMM: each K-block of the reduction reads the input rows shifted by one tap,
  so the ``K * C`` im2col matrix is never materialized; bias and an optional
  residual are added in the epilogue.
- :func:`transposed_overlap_add` finishes a transposed convolution of kernel
  ``stride`` or ``2 * stride`` whose GEMM produced ``[B, T, K, C_out]``: the
  overlap-add of consecutive inputs, the bias and the causal right trim.

Weights come from ``nn.Conv1d``/``nn.ConvTranspose1d`` via :func:`conv_gemm_weight`
and :func:`transposed_gemm_weight`.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton


def conv_gemm_weight(weight: torch.Tensor) -> torch.Tensor:
    """``nn.Conv1d`` weight ``[C_out, C_in, K]`` -> tap-major GEMM weight ``[K * C_in, C_out]``."""
    c_out, c_in, k = weight.shape
    return weight.detach().permute(2, 1, 0).reshape(k * c_in, c_out).contiguous()


def transposed_gemm_weight(weight: torch.Tensor) -> torch.Tensor:
    """``nn.ConvTranspose1d`` weight ``[C_in, C_out, K]`` -> GEMM weight ``[C_in, K * C_out]``."""
    c_in, c_out, k = weight.shape
    return weight.detach().permute(0, 2, 1).reshape(c_in, k * c_out).contiguous()


def snake_im2col_reference(
    x: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    bias: torch.Tensor | None = None,
    snake: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Eager equivalent of :func:`snake_im2col` (also the CPU path)."""
    if bias is not None:
        x = x + bias
    if snake is not None:
        exp_alpha, inv_beta = snake
        x = x + (inv_beta.float() * torch.sin(x.float() * exp_alpha.float()).square()).to(x.dtype)
    if kernel_size == 1:
        return x.reshape(-1, x.shape[-1])
    t = x.shape[1]
    xp = torch.nn.functional.pad(x, (0, 0, (kernel_size - 1) * dilation, 0))
    cols = torch.cat([xp[:, j * dilation : j * dilation + t] for j in range(kernel_size)], dim=-1)
    return cols.reshape(-1, cols.shape[-1])


def causal_conv_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    kernel_size: int,
    dilation: int = 1,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager equivalent of :func:`causal_conv`: ``[B, T, C]`` -> ``[B, T, C_out]``."""
    bsz, t, _ = x.shape
    cols = snake_im2col_reference(x, kernel_size, dilation)
    out = cols @ weight if bias is None else torch.addmm(bias, cols, weight)
    out = out.view(bsz, t, -1)
    return out if residual is None else out + residual


def transposed_overlap_add_reference(z: torch.Tensor, stride: int, bias: torch.Tensor) -> torch.Tensor:
    """Eager equivalent of :func:`transposed_overlap_add`."""
    bsz, t, k, c_out = z.shape
    out = z[:, :, :stride]
    if k > stride:
        out = out + torch.nn.functional.pad(z[:, :-1, stride:], (0, 0, 0, 0, 1, 0))
    return out.reshape(bsz, t * stride, c_out) + bias


if HAS_TRITON:

    @triton.jit
    def _snake_im2col_kernel(
        x_ptr,
        bias_ptr,
        exp_alpha_ptr,
        inv_beta_ptr,
        out_ptr,
        n_rows,
        t_len,
        c_len,
        dilation,
        K: tl.constexpr,  # noqa: N803
        HAS_BIAS: tl.constexpr,  # noqa: N803
        HAS_SNAKE: tl.constexpr,  # noqa: N803
        BLOCK_R: tl.constexpr,  # noqa: N803
        BLOCK_C: tl.constexpr,  # noqa: N803
    ):
        rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        chans = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        row_mask = rows < n_rows
        chan_mask = chans < c_len
        t = rows % t_len
        if HAS_BIAS:
            bias = tl.load(bias_ptr + chans, mask=chan_mask, other=0.0)
        if HAS_SNAKE:
            exp_alpha = tl.load(exp_alpha_ptr + chans, mask=chan_mask, other=0.0).to(tl.float32)
            inv_beta = tl.load(inv_beta_ptr + chans, mask=chan_mask, other=0.0).to(tl.float32)
        out_mask = row_mask[:, None] & chan_mask[None, :]
        for j in tl.static_range(K):
            shift = (K - 1 - j) * dilation
            valid = row_mask & (t >= shift)
            src = rows - shift
            x = tl.load(
                x_ptr + src[:, None].to(tl.int64) * c_len + chans[None, :],
                mask=valid[:, None] & chan_mask[None, :],
                other=0.0,
            )
            if HAS_BIAS:
                # Causal padding is applied to the conv input, after bias.
                x = tl.where(valid[:, None], x + bias[None, :], 0.0).to(x_ptr.dtype.element_ty)
            if HAS_SNAKE:
                s = tl.sin(x.to(tl.float32) * exp_alpha[None, :])
                x = x + (inv_beta[None, :] * s * s).to(x.dtype)
            tl.store(out_ptr + rows[:, None].to(tl.int64) * (K * c_len) + j * c_len + chans[None, :], x, mask=out_mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=st)
            for bm, bn, bk, w, st in (
                (128, 128, 64, 8, 3),
                (128, 64, 64, 4, 4),
                (64, 128, 64, 4, 4),
                (128, 32, 32, 4, 4),
                (64, 64, 32, 4, 4),
                (256, 32, 32, 8, 3),
                (256, 64, 32, 8, 3),
                (128, 16, 32, 4, 4),
            )
        ],
        key=["row_bucket", "n_cols", "c_len", "K", "dilation"],
    )
    @triton.jit
    def _causal_conv_gemm_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        res_ptr,
        out_ptr,
        n_rows,
        n_cols,
        c_len,
        t_len,
        dilation,
        row_bucket,  # autotune key only: power-of-two bucket of n_rows
        K: tl.constexpr,  # noqa: N803
        HAS_BIAS: tl.constexpr,  # noqa: N803
        HAS_RES: tl.constexpr,  # noqa: N803
        PRECISION: tl.constexpr,  # noqa: N803
        BLOCK_M: tl.constexpr,  # noqa: N803
        BLOCK_N: tl.constexpr,  # noqa: N803
        BLOCK_K: tl.constexpr,  # noqa: N803
    ):
        # Row-major tiles, N fastest: neighbouring programs share the input rows.
        pid = tl.program_id(0)
        num_n = tl.cdiv(n_cols, BLOCK_N)
        pid_m = pid // num_n
        pid_n = pid % num_n
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < n_rows
        col_mask = cols < n_cols
        t = rows % t_len
        k_blocks = tl.cdiv(c_len, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, K * k_blocks):
            tap = kk // k_blocks
            chans = (kk % k_blocks) * BLOCK_K + tl.arange(0, BLOCK_K)
            chan_mask = chans < c_len
            shift = (K - 1 - tap) * dilation
            a = tl.load(
                x_ptr + (rows - shift)[:, None] * c_len + chans[None, :],
                mask=(row_mask & (t >= shift))[:, None] & chan_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                w_ptr + (tap * c_len + chans)[:, None] * n_cols + cols[None, :],
                mask=chan_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(a, b, acc, input_precision=PRECISION)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)[None, :]
        out_offsets = rows[:, None] * n_cols + cols[None, :]
        out_mask = row_mask[:, None] & col_mask[None, :]
        if HAS_RES:
            acc += tl.load(res_ptr + out_offsets, mask=out_mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty), mask=out_mask)

    @triton.jit
    def _transposed_overlap_add_kernel(
        z_ptr,
        bias_ptr,
        out_ptr,
        n_out_rows,
        t_len,
        c_len,
        STRIDE: tl.constexpr,  # noqa: N803
        K: tl.constexpr,  # noqa: N803
        BLOCK_R: tl.constexpr,  # noqa: N803
        BLOCK_C: tl.constexpr,  # noqa: N803
    ):
        rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)  # over B * T * STRIDE output steps
        chans = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        row_mask = rows < n_out_rows
        chan_mask = chans < c_len
        mask = row_mask[:, None] & chan_mask[None, :]
        tap = rows % STRIDE
        src = rows // STRIDE  # input row b * T + t
        acc = tl.load(
            z_ptr + (src[:, None].to(tl.int64) * K + tap[:, None]) * c_len + chans[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        if K > STRIDE:
            prev_valid = row_mask & (src % t_len > 0)
            acc += tl.load(
                z_ptr + ((src[:, None].to(tl.int64) - 1) * K + tap[:, None] + STRIDE) * c_len + chans[None, :],
                mask=prev_valid[:, None] & chan_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        acc += tl.load(bias_ptr + chans, mask=chan_mask, other=0.0).to(tl.float32)[None, :]
        tl.store(
            out_ptr + rows[:, None].to(tl.int64) * c_len + chans[None, :], acc.to(out_ptr.dtype.element_ty), mask=mask
        )


def _blocks(c_len: int) -> tuple[int, int]:
    block_c = min(128, triton.next_power_of_2(c_len))
    return max(1, 4096 // block_c), block_c


def snake_im2col(
    x: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    bias: torch.Tensor | None = None,
    snake: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """``[B, T, C]`` -> ``[B * T, K * C]`` causal im2col with optional bias + SnakeBeta."""
    if not HAS_TRITON or not x.is_cuda:
        return snake_im2col_reference(x, kernel_size, dilation, bias, snake)
    x = x.contiguous()
    bsz, t, c = x.shape
    n_rows = bsz * t
    out = torch.empty(n_rows, kernel_size * c, dtype=x.dtype, device=x.device)
    block_r, block_c = _blocks(c)
    exp_alpha, inv_beta = snake if snake is not None else (x, x)
    _snake_im2col_kernel[(triton.cdiv(n_rows, block_r), triton.cdiv(c, block_c))](
        x,
        bias if bias is not None else x,
        exp_alpha,
        inv_beta,
        out,
        n_rows,
        t,
        c,
        dilation,
        K=kernel_size,
        HAS_BIAS=bias is not None,
        HAS_SNAKE=snake is not None,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
    )
    return out


def causal_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    kernel_size: int,
    dilation: int = 1,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal dilated conv ``[B, T, C]`` -> ``[B, T, C_out]`` with a :func:`conv_gemm_weight` weight.

    ``residual`` (``[B, T, C_out]``) is added in the epilogue.
    """
    if not HAS_TRITON or not x.is_cuda:
        return causal_conv_reference(x, weight, bias, kernel_size, dilation, residual)
    x = x.contiguous()
    bsz, t, c = x.shape
    n_rows, n_cols = bsz * t, weight.shape[1]
    if weight.shape[0] != kernel_size * c:
        raise ValueError(f"conv weight has {weight.shape[0]} rows, expected {kernel_size} taps x {c} channels")
    if max(n_rows * max(c, n_cols), weight.numel()) >= 2**31:
        raise ValueError("causal_conv offsets are 32-bit; split the batch")
    out = torch.empty(bsz, t, n_cols, dtype=x.dtype, device=x.device)
    if residual is not None:
        residual = residual.contiguous()

    def grid(meta):
        return (triton.cdiv(n_rows, meta["BLOCK_M"]) * triton.cdiv(n_cols, meta["BLOCK_N"]),)

    _causal_conv_gemm_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        residual if residual is not None else x,
        out,
        n_rows,
        n_cols,
        c,
        t,
        dilation,
        triton.next_power_of_2(triton.cdiv(n_rows, 1024)),
        K=kernel_size,
        HAS_BIAS=bias is not None,
        HAS_RES=residual is not None,
        PRECISION="ieee" if x.dtype == torch.float32 else "tf32",
    )
    return out


def transposed_overlap_add(z: torch.Tensor, stride: int, bias: torch.Tensor) -> torch.Tensor:
    """``[B, T, K, C_out]`` GEMM output of a kernel-``K`` transposed conv -> ``[B, T * stride, C_out]``."""
    bsz, t, k, c_out = z.shape
    if k not in (stride, 2 * stride):
        raise ValueError(f"transposed_overlap_add supports kernel stride or 2*stride, got kernel={k} stride={stride}")
    if not HAS_TRITON or not z.is_cuda:
        return transposed_overlap_add_reference(z, stride, bias)
    z = z.contiguous()
    n_out_rows = bsz * t * stride
    out = torch.empty(bsz, t * stride, c_out, dtype=z.dtype, device=z.device)
    block_r, block_c = _blocks(c_out)
    _transposed_overlap_add_kernel[(triton.cdiv(n_out_rows, block_r), triton.cdiv(c_out, block_c))](
        z, bias, out, n_out_rows, t, c_out, STRIDE=stride, K=k, BLOCK_R=block_r, BLOCK_C=block_c
    )
    return out
