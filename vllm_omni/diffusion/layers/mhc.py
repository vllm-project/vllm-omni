# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native and eager Triton mHC post-processing, without projection or tensor caches."""

import threading

import torch
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

from vllm_omni.diffusion.layers.custom_op import CustomOp

_FAILED_MHC_KERNELS: set[tuple[str, int, str]] = set()
_WARNED_MHC_KERNELS: set[tuple[str, int, str]] = set()
_MHC_WARNING_LOCK = threading.Lock()
logger = init_logger(__name__)


def _mhc_kernel_key(tensor: torch.Tensor, name: str) -> tuple[str, int, str]:
    return (tensor.device.type, tensor.device.index or 0, name)


def _is_mhc_oom_error(error: Exception) -> bool:
    message = str(error).lower()
    error_name = type(error).__name__.lower()
    return isinstance(error, MemoryError) or "out of memory" in message or "outofmemoryerror" in error_name


def _cache_mhc_kernel_failure(failure_key: tuple[str, int, str], error: Exception) -> None:
    _FAILED_MHC_KERNELS.add(failure_key)
    with _MHC_WARNING_LOCK:
        should_warn = failure_key not in _WARNED_MHC_KERNELS
        _WARNED_MHC_KERNELS.add(failure_key)
    if should_warn:
        logger.warning(
            "mHC fused kernel failed for key=%s; falling back to native: %s",
            failure_key,
            repr(error),
        )


def sinkhorn_knopp(matrix_logits: torch.Tensor, iterations: int, epsilon: float) -> torch.Tensor:
    matrix = torch.exp(matrix_logits - matrix_logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iterations):
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + epsilon)
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + epsilon)
    return matrix


@triton.jit
def _asm_add_rn_f32(a, b):
    """Opaque FP32 add that blocks FMA contraction of ``a * b + c``.

    The native expression runs the scaled-logits multiply and the bias add
    as separate kernels, so they can never fuse; an inline-asm add is
    invisible to the optimizer's contraction and keeps the kernel
    bit-equal to that boundary.
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
def _mhc_post_residual_kernel(
    post_ptr,
    residual_ptr,
    alpha_post_ptr,
    bias_post_ptr,
    alpha_residual_ptr,
    bias_residual_ptr,
    post_out_ptr,
    residual_out_ptr,
    post_stride,
    residual_stride,
    scale,
    epsilon,
    num_tokens,
    iterations: tl.constexpr,
    tokens_per_program: tl.constexpr,
    out_dtype_code: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * tokens_per_program + tl.arange(0, tokens_per_program)
    t_mask = offs_t < num_tokens
    offs_row = offs_t.to(tl.int64)
    post_row = offs_row * post_stride
    residual_row = offs_row * residual_stride
    offs_post_out = offs_row * 4
    offs_residual_out = offs_row * 16

    # Post coefficients: 2 * sigmoid((alpha_post * scale) * x + bias).
    # sigmoid is div_rn(1, 1 + exp(-z)) — tl.sigmoid is not bit-equal to
    # torch.sigmoid; libdevice exp with IEEE division is. The add sits
    # behind the asm barrier so it cannot contract into an FMA.
    ap = tl.load(alpha_post_ptr)
    post_scale = ap * scale
    post = ()
    for j in tl.static_range(4):
        x = tl.load(post_ptr + post_row + j, mask=t_mask, other=0.0)
        b = tl.load(bias_post_ptr + j)
        z = _asm_add_rn_f32(post_scale * x, b)
        post = post + (2.0 * tldevice.div_rn(1.0, 1.0 + tldevice.exp(-z)),)

    # Bit-exact Sinkhorn (per #7545): the eager formula's numerical
    # behaviors are reproduced structurally rather than left to the
    # compiler —
    #   * (alpha*scale)*x + bias keeps the eager association order behind
    #     an inline-asm barrier (the native expression runs the multiply
    #     and the add as separate kernels and can never fuse them);
    #   * exp uses the accurate libdevice expf (tl.exp may lower to the
    #     approximate ex2 path);
    #   * amax propagates NaN like torch.amax;
    #   * the strided sum(dim=-2) is a sequential ascending chain and the
    #     contiguous sum(dim=-1) is eager's interleaved lane pairing
    #     (x0 + x2) + (x1 + x3);
    #   * divisions use IEEE div_rn (Triton's / does not guarantee
    #     round-to-nearest).
    # m[i][j] holds one [TPB] vector per matrix element; Triton models
    # Python lists as immutable tuples, so updates rebuild the tuples and
    # every index is a compile-time constant.
    ar = tl.load(alpha_residual_ptr)
    residual_scale = ar * scale
    m = ()
    for i in tl.static_range(4):
        row = ()
        for j in tl.static_range(4):
            x = tl.load(residual_ptr + residual_row + i * 4 + j, mask=t_mask, other=0.0)
            b = tl.load(bias_residual_ptr + i * 4 + j)
            row = row + (_asm_add_rn_f32(residual_scale * x, b),)
        m = m + (row,)

    # NaN-propagating amax over all 16 elements, matching torch.amax.
    any_nan = m[0][0] != m[0][0]
    amax = m[0][0]
    for i in tl.static_range(4):
        for j in tl.static_range(4):
            any_nan = any_nan | (m[i][j] != m[i][j])
            amax = tl.maximum(amax, m[i][j])
    amax = tl.where(any_nan, float("nan"), amax)

    new_m = ()
    for i in tl.static_range(4):
        row = ()
        for j in tl.static_range(4):
            row = row + (tldevice.exp(m[i][j] - amax),)
        new_m = new_m + (row,)
    m = new_m

    for _ in tl.static_range(iterations):
        # matrix / (matrix.sum(dim=-2, keepdim=True) + epsilon):
        # ascending sequential chain over i per column j.
        col_sum = ()
        for j in tl.static_range(4):
            s = m[0][j]
            for i in tl.static_range(1, 4):
                s = s + m[i][j]
            col_sum = col_sum + (s,)
        new_m = ()
        for i in tl.static_range(4):
            row = ()
            for j in tl.static_range(4):
                row = row + (tldevice.div_rn(m[i][j], col_sum[j] + epsilon),)
            new_m = new_m + (row,)
        m = new_m
        # matrix / (matrix.sum(dim=-1, keepdim=True) + epsilon):
        # interleaved lane pairing (x0 + x2) + (x1 + x3).
        row_sum = ()
        for i in tl.static_range(4):
            even = m[i][0] + m[i][2]
            odd = m[i][1] + m[i][3]
            row_sum = row_sum + (even + odd,)
        new_m = ()
        for i in tl.static_range(4):
            row = ()
            for j in tl.static_range(4):
                row = row + (tldevice.div_rn(m[i][j], row_sum[i] + epsilon),)
            new_m = new_m + (row,)
        m = new_m

    for j in tl.static_range(4):
        v = post[j]
        if out_dtype_code == 1:
            v = v.to(tl.bfloat16)
        elif out_dtype_code == 2:
            v = v.to(tl.float16)
        tl.store(post_out_ptr + offs_post_out + j, v, mask=t_mask)
    for i in tl.static_range(4):
        for j in tl.static_range(4):
            v = m[i][j]
            if out_dtype_code == 1:
                v = v.to(tl.bfloat16)
            elif out_dtype_code == 2:
                v = v.to(tl.float16)
            tl.store(residual_out_ptr + offs_residual_out + i * 4 + j, v, mask=t_mask)


@triton.jit
def _mhc_mix_kernel(streams_ptr, branch_ptr, post_ptr, matrix_ptr, out_ptr, hidden, block_size: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    channels = tl.program_id(1).to(tl.int64) * block_size + tl.arange(0, block_size)
    valid = channels < hidden
    stream_ids = tl.arange(0, 4)
    mixed = tl.zeros((4, block_size), tl.float32)
    for j in tl.static_range(4):
        x = tl.load(streams_ptr + (token * 4 + j) * hidden + channels, mask=valid, other=0).to(tl.float32)
        weight = tl.load(matrix_ptr + token * 16 + stream_ids * 4 + j).to(tl.float32)
        mixed += weight[:, None] * x[None, :]
    branch = tl.load(branch_ptr + token * hidden + channels, mask=valid, other=0).to(tl.float32)
    post = tl.load(post_ptr + token * 4 + stream_ids).to(tl.float32)
    branch = post[:, None] * branch[None, :]
    # Preserve native einsum materialization before the final addition.
    mixed = mixed.to(out_ptr.dtype.element_ty).to(tl.float32)
    branch = branch.to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(
        out_ptr + (token * 4 + stream_ids[:, None]) * hidden + channels[None, :], mixed + branch, mask=valid[None, :]
    )


def _native_required(tensors: tuple[torch.Tensor, ...]) -> bool:
    return (
        torch.compiler.is_compiling()
        or tensors[0].device.type not in ("cuda", "musa")
        or (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))
    )


class MHCPostResidual(CustomOp):
    """Prepare post coefficients and the residual matrix with FP32 arithmetic."""

    @staticmethod
    def forward_native(
        post_logits: torch.Tensor,
        residual_logits: torch.Tensor,
        alpha_post: torch.Tensor,
        bias_post: torch.Tensor,
        alpha_residual: torch.Tensor,
        bias_residual: torch.Tensor,
        *,
        scale: float,
        iterations: int = 20,
        epsilon: float = 1e-12,
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        post = 2.0 * torch.sigmoid(alpha_post * scale * post_logits + bias_post.unsqueeze(0))
        residual = sinkhorn_knopp(
            alpha_residual * scale * residual_logits.float() + bias_residual.unsqueeze(0).float(),
            iterations,
            epsilon,
        )
        return post.to(out_dtype), residual.to(out_dtype)

    def forward_cuda(
        self,
        post_logits: torch.Tensor,
        residual_logits: torch.Tensor,
        alpha_post: torch.Tensor,
        bias_post: torch.Tensor,
        alpha_residual: torch.Tensor,
        bias_residual: torch.Tensor,
        *,
        scale: float,
        iterations: int = 20,
        epsilon: float = 1e-12,
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = (post_logits, residual_logits, alpha_post, bias_post, alpha_residual, bias_residual)
        if _native_required(tensors):
            return self.forward_native(
                *tensors, scale=scale, iterations=iterations, epsilon=epsilon, out_dtype=out_dtype
            )
        failure_key = _mhc_kernel_key(post_logits, "post_residual")
        if (
            not HAS_TRITON
            or failure_key in _FAILED_MHC_KERNELS
            or not (
                post_logits.ndim == 2
                and post_logits.shape[1] == 4
                and residual_logits.shape == (post_logits.shape[0], 4, 4)
                and post_logits.stride(1) == 1
                and residual_logits.stride()[1:] == (4, 1)
                and alpha_post.numel() == 1
                and alpha_residual.numel() == 1
                and alpha_post.ndim <= 1
                and alpha_residual.ndim <= 1
                and bias_post.shape == (4,)
                and bias_residual.shape == (4, 4)
                and all(t.is_contiguous() for t in tensors[2:])
                and all(t.dtype == torch.float32 and t.device == post_logits.device for t in tensors)
                and out_dtype in (torch.float32, torch.bfloat16, torch.float16)
                and isinstance(scale, (int, float))
                and isinstance(epsilon, (int, float))
                and isinstance(iterations, int)
                and 0 <= iterations <= 64
            )
        ):
            return self.forward_native(
                *tensors, scale=scale, iterations=iterations, epsilon=epsilon, out_dtype=out_dtype
            )
        tokens = post_logits.shape[0]
        post_out = torch.empty((tokens, 4), device=post_logits.device, dtype=out_dtype)
        residual_out = torch.empty((tokens, 4, 4), device=post_logits.device, dtype=out_dtype)
        if tokens:
            try:
                _mhc_post_residual_kernel[(triton.cdiv(tokens, 32),)](
                    *tensors,
                    post_out,
                    residual_out,
                    post_logits.stride(0),
                    residual_logits.stride(0),
                    scale,
                    epsilon,
                    tokens,
                    iterations=iterations,
                    tokens_per_program=32,
                    out_dtype_code={torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[out_dtype],
                    num_warps=1,
                    enable_fp_fusion=False,
                )
            except Exception as error:
                if _is_mhc_oom_error(error):
                    raise
                _cache_mhc_kernel_failure(failure_key, error)
                return self.forward_native(
                    *tensors, scale=scale, iterations=iterations, epsilon=epsilon, out_dtype=out_dtype
                )
        return post_out, residual_out

    forward_npu = forward_native


class MHCMix(CustomOp):
    """Mix four streams and add a separately rounded branch term."""

    @staticmethod
    def forward_native(
        streams: torch.Tensor,
        branch_output: torch.Tensor,
        post_coefficients: torch.Tensor,
        residual_matrix: torch.Tensor,
    ) -> torch.Tensor:
        branch = torch.einsum("tn,tc->tnc", post_coefficients, branch_output)
        mixed = torch.einsum("tij,tjc->tic", residual_matrix, streams)
        return mixed + branch

    def forward_cuda(
        self,
        streams: torch.Tensor,
        branch_output: torch.Tensor,
        post_coefficients: torch.Tensor,
        residual_matrix: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (streams, branch_output, post_coefficients, residual_matrix)
        if _native_required(tensors):
            return self.forward_native(*tensors)
        failure_key = _mhc_kernel_key(streams, "mix")
        if (
            not HAS_TRITON
            or failure_key in _FAILED_MHC_KERNELS
            or not (
                streams.ndim == 3
                and streams.shape[1] == 4
                and branch_output.shape == (streams.shape[0], streams.shape[2])
                and post_coefficients.shape == streams.shape[:2]
                and residual_matrix.shape == (streams.shape[0], 4, 4)
                and all(t.is_contiguous() and t.device == streams.device and t.dtype == streams.dtype for t in tensors)
                and streams.dtype in (torch.float32, torch.bfloat16, torch.float16)
            )
        ):
            return self.forward_native(*tensors)
        output = torch.empty_like(streams)
        if output.numel():
            try:
                _mhc_mix_kernel[(streams.shape[0], triton.cdiv(streams.shape[2], 256))](
                    *tensors,
                    output,
                    streams.shape[2],
                    block_size=256,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            except Exception as error:
                if _is_mhc_oom_error(error):
                    raise
                _cache_mhc_kernel_failure(failure_key, error)
                return self.forward_native(*tensors)
        return output

    forward_npu = forward_native
