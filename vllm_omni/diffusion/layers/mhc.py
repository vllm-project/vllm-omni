# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native and eager Triton mHC post-processing, without projection or tensor caches."""

import threading

import torch
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton

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
    iterations: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    index = tl.arange(0, 4)
    post = tl.load(post_ptr + token * post_stride + index)
    ap = tl.load(alpha_post_ptr)
    bp = tl.load(bias_post_ptr + index)
    post = 2.0 * tl.sigmoid(ap * scale * post + bp)
    offsets = index[:, None] * 4 + index[None, :]
    matrix = tl.load(residual_ptr + token * residual_stride + offsets)
    ar = tl.load(alpha_residual_ptr)
    br = tl.load(bias_residual_ptr + offsets)
    matrix = ar * scale * matrix + br
    # Native amax propagates any NaN to the whole token's normalization.
    has_nan = tl.sum(tl.sum((matrix != matrix).to(tl.int32), axis=0), axis=0) != 0
    maximum = tl.max(tl.max(matrix, axis=0), axis=0)
    maximum = tl.where(has_nan, float("nan"), maximum)
    matrix = tl.exp(matrix - maximum)
    for _ in tl.static_range(iterations):
        matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + epsilon)
        matrix = matrix / (tl.sum(matrix, axis=1)[:, None] + epsilon)
    tl.store(post_out_ptr + token * 4 + index, post)
    tl.store(residual_out_ptr + token * 16 + offsets, matrix)


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
                _mhc_post_residual_kernel[(tokens,)](
                    *tensors,
                    post_out,
                    residual_out,
                    post_logits.stride(0),
                    residual_logits.stride(0),
                    scale,
                    epsilon,
                    iterations=iterations,
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
