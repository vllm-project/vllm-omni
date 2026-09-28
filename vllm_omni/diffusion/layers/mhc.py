# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native and eager Triton mHC post-processing, without projection or tensor caches."""

import threading

import torch
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.platforms import current_omni_platform

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


if current_omni_platform.is_musa():

    @triton.jit
    def _asm_add_rn_f32(a, b):
        """Opaque FP32 add that blocks FMA contraction of ``a * b + c``.

        The MUSA assembler cannot allocate the PTX ``=f`` register class
        for inline asm, so the barrier is expressed as a correctly-rounded
        fused op instead: ``fma_rn(a, 1.0, b)`` is exactly the
        correctly-rounded ``a + b`` (``a * 1`` is exact) and is already a
        single fused multiply-add, so the optimizer cannot contract
        anything further.
        """
        return tldevice.fma_rn(a, 1.0, b)

else:

    @triton.jit
    def _asm_add_rn_f32(a, b):
        """Opaque FP32 add that blocks FMA contraction of ``a * b + c``.

        The native expression runs the scaled-logits multiply and the bias
        add as separate kernels, so they can never fuse; an inline-asm add
        is invisible to the optimizer's contraction and keeps the kernel
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
    # sigmoid is div_rn(1, 1 + exp(-z)); the add sits behind the asm
    # barrier so it cannot contract into an FMA. Fixed variables: no
    # tuple rebinding inside static_range (Triton 3.2 / MUSA).
    ap = tl.load(alpha_post_ptr)
    post_scale = ap * scale
    p0 = 2.0 * tldevice.div_rn(
        1.0,
        1.0
        + tldevice.exp(
            -_asm_add_rn_f32(
                post_scale * tl.load(post_ptr + post_row + 0, mask=t_mask, other=0.0), tl.load(bias_post_ptr + 0)
            )
        ),
    )
    p1 = 2.0 * tldevice.div_rn(
        1.0,
        1.0
        + tldevice.exp(
            -_asm_add_rn_f32(
                post_scale * tl.load(post_ptr + post_row + 1, mask=t_mask, other=0.0), tl.load(bias_post_ptr + 1)
            )
        ),
    )
    p2 = 2.0 * tldevice.div_rn(
        1.0,
        1.0
        + tldevice.exp(
            -_asm_add_rn_f32(
                post_scale * tl.load(post_ptr + post_row + 2, mask=t_mask, other=0.0), tl.load(bias_post_ptr + 2)
            )
        ),
    )
    p3 = 2.0 * tldevice.div_rn(
        1.0,
        1.0
        + tldevice.exp(
            -_asm_add_rn_f32(
                post_scale * tl.load(post_ptr + post_row + 3, mask=t_mask, other=0.0), tl.load(bias_post_ptr + 3)
            )
        ),
    )

    # Bit-exact Sinkhorn (per #7545), statically unrolled.
    ar = tl.load(alpha_residual_ptr)
    residual_scale = ar * scale
    m00 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 0, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 0),
    )
    m01 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 1, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 1),
    )
    m02 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 2, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 2),
    )
    m03 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 3, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 3),
    )
    m10 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 4, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 4),
    )
    m11 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 5, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 5),
    )
    m12 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 6, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 6),
    )
    m13 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 7, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 7),
    )
    m20 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 8, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 8),
    )
    m21 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 9, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 9),
    )
    m22 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 10, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 10),
    )
    m23 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 11, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 11),
    )
    m30 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 12, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 12),
    )
    m31 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 13, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 13),
    )
    m32 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 14, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 14),
    )
    m33 = _asm_add_rn_f32(
        residual_scale * tl.load(residual_ptr + residual_row + 15, mask=t_mask, other=0.0),
        tl.load(bias_residual_ptr + 15),
    )

    # NaN-propagating amax over all 16 elements, matching torch.amax
    # (floating-point max is order-independent; the NaN union is separate).
    any_nan = (
        (m00 != m00)
        | (m01 != m01)
        | (m02 != m02)
        | (m03 != m03)
        | (m10 != m10)
        | (m11 != m11)
        | (m12 != m12)
        | (m13 != m13)
        | (m20 != m20)
        | (m21 != m21)
        | (m22 != m22)
        | (m23 != m23)
        | (m30 != m30)
        | (m31 != m31)
        | (m32 != m32)
        | (m33 != m33)
    )
    amax = tl.maximum(
        tl.maximum(
            tl.maximum(
                tl.maximum(
                    tl.maximum(
                        tl.maximum(
                            tl.maximum(
                                tl.maximum(
                                    tl.maximum(
                                        tl.maximum(
                                            tl.maximum(
                                                tl.maximum(tl.maximum(tl.maximum(tl.maximum(m00, m01), m02), m03), m10),
                                                m11,
                                            ),
                                            m12,
                                        ),
                                        m13,
                                    ),
                                    m20,
                                ),
                                m21,
                            ),
                            m22,
                        ),
                        m23,
                    ),
                    m30,
                ),
                m31,
            ),
            m32,
        ),
        m33,
    )
    amax = tl.where(any_nan, float("nan"), amax)

    m00 = tldevice.exp(m00 - amax)
    m01 = tldevice.exp(m01 - amax)
    m02 = tldevice.exp(m02 - amax)
    m03 = tldevice.exp(m03 - amax)
    m10 = tldevice.exp(m10 - amax)
    m11 = tldevice.exp(m11 - amax)
    m12 = tldevice.exp(m12 - amax)
    m13 = tldevice.exp(m13 - amax)
    m20 = tldevice.exp(m20 - amax)
    m21 = tldevice.exp(m21 - amax)
    m22 = tldevice.exp(m22 - amax)
    m23 = tldevice.exp(m23 - amax)
    m30 = tldevice.exp(m30 - amax)
    m31 = tldevice.exp(m31 - amax)
    m32 = tldevice.exp(m32 - amax)
    m33 = tldevice.exp(m33 - amax)

    for _ in tl.static_range(iterations):
        # matrix / (matrix.sum(dim=-2, keepdim=True) + epsilon):
        # ascending sequential chain over i per column j.
        c0 = m00 + m10 + m20 + m30
        c1 = m01 + m11 + m21 + m31
        c2 = m02 + m12 + m22 + m32
        c3 = m03 + m13 + m23 + m33
        m00 = tldevice.div_rn(m00, c0 + epsilon)
        m01 = tldevice.div_rn(m01, c1 + epsilon)
        m02 = tldevice.div_rn(m02, c2 + epsilon)
        m03 = tldevice.div_rn(m03, c3 + epsilon)
        m10 = tldevice.div_rn(m10, c0 + epsilon)
        m11 = tldevice.div_rn(m11, c1 + epsilon)
        m12 = tldevice.div_rn(m12, c2 + epsilon)
        m13 = tldevice.div_rn(m13, c3 + epsilon)
        m20 = tldevice.div_rn(m20, c0 + epsilon)
        m21 = tldevice.div_rn(m21, c1 + epsilon)
        m22 = tldevice.div_rn(m22, c2 + epsilon)
        m23 = tldevice.div_rn(m23, c3 + epsilon)
        m30 = tldevice.div_rn(m30, c0 + epsilon)
        m31 = tldevice.div_rn(m31, c1 + epsilon)
        m32 = tldevice.div_rn(m32, c2 + epsilon)
        m33 = tldevice.div_rn(m33, c3 + epsilon)
        # matrix / (matrix.sum(dim=-1, keepdim=True) + epsilon):
        # interleaved lane pairing (x0 + x2) + (x1 + x3).
        r0 = (m00 + m02) + (m01 + m03)
        r1 = (m10 + m12) + (m11 + m13)
        r2 = (m20 + m22) + (m21 + m23)
        r3 = (m30 + m32) + (m31 + m33)
        m00 = tldevice.div_rn(m00, r0 + epsilon)
        m01 = tldevice.div_rn(m01, r0 + epsilon)
        m02 = tldevice.div_rn(m02, r0 + epsilon)
        m03 = tldevice.div_rn(m03, r0 + epsilon)
        m10 = tldevice.div_rn(m10, r1 + epsilon)
        m11 = tldevice.div_rn(m11, r1 + epsilon)
        m12 = tldevice.div_rn(m12, r1 + epsilon)
        m13 = tldevice.div_rn(m13, r1 + epsilon)
        m20 = tldevice.div_rn(m20, r2 + epsilon)
        m21 = tldevice.div_rn(m21, r2 + epsilon)
        m22 = tldevice.div_rn(m22, r2 + epsilon)
        m23 = tldevice.div_rn(m23, r2 + epsilon)
        m30 = tldevice.div_rn(m30, r3 + epsilon)
        m31 = tldevice.div_rn(m31, r3 + epsilon)
        m32 = tldevice.div_rn(m32, r3 + epsilon)
        m33 = tldevice.div_rn(m33, r3 + epsilon)

    # Materialize outputs at the eager out_dtype boundary.
    v = p0
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(post_out_ptr + offs_post_out + 0, v, mask=t_mask)
    v = p1
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(post_out_ptr + offs_post_out + 1, v, mask=t_mask)
    v = p2
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(post_out_ptr + offs_post_out + 2, v, mask=t_mask)
    v = p3
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(post_out_ptr + offs_post_out + 3, v, mask=t_mask)
    v = m00
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 0, v, mask=t_mask)
    v = m01
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 1, v, mask=t_mask)
    v = m02
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 2, v, mask=t_mask)
    v = m03
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 3, v, mask=t_mask)
    v = m10
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 4, v, mask=t_mask)
    v = m11
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 5, v, mask=t_mask)
    v = m12
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 6, v, mask=t_mask)
    v = m13
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 7, v, mask=t_mask)
    v = m20
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 8, v, mask=t_mask)
    v = m21
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 9, v, mask=t_mask)
    v = m22
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 10, v, mask=t_mask)
    v = m23
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 11, v, mask=t_mask)
    v = m30
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 12, v, mask=t_mask)
    v = m31
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 13, v, mask=t_mask)
    v = m32
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 14, v, mask=t_mask)
    v = m33
    if out_dtype_code == 1:
        v = v.to(tl.bfloat16)
    elif out_dtype_code == 2:
        v = v.to(tl.float16)
    tl.store(residual_out_ptr + offs_residual_out + 15, v, mask=t_mask)


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
