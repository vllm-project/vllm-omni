# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sinkhorn iterations of the four-stream mHC residual matrix in one Triton kernel.

Inductor unrolls the four-element sums of ``sinkhorn_knopp`` into left folds,
``((m0 + m1) + m2) + m3``, and emits one kernel per half-iteration because each
step reads other entries of the same token's matrix. This kernel keeps the 16
entries of a token in registers and repeats exactly those folds, the epsilon
addition and the division, so every iteration produces the same FP32 values.

The op is registered with ``torch.library.triton_op`` so compiled regions trace
the kernel instead of breaking the graph.
"""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

__all__ = ["mhc_sinkhorn_iterations"]

_BLOCK = 64
_NUM_WARPS = 2

if HAS_TRITON:

    @triton.jit
    def _mhc_sinkhorn_kernel(matrix_ptr, out_ptr, tokens, iterations, epsilon: tl.constexpr, block: tl.constexpr):
        token = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
        mask = token < tokens
        src = matrix_ptr + token * 16
        m00 = tl.load(src + 0, mask=mask, other=1.0)
        m01 = tl.load(src + 1, mask=mask, other=1.0)
        m02 = tl.load(src + 2, mask=mask, other=1.0)
        m03 = tl.load(src + 3, mask=mask, other=1.0)
        m10 = tl.load(src + 4, mask=mask, other=1.0)
        m11 = tl.load(src + 5, mask=mask, other=1.0)
        m12 = tl.load(src + 6, mask=mask, other=1.0)
        m13 = tl.load(src + 7, mask=mask, other=1.0)
        m20 = tl.load(src + 8, mask=mask, other=1.0)
        m21 = tl.load(src + 9, mask=mask, other=1.0)
        m22 = tl.load(src + 10, mask=mask, other=1.0)
        m23 = tl.load(src + 11, mask=mask, other=1.0)
        m30 = tl.load(src + 12, mask=mask, other=1.0)
        m31 = tl.load(src + 13, mask=mask, other=1.0)
        m32 = tl.load(src + 14, mask=mask, other=1.0)
        m33 = tl.load(src + 15, mask=mask, other=1.0)
        # A constexpr epsilon built as Inductor builds it: Inductor types a runtime float argument as FP64.
        eps = tl.full([1], epsilon, tl.float32)
        for _ in range(iterations):
            # matrix / (matrix.sum(dim=-2, keepdim=True) + epsilon)
            c0 = (((m00 + m10) + m20) + m30) + eps
            c1 = (((m01 + m11) + m21) + m31) + eps
            c2 = (((m02 + m12) + m22) + m32) + eps
            c3 = (((m03 + m13) + m23) + m33) + eps
            m00 = m00 / c0
            m01 = m01 / c1
            m02 = m02 / c2
            m03 = m03 / c3
            m10 = m10 / c0
            m11 = m11 / c1
            m12 = m12 / c2
            m13 = m13 / c3
            m20 = m20 / c0
            m21 = m21 / c1
            m22 = m22 / c2
            m23 = m23 / c3
            m30 = m30 / c0
            m31 = m31 / c1
            m32 = m32 / c2
            m33 = m33 / c3
            # matrix / (matrix.sum(dim=-1, keepdim=True) + epsilon)
            r0 = (((m00 + m01) + m02) + m03) + eps
            r1 = (((m10 + m11) + m12) + m13) + eps
            r2 = (((m20 + m21) + m22) + m23) + eps
            r3 = (((m30 + m31) + m32) + m33) + eps
            m00 = m00 / r0
            m01 = m01 / r0
            m02 = m02 / r0
            m03 = m03 / r0
            m10 = m10 / r1
            m11 = m11 / r1
            m12 = m12 / r1
            m13 = m13 / r1
            m20 = m20 / r2
            m21 = m21 / r2
            m22 = m22 / r2
            m23 = m23 / r2
            m30 = m30 / r3
            m31 = m31 / r3
            m32 = m32 / r3
            m33 = m33 / r3
        dst = out_ptr + token * 16
        tl.store(dst + 0, m00, mask=mask)
        tl.store(dst + 1, m01, mask=mask)
        tl.store(dst + 2, m02, mask=mask)
        tl.store(dst + 3, m03, mask=mask)
        tl.store(dst + 4, m10, mask=mask)
        tl.store(dst + 5, m11, mask=mask)
        tl.store(dst + 6, m12, mask=mask)
        tl.store(dst + 7, m13, mask=mask)
        tl.store(dst + 8, m20, mask=mask)
        tl.store(dst + 9, m21, mask=mask)
        tl.store(dst + 10, m22, mask=mask)
        tl.store(dst + 11, m23, mask=mask)
        tl.store(dst + 12, m30, mask=mask)
        tl.store(dst + 13, m31, mask=mask)
        tl.store(dst + 14, m32, mask=mask)
        tl.store(dst + 15, m33, mask=mask)

    @torch.library.triton_op("vllm_omni::magi2_mhc_sinkhorn", mutates_args=())
    def mhc_sinkhorn_iterations(matrix: torch.Tensor, iterations: int, epsilon: float) -> torch.Tensor:
        """``iterations`` column/row normalizations of FP32 ``[tokens, 4, 4]`` matrices."""
        if matrix.ndim != 3 or matrix.shape[1:] != (4, 4) or matrix.dtype != torch.float32:
            raise ValueError(f"expected FP32 [tokens, 4, 4] matrices, got {matrix.dtype} {tuple(matrix.shape)}")
        matrix = matrix.contiguous()
        out = torch.empty_like(matrix)
        tokens = matrix.shape[0]
        if tokens:
            torch.library.wrap_triton(_mhc_sinkhorn_kernel)[(triton.cdiv(tokens, _BLOCK),)](
                matrix,
                out,
                tokens,
                iterations,
                epsilon=epsilon,
                block=_BLOCK,
                num_warps=_NUM_WARPS,
            )
        return out

else:  # pragma: no cover

    def mhc_sinkhorn_iterations(*args, **kwargs):
        raise RuntimeError("Triton is not available")
