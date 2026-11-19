# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Audited BF16 action-expert QK paths, with ordinary cuBLAS as caller fallback.

Only the default expert geometry and the validated Torch/CUDA/device pairings
are selected. Scores are unscaled BF16; attention masks, softmax and PV remain
in the calling attention helper. No derived weights or persistent buffers.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

HAS_GLUON = False
if HAS_TRITON:
    try:
        from triton.experimental import gluon as g
        from triton.experimental.gluon import language as gl
    except ImportError:
        # Old Triton and CPU environments can still import the model and use
        # cuBLAS; the SM90 path does not require Gluon either.
        pass
    else:
        HAS_GLUON = True


def can_use_exact_bf16_qk(query: torch.Tensor, key: torch.Tensor, num_kv_groups: int = 8) -> bool:
    """Whether this runtime and layout match the tested action-expert QK."""
    if (
        not HAS_TRITON
        or torch.version.hip is not None
        or str(torch.__version__).split("+")[0] != "2.13.0"
        or torch.version.cuda != "13.2"
    ):
        return False
    if (
        query.shape != (1, 8, 50, 256)
        or key.shape != (1, 1, 1018, 256)
        or num_kv_groups != 8
        or query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or query.device.type != "cuda"
        or key.device != query.device
        or not query.is_contiguous()
        or not key.is_contiguous()
        or query.stride()[1:] != (12800, 256, 1)
        or key.stride()[2:] != (256, 1)
        or query.data_ptr() % 4
        or key.data_ptr() % 4
    ):
        return False
    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad):
        return False
    capability = torch.cuda.get_device_capability(query.device)
    return capability == (9, 0) or (
        capability == (12, 0) and HAS_GLUON and str(triton.__version__).split("+")[0] == "3.7.1"
    )


if HAS_GLUON:

    @g.jit
    def _native_k8_kernel(
        Q,
        K,
        OUT,
        Q_HEAD_STRIDE: gl.constexpr,
        Q_ROW_STRIDE: gl.constexpr,
        K_ROW_STRIDE: gl.constexpr,
        WM: gl.constexpr,
        WN: gl.constexpr,
        NR: gl.constexpr,
    ):
        layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
        thread = gl.arange(0, 128, layout=layout)
        lane, warp = thread % 32, thread // 32
        group, pair = lane // 4, lane % 4
        row0 = gl.program_id(0) * (16 * WM) + (warp // WN) * 16 + group
        row1 = row0 + 8
        column_base = gl.program_id(1) * (8 * WN * NR) + (warp % WN) * (8 * NR)
        # Cast pointers, not values: packed b32 operands preserve both BF16 bits.
        qp = Q.to(gl.pointer_type(gl.uint32))
        kp = K.to(gl.pointer_type(gl.uint32))
        q0 = ((row0 // 50) * Q_HEAD_STRIDE + (row0 % 50) * Q_ROW_STRIDE) // 2
        q1 = ((row1 // 50) * Q_HEAD_STRIDE + (row1 % 50) * Q_ROW_STRIDE) // 2
        zero = gl.full((128,), 0, gl.float32, layout=layout)
        accumulators = ()
        for n in gl.static_range(NR):
            accumulators += ((zero, zero, zero, zero),)
        # Exactly 32 sequential K8 accumulations. No K16 zero padding or split sums.
        for chunk in range(32):
            a0 = gl.load(qp + q0 + chunk * 4 + pair, mask=row0 < 400, other=0)
            a1 = gl.load(qp + q1 + chunk * 4 + pair, mask=row1 < 400, other=0)
            updated = ()
            for n in gl.static_range(NR):
                column_b = column_base + n * 8 + group
                b = gl.load(kp + column_b * (K_ROW_STRIDE // 2) + chunk * 4 + pair, mask=column_b < 1018, other=0)
                c0, c1, c2, c3 = accumulators[n]
                d0, d1, d2, d3 = gl.inline_asm_elementwise(
                    "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {$0,$1,$2,$3}, {$4,$5}, {$6}, {$7,$8,$9,$10};",
                    constraints="=f,=f,=f,=f,r,r,r,f,f,f,f",
                    args=[a0, a1, b, c0, c1, c2, c3],
                    dtype=(gl.float32, gl.float32, gl.float32, gl.float32),
                    is_pure=True,
                    pack=1,
                )
                updated += ((d0, d1, d2, d3),)
            accumulators = updated
        for n in gl.static_range(NR):
            column0 = column_base + n * 8 + pair * 2
            for slot in gl.static_range(4):
                row = row0 if slot < 2 else row1
                column = column0 + slot % 2
                offset = row * 1018 + column
                valid = (row < 400) & (column < 1018)
                value = accumulators[n][slot]
                gl.store(OUT + offset, value.to(gl.bfloat16), mask=valid)


def exact_bf16_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    num_kv_groups: int = 8,
    h100_score_kernel=None,
) -> torch.Tensor | None:
    """Return unscaled scores, or None so the caller keeps its original matmul.

    The SM90 JIT is supplied by modeling_pi05 to reuse its existing kernel
    without a circular import. Unsupported configurations never launch a JIT.
    """
    if not can_use_exact_bf16_qk(query, key, num_kv_groups):
        return None
    capability = torch.cuda.get_device_capability(query.device)
    if capability == (9, 0) and h100_score_kernel is None:
        return None
    output = torch.empty((1, 8, 50, 1018), dtype=query.dtype, device=query.device)
    if capability == (9, 0):
        # Same kernel and fixed BF16 tile as the validated original score path.
        h100_score_kernel[(triton.cdiv(400, 32), triton.cdiv(1018, 32))](
            query,
            key,
            output,
            400,
            1018,
            key.stride(-2),
            1018,
            1.0,
            HEAD_DIM=256,
            SCORE_DTYPE=tl.bfloat16,
            BLOCK_M=32,
            BLOCK_N=32,
            BLOCK_K=128,
            num_warps=8,
            num_stages=3,
            enable_fp_fusion=False,
        )
    else:
        _native_k8_kernel[(triton.cdiv(400, 32), triton.cdiv(1018, 32))](
            query,
            key,
            output,
            Q_HEAD_STRIDE=query.stride(1),
            Q_ROW_STRIDE=query.stride(2),
            K_ROW_STRIDE=key.stride(2),
            WM=2,
            WN=2,
            NR=2,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return output
