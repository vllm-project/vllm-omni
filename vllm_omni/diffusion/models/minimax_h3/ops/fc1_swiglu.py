# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""MiniMax-H3 DiT MLP: fc1 GEMM with the SwiGLU activation fused into its epilogue.

The reference MLP runs the merged ``[gate; up]`` fc1 as one cuBLAS GEMM into a
``[tokens, 2 * ffn]`` BF16 tensor and then ``SiluAndMul``. At the 1-GPU
benchmark shape (37,824 tokens, 5,376 -> 2 x 14,336) that intermediate is
2.2 GB per layer, written by the GEMM and read back by the activation.

This kernel computes each output tile's gate and up accumulators from one
shared activation tile and writes only ``silu(gate) * up``. It is a persistent,
warp-specialized Triton GEMM over the unmodified merged weight. The epilogue
keeps the reference rounding order -- both GEMM results rounded to BF16, silu
in FP32 rounded to BF16, the product rounded to BF16 -- so the output is
bit-identical to the reference path. On one B200 the layer takes 8.13 ms
against 8.73 ms (cuBLAS 7.71 ms + SiluAndMul 1.03 ms).

Dispatch is an allowlist: unquantized contiguous CUDA BF16 on SM100-family
devices, with the per-shard ``ffn`` a multiple of the 128-column tile. Other
inputs keep the reference path.
"""

from __future__ import annotations

import torch
from torch.library import Library
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 64
# Tiles walk the M dimension in groups of 16 so consecutive tiles on an SM
# reuse the same weight columns from L2 (8: -0.3 ms, 32: -0.2 ms, 64: +0.6 ms).
_GROUP_M = 16
_NUM_STAGES = 4
_NUM_WARPS = 4

_allocator_ready = False


def _ensure_tma_allocator() -> None:
    # Device-side TMA descriptors need a global-memory scratch allocator.
    global _allocator_ready
    if not _allocator_ready:
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        _allocator_ready = True


if HAS_TRITON:

    @triton.jit
    def _fc1_swiglu_kernel(
        a_ptr,
        w_ptr,
        out_ptr,
        M,
        N,
        K,
        NUM_SMS: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
        # Merged fc1 weight [2N, K]: rows [0, N) are gate, rows [N, 2N) are up.
        w_desc = tl.make_tensor_descriptor(w_ptr, shape=[2 * N, K], strides=[K, 1], block_shape=[BN, BK])
        o_desc = tl.make_tensor_descriptor(out_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN // 2])
        num_m = tl.cdiv(M, BM)
        num_n = tl.cdiv(N, BN)
        num_tiles = num_m * num_n
        group = GROUP_M * num_n
        for tile in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=False, warp_specialize=True):
            first_m = (tile // group) * GROUP_M
            size_m = min(num_m - first_m, GROUP_M)
            pid_m = first_m + (tile % group) % size_m
            pid_n = (tile % group) // size_m
            off_m = pid_m * BM
            off_n = pid_n * BN
            acc_gate = tl.zeros([BM, BN], dtype=tl.float32)
            acc_up = tl.zeros([BM, BN], dtype=tl.float32)
            for k in range(0, K, BK):
                a = a_desc.load([off_m, k])
                acc_gate = tl.dot(a, w_desc.load([off_n, k]).T, acc_gate)
                acc_up = tl.dot(a, w_desc.load([N + off_n, k]).T, acc_up)
            gate = acc_gate.to(tl.bfloat16).to(tl.float32)
            up = acc_up.to(tl.bfloat16).to(tl.float32)
            act = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
            # Store in two half-width subtiles so the store overlaps the next tile's mainloop.
            out = tl.reshape((act * up).to(tl.bfloat16), [BM, 2, BN // 2])
            out_lo, out_hi = tl.split(tl.permute(out, (0, 2, 1)))
            o_desc.store([off_m, off_n], out_lo)
            o_desc.store([off_m, off_n + BN // 2], out_hi)


def fc1_swiglu_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    if not HAS_TRITON or not (x.is_cuda and weight.is_cuda):
        return False
    if torch.cuda.get_device_capability(x.device)[0] != 10:
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if weight.ndim != 2 or not weight.is_contiguous() or x.shape[-1] != weight.shape[1]:
        return False
    k = weight.shape[1]
    n = weight.shape[0] // 2
    return weight.shape[0] % 2 == 0 and n % _BLOCK_N == 0 and k % 8 == 0


def _fc1_swiglu_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _ensure_tma_allocator()
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x2d.shape
    n = weight.shape[0] // 2
    out = torch.empty(m, n, device=x.device, dtype=x.dtype)
    if m:
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        _fc1_swiglu_kernel[(sms,)](
            x2d,
            weight,
            out,
            m,
            n,
            k,
            NUM_SMS=sms,
            BM=_BLOCK_M,
            BN=_BLOCK_N,
            BK=_BLOCK_K,
            GROUP_M=_GROUP_M,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
    return out.view(*x.shape[:-1], n)


def _fc1_swiglu_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], weight.shape[0] // 2)


_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "minimax_h3_fc1_swiglu"):
    direct_register_custom_op(
        op_name="minimax_h3_fc1_swiglu",
        op_func=_fc1_swiglu_impl,
        fake_impl=_fc1_swiglu_fake,
        mutates_args=[],
        target_lib=_OP_LIB,
    )


def fc1_swiglu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``silu(x @ gate.T) * (x @ up.T)`` for the merged fc1 weight ``[gate; up]``."""
    return torch.ops.vllm_omni.minimax_h3_fc1_swiglu(x, weight)
