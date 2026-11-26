# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""MiniMax-H3 DiT MLP: fc1 GEMM with the SwiGLU activation fused into its epilogue."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch.library import Library
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 64
_GROUP_M = 16
_NUM_STAGES = 4
_NUM_WARPS = 4


@contextlib.contextmanager
def _tma_allocator(device: torch.device) -> Iterator[None]:
    """Give the kernel's on-device TMA descriptors scratch on ``device`` for this launch only.

    ``triton.set_allocator`` sets a process-wide context variable; resetting it afterwards keeps whatever allocator
    other Triton kernels installed.
    """
    from triton.runtime import _allocation

    token = _allocation._allocator.set(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=device))
    try:
        yield
    finally:
        _allocation._allocator.reset(token)


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
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x2d.shape
    n = weight.shape[0] // 2
    out = torch.empty(m, n, device=x.device, dtype=x.dtype)
    if m:
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        with _tma_allocator(x.device):
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
    return x.new_empty((*x.shape[:-1], weight.shape[0] // 2))


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
