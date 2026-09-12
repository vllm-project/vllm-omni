# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Paired interleaved RoPE with Diffusers-compatible arithmetic.

This implementation supports the BF16 production tables as well as the FP32 construction-time
tables without changing either eager numerical contract.
"""

from __future__ import annotations

import torch
from torch.library import Library
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_BLOCK_SIZE = 256


if HAS_TRITON:

    @triton.jit
    def _round_bf16_to_fp32(value):
        """RNE-round FP32 to BF16 precision while retaining an FP32 value."""
        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b16 rounded;
                cvt.rn.bf16.f32 rounded, $1;
                cvt.f32.bf16 $0, rounded;
            }
            """,
            constraints="=f,f",
            args=[value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _mul_rn_f32(left, right):
        return tl.inline_asm_elementwise(
            asm="mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[left, right],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _add_rn_f32(left, right):
        return tl.inline_asm_elementwise(
            asm="add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[left, right],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sub_rn_f32(left, right):
        return tl.inline_asm_elementwise(
            asm="sub.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[left, right],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _mul_for_table_dtype(left, right, table_is_bf16: tl.constexpr):
        product = _mul_rn_f32(left, right)
        if table_is_bf16:
            # Eager BF16 * BF16 materializes a BF16 multiplication result
            # before the subsequent add/subtract kernel consumes it.
            product = _round_bf16_to_fp32(product)
        return product

    @triton.jit(
        do_not_specialize=["num_pairs", "seq_len"],
        do_not_specialize_on_alignment=[
            "q_ptr",
            "k_ptr",
            "cos_ptr",
            "sin_ptr",
            "num_pairs",
            "seq_len",
        ],
    )
    def _interleaved_rope_kernel(
        q_ptr,
        k_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        num_pairs,
        seq_len,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        table_is_bf16: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pair_indices = tl.program_id(0) * block_size + tl.arange(0, block_size)
        pair_mask = pair_indices < num_pairs
        even_offsets = pair_indices * 2
        odd_offsets = even_offsets + 1

        # Sequence length is deliberately a non-specialized runtime scalar so
        # new frame counts and resolutions reuse the same compiled kernel.
        # Heads and head dimension are model constants and remain specialized
        # so the compiler can replace integer division with cheaper indexing.
        pairs_per_head: tl.constexpr = head_dim // 2
        pair_in_head = pair_indices % pairs_per_head
        row = pair_indices // (num_heads * pairs_per_head)
        token = row % seq_len
        table_offsets = token * head_dim + pair_in_head * 2

        cos = tl.load(cos_ptr + table_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + table_offsets + 1, mask=pair_mask, other=0.0).to(tl.float32)

        q_even = tl.load(q_ptr + even_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        q_odd = tl.load(q_ptr + odd_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        q_even_out = _sub_rn_f32(
            _mul_for_table_dtype(q_even, cos, table_is_bf16),
            _mul_for_table_dtype(q_odd, sin, table_is_bf16),
        )
        q_odd_out = _add_rn_f32(
            _mul_for_table_dtype(q_even, sin, table_is_bf16),
            _mul_for_table_dtype(q_odd, cos, table_is_bf16),
        )
        tl.store(q_out_ptr + even_offsets, q_even_out, mask=pair_mask)
        tl.store(q_out_ptr + odd_offsets, q_odd_out, mask=pair_mask)

        k_even = tl.load(k_ptr + even_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        k_odd = tl.load(k_ptr + odd_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        k_even_out = _sub_rn_f32(
            _mul_for_table_dtype(k_even, cos, table_is_bf16),
            _mul_for_table_dtype(k_odd, sin, table_is_bf16),
        )
        k_odd_out = _add_rn_f32(
            _mul_for_table_dtype(k_even, sin, table_is_bf16),
            _mul_for_table_dtype(k_odd, cos, table_is_bf16),
        )
        tl.store(k_out_ptr + even_offsets, k_even_out, mask=pair_mask)
        tl.store(k_out_ptr + odd_offsets, k_odd_out, mask=pair_mask)


def can_use_fused_interleaved_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Return whether the CUDA kernel supports these exact input semantics."""
    if q.ndim != 4:
        return False
    expected_table_shape = (1, q.shape[1], 1, q.shape[3])
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and q.dtype is torch.bfloat16
        and k.dtype is q.dtype
        and q.is_cuda
        and k.is_cuda
        and q.device == k.device == cos.device == sin.device
        and k.shape == q.shape
        and all(size > 0 for size in q.shape)
        and q.shape[-1] % 2 == 0
        and q.is_contiguous()
        and k.is_contiguous()
        and cos.dtype in (torch.bfloat16, torch.float32)
        and sin.dtype is cos.dtype
        and cos.shape == expected_table_shape
        and sin.shape == expected_table_shape
        and cos.is_contiguous()
        and sin.is_contiguous()
        and not any(tensor.requires_grad for tensor in (q, k, cos, sin))
    )


def _fused_interleaved_rope_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    num_pairs = q.numel() // 2
    grid = (triton.cdiv(num_pairs, _BLOCK_SIZE),)
    _interleaved_rope_kernel[grid](
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        num_pairs,
        seq_len=q.shape[1],
        num_heads=q.shape[2],
        head_dim=q.shape[3],
        table_is_bf16=cos.dtype is torch.bfloat16,
        block_size=_BLOCK_SIZE,
        num_warps=4,
    )
    return q_out, k_out


def _fused_interleaved_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del cos, sin
    return torch.empty_like(q), torch.empty_like(k)


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "fused_interleaved_rope"):
    direct_register_custom_op(
        op_name="fused_interleaved_rope",
        op_func=_fused_interleaved_rope_impl,
        fake_impl=_fused_interleaved_rope_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def fused_interleaved_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch paired interleaved RoPE for inputs accepted by the predicate."""
    if not can_use_fused_interleaved_rope(q, k, cos, sin):
        raise ValueError("Unsupported inputs for fused interleaved RoPE")
    return torch.ops.vllm_omni.fused_interleaved_rope(q, k, cos, sin)


__all__ = ["can_use_fused_interleaved_rope", "fused_interleaved_rope"]
