# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bit-exact paired Q/K full-width interleaved RoPE for CUDA.

The operator accepts already-normalized BF16 query and key tensors in BSND
layout and full-width FP32 cosine/sine tables.  It deliberately performs two
round-to-nearest FP32 multiplies followed by one round-to-nearest FP32 add,
then rounds only once when storing BF16 output.  This matches Diffusers'
``apply_rotary_emb`` arithmetic while combining its separate Q and K calls.
"""

from __future__ import annotations

import torch
from torch.library import Library
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_HEAD_DIM = 128
_HEADS_PER_PROGRAM = 4


if HAS_TRITON:

    @triton.jit
    def _mul_rn_f32(x, y):
        return tl.inline_asm_elementwise(
            asm="mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _add_rn_f32(x, y):
        return tl.inline_asm_elementwise(
            asm="add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _qk_rope_exact_kernel(
        q_ptr,
        k_ptr,
        cos_ptr,
        sin_ptr,
        q_output_ptr,
        k_output_ptr,
        q_stride_token,
        q_stride_head,
        q_stride_dim,
        k_stride_token,
        k_stride_head,
        k_stride_dim,
        q_output_stride_token,
        q_output_stride_head,
        q_output_stride_dim,
        k_output_stride_token,
        k_output_stride_head,
        k_output_stride_dim,
        table_stride_sequence,
        table_stride_dim,
        sequence,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        heads_per_program: tl.constexpr,
    ):
        token = tl.program_id(0)
        group = tl.program_id(1)
        is_key = tl.program_id(2) == 1
        heads = group * heads_per_program + tl.arange(0, heads_per_program)
        dims = tl.arange(0, head_dim)
        valid = heads[:, None] < num_heads

        q_offsets = token * q_stride_token + heads[:, None] * q_stride_head + dims[None, :] * q_stride_dim
        k_offsets = token * k_stride_token + heads[:, None] * k_stride_head + dims[None, :] * k_stride_dim
        input_ptrs = tl.where(is_key, k_ptr + k_offsets, q_ptr + q_offsets)

        pair_dims = dims ^ 1
        q_pair_offsets = token * q_stride_token + heads[:, None] * q_stride_head + pair_dims[None, :] * q_stride_dim
        k_pair_offsets = token * k_stride_token + heads[:, None] * k_stride_head + pair_dims[None, :] * k_stride_dim
        pair_ptrs = tl.where(is_key, k_ptr + k_pair_offsets, q_ptr + q_pair_offsets)
        values = tl.load(input_ptrs, mask=valid, other=0.0).to(tl.float32)
        pairs = tl.load(pair_ptrs, mask=valid, other=0.0).to(tl.float32)
        signed_pairs = tl.where((dims[None, :] & 1) == 0, -pairs, pairs)

        sequence_index = token % sequence
        table_offsets = sequence_index * table_stride_sequence + dims * table_stride_dim
        cos = tl.load(cos_ptr + table_offsets).to(tl.float32)[None, :]
        sin = tl.load(sin_ptr + table_offsets).to(tl.float32)[None, :]

        # Do not let the compiler contract this expression into an FMA.  The
        # reference has two independently rounded FP32 multiplies and one
        # independently rounded FP32 add, followed by the final BF16 store.
        first = _mul_rn_f32(values, cos)
        second = _mul_rn_f32(signed_pairs, sin)
        output = _add_rn_f32(first, second)
        q_output_offsets = (
            token * q_output_stride_token + heads[:, None] * q_output_stride_head + dims[None, :] * q_output_stride_dim
        )
        k_output_offsets = (
            token * k_output_stride_token + heads[:, None] * k_output_stride_head + dims[None, :] * k_output_stride_dim
        )
        output_ptrs = tl.where(
            is_key,
            k_output_ptr + k_output_offsets,
            q_output_ptr + q_output_offsets,
        )
        tl.store(output_ptrs, output, mask=valid)


def fused_qk_rope_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Return whether the exact CUDA kernel can consume these tensors."""

    if not (
        HAS_TRITON
        and torch.version.hip is None
        and current_platform.is_cuda()
        and not torch.is_grad_enabled()
        and q.is_cuda
        and k.is_cuda
        and q.device == k.device
        and q.dtype is torch.bfloat16
        and k.dtype is q.dtype
        and q.ndim == 4
        and k.shape == q.shape
        and q.numel() > 0
        and q.is_contiguous()
        and k.is_contiguous()
    ):
        return False

    batch, sequence, _heads, head_dim = q.shape
    del batch
    if head_dim != _HEAD_DIM:
        return False
    if not (
        cos.shape == (sequence, head_dim)
        and sin.shape == cos.shape
        and cos.device == q.device
        and sin.device == q.device
        and cos.dtype is torch.float32
        and sin.dtype is torch.float32
        and cos.is_contiguous()
        and sin.is_contiguous()
    ):
        return False
    return True


def _launch_fused_qk_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_output = torch.empty_like(q)
    k_output = torch.empty_like(k)
    tokens = q.shape[0] * q.shape[1]
    head_groups = triton.cdiv(q.shape[2], _HEADS_PER_PROGRAM)
    grid = (tokens, head_groups, 2)
    with torch.accelerator.device_index(q.device.index):
        _qk_rope_exact_kernel[grid](
            q,
            k,
            cos,
            sin,
            q_output,
            k_output,
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            q_output.stride(1),
            q_output.stride(2),
            q_output.stride(3),
            k_output.stride(1),
            k_output.stride(2),
            k_output.stride(3),
            cos.stride(0),
            cos.stride(1),
            q.shape[1],
            num_heads=q.shape[2],
            head_dim=q.shape[-1],
            heads_per_program=_HEADS_PER_PROGRAM,
            num_warps=4,
        )
    return q_output, k_output


def _fused_qk_rope_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not fused_qk_rope_supported(q, k, cos, sin):
        raise ValueError("fused_qk_rope received unsupported inputs")
    return _launch_fused_qk_rope(q, k, cos, sin)


def _fused_qk_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del cos, sin
    return torch.empty_like(q), torch.empty_like(k)


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "fused_qk_rope"):
    direct_register_custom_op(
        op_name="fused_qk_rope",
        op_func=_fused_qk_rope_impl,
        fake_impl=_fused_qk_rope_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def fused_qk_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply paired full-width adjacent/interleaved RoPE on CUDA."""

    if not fused_qk_rope_supported(q, k, cos, sin):
        raise ValueError("fused_qk_rope requires contiguous CUDA BF16 Q/K and contiguous full-width CUDA FP32 tables")
    return torch.ops.vllm_omni.fused_qk_rope(q, k, cos, sin)


__all__ = ["fused_qk_rope", "fused_qk_rope_supported"]
