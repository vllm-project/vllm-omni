# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exact fused Q/K RMSNorm and RoPE for the MiniMax H3 video VAE."""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _f32_add(left, right):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [left, right],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _f32_fma(left, right, acc):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            [left, right, acc],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _fold_high_plus_low(values, half: tl.constexpr, heads_per_program: tl.constexpr):
        # Shuffle-down by half the active width adds the high lane into the low lane.
        view = tl.reshape(values, [2, half, heads_per_program])
        lane = tl.arange(0, 2)[:, None, None]
        high = tl.sum(tl.where(lane == 1, view, 0.0), axis=0)
        low = tl.sum(tl.where(lane == 0, view, 0.0), axis=0)
        return _f32_add(high, low)

    @triton.jit
    def _rms_sum_squares(values, heads_per_program: tl.constexpr):
        """Sum squares in PyTorch's N=64 vectorized RMSNorm order.

        Each group of four lanes is an ``fma.rn`` chain, matching one float4
        thread. Groups then combine like a warp shuffle-down. ``tl.sum`` does
        not keep that parenthesization.
        """

        dims = tl.arange(0, 64)[None, :]
        chunk_ids = tl.arange(0, 16)[:, None]
        chunks = tl.zeros([16, heads_per_program], dtype=tl.float32)
        for group in tl.static_range(16):
            acc = tl.zeros([heads_per_program], dtype=tl.float32)
            for offset in tl.static_range(4):
                val = tl.sum(tl.where(dims == group * 4 + offset, values, 0.0), axis=1)
                acc = _f32_fma(val, val, acc)
            chunks = tl.where(chunk_ids == group, acc[None, :], chunks)
        stage = _fold_high_plus_low(chunks, 8, heads_per_program)
        stage = _fold_high_plus_low(stage, 4, heads_per_program)
        stage = _fold_high_plus_low(stage, 2, heads_per_program)
        stage = _fold_high_plus_low(stage, 1, heads_per_program)
        return tl.reshape(stage, [heads_per_program])

    @triton.jit
    def _qk_rms_norm_rope_exact_kernel(
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
        rope_stride_token,
        rope_stride_dim,
        q_output_stride_token,
        q_output_stride_head,
        q_output_stride_dim,
        k_output_stride_token,
        k_output_stride_head,
        k_output_stride_dim,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        eps: tl.constexpr,
        heads_per_program: tl.constexpr,
    ):
        token = tl.program_id(0)
        head_group = tl.program_id(1)
        qk_domain = tl.program_id(2)
        heads = head_group * heads_per_program + tl.arange(0, heads_per_program)
        dims = tl.arange(0, head_dim)
        mask = heads[:, None] < num_heads

        q_offsets = token * q_stride_token + heads[:, None] * q_stride_head + dims[None, :] * q_stride_dim
        k_offsets = token * k_stride_token + heads[:, None] * k_stride_head + dims[None, :] * k_stride_dim
        x_ptrs = tl.where(qk_domain == 0, q_ptr + q_offsets, k_ptr + k_offsets)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        inv_rms = tl.rsqrt(_rms_sum_squares(x, heads_per_program) / head_dim + eps)
        normalized = (x * inv_rms[:, None]).to(tl.float16)

        rotary_half = rotary_dim // 2
        pair_dims = tl.where(
            dims < rotary_half,
            dims + rotary_half,
            tl.where(dims < rotary_dim, dims - rotary_half, dims),
        )
        q_pair_offsets = token * q_stride_token + heads[:, None] * q_stride_head + pair_dims[None, :] * q_stride_dim
        k_pair_offsets = token * k_stride_token + heads[:, None] * k_stride_head + pair_dims[None, :] * k_stride_dim
        pair_x_ptrs = tl.where(qk_domain == 0, q_ptr + q_pair_offsets, k_ptr + k_pair_offsets)
        pair_x = tl.load(pair_x_ptrs, mask=mask, other=0.0).to(tl.float32)
        pair_normalized = (pair_x * inv_rms[:, None]).to(tl.float16)

        rope_mask = dims < rotary_dim
        rope_offsets = token * rope_stride_token + dims * rope_stride_dim
        cos = tl.load(cos_ptr + rope_offsets, mask=rope_mask, other=0.0).to(tl.float16)
        sin = tl.load(sin_ptr + rope_offsets, mask=rope_mask, other=0.0).to(tl.float16)
        signed_pair = tl.where(
            dims[None, :] < rotary_half,
            -pair_normalized,
            pair_normalized,
        )
        # Preserve multiply-round, multiply-round, add-round. Ordinary Triton
        # arithmetic may contract this expression into an FMA.
        rotated = tl.inline_asm_elementwise(
            """
            {
                .reg .b32 first;
                .reg .b32 second;
                mul.rn.f16x2 first, $1, $2;
                mul.rn.f16x2 second, $3, $4;
                add.rn.f16x2 $0, first, second;
            }
            """,
            constraints="=r,r,r,r,r",
            args=[normalized, cos[None, :], signed_pair, sin[None, :]],
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )
        output = tl.where(rope_mask[None, :], rotated, normalized)

        q_output_offsets = (
            token * q_output_stride_token + heads[:, None] * q_output_stride_head + dims[None, :] * q_output_stride_dim
        )
        k_output_offsets = (
            token * k_output_stride_token + heads[:, None] * k_output_stride_head + dims[None, :] * k_output_stride_dim
        )
        output_ptrs = tl.where(
            qk_domain == 0,
            q_output_ptr + q_output_offsets,
            k_output_ptr + k_output_offsets,
        )
        tl.store(output_ptrs, output, mask=mask)


def _supported_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
) -> bool:
    cos, sin = rotary_pos_emb
    return (
        HAS_TRITON
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and q.is_cuda
        and k.is_cuda
        and q.device == k.device
        and q.dtype == torch.float16
        and k.dtype == q.dtype
        and q.ndim == 4
        and k.shape == q.shape
        and q.numel() > 0
        and q.shape[-1] == 64
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and cos.shape == sin.shape
        and cos.ndim == 4
        and cos.shape[:2] == q.shape[:2]
        and cos.shape[2] == 1
        and cos.shape[-1] == 48
        and cos.device == q.device
        and sin.device == q.device
        and cos.dtype == q.dtype
        and sin.dtype == q.dtype
        and cos.is_contiguous()
        and sin.is_contiguous()
        and cos.stride() == sin.stride()
    )


def try_qk_norm_rope_exact(
    q: torch.Tensor,
    k: torch.Tensor,
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the fused bit-exact result, or ``None`` outside its contract."""

    if not _supported_inputs(q, k, rotary_pos_emb):
        return None

    output_shape = q.shape
    batch, sequence, heads, head_dim = output_shape
    tokens = batch * sequence
    cos, sin = rotary_pos_emb
    q = q.reshape(tokens, heads, head_dim)
    k = k.reshape(tokens, heads, head_dim)
    cos = cos.reshape(tokens, cos.shape[-1])
    sin = sin.reshape(tokens, sin.shape[-1])
    q_output = torch.empty_like(q)
    k_output = torch.empty_like(k)
    # The RMS sum is an explicit PyTorch vectorized-RMS parenthesization, so
    # these launch dimensions no longer change its rounding. They stay fixed
    # at the shape validated on SM90 and SM103.
    heads_per_program = 8
    grid = (tokens, triton.cdiv(heads, heads_per_program), 2)
    launch_args = {
        "num_heads": heads,
        "head_dim": head_dim,
        "rotary_dim": cos.shape[-1],
        "eps": eps,
        "heads_per_program": heads_per_program,
        "num_warps": 4,
    }
    _qk_rms_norm_rope_exact_kernel[grid](
        q,
        k,
        cos,
        sin,
        q_output,
        k_output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        cos.stride(0),
        cos.stride(1),
        q_output.stride(0),
        q_output.stride(1),
        q_output.stride(2),
        k_output.stride(0),
        k_output.stride(1),
        k_output.stride(2),
        **launch_args,
    )
    return q_output.reshape(output_shape), k_output.reshape(output_shape)


__all__ = ["try_qk_norm_rope_exact"]
