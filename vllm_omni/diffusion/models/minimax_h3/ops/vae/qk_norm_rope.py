# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exact fused Q/K RMSNorm and RoPE for the MiniMax H3 video VAE."""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _qk_rms_norm_rope_exact_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        output_ptr,
        x_stride_token,
        x_stride_head,
        x_stride_dim,
        rope_stride_token,
        rope_stride_dim,
        output_stride_token,
        output_stride_head,
        output_stride_dim,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        eps: tl.constexpr,
        heads_per_program: tl.constexpr,
    ):
        token = tl.program_id(0)
        head_group = tl.program_id(1)
        heads = head_group * heads_per_program + tl.arange(0, heads_per_program)
        dims = tl.arange(0, head_dim)
        mask = heads[:, None] < num_heads

        offsets = token * x_stride_token + heads[:, None] * x_stride_head + dims[None, :] * x_stride_dim
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / head_dim + eps)
        normalized = (x * inv_rms[:, None]).to(tl.float16)

        rotary_half = rotary_dim // 2
        pair_dims = tl.where(
            dims < rotary_half,
            dims + rotary_half,
            tl.where(dims < rotary_dim, dims - rotary_half, dims),
        )
        pair_offsets = token * x_stride_token + heads[:, None] * x_stride_head + pair_dims[None, :] * x_stride_dim
        pair_x = tl.load(x_ptr + pair_offsets, mask=mask, other=0.0).to(tl.float32)
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

        output_offsets = (
            token * output_stride_token + heads[:, None] * output_stride_head + dims[None, :] * output_stride_dim
        )
        tl.store(output_ptr + output_offsets, output, mask=mask)


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
    # This launch layout is part of the exactness contract on the validated
    # SM90 and SM103 targets: changing the warp-to-row mapping changes the
    # FP32 RMS reduction order.
    heads_per_program = 8
    grid = (tokens, triton.cdiv(heads, heads_per_program))
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
        cos,
        sin,
        q_output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        cos.stride(0),
        cos.stride(1),
        q_output.stride(0),
        q_output.stride(1),
        q_output.stride(2),
        **launch_args,
    )
    _qk_rms_norm_rope_exact_kernel[grid](
        k,
        cos,
        sin,
        k_output,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        cos.stride(0),
        cos.stride(1),
        k_output.stride(0),
        k_output.stride(1),
        k_output.stride(2),
        **launch_args,
    )
    return q_output.reshape(output_shape), k_output.reshape(output_shape)


__all__ = ["try_qk_norm_rope_exact"]
