# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused attention prologue for models with a packed QKV projection.

A Qwen3-style attention layer with GQA does six things between its projection
and its SDPA call: split the packed QKV, RMSNorm Q and K per head, rotate both
with RoPE, broadcast K and V across their query groups, and transpose all three
into the ``[batch, heads, positions, head_dim]`` layout SDPA wants. Every one of
those is cheap arithmetic over the whole activation, so together they cost
several full round-trips of Q, K and V through memory.

``fused_qkv_norm_rope`` does all of it in a single Triton pass. Each program
owns one tile of heads of one token: it reads that tile out of the packed
projection once, and writes the finished tensor straight into the attention
layout, broadcasting K and V to their query groups from registers. RoPE's
rotate-half partner is re-read from the same row rather than materialized as a
concatenation, and the RMS is a per-head scalar, so the partner needs no second
reduction.

Measured on OmniVoice (28 layers x 32 unmasking steps, A800, float16), the
unfused path launches 18 kernels between the packed projection and attention
and this replaces them with 1, cutting the generator's forward by 25-36%
depending on target length.

Falls back to an eager reference for non-CUDA tensors, missing Triton, or a
geometry the tiling cannot express.

Nothing here is OmniVoice-specific beyond the calling convention, but OmniVoice
is its only consumer today: every other model in this repo with a packed QKV
projection runs on vLLM's own attention stack, whose backends already handle
GQA and RoPE, so none of them performs the sequence this replaces. It therefore
lives beside its consumer rather than in models/common/, the same placement
fused_qk_norm_rope took under diffusion/. Lift it out when a second model wants
it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON, tl, triton

# One tile of heads per program. Head counts must be a multiple of this so that
# a tile never straddles the Q/K/V boundary, which keeps the branch uniform.
_HEAD_TILE = 8


if HAS_TRITON:

    @triton.jit
    def _qkv_norm_rope_kernel(
        qkv_ptr,
        q_weight_ptr,
        k_weight_ptr,
        rope_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        qkv_stride_token,
        qkv_stride_head,
        rope_stride_pos,
        out_stride_batch,
        out_stride_head,
        out_stride_pos,
        seq_len,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_half: tl.constexpr,
        kv_repeat: tl.constexpr,
        head_tile: tl.constexpr,
        eps: tl.constexpr,
    ):
        token = tl.program_id(0)
        tile_id = tl.program_id(1)
        batch = token // seq_len
        pos = token % seq_len

        dims = tl.arange(0, head_dim)
        heads = tile_id * head_tile + tl.arange(0, head_tile)
        src = qkv_ptr + token * qkv_stride_token + heads[:, None] * qkv_stride_head + dims[None, :]

        # rotate_half pairs dim d with d +/- rotary_half, negating the lower half.
        partner = tl.where(dims < rotary_half, dims + rotary_half, dims - rotary_half)
        sign = tl.where(dims < rotary_half, -1.0, 1.0)
        freq = tl.where(dims < rotary_half, dims, dims - rotary_half)
        cos = tl.load(rope_ptr + pos * rope_stride_pos + freq)
        sin = tl.load(rope_ptr + pos * rope_stride_pos + rotary_half + freq)

        q_tiles: tl.constexpr = num_heads // head_tile
        kv_tiles: tl.constexpr = num_kv_heads // head_tile

        if tile_id < q_tiles:
            x = tl.load(src).to(tl.float32)
            weight = tl.load(q_weight_ptr + dims).to(tl.float32)
            inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / head_dim + eps)
            normed = x * inv_rms[:, None] * weight[None, :]
            pair_x = tl.load(
                qkv_ptr + token * qkv_stride_token + heads[:, None] * qkv_stride_head + partner[None, :]
            ).to(tl.float32)
            pair_weight = tl.load(q_weight_ptr + partner).to(tl.float32)
            pair_normed = pair_x * inv_rms[:, None] * pair_weight[None, :]
            out = normed * cos[None, :].to(tl.float32) + sign[None, :] * pair_normed * sin[None, :].to(tl.float32)
            dst = (
                q_out_ptr
                + batch * out_stride_batch
                + heads[:, None] * out_stride_head
                + pos * out_stride_pos
                + dims[None, :]
            )
            tl.store(dst, out.to(q_out_ptr.dtype.element_ty))

        elif tile_id < q_tiles + kv_tiles:
            # K is broadcast across its query group here, from registers.
            # Measured alternative: SDPA's enable_gqa does the broadcast with no
            # copy at all, but on torch 2.13 no fused kernel accepts mismatched
            # head counts, so it falls back to the math backend and runs 2.0-4.9x
            # slower than materializing K and V. Hence the broadcast, but folded
            # into a store that was happening anyway.
            kv_head = heads - num_heads
            x = tl.load(src).to(tl.float32)
            weight = tl.load(k_weight_ptr + dims).to(tl.float32)
            inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / head_dim + eps)
            normed = x * inv_rms[:, None] * weight[None, :]
            pair_x = tl.load(
                qkv_ptr + token * qkv_stride_token + heads[:, None] * qkv_stride_head + partner[None, :]
            ).to(tl.float32)
            pair_weight = tl.load(k_weight_ptr + partner).to(tl.float32)
            pair_normed = pair_x * inv_rms[:, None] * pair_weight[None, :]
            out = normed * cos[None, :].to(tl.float32) + sign[None, :] * pair_normed * sin[None, :].to(tl.float32)
            stored = out.to(k_out_ptr.dtype.element_ty)
            # Broadcast to the query group straight from registers.
            for rep in tl.static_range(kv_repeat):
                dst = (
                    k_out_ptr
                    + batch * out_stride_batch
                    + (kv_head[:, None] * kv_repeat + rep) * out_stride_head
                    + pos * out_stride_pos
                    + dims[None, :]
                )
                tl.store(dst, stored)

        else:
            kv_head = heads - num_heads - num_kv_heads
            stored = tl.load(src)
            for rep in tl.static_range(kv_repeat):
                dst = (
                    v_out_ptr
                    + batch * out_stride_batch
                    + (kv_head[:, None] * kv_repeat + rep) * out_stride_head
                    + pos * out_stride_pos
                    + dims[None, :]
                )
                tl.store(dst, stored)


def _eager_qkv_norm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The six unfused steps, kept as the reference the kernel is checked against."""
    head_dim = qkv.shape[-1]
    half = head_dim // 2
    q, k, v = qkv.split([num_heads, num_kv_heads, num_kv_heads], dim=2)

    cos = rope_table[: qkv.shape[1], :half]
    sin = rope_table[: qkv.shape[1], half:]
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(2)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(2)

    def norm_rope(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        normed = F.rms_norm(x.to(torch.float32), (head_dim,), weight.to(torch.float32), eps)
        rotated = torch.cat([-normed[..., half:], normed[..., :half]], dim=-1)
        return (normed * cos + rotated * sin).to(x.dtype)

    repeat = num_heads // num_kv_heads
    q = norm_rope(q, q_weight)
    k = norm_rope(k, k_weight).repeat_interleave(repeat, dim=2)
    v = v.repeat_interleave(repeat, dim=2)
    return (q.permute(0, 2, 1, 3).contiguous(), k.permute(0, 2, 1, 3).contiguous(), v.permute(0, 2, 1, 3).contiguous())


def fused_cuda_supported(qkv: torch.Tensor, num_heads: int, num_kv_heads: int) -> bool:
    """Whether the Triton path can express this geometry."""
    head_dim = qkv.shape[-1]
    return (
        HAS_TRITON
        and qkv.is_cuda
        and qkv.is_contiguous()
        and head_dim > 0
        and head_dim & (head_dim - 1) == 0  # a single tl.arange tiles the head
        and num_heads % _HEAD_TILE == 0
        and num_kv_heads % _HEAD_TILE == 0  # tiles never straddle the Q/K/V boundary
        and num_heads % num_kv_heads == 0
    )


def fused_qkv_norm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a packed QKV projection into attention-ready Q, K and V.

    Args:
        qkv: ``[batch, positions, num_heads + 2 * num_kv_heads, head_dim]``, the
            output of one packed projection.
        q_weight: ``[head_dim]`` per-head RMSNorm weight for Q.
        k_weight: ``[head_dim]`` per-head RMSNorm weight for K.
        rope_table: ``[positions, head_dim]``, the first half of each row holding
            ``cos(theta)`` and the second ``sin(theta)``, each ``head_dim // 2``
            wide. One row per position; the kernel repeats it across the batch.
        eps: RMSNorm epsilon.
        num_heads: number of query heads.
        num_kv_heads: number of key/value heads; must divide ``num_heads``.

    Returns:
        ``(q, k, v)``, each ``[batch, num_heads, positions, head_dim]`` and
        contiguous, with K and V already broadcast across their query groups.
    """
    batch, seq_len, total_heads, head_dim = qkv.shape
    if total_heads != num_heads + 2 * num_kv_heads:
        raise ValueError(f"packed QKV has {total_heads} heads, expected {num_heads} + 2 * {num_kv_heads}")
    if rope_table.shape[0] < seq_len or rope_table.shape[-1] != head_dim:
        raise ValueError(f"rope_table {tuple(rope_table.shape)} does not cover [{seq_len}, {head_dim}]")

    if not fused_cuda_supported(qkv, num_heads, num_kv_heads):
        return _eager_qkv_norm_rope(qkv, q_weight, k_weight, rope_table, eps, num_heads, num_kv_heads)

    out_shape = (batch, num_heads, seq_len, head_dim)
    q_out = torch.empty(out_shape, dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty(out_shape, dtype=qkv.dtype, device=qkv.device)
    v_out = torch.empty(out_shape, dtype=qkv.dtype, device=qkv.device)

    rope = rope_table[:seq_len].contiguous()
    grid = (batch * seq_len, total_heads // _HEAD_TILE)
    _qkv_norm_rope_kernel[grid](
        qkv,
        q_weight,
        k_weight,
        rope,
        q_out,
        k_out,
        v_out,
        qkv.stride(1),
        qkv.stride(2),
        rope.stride(0),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_half=head_dim // 2,
        kv_repeat=num_heads // num_kv_heads,
        head_tile=_HEAD_TILE,
        eps=eps,
        num_warps=4,
    )
    return q_out, k_out, v_out


__all__ = ["fused_qkv_norm_rope", "fused_cuda_supported"]
