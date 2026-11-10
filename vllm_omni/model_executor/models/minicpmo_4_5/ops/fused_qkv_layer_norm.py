# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# ruff: noqa: N803

"""Fused packed-QKV split + per-head LayerNorm of q/k into the CFM KV cache."""

import torch
from vllm.triton_utils import tl, triton

from .fused_residual_layer_norm import _use_triton


@triton.jit
def _head_layer_norm(x, mask, dims, weight_ptr, bias_ptr, eps, HEAD_DIM: tl.constexpr, HAS_AFFINE: tl.constexpr):
    """LayerNorm over the last axis of ``(heads, dims)`` in fp32, with an optional per-dim affine."""
    x = x.to(tl.float32)
    mean = tl.sum(x, axis=1) / HEAD_DIM
    centered = tl.where(mask, x - mean[:, None], 0.0)
    variance = tl.sum(centered * centered, axis=1) / HEAD_DIM
    x = centered * tl.rsqrt(variance + eps)[:, None]
    if HAS_AFFINE:
        x = x * tl.load(weight_ptr + dims, mask=dims < HEAD_DIM, other=1.0).to(tl.float32)[None, :]
        x = x + tl.load(bias_ptr + dims, mask=dims < HEAD_DIM, other=0.0).to(tl.float32)[None, :]
    return x


@triton.jit
def _qkv_head_layer_norm_kernel(
    q_out_ptr,
    kv_ptr,
    qkv_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    rows_ptr,
    positions_ptr,
    frames,
    stride_qkv_n,
    stride_qkv_t,
    stride_qn,
    stride_qh,
    stride_qt,
    stride_kvn,
    stride_kvh,
    stride_kvt,
    q_eps,
    k_eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_Q_AFFINE: tl.constexpr,
    HAS_K_AFFINE: tl.constexpr,
    HAS_ROWS: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
):
    row = tl.program_id(0)
    n = row // frames
    t = row % frames
    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    mask = (heads[:, None] < HEADS) & (dims[None, :] < HEAD_DIM)
    inner = HEADS * HEAD_DIM
    src = qkv_ptr + n * stride_qkv_n + t * stride_qkv_t + heads[:, None] * HEAD_DIM + dims[None, :]

    q = tl.load(src, mask=mask, other=0.0)
    q = _head_layer_norm(q, mask, dims, q_weight_ptr, q_bias_ptr, q_eps, HEAD_DIM, HAS_Q_AFFINE)
    tl.store(q_out_ptr + n * stride_qn + heads[:, None] * stride_qh + t * stride_qt + dims[None, :], q, mask=mask)

    k = tl.load(src + inner, mask=mask, other=0.0)
    k = _head_layer_norm(k, mask, dims, k_weight_ptr, k_bias_ptr, k_eps, HEAD_DIM, HAS_K_AFFINE)
    kv_row = n
    if HAS_ROWS:
        kv_row = tl.load(rows_ptr + n).to(tl.int64)
    frame = t
    if HAS_POSITIONS:
        frame = tl.load(positions_ptr + row).to(tl.int64)
        mask = mask & (frame >= 0)
    dst = kv_ptr + kv_row * stride_kvn + heads[:, None] * stride_kvh + frame * stride_kvt + dims[None, :]
    tl.store(dst, k, mask=mask)

    v = tl.load(src + 2 * inner, mask=mask, other=0.0)
    tl.store(dst + HEAD_DIM, v, mask=mask)


def qkv_head_layer_norm(
    qkv: torch.Tensor,
    kv: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    q_norm: torch.nn.LayerNorm,
    k_norm: torch.nn.LayerNorm,
    rows: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Split packed ``qkv``, LayerNorm q/k per head, write q and interleaved ``kv``."""
    batch, frames, width = (int(dim) for dim in qkv.shape)
    inner = num_heads * head_dim
    if width != 3 * inner or qkv.stride(2) != 1:
        raise ValueError(f"qkv_head_layer_norm: qkv must be (N, T, 3*{inner}) with contiguous channels")
    if (
        kv.dim() != 4
        or (rows is None and int(kv.shape[0]) != batch)
        or int(kv.shape[1]) != num_heads
        or (positions is None and int(kv.shape[2]) < frames)
        or int(kv.shape[3]) != 2 * head_dim
        or kv.stride(3) != 1
    ):
        raise ValueError(f"qkv_head_layer_norm: kv must be (N, H, >= T, 2*D), got {tuple(kv.shape)}")
    if rows is not None and tuple(rows.shape) != (batch,):
        raise ValueError(f"qkv_head_layer_norm: rows must be ({batch},), got {tuple(rows.shape)}")
    if positions is not None and (tuple(positions.shape) != (batch, frames) or not positions.is_contiguous()):
        raise ValueError(f"qkv_head_layer_norm: positions must be contiguous ({batch}, {frames})")
    q_out = torch.empty((batch, num_heads, frames, head_dim), device=qkv.device, dtype=qkv.dtype)

    if not _use_triton(qkv):
        heads = qkv.view(batch, frames, 3, num_heads, head_dim)
        q_out.copy_(q_norm(heads[:, :, 0].transpose(1, 2)))
        keys = k_norm(heads[:, :, 1].transpose(1, 2))
        values = heads[:, :, 2].transpose(1, 2)
        if rows is None and positions is None:
            kv[:, :, :frames, :head_dim].copy_(keys)
            kv[:, :, :frames, head_dim:].copy_(values)
            return q_out
        pool_rows = rows.tolist() if rows is not None else range(batch)
        for n, pool_row in enumerate(pool_rows):
            frame_index = positions[n] if positions is not None else torch.arange(frames, device=kv.device)
            written = frame_index >= 0
            destination = frame_index[written].to(kv.device)
            kv[pool_row, :, destination, :head_dim] = keys[n][:, written]
            kv[pool_row, :, destination, head_dim:] = values[n][:, written]
        return q_out
    if batch * frames == 0:
        return q_out
    q_weight, q_bias, q_eps = q_norm.weight, q_norm.bias, float(q_norm.eps)
    k_weight, k_bias, k_eps = k_norm.weight, k_norm.bias, float(k_norm.eps)
    if (q_weight is None) != (q_bias is None) or (k_weight is None) != (k_bias is None):
        raise ValueError("qkv_head_layer_norm: LayerNorm affine needs both weight and bias")
    _qkv_head_layer_norm_kernel[(batch * frames,)](
        q_out,
        kv,
        qkv,
        q_weight if q_weight is not None else qkv,
        q_bias if q_bias is not None else qkv,
        k_weight if k_weight is not None else qkv,
        k_bias if k_bias is not None else qkv,
        rows if rows is not None else qkv,
        positions if positions is not None else qkv,
        frames,
        qkv.stride(0),
        qkv.stride(1),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        q_eps,
        k_eps,
        HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_H=triton.next_power_of_2(num_heads),
        BLOCK_D=triton.next_power_of_2(head_dim),
        HAS_Q_AFFINE=q_weight is not None,
        HAS_K_AFFINE=k_weight is not None,
        HAS_ROWS=rows is not None,
        HAS_POSITIONS=positions is not None,
        num_warps=4,
    )
    return q_out
