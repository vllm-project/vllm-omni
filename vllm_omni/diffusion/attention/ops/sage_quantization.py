# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from Block-Sparse-Attention's csrc/fwd/sm120_blk64/bsa_quant_sm120_sage.py
# and bsa_sage_quant.py: Triton quantization kernels that produce the exact
# QK-INT8 / PV-FP8 operand layout consumed by
# flash_fwd_sm120_sage.BlockSparseAttnForwardSageSm120Blk64.
#
# Phase-1 scope note: unlike upstream, this module has no AOT-compiled quant
# runtime fallback and no external-workspace reuse parameter -- both are
# optional performance paths, not required for correctness. Only the plain
# Triton JIT path is ported here.

from collections.abc import Sequence

import torch
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice

from vllm_omni.platforms import current_omni_platform

SAGE_Q_GROUP_SIZE = 32
SAGE_Q_BLOCK_SIZE = 128
SAGE_K_BLOCK_SIZE = 64
SAGE_KV_STATS_CHUNK = 256
SAGE_HEAD_DIM = 128
SAGE_V_SCALE_MAX = 2.25


# =============================================================================
# Triton kernels
# =============================================================================


@triton.jit
def _quantize_sage_q_kernel(
    q_ptr,
    q8_ptr,
    scale_ptr,
    q_stride_b,
    q_stride_h,
    q_stride_s,
    q8_stride_b,
    q8_stride_h,
    q8_stride_s,
    scale_stride_b,
    scale_stride_h,
    seqlen_q,
    batch_size,
    HEAD_DIM: tl.constexpr,  # noqa: N803
    GROUP_SIZE: tl.constexpr,  # noqa: N803
):
    group_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    row_idx = group_idx * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    dim_idx = tl.arange(0, HEAD_DIM)
    valid = row_idx < seqlen_q
    q = tl.load(
        q_ptr + batch_idx * q_stride_b + head_idx * q_stride_h + row_idx[:, None] * q_stride_s + dim_idx[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    row_max = tl.max(tl.abs(q), axis=1)
    amax = tl.maximum(tl.max(row_max, axis=0), 1.0e-7)
    scale = amax / 127.0
    q_quant = tl.maximum(
        tl.minimum(libdevice.rint(q / scale), 127.0),
        -127.0,
    )
    tl.store(
        q8_ptr + batch_idx * q8_stride_b + head_idx * q8_stride_h + row_idx[:, None] * q8_stride_s + dim_idx[None, :],
        q_quant.to(tl.int8),
        mask=valid[:, None],
    )
    tl.store(
        scale_ptr + batch_idx * scale_stride_b + head_idx * scale_stride_h + group_idx,
        scale,
    )


@triton.jit
def _sage_kv_stats_partial_kernel(
    k_ptr,
    v_ptr,
    k_partial_ptr,
    v_partial_ptr,
    k_stride_b,
    k_stride_h,
    k_stride_s,
    v_stride_b,
    v_stride_h,
    v_stride_s,
    partial_stride_b,
    partial_stride_h,
    partial_stride_c,
    seqlen_k,
    heads,
    batch_size,
    HEAD_DIM: tl.constexpr,  # noqa: N803
    CHUNK_SIZE: tl.constexpr,  # noqa: N803
    DIM_TILE: tl.constexpr,  # noqa: N803
):
    chunk_idx = tl.program_id(0)
    dim_tile_idx = tl.program_id(1)
    batch_head_idx = tl.program_id(2)
    batch_idx = batch_head_idx // heads
    head_idx = batch_head_idx - batch_idx * heads

    row_idx = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    dim_idx = dim_tile_idx * DIM_TILE + tl.arange(0, DIM_TILE)
    valid = row_idx < seqlen_k
    k = tl.load(
        k_ptr + batch_idx * k_stride_b + head_idx * k_stride_h + row_idx[:, None] * k_stride_s + dim_idx[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + batch_idx * v_stride_b + head_idx * v_stride_h + row_idx[:, None] * v_stride_s + dim_idx[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    partial_base = batch_idx * partial_stride_b + head_idx * partial_stride_h + chunk_idx * partial_stride_c + dim_idx
    tl.store(partial_base + k_partial_ptr, tl.sum(k, axis=0))
    tl.store(partial_base + v_partial_ptr, tl.max(tl.abs(v), axis=0))


@triton.jit
def _sage_kv_stats_finalize_kernel(
    k_partial_ptr,
    v_partial_ptr,
    k_mean_ptr,
    v_scale_ptr,
    partial_stride_b,
    partial_stride_h,
    partial_stride_c,
    mean_stride_b,
    mean_stride_h,
    scale_stride_b,
    scale_stride_h,
    num_chunks,
    seqlen_k,
    heads,
    batch_size,
    DIM_TILE: tl.constexpr,  # noqa: N803
    REDUCE_TILE: tl.constexpr,  # noqa: N803
    SCALE_MAX: tl.constexpr,  # noqa: N803
):
    dim_tile_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    batch_idx = batch_head_idx // heads
    head_idx = batch_head_idx - batch_idx * heads
    dim_idx = dim_tile_idx * DIM_TILE + tl.arange(0, DIM_TILE)

    sum_acc = tl.zeros((DIM_TILE,), tl.float32)
    max_acc = tl.zeros((DIM_TILE,), tl.float32)
    chunk_base = 0
    while chunk_base < num_chunks:
        chunk_idx = chunk_base + tl.arange(0, REDUCE_TILE)
        mask = chunk_idx < num_chunks
        partial_offset = (
            batch_idx * partial_stride_b
            + head_idx * partial_stride_h
            + chunk_idx[:, None] * partial_stride_c
            + dim_idx[None, :]
        )
        partial_sum = tl.load(
            k_partial_ptr + partial_offset,
            mask=mask[:, None],
            other=0.0,
        )
        partial_max = tl.load(
            v_partial_ptr + partial_offset,
            mask=mask[:, None],
            other=0.0,
        )
        sum_acc += tl.sum(partial_sum, axis=0)
        max_acc = tl.maximum(max_acc, tl.max(partial_max, axis=0))
        chunk_base += REDUCE_TILE

    mean = sum_acc / seqlen_k
    scale = tl.maximum(max_acc, 1.0e-7) / SCALE_MAX
    tl.store(
        k_mean_ptr + batch_idx * mean_stride_b + head_idx * mean_stride_h + dim_idx,
        mean,
    )
    tl.store(
        v_scale_ptr + batch_idx * scale_stride_b + head_idx * scale_stride_h + dim_idx,
        scale,
    )


@triton.jit
def _quantize_sage_kv_kernel(
    k_ptr,
    v_ptr,
    k_mean_ptr,
    v_scale_ptr,
    k8_ptr,
    v8_ptr,
    k_scale_ptr,
    k_stride_b,
    k_stride_h,
    k_stride_s,
    v_stride_b,
    v_stride_h,
    v_stride_s,
    mean_stride_b,
    mean_stride_h,
    v_scale_stride_b,
    v_scale_stride_h,
    k8_stride_b,
    k8_stride_h,
    k8_stride_s,
    v8_stride_b,
    v8_stride_h,
    v8_stride_d,
    k_scale_stride_b,
    k_scale_stride_h,
    seqlen_k,
    batch_size,
    HEAD_DIM: tl.constexpr,  # noqa: N803
    BLOCK_SIZE: tl.constexpr,  # noqa: N803
):
    block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    local_row = tl.arange(0, BLOCK_SIZE)
    row_idx = block_idx * BLOCK_SIZE + local_row
    dim_idx = tl.arange(0, HEAD_DIM)
    valid = row_idx < seqlen_k

    mean = tl.load(k_mean_ptr + batch_idx * mean_stride_b + head_idx * mean_stride_h + dim_idx).to(tl.float32)
    v_scale = tl.load(v_scale_ptr + batch_idx * v_scale_stride_b + head_idx * v_scale_stride_h + dim_idx).to(tl.float32)
    k = tl.load(
        k_ptr + batch_idx * k_stride_b + head_idx * k_stride_h + row_idx[:, None] * k_stride_s + dim_idx[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + batch_idx * v_stride_b + head_idx * v_stride_h + row_idx[:, None] * v_stride_s + dim_idx[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    k_centered = tl.where(valid[:, None], k - mean[None, :], 0.0)
    row_max = tl.max(tl.abs(k_centered), axis=1)
    k_amax = tl.maximum(tl.max(row_max, axis=0), 1.0e-7)
    k_scale = k_amax / 127.0
    k_quant = tl.maximum(
        tl.minimum(libdevice.rint(k_centered / k_scale), 127.0),
        -127.0,
    )
    tl.store(
        k8_ptr + batch_idx * k8_stride_b + head_idx * k8_stride_h + row_idx[:, None] * k8_stride_s + dim_idx[None, :],
        k_quant.to(tl.int8),
        mask=valid[:, None],
    )
    tl.store(
        k_scale_ptr + batch_idx * k_scale_stride_b + head_idx * k_scale_stride_h + block_idx,
        k_scale,
    )

    # This 16-token physical permutation must stay bit-for-bit consistent
    # with the P-fragment byte-lane shuffle in
    # flash_fwd_sm120_sage._make_acc_into_fp8_op -- the two together make the
    # FP8 PV MMA's A-operand mapping lane-local without a runtime transpose.
    row_mod = local_row % 16
    physical_row = (
        block_idx * BLOCK_SIZE + (local_row // 16) * 16 + (row_mod // 8) * 2 + ((row_mod // 2) % 4) * 4 + row_mod % 2
    )
    v_quant = tl.where(valid[:, None], v / v_scale[None, :], 0.0)
    tl.store(
        v8_ptr
        + batch_idx * v8_stride_b
        + head_idx * v8_stride_h
        + dim_idx[:, None] * v8_stride_d
        + physical_row[None, :],
        tl.trans(v_quant).to(v8_ptr.dtype.element_ty),
    )


# =============================================================================
# Public API
# =============================================================================


def _require_sm120_bf16_bhsd(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be a rank-4 BHSD tensor")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must use torch.bfloat16")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have a contiguous head-dim (stride(-1) == 1)")
    if tensor.shape[-1] != SAGE_HEAD_DIM:
        raise ValueError(f"{name} must have head dimension 128")
    if tensor.shape[0] < 1 or tensor.shape[1] < 1 or tensor.shape[2] < 1:
        raise ValueError(f"{name} requires positive B, H, and sequence length")
    if current_omni_platform.get_device_capability(
        tensor.device.index if tensor.device.index is not None else torch.accelerator.current_device_index()
    ) != (12, 0):
        raise RuntimeError("SM120 Sage quantization requires compute capability 12.0")


def _require_output(
    name: str,
    tensor: torch.Tensor,
    shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _normalize_out(
    out: Sequence[torch.Tensor] | None,
    specs: tuple,
    device: torch.device,
) -> tuple:
    if out is None:
        return tuple(torch.empty(shape, dtype=dtype, device=device) for _, shape, dtype in specs)
    if len(out) != len(specs):
        raise ValueError(f"out must contain {len(specs)} tensors")
    result = tuple(out)
    for tensor, (name, shape, dtype) in zip(result, specs, strict=True):
        _require_output(name, tensor, shape, dtype, device)
    return result


def quantize_sage_q_sm120(
    q: torch.Tensor,
    *,
    out: Sequence[torch.Tensor] | None = None,
):
    """Quantize BF16 BHSD Q to Sage per-32-token-group INT8 on SM120.

    Args:
        q: Query tensor, contiguous BHSD bfloat16, head_dim=128.
        out: Optional pre-allocated ``(q_int8, q_scale)`` output tensors.

    Returns:
        ``(q_int8, q_scale)``: ``q_int8`` is int8 with the same BHSD shape as
        ``q``. ``q_scale`` is float32 with shape
        ``(B, H, ceil(Sq/128) * 4)`` -- one scale per 32-token group, padded
        to four groups per 128-query tile.
    """
    _require_sm120_bf16_bhsd("Q", q)
    batch, heads, seqlen_q, head_dim = q.shape
    num_groups = triton.cdiv(seqlen_q, SAGE_Q_BLOCK_SIZE) * (SAGE_Q_BLOCK_SIZE // SAGE_Q_GROUP_SIZE)
    q_int8, q_scale = _normalize_out(
        out,
        (
            ("q_int8", tuple(q.shape), torch.int8),
            ("q_scale", (batch, heads, num_groups), torch.float32),
        ),
        q.device,
    )
    _quantize_sage_q_kernel[(num_groups, heads, batch)](
        q,
        q_int8,
        q_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q_int8.stride(0),
        q_int8.stride(1),
        q_int8.stride(2),
        q_scale.stride(0),
        q_scale.stride(1),
        seqlen_q,
        batch,
        HEAD_DIM=head_dim,
        GROUP_SIZE=SAGE_Q_GROUP_SIZE,
        num_warps=8,
    )
    return q_int8, q_scale


def quantize_sage_kv_sm120(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: Sequence[torch.Tensor] | None = None,
):
    """Quantize BF16 BHSD K/V to Sage K64 INT8 and per-channel FP8 on SM120.

    K is channel-mean-centered before INT8 quantization (one scale per K64
    tile). V is FP8 E4M3 with a per-``[B,H,D]``-channel scale, stored in
    Sage's ``[B, H, D, round_up(Sk, 64)]`` layout with the 16-token physical
    permutation baked in by the kernel used to feed the FP8 PV MMA directly
    (see flash_fwd_sm120_sage._make_acc_into_fp8_op).

    Args:
        k: Key tensor, contiguous BHSD bfloat16, head_dim=128.
        v: Value tensor, same shape/dtype/layout as ``k``.
        out: Optional pre-allocated ``(k_int8, v_fp8, k_scale, v_scale)``
            output tensors.

    Returns:
        ``(k_int8, v_fp8, k_scale, v_scale)``.
    """
    _require_sm120_bf16_bhsd("K", k)
    _require_sm120_bf16_bhsd("V", v)
    if v.shape != k.shape:
        raise ValueError("V must have the same shape as K")
    if v.device != k.device:
        raise ValueError("K and V must be on the same CUDA device")

    batch, heads, seqlen_k, head_dim = k.shape
    padded_len = triton.cdiv(seqlen_k, SAGE_K_BLOCK_SIZE) * SAGE_K_BLOCK_SIZE
    num_blocks = padded_len // SAGE_K_BLOCK_SIZE
    specs = (
        ("k_int8", tuple(k.shape), torch.int8),
        ("v_fp8", (batch, heads, head_dim, padded_len), torch.float8_e4m3fn),
        ("k_scale", (batch, heads, num_blocks), torch.float32),
        ("v_scale", (batch, heads, head_dim), torch.float32),
    )
    k_int8, v_fp8, k_scale, v_scale = _normalize_out(out, specs, k.device)

    num_chunks = triton.cdiv(seqlen_k, SAGE_KV_STATS_CHUNK)
    dim_tile = 32
    k_partial = torch.empty((batch, heads, num_chunks, head_dim), dtype=torch.float32, device=k.device)
    v_partial = torch.empty_like(k_partial)
    k_mean = torch.empty((batch, heads, head_dim), dtype=torch.bfloat16, device=k.device)

    _sage_kv_stats_partial_kernel[(num_chunks, head_dim // dim_tile, batch * heads)](
        k,
        v,
        k_partial,
        v_partial,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k_partial.stride(0),
        k_partial.stride(1),
        k_partial.stride(2),
        seqlen_k,
        heads,
        batch,
        HEAD_DIM=head_dim,
        CHUNK_SIZE=SAGE_KV_STATS_CHUNK,
        DIM_TILE=dim_tile,
        num_warps=8,
    )
    _sage_kv_stats_finalize_kernel[(head_dim // dim_tile, batch * heads)](
        k_partial,
        v_partial,
        k_mean,
        v_scale,
        k_partial.stride(0),
        k_partial.stride(1),
        k_partial.stride(2),
        k_mean.stride(0),
        k_mean.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        num_chunks,
        seqlen_k,
        heads,
        batch,
        DIM_TILE=dim_tile,
        REDUCE_TILE=16,
        SCALE_MAX=SAGE_V_SCALE_MAX,
        num_warps=4,
    )
    _quantize_sage_kv_kernel[(num_blocks, heads, batch)](
        k,
        v,
        k_mean,
        v_scale,
        k_int8,
        v_fp8,
        k_scale,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k_mean.stride(0),
        k_mean.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        k_int8.stride(0),
        k_int8.stride(1),
        k_int8.stride(2),
        v_fp8.stride(0),
        v_fp8.stride(1),
        v_fp8.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        seqlen_k,
        batch,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=SAGE_K_BLOCK_SIZE,
        num_warps=8,
    )
    return k_int8, v_fp8, k_scale, v_scale


def quantize_sage_qkv_sm120(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: Sequence[torch.Tensor] | None = None,
):
    """Quantize BF16 BHSD Q/K/V using the native Sage SM120 contract.

    Convenience wrapper around :func:`quantize_sage_q_sm120` and
    :func:`quantize_sage_kv_sm120`.

    Returns:
        ``(q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale)``.
    """
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
        raise ValueError("Q, K, and V must use matching batch/head dimensions")
    if out is None:
        q_out = None
        kv_out = None
    else:
        if len(out) != 6:
            raise ValueError("out must contain six tensors")
        q_out = (out[0], out[3])
        kv_out = (out[1], out[2], out[4], out[5])

    q_int8, q_scale = quantize_sage_q_sm120(q, out=q_out)
    k_int8, v_fp8, k_scale, v_scale = quantize_sage_kv_sm120(k, v, out=kv_out)
    return q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale


__all__ = [
    "quantize_sage_kv_sm120",
    "quantize_sage_q_sm120",
    "quantize_sage_qkv_sm120",
]
