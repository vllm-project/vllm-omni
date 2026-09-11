# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Explicit Triton glue operators for the MOSS Audio Tokenizer v2 decoder.

These ops replace small glue regions of the streaming decoder hot path
(unpack+RoPE and boolean causal-mask construction for the packed ring layout)
with single kernel launches. Every op keeps an eager fallback with semantics
identical to the unfused reference.
"""

import math

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

libdevice = tl.extra.libdevice

# =============================================================================
# K1: unpack (B,T,3,H,D) in_proj output + interleaved GPT-J RoPE on Q/K
# =============================================================================


@triton.jit
def _rope_unpack_qkv_kernel(
    projected_ptr,
    offset_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    pair_elements,
    num_heads,
    num_frames,
    log_period_scale: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One element per (b, h, t, pair): rotate Q/K pairs and copy V pairs.

    ``projected`` holds the in_proj GEMM output viewed as ``(B,T,3,H,D)``;
    Q/K/V are written as three contiguous ``(B,H,T,D)`` tensors so the flat
    pair index maps to consecutive output addresses.
    """
    pairs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = pairs < pair_elements
    pair_index = pairs % half_dim
    row = pairs // half_dim
    time_index = row % num_frames
    head_row = row // num_frames
    head = head_row % num_heads
    batch = head_row // num_heads

    position = tl.load(offset_ptr + batch, mask=mask, other=0).to(tl.float32) + time_index
    # libdevice exp/cos/sin and disabled FP fusion keep every fp32 op
    # bit-identical to the eager torch mul/exp/cos/sin chain.
    frequency = libdevice.exp(pair_index.to(tl.float32) * log_period_scale)
    angle = position * frequency
    cosine = libdevice.cos(angle)
    sine = libdevice.sin(angle)

    dim = pair_index * 2
    head_dim = half_dim * 2
    input_base = ((batch * num_frames + time_index) * 3 * num_heads + head) * head_dim + dim
    head_stride = num_heads * head_dim
    query_real = tl.load(projected_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
    query_imag = tl.load(projected_ptr + input_base + 1, mask=mask, other=0.0).to(tl.float32)
    key_real = tl.load(projected_ptr + input_base + head_stride, mask=mask, other=0.0).to(tl.float32)
    key_imag = tl.load(projected_ptr + input_base + head_stride + 1, mask=mask, other=0.0).to(tl.float32)
    value_real = tl.load(projected_ptr + input_base + 2 * head_stride, mask=mask, other=0.0)
    value_imag = tl.load(projected_ptr + input_base + 2 * head_stride + 1, mask=mask, other=0.0)

    out_dtype = q_ptr.dtype.element_ty
    output_offset = row * head_dim + dim
    tl.store(q_ptr + output_offset, (query_real * cosine - query_imag * sine).to(out_dtype), mask=mask)
    tl.store(q_ptr + output_offset + 1, (query_real * sine + query_imag * cosine).to(out_dtype), mask=mask)
    tl.store(k_ptr + output_offset, (key_real * cosine - key_imag * sine).to(out_dtype), mask=mask)
    tl.store(k_ptr + output_offset + 1, (key_real * sine + key_imag * cosine).to(out_dtype), mask=mask)
    tl.store(v_ptr + output_offset, value_real, mask=mask)
    tl.store(v_ptr + output_offset + 1, value_imag, mask=mask)


def _eager_rope_unpack_qkv(
    projected: torch.Tensor, offset: torch.Tensor, max_period: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference path: permuted views plus the eager interleaved rotation."""
    q, k, v = projected.permute(2, 0, 3, 1, 4).unbind(0)
    batch_size, num_heads, query_length, head_dim = q.shape
    ds = torch.arange(head_dim // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / head_dim))
    ts = offset.float().view(-1, 1) + torch.arange(query_length, device=q.device, dtype=torch.float32)
    ts = ts.view(batch_size, 1, query_length, 1)
    dims = q.shape[:-1]
    q = q.view(*dims, head_dim // 2, 2)
    k = k.view(*dims, head_dim // 2, 2)
    qr, qi = q[..., 0].float(), q[..., 1].float()
    kr, ki = k[..., 0].float(), k[..., 1].float()
    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    qo = torch.stack([(qr * rotr - qi * roti).to(q.dtype), (qr * roti + qi * rotr).to(q.dtype)], dim=-1)
    ko = torch.stack([(kr * rotr - ki * roti).to(k.dtype), (kr * roti + ki * rotr).to(k.dtype)], dim=-1)
    return qo.view(*dims, head_dim), ko.view(*dims, head_dim), v


@torch.library.custom_op("moss_codec::rope_unpack_qkv", mutates_args=(), device_types="cuda")
def codec_rope_unpack_qkv(
    projected: torch.Tensor, offset: torch.Tensor, max_period: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack the interleaved in_proj output and rotate Q/K in one launch.

    ``projected`` is the contiguous ``(B,T,3,H,D)`` view of the in_proj GEMM
    output; ``offset`` holds the per-request absolute base positions. Returns
    contiguous ``(B,H,T,D)`` Q/K (RoPE applied with the same fp32 pair math as
    ``apply_rope``) and V.
    """
    batch_size, query_length, _, num_heads, head_dim = projected.shape
    if (
        not HAS_TRITON
        or not projected.is_cuda
        or projected.dtype not in (torch.float16, torch.bfloat16)
        or not projected.is_contiguous()
        or offset.dtype != torch.long
        or offset.shape != (batch_size,)
        or not offset.is_contiguous()
        or head_dim % 2
    ):
        return _eager_rope_unpack_qkv(projected, offset, max_period)

    q = torch.empty((batch_size, num_heads, query_length, head_dim), device=projected.device, dtype=projected.dtype)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    pair_elements = q.numel() // 2
    block_size = 256
    _rope_unpack_qkv_kernel[(triton.cdiv(pair_elements, block_size),)](
        projected,
        offset,
        q,
        k,
        v,
        pair_elements,
        num_heads,
        query_length,
        log_period_scale=-math.log(max_period) * 2 / head_dim,
        half_dim=head_dim // 2,
        BLOCK=block_size,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q, k, v


@codec_rope_unpack_qkv.register_fake
def _codec_rope_unpack_qkv_fake(projected, offset, max_period):
    batch_size, query_length, _, num_heads, head_dim = projected.shape
    shape = (batch_size, num_heads, query_length, head_dim)
    return (
        projected.new_empty(shape),
        projected.new_empty(shape),
        projected.new_empty(shape),
    )


# =============================================================================
# K2: boolean causal/context mask for the packed ring-physical KV layout
# =============================================================================


@triton.jit
def _packed_causal_mask_kernel(
    mask_ptr,
    end_offset_ptr,
    query_offset_ptr,
    valid_ptr,
    query_length,
    capacity,
    context: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Integer-exact physical-ring causal mask from pre-advance end offsets.

    Replicates the reference chain: chronological positions for each physical
    ring column (``RingKVCache.complete``), invalidation of columns at or past
    the post-chunk end offset, then the query/key delta causal+context test in
    ``MossAudioTokenizerMultiheadAttention.forward``.
    """
    columns = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid_column = columns < capacity
    batch = tl.program_id(2).to(tl.int64)
    time_index = tl.program_id(1)

    end_offset = tl.load(end_offset_ptr + batch)
    is_valid = tl.load(valid_ptr + batch).to(tl.int32)
    next_offset = tl.where(is_valid != 0, end_offset + query_length, end_offset)
    last_offset = end_offset + query_length - 1
    end_index = last_offset % capacity
    delta_index = columns - end_index
    key_position = tl.where(delta_index <= 0, last_offset + delta_index, last_offset + delta_index - capacity)
    key_position = tl.where(columns >= next_offset, -1, key_position)
    query_position = tl.load(query_offset_ptr + batch) + time_index
    delta = query_position - key_position
    allowed = (key_position >= 0) & (delta >= 0)
    if context > 0:
        allowed &= delta < context
    output = (batch * query_length + time_index) * capacity + columns
    tl.store(mask_ptr + output, allowed, mask=valid_column)


def _eager_packed_causal_mask(
    offset_end: torch.Tensor,
    query_offset: torch.Tensor,
    valid_rows: torch.Tensor,
    query_length: int,
    capacity: int,
    context: int,
) -> torch.Tensor:
    """Reference mask identical to the attention-forward position math."""
    device = offset_end.device
    cache_indexes = torch.arange(capacity, device=device, dtype=torch.long)
    next_offset = torch.where(valid_rows, offset_end + query_length, offset_end)
    last_offset = offset_end.view(-1, 1) + query_length - 1
    end_index = last_offset % capacity
    delta_index = cache_indexes - end_index
    pos_k = torch.where(
        delta_index <= 0,
        last_offset + delta_index,
        last_offset + delta_index - capacity,
    )
    invalid = cache_indexes >= next_offset.view(-1, 1)
    pos_k = torch.where(invalid, torch.full_like(pos_k, -1), pos_k)
    pos_q = query_offset.view(-1, 1, 1) + torch.arange(query_length, device=device, dtype=torch.long).view(-1, 1)
    delta = pos_q - pos_k[:, None]
    attn_bias = (pos_k[:, None] >= 0) & (delta >= 0)
    if context > 0:
        attn_bias = attn_bias & (delta < context)
    return attn_bias[:, None]


@torch.library.custom_op("moss_codec::packed_causal_mask", mutates_args=(), device_types="cuda")
def codec_causal_mask(
    offset_end: torch.Tensor,
    query_offset: torch.Tensor,
    valid_rows: torch.Tensor,
    query_length: int,
    capacity: int,
    context: int,
) -> torch.Tensor:
    """Build the ``(B,1,T,C)`` bool causal mask for packed ring-physical KV.

    ``offset_end`` are the ring end offsets gathered for the execution rows
    *before* the chunk is written/advanced; ``query_offset`` are the per-row
    query base positions; ``context`` <= 0 disables the context-window bound.
    Integer math only, bitwise identical to the unfused reference.
    """
    batch_size = offset_end.shape[0]
    if (
        not HAS_TRITON
        or not offset_end.is_cuda
        or offset_end.dtype != torch.long
        or query_offset.dtype != torch.long
        or valid_rows.dtype != torch.bool
        or query_offset.shape != (batch_size,)
        or valid_rows.shape != (batch_size,)
        or not offset_end.is_contiguous()
        or not query_offset.is_contiguous()
        or not valid_rows.is_contiguous()
    ):
        return _eager_packed_causal_mask(offset_end, query_offset, valid_rows, query_length, capacity, context)

    mask = torch.empty(
        (batch_size, 1, query_length, capacity),
        device=offset_end.device,
        dtype=torch.bool,
    )
    block_size = 256
    _packed_causal_mask_kernel[(triton.cdiv(capacity, block_size), query_length, batch_size)](
        mask,
        offset_end,
        query_offset,
        valid_rows,
        query_length,
        capacity,
        context=context,
        BLOCK=block_size,
        num_warps=4,
    )
    return mask


@codec_causal_mask.register_fake
def _codec_causal_mask_fake(offset_end, query_offset, valid_rows, query_length, capacity, context):
    return offset_end.new_empty((offset_end.shape[0], 1, query_length, capacity), dtype=torch.bool)


__all__ = [
    "codec_causal_mask",
    "codec_rope_unpack_qkv",
]
