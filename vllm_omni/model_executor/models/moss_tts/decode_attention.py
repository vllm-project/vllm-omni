# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Paged attention for TRITON_ATTN decoders with a head-grouped decode path.

Drop-in replacement for the 2D path of vLLM's ``unified_attention`` for plain
causal GQA attention (no ALiBi, sinks, softcap, sliding window, quantized KV,
...). The prefill path reproduces ``kernel_unified_attention`` (fp32 online
softmax, bf16 P@V, same tiles). Decode rows (query length 1) take a separate
path in the same launch:

* one program handles ``DECODE_HEADS`` KV heads of a sequence, so each KV tile
  is one contiguous span of the LBNHC page (all heads of a token are adjacent)
  and the launch needs fewer, longer-lived programs (one wave on H200 for a
  128-row decode batch);
* the score / P@V products use a block-diagonal layout: query rows of head ``g``
  only see the columns of KV head ``g`` (other columns are masked to -inf, so
  they contribute exact zeros);
* sequence lookup loads the whole ``query_start_loc`` prefix as one vector
  instead of a dependent binary search.
"""

from __future__ import annotations

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
from vllm.v1.kv_cache_interface import KVQuantMode

_gdc_launch = tl.extra.cuda.gdc_launch_dependents
_gdc_wait = tl.extra.cuda.gdc_wait

# Hopper+ supports programmatic dependent launch. The kernel waits for its
# predecessor before any global access and lets its successor launch at once; results are unchanged.
_USE_PDL = current_platform.is_cuda() and current_platform.has_device_capability(90)


@triton.jit
def _online_softmax_step(S, M, L, acc, V):
    m_j = tl.maximum(M, tl.max(S, axis=1))
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    P = tl.exp(S - m_j[:, None])
    l_j = tl.sum(P, axis=1)
    alpha = tl.exp(M - m_j)
    L = L * alpha + l_j
    acc = acc * alpha[:, None]
    acc += tl.dot(P.to(V.dtype), V)
    return m_j, L, acc


@triton.jit
def _prefill_block(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    scale,
    seq_idx,
    kv_head_idx,
    q_block_local_idx,
    cur_start,
    cur_batch_query_len,
    seq_len,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride,
    query_stride_0,
    query_stride_1,
    output_stride_0,
    output_stride_1,
    stride_k_cache_0,
    stride_k_cache_1,
    stride_k_cache_2,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0,
    stride_v_cache_1,
    stride_v_cache_2,
    stride_v_cache_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Same math and tiling as vLLM's kernel_unified_attention (2D, causal).
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = cur_start + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = query_offset_0[:, None] * query_stride_0 + query_offset_1[:, None] * query_stride_1 + offs_d[None, :]
    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)
    Q = tl.load(
        query_ptr + query_offset, mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None], other=0.0
    )
    block_table_offset = seq_idx * block_table_stride
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    context_len = seq_len - cur_batch_query_len
    max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = (max_seq_prefix_len + TILE_SIZE - 1) // TILE_SIZE
    query_abs_pos = context_len + query_pos[:, None]
    row_mask = query_mask_1[:, None] & query_mask_0[:, None]
    for j in range(0, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE).to(tl.int64)
        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        K = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
        V = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :] & tile_mask[:, None], other=0.0)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)
        S = tl.where(row_mask & (seq_offset[None, :] <= query_abs_pos), S, float("-inf"))
        M, L, acc = _online_softmax_step(S, M, L, acc, V)
    acc = acc / L[:, None]
    output_offset = (
        query_offset_0[:, None] * output_stride_0 + query_offset_1[:, None] * output_stride_1 + offs_d[None, :]
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None])


@triton.jit
def _load_kv_group(
    key_cache_ptr,
    value_cache_ptr,
    bt_ptr,
    tile_start,
    seq_len,
    col_t,
    k_head,
    offs_d,
    dim_mask,
    stride_k_cache_0,
    stride_k_cache_1,
    stride_k_cache_2,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0,
    stride_v_cache_1,
    stride_v_cache_2,
    stride_v_cache_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DECODE_TILE: tl.constexpr,
):
    seq_offset = tile_start + col_t
    tile_mask = seq_offset < seq_len
    physical_block_idx = tl.load(bt_ptr + seq_offset // BLOCK_SIZE, mask=tile_mask, other=0).to(tl.int64)
    slot = (seq_offset % BLOCK_SIZE).to(tl.int64)
    k_offset = (
        physical_block_idx[None, :] * stride_k_cache_0
        + slot[None, :] * stride_k_cache_1
        + k_head[None, :] * stride_k_cache_2
        + offs_d[:, None] * stride_k_cache_3
    )
    v_offset = (
        physical_block_idx[:, None] * stride_v_cache_0
        + slot[:, None] * stride_v_cache_1
        + k_head[:, None] * stride_v_cache_2
        + offs_d[None, :] * stride_v_cache_3
    )
    K = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
    V = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :] & tile_mask[:, None], other=0.0)
    return K, V, tile_mask


@triton.jit
def _decode_group(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    scale,
    seq_idx,
    kv_head_idx,
    cur_start,
    seq_len,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride,
    query_stride_0,
    query_stride_1,
    output_stride_0,
    output_stride_1,
    stride_k_cache_0,
    stride_k_cache_1,
    stride_k_cache_2,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0,
    stride_v_cache_1,
    stride_v_cache_2,
    stride_v_cache_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    DECODE_HEADS: tl.constexpr,
    DECODE_TILE: tl.constexpr,
    DECODE_M: tl.constexpr,
):
    # One query token, DECODE_HEADS consecutive KV heads starting at kv_head_idx.
    # Columns are head-major: column c belongs to KV head c // DECODE_TILE.
    TG: tl.constexpr = DECODE_HEADS * DECODE_TILE
    offs_m = tl.arange(0, DECODE_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_c = tl.arange(0, TG)
    col_g = offs_c // DECODE_TILE
    col_t = offs_c % DECODE_TILE
    row_g = offs_m // num_queries_per_kv
    q_head = kv_head_idx * num_queries_per_kv + offs_m
    row_valid = (offs_m < DECODE_HEADS * num_queries_per_kv) & (q_head < num_query_heads)
    dim_mask = offs_d < HEAD_SIZE
    Q = tl.load(
        query_ptr + cur_start * query_stride_0 + q_head[:, None] * query_stride_1 + offs_d[None, :],
        mask=row_valid[:, None] & dim_mask[None, :],
        other=0.0,
    )
    block_table_offset = seq_idx * block_table_stride
    k_head = (kv_head_idx + col_g).to(tl.int64)
    M = tl.full([DECODE_M], float("-inf"), dtype=tl.float32)
    L = tl.full([DECODE_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([DECODE_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    num_tiles = (seq_len + DECODE_TILE - 1) // DECODE_TILE
    same_head = row_valid[:, None] & (row_g[:, None] == col_g[None, :])
    bt_ptr = block_tables_ptr + block_table_offset
    for j in range(0, num_tiles):
        K, V, tile_mask = _load_kv_group(
            key_cache_ptr,
            value_cache_ptr,
            bt_ptr,
            j * DECODE_TILE,
            seq_len,
            col_t,
            k_head,
            offs_d,
            dim_mask,
            stride_k_cache_0,
            stride_k_cache_1,
            stride_k_cache_2,
            stride_k_cache_3,
            stride_v_cache_0,
            stride_v_cache_1,
            stride_v_cache_2,
            stride_v_cache_3,
            BLOCK_SIZE,
            DECODE_TILE,
        )
        S = tl.zeros(shape=(DECODE_M, TG), dtype=tl.float32)
        S += scale * tl.dot(Q, K)
        S = tl.where(same_head & tile_mask[None, :], S, float("-inf"))
        M, L, acc = _online_softmax_step(S, M, L, acc, V)
    acc = acc / L[:, None]
    tl.store(
        output_ptr + cur_start * output_stride_0 + q_head[:, None] * output_stride_1 + offs_d[None, :],
        acc,
        mask=row_valid[:, None] & dim_mask[None, :],
    )


@triton.jit
def _paged_attention_kernel(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    scale,
    num_seqs,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_SEQS_PADDED: tl.constexpr,
    DECODE_HEADS: tl.constexpr,
    DECODE_TILE: tl.constexpr,
    DECODE_M: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    # Programs [0, num_seqs) take the first query block of sequence pid0 (no
    # lookup: decode rows start immediately); programs past num_seqs take the
    # remaining query blocks of prefill rows. Sequence s owns the extra range
    # [cu[s] // BLOCK_Q, cu[s + 1] // BLOCK_Q), which covers its
    # ceil(len / BLOCK_Q) - 1 extra blocks, so num_seqs + T // BLOCK_Q programs
    # (vLLM's grid) suffice.
    if USE_PDL:
        _gdc_wait()
        _gdc_launch()
    pid0 = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    if pid0 < num_seqs:
        seq_idx = pid0
        cur_start = tl.load(query_start_len_ptr + seq_idx)
        q_block_local_idx = 0
    else:
        extra_idx = pid0 - num_seqs
        offs = tl.arange(0, MAX_SEQS_PADDED)
        vals = tl.load(query_start_len_ptr + offs, mask=offs < num_seqs, other=0x3FFFFFFF)
        seq_idx = tl.sum((vals // BLOCK_Q <= extra_idx).to(tl.int32), axis=0) - 1
        cur_start = tl.load(query_start_len_ptr + seq_idx)
        q_block_local_idx = extra_idx - cur_start // BLOCK_Q + 1
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_stop - cur_start
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if DECODE_HEADS > 0 and cur_batch_query_len == 1:
        if kv_head_idx * DECODE_HEADS < num_query_heads // num_queries_per_kv:
            _decode_group(
                output_ptr,
                query_ptr,
                key_cache_ptr,
                value_cache_ptr,
                block_tables_ptr,
                scale,
                seq_idx,
                kv_head_idx * DECODE_HEADS,
                cur_start,
                seq_len,
                num_query_heads,
                num_queries_per_kv,
                block_table_stride,
                query_stride_0,
                query_stride_1,
                output_stride_0,
                output_stride_1,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                BLOCK_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
                DECODE_HEADS,
                DECODE_TILE,
                DECODE_M,
            )
    else:
        _prefill_block(
            output_ptr,
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            block_tables_ptr,
            scale,
            seq_idx,
            kv_head_idx,
            q_block_local_idx,
            cur_start,
            cur_batch_query_len,
            seq_len,
            num_query_heads,
            num_queries_per_kv,
            block_table_stride,
            query_stride_0,
            query_stride_1,
            output_stride_0,
            output_stride_1,
            stride_k_cache_0,
            stride_k_cache_1,
            stride_k_cache_2,
            stride_k_cache_3,
            stride_v_cache_0,
            stride_v_cache_1,
            stride_v_cache_2,
            stride_v_cache_3,
            BLOCK_SIZE,
            TILE_SIZE,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
            BLOCK_Q,
            BLOCK_M,
        )


# Grouping two KV heads per decode program halves the launch to one wave for
# ~128-row decode batches on H200; below that the per-head programs are faster.
_GROUPED_DECODE_MIN_SEQS = 96
# Launch of a pure decode batch (query length 1) with grouped heads. Measured
# on H200 at 120 rows, ctx 100-300, GQA 32/8, D 128: 2 warps / 3 stages runs
# 31 us vs 38 us for the vLLM 2D kernel. Mixed batches keep Triton defaults.
_DECODE_NUM_WARPS = 2
_DECODE_NUM_STAGES = 3


def paged_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    softmax_scale,
    *,
    decode_heads=None,
    decode_tile=32,
    tile_size=32,
    num_warps=None,
    num_stages=None,
):
    """Causal paged attention for any mix of query lengths (see module docstring).

    ``decode_heads=None`` picks the grouping from the (padded) batch size;
    ``0`` sends decode rows through the vLLM-equivalent prefill path.
    """
    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    if decode_heads is None:
        decode_heads = 2 if num_seqs >= _GROUPED_DECODE_MIN_SEQS and num_kv_heads % 2 == 0 else 1
    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    decode_m = max(16, triton.next_power_of_2(num_queries_per_kv * max(decode_heads, 1)))
    kwargs = {}
    if num_warps is not None:
        kwargs["num_warps"] = num_warps
    if num_stages is not None:
        kwargs["num_stages"] = num_stages
    grid = (q.shape[0] // BLOCK_Q + num_seqs, num_kv_heads)
    _paged_attention_kernel[grid](
        out,
        q,
        k,
        v,
        block_table,
        seqused_k,
        cu_seqlens_q,
        softmax_scale,
        num_seqs,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        # One specialization for every batch up to 255 sequences (graph capture).
        MAX_SEQS_PADDED=max(256, triton.next_power_of_2(num_seqs + 1)),
        DECODE_HEADS=decode_heads,
        DECODE_TILE=decode_tile,
        DECODE_M=decode_m,
        USE_PDL=_USE_PDL,
        launch_pdl=_USE_PDL,
        **kwargs,
    )


def _uses_3d(meta) -> bool:
    # Keep upstream's split-KV path for small decode batches.
    from vllm.v1.attention.ops import triton_unified_attention as tua

    return not (
        meta.seq_threshold_3D is None
        or meta.num_par_softmax_segments is None
        or meta.softmax_segm_output is None
        or meta.softmax_segm_max is None
        or meta.softmax_segm_expsum is None
        or meta.max_query_len > 1
        or len(meta.seq_lens) > meta.seq_threshold_3D
        or tua.is_batch_invariant
    )


@triton.jit
def _cache_write_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    head_stride: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    # vLLM's reshape_and_cache_kernel_flash (bf16, [blocks, slots, heads, dim] pages) with a PDL launch: it waits
    # for the RoPE kernel that produced key/value and lets the decode attention launch behind it.
    if USE_PDL:
        _gdc_wait()
        _gdc_launch()
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    tile_pos = tl.program_id(axis=1) * TILE_SIZE + tl.arange(0, TILE_SIZE)
    cur_head = tile_pos // head_size
    cur_dim = tile_pos % head_size
    tgt = block_idx * block_stride + block_offset * page_stride + cur_head * head_stride + cur_dim
    mask = tile_pos < (num_heads * head_size)
    k = tl.load(key_ptr + token_idx * key_stride + tile_pos, mask=mask)
    v = tl.load(value_ptr + token_idx * value_stride + tile_pos, mask=mask)
    tl.store(key_cache_ptr + tgt, k, mask=mask)
    tl.store(value_cache_ptr + tgt, v, mask=mask)


def _cache_write_pdl(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale, upstream=None):
    native = kv_cache_dtype in ("auto", "bfloat16", "float16") and key_cache.dtype == key.dtype
    if not (_USE_PDL and native and key_cache.ndim == 4 and key.stride(-1) == 1 and key_cache.stride(-1) == 1):
        return upstream(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)
    num_heads, head_size = key.shape[1], key.shape[2]
    n = num_heads * head_size
    tile = min(2048, triton.next_power_of_2(n))
    _cache_write_kernel[(slot_mapping.shape[0], triton.cdiv(n, tile))](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key.stride(0),
        value.stride(0),
        key_cache.stride(0),
        key_cache.stride(2),
        key_cache.stride(1),
        num_heads=num_heads,
        head_size=head_size,
        block_size=key_cache.shape[1],
        TILE_SIZE=tile,
        USE_PDL=True,
        launch_pdl=True,
        num_warps=16,
        num_stages=10,
    )


class MossLocalAttentionImpl(TritonAttentionImpl):
    """Use the grouped kernel only on this MOSS layer; retain upstream fallbacks."""

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None
    ):
        meta = attn_metadata
        if (
            meta is None
            or output is None
            or output_block_scale is not None
            or self.attn_type != AttentionType.DECODER
            or self._kv_quant_mode != KVQuantMode.NONE
        ):
            return super().forward(layer, query, key, value, kv_cache, meta, output, output_scale, output_block_scale)
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        supported = (
            meta.causal is True
            and not meta.use_cascade
            and self.sliding_window[0] < 0
            and not self.logits_soft_cap
            and self.alibi_slopes is None
            and output_scale is None
            and self.sinks is None
            and meta.mm_prefix_range_tensor is None
            and (meta.rswa_window is None or meta.rswa_prefix_lens is None)
            and self.chunk_lookback == -1
            and not self.use_td
            and not getattr(layer, "mm_prefix_clamp_sliding_window", False)
            and query.dtype in (torch.bfloat16, torch.float16)
            and key_cache.dtype == query.dtype == output.dtype
            and self.head_size == triton.next_power_of_2(self.head_size)
            and self.num_heads % self.num_kv_heads == 0
            and self.num_queries_per_kv <= 16
        )
        if not supported or _uses_3d(meta):
            return super().forward(layer, query, key, value, kv_cache, meta, output, output_scale, output_block_scale)
        launch = {}
        if meta.max_query_len == 1 and len(meta.seq_lens) >= _GROUPED_DECODE_MIN_SEQS:
            launch = {"num_warps": _DECODE_NUM_WARPS, "num_stages": _DECODE_NUM_STAGES}
        n = meta.num_actual_tokens
        paged_attention(
            query[:n],
            key_cache,
            value_cache,
            output[:n],
            meta.query_start_loc,
            meta.seq_lens,
            meta.block_table,
            self.scale,
            **launch,
        )
        return output

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if self.attn_type != AttentionType.DECODER or self._kv_quant_mode != KVQuantMode.NONE:
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash

        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        _cache_write_pdl(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
            upstream=triton_reshape_and_cache_flash,
        )


def use_moss_local_attention(model: torch.nn.Module) -> None:
    """Replace plain Triton implementations on this backbone's layers only.

    The backend and metadata builder stay unchanged. No vLLM module globals,
    backend registry entries or other models' attention layers are modified.
    """
    from vllm.model_executor.layers.attention import Attention

    for layer in model.modules():
        if not isinstance(layer, Attention):
            continue
        impl = layer.impl
        if (
            type(impl) is not TritonAttentionImpl
            or impl.attn_type != AttentionType.DECODER
            or impl.alibi_slopes is not None
            or impl.sliding_window != (-1, -1)
            or impl.sinks is not None
        ):
            continue
        layer.impl = MossLocalAttentionImpl(
            num_heads=impl.num_heads,
            head_size=impl.head_size,
            scale=impl.scale,
            num_kv_heads=impl.num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=impl.kv_cache_dtype,
            logits_soft_cap=impl.logits_soft_cap,
            attn_type=impl.attn_type,
            kv_sharing_target_layer_name=impl.kv_sharing_target_layer_name,
            use_alibi_sqrt=impl.use_alibi_sqrt,
            chunk_lookback=impl.chunk_lookback,
        )
