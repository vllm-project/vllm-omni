# ruff: noqa: N803

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - depends on optional Triton install
    triton = None
    tl = None
    _LOAD_ERROR: Exception | None = exc
else:
    _LOAD_ERROR = None

_BLOCK_SIZE = 16
_HEAD_DIM = 128
_BLOCK_N = 128


def is_available() -> bool:
    return _LOAD_ERROR is None and triton is not None and tl is not None


def load_error() -> Exception | None:
    return _LOAD_ERROR


if is_available():

    @triton.jit
    def _small_decode_kernel(
        Q,
        K,
        V,
        BLOCK_TABLE,
        SEQ_LENS,
        OUT,
        SCALE: tl.constexpr,
        MAX_SEQ_LEN: tl.constexpr,
        MAX_BLOCKS_PER_SEQ: tl.constexpr,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        KV_GROUP: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        GUARD_MAX_SEQ_LEN: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        kv_head = tl.program_id(1)

        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)
        q_heads = kv_head * KV_GROUP + offs_h
        mask_h = offs_h < KV_GROUP

        q_offsets = batch_id * NUM_Q_HEADS * BLOCK_D + q_heads[:, None] * BLOCK_D + offs_d[None, :]
        q = tl.load(Q + q_offsets, mask=mask_h[:, None], other=0.0)

        seq_len = tl.load(SEQ_LENS + batch_id)
        if GUARD_MAX_SEQ_LEN and seq_len > GUARD_MAX_SEQ_LEN:
            return

        m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_H,), tl.float32)
        acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

        start_n = 0
        while start_n < seq_len:
            cur_n = start_n + offs_n
            mask_n = cur_n < seq_len
            logical_block = cur_n // 16
            block_offset = cur_n - logical_block * 16
            physical_block = tl.load(
                BLOCK_TABLE + batch_id * MAX_BLOCKS_PER_SEQ + logical_block,
                mask=mask_n,
                other=0,
            )

            kv_offsets = (
                (physical_block[:, None] * 16 + block_offset[:, None]) * NUM_KV_HEADS + kv_head
            ) * BLOCK_D + offs_d[None, :]
            k = tl.load(K + kv_offsets, mask=mask_n[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * SCALE
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, -float("inf"))

            v = tl.load(V + kv_offsets, mask=mask_n[:, None], other=0.0)
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new
            start_n += BLOCK_N

        denom = tl.where(l_i > 0.0, l_i, 1.0)
        result = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)
        out_offsets = batch_id * NUM_Q_HEADS * BLOCK_D + q_heads[:, None] * BLOCK_D + offs_d[None, :]
        tl.store(OUT + out_offsets, result, mask=mask_h[:, None])

    @triton.jit
    def _split_partial_decode_kernel(
        Q,
        K,
        V,
        BLOCK_TABLE,
        SEQ_LENS,
        PARTIAL_M,
        PARTIAL_L,
        PARTIAL_ACC,
        SCALE: tl.constexpr,
        MAX_BLOCKS_PER_SEQ: tl.constexpr,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        KV_GROUP: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SPLIT_TOKENS: tl.constexpr,
        BATCH_SIZE: tl.constexpr,
        MIN_SEQ_LEN: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        kv_head = tl.program_id(1)
        split_id = tl.program_id(2)

        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)
        q_heads = kv_head * KV_GROUP + offs_h
        mask_h = offs_h < KV_GROUP

        partial_head_offsets = (split_id * BATCH_SIZE + batch_id) * NUM_Q_HEADS + q_heads
        partial_acc_offsets = partial_head_offsets[:, None] * BLOCK_D + offs_d[None, :]

        seq_len = tl.load(SEQ_LENS + batch_id)
        if MIN_SEQ_LEN and seq_len < MIN_SEQ_LEN:
            return

        m_i = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_H,), tl.float32)
        acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

        split_begin = split_id * SPLIT_TOKENS
        if split_begin >= seq_len:
            tl.store(PARTIAL_M + partial_head_offsets, m_i, mask=mask_h)
            tl.store(PARTIAL_L + partial_head_offsets, l_i, mask=mask_h)
            tl.store(PARTIAL_ACC + partial_acc_offsets, acc, mask=mask_h[:, None])
            return

        q_offsets = batch_id * NUM_Q_HEADS * BLOCK_D + q_heads[:, None] * BLOCK_D + offs_d[None, :]
        q = tl.load(Q + q_offsets, mask=mask_h[:, None], other=0.0)

        for split_offset in tl.range(0, SPLIT_TOKENS, BLOCK_N):
            cur_n = split_begin + split_offset + offs_n
            mask_n = cur_n < seq_len
            logical_block = cur_n // 16
            block_offset = cur_n - logical_block * 16
            physical_block = tl.load(
                BLOCK_TABLE + batch_id * MAX_BLOCKS_PER_SEQ + logical_block,
                mask=mask_n,
                other=0,
            )

            kv_offsets = (
                (physical_block[:, None] * 16 + block_offset[:, None]) * NUM_KV_HEADS + kv_head
            ) * BLOCK_D + offs_d[None, :]
            k = tl.load(K + kv_offsets, mask=mask_n[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * SCALE
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, -float("inf"))

            v = tl.load(V + kv_offsets, mask=mask_n[:, None], other=0.0)
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        tl.store(PARTIAL_M + partial_head_offsets, m_i, mask=mask_h)
        tl.store(PARTIAL_L + partial_head_offsets, l_i, mask=mask_h)
        tl.store(PARTIAL_ACC + partial_acc_offsets, acc, mask=mask_h[:, None])

    @triton.jit
    def _split_combine_decode_kernel(
        SEQ_LENS,
        PARTIAL_M,
        PARTIAL_L,
        PARTIAL_ACC,
        OUT,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        KV_GROUP: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BATCH_SIZE: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        MIN_SEQ_LEN: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        kv_head = tl.program_id(1)

        seq_len = tl.load(SEQ_LENS + batch_id)
        if MIN_SEQ_LEN and seq_len < MIN_SEQ_LEN:
            return

        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)
        q_heads = kv_head * KV_GROUP + offs_h
        mask_h = offs_h < KV_GROUP

        global_m = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        for split_id in tl.range(0, NUM_SPLITS):
            partial_head_offsets = (split_id * BATCH_SIZE + batch_id) * NUM_Q_HEADS + q_heads
            split_m = tl.load(PARTIAL_M + partial_head_offsets, mask=mask_h, other=-float("inf"))
            global_m = tl.maximum(global_m, split_m)

        global_l = tl.zeros((BLOCK_H,), tl.float32)
        acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
        for split_id in tl.range(0, NUM_SPLITS):
            partial_head_offsets = (split_id * BATCH_SIZE + batch_id) * NUM_Q_HEADS + q_heads
            split_m = tl.load(PARTIAL_M + partial_head_offsets, mask=mask_h, other=-float("inf"))
            split_l = tl.load(PARTIAL_L + partial_head_offsets, mask=mask_h, other=0.0)
            weight = tl.exp(split_m - global_m)
            partial_acc_offsets = partial_head_offsets[:, None] * BLOCK_D + offs_d[None, :]
            split_acc = tl.load(PARTIAL_ACC + partial_acc_offsets, mask=mask_h[:, None], other=0.0)
            global_l += split_l * weight
            acc += split_acc * weight[:, None]

        denom = tl.where(global_l > 0.0, global_l, 1.0)
        result = tl.where(global_l[:, None] > 0.0, acc / denom[:, None], 0.0)
        out_offsets = batch_id * NUM_Q_HEADS * BLOCK_D + q_heads[:, None] * BLOCK_D + offs_d[None, :]
        tl.store(OUT + out_offsets, result, mask=mask_h[:, None])


def fish_decode_kvcache_attn_triton(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    scale: float,
    max_seq_len: int,
    small_path_max_seq_len: int,
    long_split_tokens: int,
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
) -> torch.Tensor:
    if not is_available():
        raise RuntimeError(f"Fish Triton attention is unavailable: {_LOAD_ERROR!r}")
    batch_size, num_q_heads, head_dim = query.shape
    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    if head_dim != _HEAD_DIM or block_size != _BLOCK_SIZE:
        raise RuntimeError("Fish Triton attention only supports head_dim=128 and block_size=16")
    if num_q_heads % num_kv_heads != 0:
        raise RuntimeError("num_q_heads must be divisible by num_kv_heads")

    kv_group = num_q_heads // num_kv_heads
    block_h = triton.next_power_of_2(kv_group)
    max_blocks_per_seq = block_table.shape[1]

    if max_seq_len <= small_path_max_seq_len:
        _small_decode_kernel[(batch_size, num_kv_heads)](
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            out,
            SCALE=float(scale),
            MAX_SEQ_LEN=int(max_seq_len),
            MAX_BLOCKS_PER_SEQ=max_blocks_per_seq,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            KV_GROUP=kv_group,
            BLOCK_H=block_h,
            BLOCK_D=head_dim,
            BLOCK_N=_BLOCK_N,
            GUARD_MAX_SEQ_LEN=0,
            num_warps=4,
            num_stages=3,
        )
        return out

    _small_decode_kernel[(batch_size, num_kv_heads)](
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        out,
        SCALE=float(scale),
        MAX_SEQ_LEN=small_path_max_seq_len,
        MAX_BLOCKS_PER_SEQ=max_blocks_per_seq,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        KV_GROUP=kv_group,
        BLOCK_H=block_h,
        BLOCK_D=head_dim,
        BLOCK_N=_BLOCK_N,
        GUARD_MAX_SEQ_LEN=small_path_max_seq_len,
        num_warps=4,
        num_stages=3,
    )
    num_splits = triton.cdiv(int(max_seq_len), long_split_tokens)
    expected_partial_acc_shape = (num_splits, batch_size, num_q_heads, head_dim)
    if tuple(partial_acc.shape) != expected_partial_acc_shape:
        raise RuntimeError(
            "Fish Triton long attention workspace shape mismatch: "
            f"expected partial_acc={expected_partial_acc_shape}, got {tuple(partial_acc.shape)}"
        )
    _split_partial_decode_kernel[(batch_size, num_kv_heads, num_splits)](
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partial_m,
        partial_l,
        partial_acc,
        SCALE=float(scale),
        MAX_BLOCKS_PER_SEQ=max_blocks_per_seq,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        KV_GROUP=kv_group,
        BLOCK_H=block_h,
        BLOCK_D=head_dim,
        BLOCK_N=_BLOCK_N,
        SPLIT_TOKENS=long_split_tokens,
        BATCH_SIZE=batch_size,
        MIN_SEQ_LEN=small_path_max_seq_len + 1,
        num_warps=4,
        num_stages=3,
    )
    _split_combine_decode_kernel[(batch_size, num_kv_heads)](
        seq_lens,
        partial_m,
        partial_l,
        partial_acc,
        out,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        KV_GROUP=kv_group,
        BLOCK_H=block_h,
        BLOCK_D=head_dim,
        BATCH_SIZE=batch_size,
        NUM_SPLITS=num_splits,
        MIN_SEQ_LEN=small_path_max_seq_len + 1,
        num_warps=4,
        num_stages=3,
    )
    return out
