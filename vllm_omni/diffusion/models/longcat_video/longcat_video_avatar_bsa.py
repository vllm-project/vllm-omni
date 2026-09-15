# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: N803

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_TRITON_AUTOTUNE_ENABLE_ENV = "TRITON_AUTOTUNE_ENABLE"
_LEGACY_TRITON_AUTOTUNE_ENABLE_ENV = "TRITON_AUTOTUNE_ENBALE"


def _triton_autotune_enabled() -> bool:
    return (
        os.environ.get(_TRITON_AUTOTUNE_ENABLE_ENV, "0") == "1"
        or os.environ.get(_LEGACY_TRITON_AUTOTUNE_ENABLE_ENV, "0") == "1"
    )


if _triton_autotune_enabled():
    autotune = triton.autotune
else:

    def autotune(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


_GATING_CONFIG_PRESET = {
    "default": {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "num_stages": 3,
        "num_warps": 8,
    }
}

_GATING_CONFIGS = [
    triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n}, num_stages=stages, num_warps=warps)
    for block_m in [64, 128]
    for block_n in [32, 64]
    for stages in [2, 3, 4, 5]
    for warps in [4, 8]
]

_GATING_REEVALUATE_KEYS = ["M", "N"] if os.environ.get("TRITON_REEVALUATE_KEY", "0") == "1" else []


@autotune(_GATING_CONFIGS, key=_GATING_REEVALUATE_KEYS)
@triton.jit
def _attn_fwd_gating(
    Q,
    K,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    H,
    M,
    N,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(M, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    out_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(M, N),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    q = tl.load(q_block_ptr, boundary_check=(0,))
    for start_n in range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_block_ptr, boundary_check=(1,))
        qk = tl.dot(q, k)
        tl.store(out_block_ptr, qk.to(Out.type.element_ty), boundary_check=(0, 1))
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
        out_block_ptr = tl.advance(out_block_ptr, (0, BLOCK_N))


_FWD_BSA_VARLEN_CONFIG_PRESET = {
    "default": {
        "BLOCK_N": 64,
        "num_stages": 3,
        "num_warps": 8,
    },
    "BLOCK_N_LG=64": {
        "BLOCK_N": 64,
        "num_stages": 3,
        "num_warps": 4,
    },
}
_FWD_BSA_VARLEN_CONFIGS = [
    triton.Config({"BLOCK_N": block_n}, num_stages=stages, num_warps=warps)
    for block_n in [32, 64, 128]
    for stages in [2, 3, 4, 5]
    for warps in [4, 8]
]
_FWD_BSA_REEVALUATE_KEYS = (
    ["N_CTX", "BLOCK_M", "BLOCK_N_LG", "SPARSITY"] if os.environ.get("TRITON_REEVALUATE_KEY", "0") == "1" else []
)


@autotune(_FWD_BSA_VARLEN_CONFIGS, key=_FWD_BSA_REEVALUATE_KEYS)
@triton.jit
def _attn_fwd_bsa_varlen(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,
    block_indices,
    block_indices_lens,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bs,
    stride_lz,
    stride_lh,
    stride_lm,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N_LG: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPARSITY: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    b_offset = off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh
    l_offset = off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh

    q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    kt_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    out_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    block_indices += b_offset + start_m * stride_bm
    block_indices_lens += l_offset + start_m * stride_lm
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_block_ptr)
    selected_blocks = tl.load(block_indices_lens)
    for i in range(selected_blocks):
        block_id = tl.load(block_indices + i * stride_bs).to(tl.int32)
        lo = block_id * BLOCK_N_LG
        hi = (block_id + 1) * BLOCK_N_LG
        lo = tl.multiple_of(lo, BLOCK_N)
        kt_block_ptr_i = tl.advance(kt_block_ptr, (0, lo))
        v_block_ptr_i = tl.advance(v_block_ptr, (lo, 0))

        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            kt = tl.load(kt_block_ptr_i)
            qkt = tl.dot(q, kt)

            m_ij = tl.maximum(m_i, tl.max(qkt, 1) * qk_scale)
            qkt = qkt * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qkt)

            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(v_block_ptr_i)
            acc = tl.dot(p.to(v.dtype), v, acc)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            v_block_ptr_i = tl.advance(v_block_ptr_i, (BLOCK_N, 0))
            kt_block_ptr_i = tl.advance(kt_block_ptr_i, (0, BLOCK_N))

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(out_block_ptr, acc.to(Out.type.element_ty))


_FWD_BSA_VARLEN_ALIGN_CONFIG_PRESET = {
    "default": {
        "num_stages": 3,
        "num_warps": 8,
    },
    "BLOCK_N_LG=64": {
        "num_stages": 3,
        "num_warps": 4,
    },
}
_FWD_BSA_VARLEN_ALIGN_CONFIGS = [
    triton.Config({}, num_stages=stages, num_warps=warps) for stages in [2, 3, 4, 5] for warps in [4, 8]
]


@autotune(_FWD_BSA_VARLEN_ALIGN_CONFIGS, key=_FWD_BSA_REEVALUATE_KEYS)
@triton.jit
def _attn_fwd_bsa_varlen_align(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,
    block_indices,
    block_indices_lens,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bs,
    stride_lz,
    stride_lh,
    stride_lm,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N_LG: tl.constexpr,
    SPARSITY: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    b_offset = off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh
    l_offset = off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh

    q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N_LG, HEAD_DIM),
        order=(1, 0),
    )
    kt_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N_LG),
        order=(0, 1),
    )
    out_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    block_indices += b_offset + start_m * stride_bm
    block_indices_lens += l_offset + start_m * stride_lm
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_block_ptr)
    selected_blocks = tl.load(block_indices_lens)
    for i in range(selected_blocks):
        block_id = tl.load(block_indices + i * stride_bs).to(tl.int32)
        lo = block_id * BLOCK_N_LG
        lo = tl.multiple_of(lo, BLOCK_N_LG)
        kt_block_ptr_i = tl.advance(kt_block_ptr, (0, lo))
        v_block_ptr_i = tl.advance(v_block_ptr, (lo, 0))

        kt = tl.load(kt_block_ptr_i)
        qkt = tl.dot(q, kt)

        m_ij = tl.maximum(m_i, tl.max(qkt, 1) * qk_scale)
        qkt = qkt * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qkt)

        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_block_ptr_i)
        acc = tl.dot(p.to(v.dtype), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(out_block_ptr, acc.to(Out.type.element_ty))


def _mean_pooling_compression(x: torch.Tensor, block_size: int) -> torch.Tensor:
    batch, heads, seq_len = x.shape[:3]
    num_blocks = math.ceil(seq_len / block_size)
    if seq_len % block_size != 0:
        x = F.pad(x, (0, 0, 0, num_blocks * block_size - seq_len))
    return x.view(batch, heads, num_blocks, block_size, -1).mean(dim=3)


def _cal_score_triton(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_q, head_dim = q.shape
    seq_k = k.shape[2]
    score = torch.empty(batch, heads, seq_q, seq_k, device=q.device, dtype=q.dtype)
    kernel_config = {} if _triton_autotune_enabled() else _GATING_CONFIG_PRESET["default"]

    def grid(args):
        return (triton.cdiv(seq_q, args["BLOCK_M"]), batch * heads, 1)

    _attn_fwd_gating[grid](
        q,
        k,
        score,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        score.stride(3),
        heads,
        seq_q,
        seq_k,
        HEAD_DIM=head_dim,
        **kernel_config,
    )
    return score


def _get_select_indices_from_score(
    score: torch.Tensor,
    sparsity: float | None,
    cdf_threshold: float | None,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sparsity is None and cdf_threshold is None:
        raise ValueError("Either sparsity or cdf_threshold must be set for LongCat BSA.")

    selected_topk = None
    if sparsity is not None:
        selected_topk = max(1, int((1 - sparsity) * score.shape[-1]))

    if cdf_threshold is None:
        block_indices = torch.topk(score, selected_topk)[1]
        block_indices_lens = torch.full(
            score.shape[:3],
            selected_topk,
            dtype=torch.int32,
            device=score.device,
        )
        return block_indices, block_indices_lens

    weights = torch.softmax(score * sm_scale, dim=-1)
    batch, heads, seq_q, _ = weights.shape
    threshold = torch.full(
        (heads,),
        cdf_threshold,
        device=weights.device,
    ).view(1, heads, 1, 1)
    threshold = threshold.expand(batch, -1, seq_q, -1)
    weights_sorted = torch.sort(weights, dim=-1, descending=True)
    cdf = torch.cumsum(weights_sorted.values, dim=-1)
    num_selected = torch.searchsorted(cdf, threshold, right=True).squeeze(-1)
    if selected_topk is not None:
        num_selected[num_selected < selected_topk] = selected_topk
    block_indices_lens = num_selected.to(torch.int32)
    return weights_sorted.indices, block_indices_lens


def _attn_fwd_bsa_varlen_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    block_indices: torch.Tensor,
    block_indices_lens: torch.Tensor,
    chunk_size_q: int,
    chunk_size_k: int,
    sparsity: float | None,
) -> torch.Tensor:
    batch, heads, seq_q, head_dim = q.shape
    output = torch.empty_like(q)
    softmax_lse = torch.empty((batch, heads, seq_q), device=q.device, dtype=torch.float32)

    def grid(args):
        return (triton.cdiv(seq_q, args["BLOCK_M"]), batch * heads, 1)

    config_key = "BLOCK_N_LG=64" if chunk_size_k == 64 else "default"
    if chunk_size_k > 128:
        fwd_func = _attn_fwd_bsa_varlen
        kernel_config = {} if _triton_autotune_enabled() else _FWD_BSA_VARLEN_CONFIG_PRESET[config_key]
    else:
        fwd_func = _attn_fwd_bsa_varlen_align
        kernel_config = {} if _triton_autotune_enabled() else _FWD_BSA_VARLEN_ALIGN_CONFIG_PRESET[config_key]

    block_indices = block_indices.contiguous()
    block_indices_lens = block_indices_lens.contiguous()
    fwd_func[grid](
        q,
        k,
        v,
        sm_scale,
        softmax_lse,
        output,
        block_indices,
        block_indices_lens,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        block_indices.stride(0),
        block_indices.stride(1),
        block_indices.stride(2),
        block_indices.stride(3),
        block_indices_lens.stride(0),
        block_indices_lens.stride(1),
        block_indices_lens.stride(2),
        heads,
        seq_q,
        head_dim,
        BLOCK_M=chunk_size_q,
        BLOCK_N_LG=chunk_size_k,
        SPARSITY=0.0 if sparsity is None else sparsity,
        **kernel_config,
    )
    return output


def _rearrange_thw_to_3d_block(
    x: torch.Tensor,
    num_t_blocks: int,
    num_h_blocks: int,
    num_w_blocks: int,
    chunk_t: int,
    chunk_h: int,
    chunk_w: int,
) -> torch.Tensor:
    batch, heads, _, head_dim = x.shape
    x = x.view(batch, heads, num_t_blocks, chunk_t, num_h_blocks, chunk_h, num_w_blocks, chunk_w, head_dim)
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7, 8)
    seq_len = num_t_blocks * num_h_blocks * num_w_blocks * chunk_t * chunk_h * chunk_w
    return x.contiguous().view(batch, heads, seq_len, head_dim)


def _rearrange_3d_block_to_thw(
    x: torch.Tensor,
    num_t_blocks: int,
    num_h_blocks: int,
    num_w_blocks: int,
    chunk_t: int,
    chunk_h: int,
    chunk_w: int,
) -> torch.Tensor:
    batch, heads, _, head_dim = x.shape
    x = x.view(batch, heads, num_t_blocks, num_h_blocks, num_w_blocks, chunk_t, chunk_h, chunk_w, head_dim)
    x = x.permute(0, 1, 2, 5, 3, 6, 4, 7, 8)
    seq_len = num_t_blocks * chunk_t * num_h_blocks * chunk_h * num_w_blocks * chunk_w
    return x.contiguous().view(batch, heads, seq_len, head_dim)


def flash_attn_bsa_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    latent_shape_q: tuple[int, int, int],
    latent_shape_k: tuple[int, int, int],
    *,
    sparsity: float | None = 0.875,
    cdf_threshold: float | None = None,
    chunk_3d_shape_q: tuple[int, int, int] | list[int] = (4, 4, 8),
    chunk_3d_shape_k: tuple[int, int, int] | list[int] = (4, 4, 8),
) -> torch.Tensor:
    _, _, seq_q, head_dim_q = q.shape
    _, _, seq_k, head_dim_k = k.shape
    if head_dim_q != head_dim_k:
        raise ValueError("LongCat BSA requires q and k to use the same head dimension.")

    t_q, h_q, w_q = latent_shape_q
    t_k, h_k, w_k = latent_shape_k
    if t_q * h_q * w_q != seq_q:
        raise ValueError("LongCat BSA q sequence length does not match latent_shape_q.")
    if t_k * h_k * w_k != seq_k:
        raise ValueError("LongCat BSA k sequence length does not match latent_shape_k.")

    chunk_t_q, chunk_h_q, chunk_w_q = tuple(chunk_3d_shape_q)
    chunk_t_k, chunk_h_k, chunk_w_k = tuple(chunk_3d_shape_k)
    if t_q % chunk_t_q != 0 or h_q % chunk_h_q != 0 or w_q % chunk_w_q != 0:
        raise ValueError("LongCat BSA q latent shape must be divisible by chunk_3d_shape_q.")
    if t_k % chunk_t_k != 0 or h_k % chunk_h_k != 0 or w_k % chunk_w_k != 0:
        raise ValueError("LongCat BSA k latent shape must be divisible by chunk_3d_shape_k.")

    num_t_q = t_q // chunk_t_q
    num_h_q = h_q // chunk_h_q
    num_w_q = w_q // chunk_w_q
    num_t_k = t_k // chunk_t_k
    num_h_k = h_k // chunk_h_k
    num_w_k = w_k // chunk_w_k

    q = _rearrange_thw_to_3d_block(q, num_t_q, num_h_q, num_w_q, chunk_t_q, chunk_h_q, chunk_w_q)
    k = _rearrange_thw_to_3d_block(k, num_t_k, num_h_k, num_w_k, chunk_t_k, chunk_h_k, chunk_w_k)
    v = _rearrange_thw_to_3d_block(v, num_t_k, num_h_k, num_w_k, chunk_t_k, chunk_h_k, chunk_w_k)

    chunk_size_q = chunk_t_q * chunk_h_q * chunk_w_q
    chunk_size_k = chunk_t_k * chunk_h_k * chunk_w_k
    q_cmp = _mean_pooling_compression(q, chunk_size_q)
    k_cmp = _mean_pooling_compression(k, chunk_size_k)
    score = _cal_score_triton(q_cmp, k_cmp)
    block_indices, block_indices_lens = _get_select_indices_from_score(
        score,
        sparsity,
        cdf_threshold,
        1 / math.sqrt(head_dim_q),
    )
    output = _attn_fwd_bsa_varlen_triton(
        q,
        k,
        v,
        1 / math.sqrt(head_dim_q),
        block_indices,
        block_indices_lens,
        chunk_size_q,
        chunk_size_k,
        sparsity,
    )
    return _rearrange_3d_block_to_thw(output, num_t_q, num_h_q, num_w_q, chunk_t_q, chunk_h_q, chunk_w_q)
