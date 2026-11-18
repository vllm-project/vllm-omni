# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fuse SOL residual stages while batching original-sized FP32 GEMMs."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _merge_residual(
    residual,
    exact_output,
    exact_weights,
    output,
    num_elements: tl.constexpr,
    chunk_len: tl.constexpr,
    start_query,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    exact_stride_b: tl.constexpr,
    exact_stride_q: tl.constexpr,
    exact_stride_h: tl.constexpr,
    exact_stride_d: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_q: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    block_size: tl.constexpr,
):
    ids = tl.program_id(0) * block_size + tl.arange(0, block_size)
    d = ids % head_dim
    h = ids // head_dim % num_heads
    q_local = ids // (head_dim * num_heads) % chunk_len
    b = ids // (head_dim * num_heads * chunk_len)
    q = q_local + start_query
    row = (b * num_heads + h) * chunk_len + q_local
    valid = ids < num_elements
    r = tl.load(residual + row * head_dim + d, valid, other=0)
    e = tl.load(
        exact_output + b * exact_stride_b + q * exact_stride_q + h * exact_stride_h + d * exact_stride_d, valid, other=0
    ).to(tl.float32)
    weight = tl.load(exact_weights + row, valid, other=0)
    value = e * weight + r
    tl.store(output + b * out_stride_b + q * out_stride_q + h * out_stride_h + d * out_stride_d, value, valid)


@triton.jit
def _finish_lse(
    sums,
    maxima,
    exact_lse,
    joint,
    weights,
    rows_count: tl.constexpr,
    chunk: tl.constexpr,
    start,
    heads: tl.constexpr,
    sb: tl.constexpr,
    sh: tl.constexpr,
    sq: tl.constexpr,
    bs: tl.constexpr,
):
    ids = tl.program_id(0) * bs + tl.arange(0, bs)
    valid = ids < rows_count
    q = ids % chunk + start
    h = ids // chunk % heads
    b = ids // (chunk * heads)
    x = tl.extra.cuda.libdevice.log(tl.load(sums + ids, valid, other=1)) + tl.load(maxima + ids, valid, other=0)
    y = tl.load(exact_lse + b * sb + h * sh + q * sq, valid, other=0)
    maximum = tl.maximum(x, y)
    merged = maximum + tl.extra.cuda.libdevice.log1p(tl.extra.cuda.libdevice.exp(-tl.abs(x - y)))
    merged = tl.where((x == y) & (tl.abs(x) == float("inf")), x, merged)
    tl.store(joint + ids, merged, valid)
    tl.store(weights + ids, tl.extra.cuda.libdevice.exp(y - merged), valid)


@triton.jit
def _prepare_sum(
    scores,
    selected,
    log_sizes,
    exponentials,
    maxima,
    rows_count: tl.constexpr,
    chunk: tl.constexpr,
    start,
    heads: tl.constexpr,
    blocks: tl.constexpr,
    sb: tl.constexpr,
    sh: tl.constexpr,
    sq: tl.constexpr,
    sk: tl.constexpr,
    scale: tl.constexpr,
    bm: tl.constexpr,
    bn: tl.constexpr,
):
    rows = tl.program_id(0) * bm + tl.arange(0, bm)
    cols = tl.arange(0, bn)
    q = rows % chunk + start
    h = rows // chunk % heads
    b = rows // (chunk * heads)
    valid = (rows[:, None] < rows_count) & (cols[None, :] < blocks)
    values = tl.load(scores + rows[:, None] * blocks + cols[None, :], valid, other=0) * scale
    mask = tl.load(
        selected + b[:, None] * sb + h[:, None] * sh + (q[:, None] // 64) * sq + cols[None, :] * sk, valid, other=1
    )
    values = tl.where(valid & ~mask, values, -float("inf"))
    weighted = values + tl.load(log_sizes + cols, cols < blocks, other=0)[None, :]
    maximum = tl.max(weighted, 1)
    maximum = tl.where(tl.abs(maximum) == float("inf"), 0.0, maximum)
    exps = tl.extra.cuda.libdevice.exp(weighted - maximum[:, None])
    tl.store(scores + rows[:, None] * blocks + cols[None, :], values, valid)
    tl.store(exponentials + rows[:, None] * blocks + cols[None, :], exps, valid)
    tl.store(maxima + rows, maximum, rows < rows_count)


@triton.jit
def _probabilities(scores, joint, rows_count: tl.constexpr, blocks: tl.constexpr, bs: tl.constexpr):
    ids = tl.program_id(0) * bs + tl.arange(0, bs)
    valid = ids < rows_count * blocks
    values = tl.load(scores + ids, valid, other=0)
    lse = tl.load(joint + ids // blocks, valid, other=0)
    probs = tl.extra.cuda.libdevice.exp(values - lse)
    tl.store(scores + ids, probs, valid)


def merge_sol_residual(
    query: torch.Tensor,
    key_centroids: torch.Tensor,
    value_sums: torch.Tensor,
    selected: torch.Tensor,
    sizes: torch.Tensor,
    exact_output: torch.Tensor,
    exact_lse: torch.Tensor,
    scale: float,
    *,
    chunks_per_launch: int = 8,
) -> torch.Tensor:
    """Keep FP32 GEMMs and the eager normalization's reduction/rounding order.

    Value sums already include block multiplicity; sizes affect only the
    denominator. Preserve the separate residual LSE and logaddexp merge:
    regrouping them changes BF16 rounding and subsequent sparse routing.
    """
    batch, tokens, heads, dim = query.shape
    blocks = sizes.numel()
    output = torch.empty_like(query)
    keys = key_centroids.float().transpose(1, 2).transpose(-1, -2)
    values = value_sums.float().transpose(1, 2)
    log_sizes = sizes.float().log()
    bn = triton.next_power_of_2(blocks)
    bm = 4 if bn <= 2048 else 1
    chunk_size = 256 * chunks_per_launch
    # Batch 256-row GEMMs without changing their reduction/rounding shape.
    full = tokens // 256 * 256
    windows = [(start, min(start + chunk_size, full)) for start in range(0, full, chunk_size)]
    if full < tokens:
        windows.append((full, tokens))
    for start, stop in windows:
        chunk = stop - start
        group_len = min(chunk, 256)
        groups = chunk // group_len
        grouped_query = (
            query[:, start:stop].float().reshape(batch, groups, group_len, heads, dim).permute(0, 3, 1, 2, 4)
        )
        scores = (grouped_query @ keys.unsqueeze(2)).reshape(batch, heads, chunk, blocks)
        exps = torch.empty_like(scores)
        maxima = torch.empty((batch, heads, chunk), device=query.device, dtype=torch.float32)
        _prepare_sum[(triton.cdiv(batch * heads * chunk, bm),)](
            scores,
            selected,
            log_sizes,
            exps,
            maxima,
            batch * heads * chunk,
            chunk,
            start,
            heads,
            blocks,
            *selected.stride(),
            scale,
            bm,
            bn,
            num_warps=8 if bn >= 2048 else 4,
            enable_fp_fusion=False,
        )
        # Keep the PyTorch sum reduction; scalar operations retain eager order.
        sums = exps.sum(-1)
        joint_lse = torch.empty_like(sums)
        exact_weights = torch.empty_like(sums)
        _finish_lse[(triton.cdiv(batch * heads * chunk, 256),)](
            sums,
            maxima,
            exact_lse,
            joint_lse,
            exact_weights,
            batch * heads * chunk,
            chunk,
            start,
            heads,
            *exact_lse.stride(),
            256,
            enable_fp_fusion=False,
        )
        _probabilities[(triton.cdiv(scores.numel(), 1024),)](
            scores,
            joint_lse,
            batch * heads * chunk,
            blocks,
            1024,
            enable_fp_fusion=False,
        )
        residual = (scores.reshape(batch, heads, groups, group_len, blocks) @ values.unsqueeze(2)).reshape(
            batch, heads, chunk, dim
        )
        elements = batch * chunk * heads * dim
        _merge_residual[(triton.cdiv(elements, 512),)](
            residual,
            exact_output,
            exact_weights,
            output,
            elements,
            chunk,
            start,
            heads,
            dim,
            *exact_output.stride(),
            *output.stride(),
            512,
            enable_fp_fusion=False,
        )
    return output
