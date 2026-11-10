# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gather paged BF16 K/V directly into dense head-major EagleFIA inputs."""

from vllm.triton_utils import tl, triton


@triton.jit
def _gather(
    key,
    value,
    block_ids,
    out_k,
    out_v,
    length,
    start,
    page_size: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    tile: tl.constexpr,
):
    head = tl.program_id(0)
    token = start + tl.program_id(1) * tile + tl.arange(0, tile)
    valid = token < length
    page = tl.load(block_ids + token // page_size, valid, other=0)
    physical = page * page_size + token % page_size
    channel = tl.arange(0, dim)
    src = ((physical * heads + head) * dim)[:, None] + channel[None, :]
    dst = ((head * length + token) * dim)[:, None] + channel[None, :]
    k = tl.load(key + src, valid[:, None], other=0)
    v = tl.load(value + src, valid[:, None], other=0)
    tl.store(out_k + dst, k, valid[:, None])
    tl.store(out_v + dst, v, valid[:, None])


def gather_kv_bnsd(key, value, block_ids, length, stage_key, stage_value, *, first_block):
    """Update a logical suffix in packed [H, S, D] staging storage."""
    _, page_size, heads, dim = key.shape
    # Slice the backing storage before reshaping so the head stride is
    # length * dim, even when the allocation can hold a larger window.
    out_k = stage_key[:length].view(heads, length, dim)
    out_v = stage_value[:length].view(heads, length, dim)
    start = first_block * page_size
    tile = 64
    _gather[(heads, triton.cdiv(length - start, tile))](
        key, value, block_ids, out_k, out_v, length, start, page_size, heads, dim, tile
    )
    return out_k, out_v
