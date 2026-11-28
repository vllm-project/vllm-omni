# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lossless 16-bit K/V packet movement and sparse Ring partial merge."""

import torch
from vllm.triton_utils import tl, triton


def kv_block_bytes(dim: int) -> int:
    """Storage for one head's K64 and V64 blocks in either 16-bit input dtype."""
    return 64 * dim * 4


@triton.jit
def _pack(
    k,
    v,
    heads,
    blocks,
    wire,
    count,
    k_head_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    k_dim_stride: tl.constexpr,
    v_head_stride: tl.constexpr,
    v_token_stride: tl.constexpr,
    v_dim_stride: tl.constexpr,
    dim: tl.constexpr,
    dynamic: tl.constexpr,
    rows: tl.constexpr = 32,
):
    block = tl.program_id(0)
    if dynamic:
        valid = block < tl.load(count)
    else:
        valid = True
    row = tl.program_id(1) * rows + tl.arange(0, rows)
    channel = tl.arange(0, dim)
    head = tl.load(heads + block, mask=valid, other=0).to(tl.int64)
    source_block = tl.load(blocks + block, mask=valid, other=0).to(tl.int64)
    token = source_block * 64 + row[:, None]
    k_source = head * k_head_stride + token * k_token_stride + channel[None, :] * k_dim_stride
    v_source = head * v_head_stride + token * v_token_stride + channel[None, :] * v_dim_stride
    offset = row[:, None] * dim + channel[None, :]
    base = wire + block.to(tl.int64) * (64 * dim * 4)
    bits = tl.pointer_type(tl.uint16)
    tl.store(base.to(bits) + offset, tl.load(k.to(bits) + k_source, mask=valid, other=0), mask=valid)
    tl.store((base + 64 * dim * 2).to(bits) + offset, tl.load(v.to(bits) + v_source, mask=valid, other=0), mask=valid)


def pack_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    heads: torch.Tensor,
    blocks: torch.Tensor,
    wire: torch.Tensor,
    count: torch.Tensor | None = None,
) -> None:
    """Pack selected K/V blocks without converting their 16-bit input bits."""
    if heads.numel():
        _pack[(heads.numel(), 2)](
            k,
            v,
            heads,
            blocks,
            wire,
            count,
            *k.stride()[-3:],
            *v.stride()[-3:],
            k.shape[-1],
            count is not None,
        )


@triton.jit
def _place(wire, heads, blocks, key, scale, value, tokens: tl.constexpr, dim: tl.constexpr, bf16: tl.constexpr):
    block = tl.program_id(0)
    row = tl.arange(0, 64)
    channel = tl.arange(0, dim)
    head = tl.load(heads + block).to(tl.int64)
    target_block = tl.load(blocks + block).to(tl.int64)
    offset = row[:, None] * dim + channel[None, :]
    target = (head * tokens + target_block * 64 + row[:, None]) * dim + channel[None, :]
    base = wire + block.to(tl.int64) * (64 * dim * 4)
    dtype = tl.bfloat16 if bf16 else tl.float16
    # Match the common SpargeAttn compute conversion after receiving the raw bits.
    x = tl.load(base.to(tl.pointer_type(dtype)) + offset).to(tl.float16).to(tl.float32)
    s = tl.max(tl.abs(x)) / 127.0 + 1e-7
    normalized = x / s
    normalized += 0.5 * tl.where(normalized >= 0, 1, -1)
    tl.store(key + target, normalized.to(tl.int8))
    tl.store(scale + head * (tokens // 64) + target_block, s)
    tl.store(value + target, tl.load((base + 64 * dim * 2).to(tl.pointer_type(dtype)) + offset).to(tl.float16))


def place_kv(
    wire: torch.Tensor,
    heads: torch.Tensor,
    blocks: torch.Tensor,
    key: torch.Tensor,
    scale: torch.Tensor,
    value: torch.Tensor,
    wire_dtype: torch.dtype,
) -> None:
    if heads.numel():
        _place[(heads.numel(),)](
            wire, heads, blocks, key, scale, value, value.shape[-2], value.shape[-1], wire_dtype == torch.bfloat16
        )


@triton.jit
def _merge(partials, lses, output, rows: tl.constexpr, dim: tl.constexpr, count: tl.constexpr, tile: tl.constexpr):
    offsets = tl.program_id(0) * tile + tl.arange(0, tile)
    valid = offsets < rows * dim
    row = offsets // dim
    maximum = tl.full((tile,), -float("inf"), tl.float32)
    for peer in tl.static_range(count):
        lse = tl.load(lses + peer * rows + row, mask=valid, other=-float("inf"))
        maximum = tl.maximum(maximum, lse)
    denominator = tl.zeros((tile,), tl.float32)
    accumulator = tl.zeros((tile,), tl.float32)
    for peer in tl.static_range(count):
        lse = tl.load(lses + peer * rows + row, mask=valid, other=-float("inf"))
        weight = tl.where(maximum > -float("inf"), tl.exp(lse - maximum), 0.0)
        # Empty partials need not initialize their output rows. Never read
        # those bytes, including NaNs left by a previous unrelated call.
        value = tl.load(partials + peer * rows * dim + offsets, mask=valid & (weight > 0), other=0).to(tl.float32)
        accumulator += value * weight
        denominator += weight
    result = accumulator / tl.where(denominator > 0, denominator, 1.0)
    tl.store(output + offsets, result, mask=valid)


def merge_partials(partials: torch.Tensor, lses: torch.Tensor) -> torch.Tensor:
    count, _, heads, tokens, dim = partials.shape
    output = torch.empty_like(partials[0])
    _merge[(triton.cdiv(heads * tokens * dim, 1024),)](
        partials, lses, output, heads * tokens, dim, count, 1024, enable_fp_fusion=False
    )
    return output
