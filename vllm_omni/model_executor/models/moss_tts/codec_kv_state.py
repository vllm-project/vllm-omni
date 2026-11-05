# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Masked persistent writes for shared codec offsets and immutable padding.

Each layer/request keeps its own K/V data. CUDA writes only touched ring
entries; other devices use fixed-shape masked scatters. Padding rows
preserve the null slot's values.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _commit_cache(
    rows,
    cache,
    slots,
    valid,
    offsets,
    batch: tl.constexpr,
    heads: tl.constexpr,
    capacity: tl.constexpr,
    dim: tl.constexpr,
    pool_size: tl.constexpr,
    frames: tl.constexpr,
    block: tl.constexpr,
):
    row, head = tl.program_id(1), tl.program_id(2)
    if tl.load(valid + row):
        slot = tl.load(slots + row)
        offset = tl.load(offsets + row)
        i = tl.program_id(0) * block + tl.arange(0, block)
        touched = ((i // dim - offset % capacity + capacity) % capacity) < frames
        keep = (i < capacity * dim) & touched
        src = (row * heads + head) * capacity * dim + i
        dst = (slot * heads + head) * capacity * dim + i
        k = tl.load(rows + src, keep, 0)
        v = tl.load(rows + batch * heads * capacity * dim + src, keep, 0)
        tl.store(cache + dst, k, keep)
        tl.store(cache + pool_size * heads * capacity * dim + dst, v, keep)


@torch.library.custom_op("vllm_omni::moss_codec_commit_cache", mutates_args=("cache",), device_types="cuda")
def _commit_cache_cuda(
    rows: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    valid: torch.Tensor,
    offsets: torch.Tensor,
    frames: int,
) -> None:
    _, batch, heads, capacity, dim = rows.shape
    _commit_cache[(triton.cdiv(capacity * dim, 256), batch, heads)](
        rows,
        cache,
        slots,
        valid,
        offsets,
        batch,
        heads,
        capacity,
        dim,
        cache.shape[1],
        frames,
        256,
    )


@_commit_cache_cuda.register_fake
def _commit_cache_fake(
    rows: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    valid: torch.Tensor,
    offsets: torch.Tensor,
    frames: int,
) -> None:
    return None


def commit_cache(
    rows: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    valid: torch.Tensor,
    offsets: torch.Tensor,
    frames: int,
) -> None:
    if cache.is_cuda:
        _commit_cache_cuda(rows, cache, slots, valid, offsets, frames)
    else:
        # Invalid rows all address the final pool slot. Every duplicate write
        # copies its unchanged values, without dynamic-shape boolean indexing.
        rows = torch.where(valid[None, :, None, None, None], rows, cache[:, -1:])
        indexes = slots.view(1, -1, 1, 1, 1).expand_as(rows)
        cache.scatter_(1, indexes, rows)


@triton.jit
def _commit_offsets(values, pool, slots, validity, batch: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0) * block + tl.arange(0, block)
    valid = tl.load(validity + row, row < batch, 0)
    slot = tl.load(slots + row, row < batch, 0)
    value = tl.load(values + row, row < batch, 0)
    tl.store(pool + slot, value, (row < batch) & valid)


@torch.library.custom_op("vllm_omni::moss_codec_commit_offsets", mutates_args=("pool",), device_types="cuda")
def _commit_offsets_cuda(
    values: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    _commit_offsets[(triton.cdiv(slots.numel(), 256),)](values, pool, slots, valid, slots.numel(), 256)


@_commit_offsets_cuda.register_fake
def _commit_offsets_fake(
    values: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    return None


def commit_offsets(
    values: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    if pool.is_cuda:
        _commit_offsets_cuda(values, pool, slots, valid)
    else:
        pool.scatter_(0, slots, torch.where(valid, values, pool[-1]))
