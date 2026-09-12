# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Masked state writes for shared codec metadata and read-only padding slots.

Attention and the reference scatter (including duplicate destinations for
oversized chunks) are deliberately left to PyTorch. Only persistent writeback
is changed here. Invalid execution rows must never write even a shared null slot.
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
        # T >= C touches the entire ring. Preserve whichever values the
        # reference scatter produced, rather than inventing a last-writer rule.
        touched = ((i // dim - offset % capacity + capacity) % capacity) < frames
        keep = (i < capacity * dim) & touched
        src = (row * heads + head) * capacity * dim + i
        dst = (slot * heads + head) * capacity * dim + i
        k = tl.load(rows + src, keep, 0)
        v = tl.load(rows + batch * heads * capacity * dim + src, keep, 0)
        tl.store(cache + dst, k, keep)
        tl.store(cache + pool_size * heads * capacity * dim + dst, v, keep)


@torch.library.custom_op("moss_codec_state::commit_cache", mutates_args=("cache",), device_types="cuda")
def _commit_cache_cuda(
    rows: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    valid: torch.Tensor,
    offsets: torch.Tensor,
    frames: int,
) -> None:
    _, b, h, c, d = rows.shape
    _commit_cache[(triton.cdiv(c * d, 256), b, h)](
        rows,
        cache,
        slots,
        valid,
        offsets,
        b,
        h,
        c,
        d,
        cache.shape[1],
        frames,
        256,
    )


@_commit_cache_cuda.register_fake
def _commit_cache_fake(rows, cache, slots, valid, offsets, frames):
    return None


def commit_cache(rows, cache, slots, valid, offsets, frames):
    if cache.is_cuda:
        _commit_cache_cuda(rows, cache, slots, valid, offsets, frames)
    else:
        cache.index_copy_(1, slots[valid], rows[:, valid])


@triton.jit
def _commit_offsets(values, pool, slots, validity, batch: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0) * block + tl.arange(0, block)
    valid = tl.load(validity + row, row < batch, 0)
    slot = tl.load(slots + row, row < batch, 0)
    value = tl.load(values + row, row < batch, 0)
    tl.store(pool + slot, value, (row < batch) & valid)


@torch.library.custom_op("moss_codec_state::commit_offsets", mutates_args=("pool",), device_types="cuda")
def _commit_offsets_cuda(values: torch.Tensor, pool: torch.Tensor, slots: torch.Tensor, valid: torch.Tensor) -> None:
    _commit_offsets[(triton.cdiv(slots.numel(), 256),)](values, pool, slots, valid, slots.numel(), 256)


@_commit_offsets_cuda.register_fake
def _commit_offsets_fake(values, pool, slots, valid):
    return None


def commit_offsets(values, pool, slots, valid):
    if pool.is_cuda:
        _commit_offsets_cuda(values, pool, slots, valid)
    else:
        pool.index_copy_(0, slots[valid], values[valid])
