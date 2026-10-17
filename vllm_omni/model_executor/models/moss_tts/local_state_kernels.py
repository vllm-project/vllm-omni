# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Fused MOSS Local MRV2 slot-state copies driven by UVA (host-mapped) indices.

Each kernel replaces several eager index_select/index_copy/fill launches and the
host-to-device index upload that preceded them. Indices live in pinned,
device-mapped host memory, so the CPU writes them without a copy launch. All
operations are exact copies/fills.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _reset_slots(
    Slots,
    Hidden,
    Codes,
    Active,
    Keep,
    EagerEmb,
    H: tl.constexpr,
    NVQ: tl.constexpr,
    PAD: tl.constexpr,
    BH: tl.constexpr,
    BQ: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(Slots + i)
    h = tl.arange(0, BH)
    for start in range(0, H, BH):
        mask = start + h < H
        tl.store(Hidden + slot * H + start + h, tl.zeros((BH,), Hidden.dtype.element_ty), mask)
        tl.store(EagerEmb + slot * H + start + h, tl.zeros((BH,), EagerEmb.dtype.element_ty), mask)
    q = tl.arange(0, BQ)
    tl.store(Codes + slot * NVQ + q, tl.full((BQ,), PAD, Codes.dtype.element_ty), q < NVQ)
    tl.store(Active + slot, 1)
    tl.store(Keep + slot, 0)


def reset_slots(slots: torch.Tensor, hidden, codes, active, keep, eager_emb, pad: int) -> None:
    """Admission reset of request slots (``slots`` may be a UVA view)."""
    n = slots.shape[0]
    if n == 0:
        return
    h = hidden.shape[1]
    nvq = codes.shape[1]
    _reset_slots[(n,)](
        slots,
        hidden,
        codes,
        active.view(torch.uint8),
        keep.view(torch.uint8),
        eager_emb,
        h,
        nvq,
        pad,
        1024,
        triton.next_power_of_2(nvq),
        num_warps=4,
    )


@triton.jit
def _gather_mtp(
    Index,
    InputIds,
    Embeds,
    HiddenPool,
    ActivePool,
    Ids,
    Emb,
    Hidden,
    Step,
    H: tl.constexpr,
    STEP_STRIDE: tl.constexpr,
    BH: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(Index + 2 * i)
    offset = tl.load(Index + 2 * i + 1)
    tl.store(Ids + i, tl.load(InputIds + offset).to(Ids.dtype.element_ty))
    h = tl.arange(0, BH)
    for start in range(0, H, BH):
        mask = start + h < H
        tl.store(Emb + i * H + start + h, tl.load(Embeds + offset * H + start + h, mask), mask)
        tl.store(Hidden + i * H + start + h, tl.load(HiddenPool + slot * H + start + h, mask), mask)
    active = tl.load(ActivePool + slot)
    tl.store(Step + i * STEP_STRIDE, active.to(Step.dtype.element_ty))


def gather_mtp(index, input_ids, embeds, hidden_pool, active_pool, ids, emb, hidden, step) -> None:
    """index: (B, 2) int64 [slot, token offset] rows (may be a UVA view).

    ids[i] = input_ids[offset]; emb[i] = embeds[offset]; hidden[i] = hidden_pool[slot];
    step[i, 0] = active_pool[slot].
    """
    n = index.shape[0]
    if n == 0:
        return
    _gather_mtp[(n,)](
        index,
        input_ids,
        embeds,
        hidden_pool,
        active_pool.view(torch.uint8),
        ids,
        emb,
        hidden,
        step,
        embeds.shape[1],
        step.stride(0),
        1024,
        num_warps=4,
    )


@triton.jit
def _scatter_mtp(
    Index,
    NewEmb,
    Codes,
    Embeds,
    CodesPool,
    H: tl.constexpr,
    NVQ: tl.constexpr,
    BH: tl.constexpr,
    BQ: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(Index + 2 * i)
    offset = tl.load(Index + 2 * i + 1)
    h = tl.arange(0, BH)
    for start in range(0, H, BH):
        mask = start + h < H
        tl.store(Embeds + offset * H + start + h, tl.load(NewEmb + i * H + start + h, mask), mask)
    q = tl.arange(0, BQ)
    tl.store(CodesPool + slot * NVQ + q, tl.load(Codes + i * NVQ + q, q < NVQ), q < NVQ)


def scatter_mtp(index, new_emb, codes, embeds, codes_pool) -> None:
    """embeds[offset] = new_emb[i]; codes_pool[slot] = codes[i] for each index row."""
    n = index.shape[0]
    if n == 0:
        return
    nvq = codes_pool.shape[1]
    _scatter_mtp[(n,)](
        index,
        new_emb,
        codes,
        embeds,
        codes_pool,
        embeds.shape[1],
        nvq,
        1024,
        triton.next_power_of_2(nvq),
        num_warps=4,
    )


@triton.jit
def _add_rows(
    SrcIndex,
    DstIndex,
    Src,
    Dst,
    H: tl.constexpr,
    SRC_STRIDE: tl.constexpr,
    DST_STRIDE: tl.constexpr,
    HAS_SRC_INDEX: tl.constexpr,
    BH: tl.constexpr,
):
    i = tl.program_id(0)
    if HAS_SRC_INDEX:
        src = tl.load(SrcIndex + i * SRC_STRIDE)
    else:
        src = i
    dst = tl.load(DstIndex + i * DST_STRIDE)
    h = tl.arange(0, BH)
    for start in range(0, H, BH):
        mask = start + h < H
        a = tl.load(Dst + dst * H + start + h, mask).to(tl.float32)
        b = tl.load(Src + src * H + start + h, mask).to(tl.float32)
        tl.store(Dst + dst * H + start + h, (a + b).to(Dst.dtype.element_ty), mask)


def add_rows(dst_index: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, src_index: torch.Tensor | None = None):
    """dst[dst_index[i]] += src[src_index[i] if given else i], rounded once like ``dst[idx] + src[...]``.

    Index tensors may be strided views (e.g. one column of a UVA ``(n, 2)`` pool);
    destination rows must be distinct.
    """
    n = dst_index.shape[0]
    if n == 0:
        return
    _add_rows[(n,)](
        src_index if src_index is not None else dst_index,
        dst_index,
        src,
        dst,
        dst.shape[1],
        src_index.stride(0) if src_index is not None else 0,
        dst_index.stride(0),
        src_index is not None,
        1024,
        num_warps=4,
    )


@triton.jit
def _snapshot_hidden(
    QueryStartLoc,
    IdxMapping,
    Hidden,
    HiddenPool,
    H: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    BH: tl.constexpr,
):
    i = tl.program_id(0)
    last = tl.load(QueryStartLoc + i + 1).to(tl.int64) - 1
    slot = tl.load(IdxMapping + i).to(tl.int64)
    h = tl.arange(0, BH)
    for start in range(0, H, BH):
        mask = start + h < H
        tl.store(HiddenPool + slot * H + start + h, tl.load(Hidden + last * HIDDEN_STRIDE + start + h, mask), mask)


def snapshot_hidden(query_start_loc, idx_mapping, hidden_states, hidden_pool, num_reqs: int) -> None:
    """hidden_pool[idx_mapping[i]] = hidden_states[query_start_loc[i + 1] - 1] for i < num_reqs."""
    if num_reqs == 0:
        return
    _snapshot_hidden[(num_reqs,)](
        query_start_loc,
        idx_mapping,
        hidden_states,
        hidden_pool,
        hidden_pool.shape[1],
        hidden_states.stride(0),
        1024,
        num_warps=4,
    )


@triton.jit
def _gather_eager_mtp(
    Index,
    HiddenPool,
    KeepPool,
    ActivePool,
    Ids,
    Emb,
    Hidden,
    Step,
    H: tl.constexpr,
    STEP_STRIDE: tl.constexpr,
    BH: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(Index + 2 * i)
    first = tl.load(Index + 2 * i + 1)
    tl.store(Ids + i, tl.zeros((), Ids.dtype.element_ty))
    h = tl.arange(0, BH)
    for start in range(0, H, BH):
        mask = start + h < H
        tl.store(Emb + i * H + start + h, tl.zeros((BH,), Emb.dtype.element_ty), mask)
        tl.store(Hidden + i * H + start + h, tl.load(HiddenPool + slot * H + start + h, mask), mask)
    gate = (tl.load(KeepPool + slot) != 0) | (first != 0)
    active = (tl.load(ActivePool + slot) != 0) & gate
    tl.store(Step + i * STEP_STRIDE, active.to(Step.dtype.element_ty))


def gather_eager_mtp(index, hidden_pool, keep_pool, active_pool, ids, emb, hidden, step) -> None:
    """index: (B, 2) int64 [slot, completes prefill] rows (may be a UVA view).

    ids[i] = 0; emb[i] = 0; hidden[i] = hidden_pool[slot];
    step[i, 0] = active_pool[slot] & (keep_pool[slot] | first).
    """
    n = index.shape[0]
    if n == 0:
        return
    _gather_eager_mtp[(n,)](
        index,
        hidden_pool,
        keep_pool.view(torch.uint8),
        active_pool.view(torch.uint8),
        ids,
        emb,
        hidden,
        step,
        hidden_pool.shape[1],
        step.stride(0),
        1024,
        num_warps=4,
    )


@triton.jit
def _scatter_eager_mtp(
    Index,
    Contribution,
    Codes,
    EagerEmb,
    CodesPool,
    KeepPool,
    H: tl.constexpr,
    NVQ: tl.constexpr,
    CONTRIBUTION_STRIDE: tl.constexpr,
    CODES_STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    BH: tl.constexpr,
    BQ: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(Index + 2 * i)
    h = tl.arange(0, BH)
    for start in range(0, H, BH):
        mask = start + h < H
        row = tl.load(Contribution + i * CONTRIBUTION_STRIDE + start + h, mask)
        tl.store(EagerEmb + slot * H + start + h, row, mask)
    q = tl.arange(0, BQ)
    codes = tl.load(Codes + i * CODES_STRIDE + q, q < NVQ, PAD)
    tl.store(CodesPool + slot * NVQ + q, codes, q < NVQ)
    tl.store(KeepPool + slot, tl.max((codes != PAD).to(tl.uint8), axis=0))


def scatter_eager_mtp(index, contribution, codes, eager_emb, codes_pool, keep_pool, pad: int) -> None:
    """For each index row's slot: eager_emb = contribution[i]; codes_pool = codes[i];
    keep_pool = any(codes[i] != pad)."""
    n = index.shape[0]
    if n == 0:
        return
    nvq = codes_pool.shape[1]
    _scatter_eager_mtp[(n,)](
        index,
        contribution,
        codes,
        eager_emb,
        codes_pool,
        keep_pool.view(torch.uint8),
        eager_emb.shape[1],
        nvq,
        contribution.stride(0),
        codes.stride(0),
        pad,
        1024,
        triton.next_power_of_2(nvq),
        num_warps=4,
    )


@triton.jit
def _gather_codes(
    Index,
    CodesPool,
    Out,
    Keep,
    NVQ: tl.constexpr,
    PAD: tl.constexpr,
    BQ: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(Index + 2 * i)
    row = tl.load(Index + 2 * i + 1)
    q = tl.arange(0, BQ)
    codes = tl.load(CodesPool + slot * NVQ + q, q < NVQ, PAD)
    tl.store(Out + i * NVQ + q, codes, q < NVQ)
    tl.store(Keep + row, tl.max((codes != PAD).to(tl.uint8), axis=0))


def gather_codes(index, codes_pool, out, keep, pad: int) -> None:
    """index: (B, 2) int64 [slot, batch row]. out[i] = codes_pool[slot]; keep[row] = any(out[i] != pad)."""
    n = index.shape[0]
    if n == 0:
        return
    nvq = codes_pool.shape[1]
    _gather_codes[(n,)](
        index, codes_pool, out, keep.view(torch.uint8), nvq, pad, triton.next_power_of_2(nvq), num_warps=1
    )
