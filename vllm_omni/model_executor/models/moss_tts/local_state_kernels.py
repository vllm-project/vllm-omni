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
