# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803 - Triton constexpr parameters use kernel-style capitals.
"""Gather/scatter kernels around the eager Talker-MTP call.

Each engine step, the rows sampled this step need their CB0, codec embedding
and last hidden state gathered into the MTP inputs, and the MTP results
scattered back into the eager-frame embeddings and the retained outputs. As
separate torch ops that is a few dozen small launches per step, and the engine
loop is host-bound, so each costs the whole batch. These two kernels do the
same indexing in one launch each; the row metadata arrives in one H2D copy.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _pre_kernel(
    meta_ptr,  # int64 [4, n]: batch row, request index, prefill flag, first-audio scheduled
    n,
    qsl_ptr,  # query_start_loc
    sampled_ptr,  # [num_reqs, s_stride]
    s_stride,
    hidden_ptr,  # [T, H]
    emb_w_ptr,  # [V, H]
    ids_ptr,  # [n] MTP input ids
    emb_ptr,  # [n, H]
    mtp_hidden_ptr,  # [n, H]
    step_ptr,  # [n, H]
    last_ptr,  # int64 [n] out
    layer0_ptr,  # int64 [n] out
    H: tl.constexpr,
    BH: tl.constexpr,
):
    b = tl.program_id(0)
    row = tl.load(meta_ptr + b)
    last = tl.load(qsl_ptr + row + 1).to(tl.int64) - 1
    l0 = tl.load(sampled_ptr + row * s_stride).to(tl.int64)
    tl.store(ids_ptr + b, l0.to(ids_ptr.dtype.element_ty))
    tl.store(last_ptr + b, last)
    tl.store(layer0_ptr + b, l0)
    for h0 in tl.static_range(0, H, BH):
        c = h0 + tl.arange(0, BH)
        m = c < H
        tl.store(emb_ptr + b * H + c, tl.load(emb_w_ptr + l0 * H + c, mask=m).to(emb_ptr.dtype.element_ty), mask=m)
        tl.store(mtp_hidden_ptr + b * H + c, tl.load(hidden_ptr + last * H + c, mask=m), mask=m)
        tl.store(step_ptr + b * H + c, tl.zeros([BH], dtype=step_ptr.dtype.element_ty), mask=m)


@triton.jit
def _post_kernel(
    meta_ptr,
    n,
    last_ptr,
    layer0_ptr,
    frame_ptr,  # [>=n, H] MTP frame embeddings
    eager_ptr,  # [R, H]
    codes_ptr,  # [>=n, Q]
    codes_out_ptr,  # [T, Q]
    input_ids_ptr,  # [T] this step's input ids
    valid_out_ptr,  # [T]
    valid_ptr,  # [n] bool out
    fa_ptr,  # [T] first_audio output or dummy
    fa_valid_ptr,  # [R] bool or dummy
    codes32_ptr,  # [n, Q] int32 copy of the codes for the codec, or dummy
    vocab,
    H: tl.constexpr,
    Q: tl.constexpr,
    QP: tl.constexpr,
    BH: tl.constexpr,
    WRITE_FA: tl.constexpr,
    HAS_FA_VALID: tl.constexpr,
    CODES32: tl.constexpr,
):
    b = tl.program_id(0)
    req = tl.load(meta_ptr + n + b)
    prefill = tl.load(meta_ptr + 2 * n + b) != 0
    last = tl.load(last_ptr + b)
    l0 = tl.load(layer0_ptr + b)
    for h0 in tl.static_range(0, H, BH):
        c = h0 + tl.arange(0, BH)
        m = c < H
        v = tl.load(frame_ptr + b * H + c, mask=m)
        tl.store(eager_ptr + req * H + c, v.to(eager_ptr.dtype.element_ty), mask=m)
    q = tl.arange(0, QP)
    qm = q < Q
    cv = tl.load(codes_ptr + b * Q + q, mask=qm)
    tl.store(codes_out_ptr + last * Q + q, cv.to(codes_out_ptr.dtype.element_ty), mask=qm)
    if CODES32:
        tl.store(codes32_ptr + b * Q + q, cv.to(tl.int32), mask=qm)
    inp = tl.load(input_ids_ptr + last).to(tl.int64)
    in_valid = ((inp >= 0) & (inp < vocab)) | prefill
    valid = (l0 >= 0) & (l0 < vocab) & in_valid
    tl.store(valid_out_ptr + last, valid.to(valid_out_ptr.dtype.element_ty))
    tl.store(valid_ptr + b, valid)
    if WRITE_FA:
        scheduled = tl.load(meta_ptr + 3 * n + b) != 0
        if HAS_FA_VALID:
            scheduled = scheduled & (tl.load(fa_valid_ptr + req) != 0)
        else:
            scheduled = scheduled & False
        tl.store(fa_ptr + last, scheduled.to(fa_ptr.dtype.element_ty))


def eager_pre(meta, n, qsl, sampled, hidden, emb_w, ids, emb, mtp_hidden, step):
    last = torch.empty(n, dtype=torch.long, device=hidden.device)
    layer0 = torch.empty(n, dtype=torch.long, device=hidden.device)
    H = hidden.shape[-1]
    _pre_kernel[(n,)](
        meta, n, qsl, sampled, sampled.stride(0), hidden, emb_w, ids, emb, mtp_hidden, step, last, layer0,
        H=H, BH=min(1024, triton.next_power_of_2(H)),
    )  # fmt: skip
    return last, layer0


def eager_post(
    meta, n, last, layer0, frames, eager, codes, codes_out, input_ids, valid_out, fa, fa_valid, vocab, codes32=None
):
    valid = torch.empty(n, dtype=torch.bool, device=last.device)
    H = eager.shape[-1]
    Q = codes_out.shape[-1]
    dummy = valid
    _post_kernel[(n,)](
        meta, n, last, layer0, frames, eager, codes, codes_out, input_ids, valid_out, valid,
        fa if fa is not None else dummy, fa_valid if fa_valid is not None else dummy,
        codes32 if codes32 is not None else dummy, int(vocab),
        H=H, Q=Q, QP=triton.next_power_of_2(Q), BH=min(1024, triton.next_power_of_2(H)),
        WRITE_FA=fa is not None, HAS_FA_VALID=fa_valid is not None, CODES32=codes32 is not None,
    )  # fmt: skip
    return valid


@triton.jit
def _settled_kernel(
    meta_ptr,  # int64 [2, n]: request slot, token row
    n,
    eager_ptr,  # [R, H] eager-frame embeddings
    step_ptr,  # [H] constant text step
    embeds_ptr,  # [T, H]
    H: tl.constexpr,
    BH: tl.constexpr,
):
    b = tl.program_id(0)
    slot = tl.load(meta_ptr + b)
    row = tl.load(meta_ptr + n + b)
    for h0 in tl.static_range(0, H, BH):
        c = h0 + tl.arange(0, BH)
        m = c < H
        frame = tl.load(eager_ptr + slot * H + c, mask=m).to(tl.float32)
        step = tl.load(step_ptr + c, mask=m).to(eager_ptr.dtype.element_ty).to(tl.float32)
        value = (frame + step).to(eager_ptr.dtype.element_ty)
        tl.store(embeds_ptr + row * H + c, value.to(embeds_ptr.dtype.element_ty), mask=m)


@triton.jit
def _record_kernel(
    index_ptr,  # int64 [3, n]: request slot, position, token row
    n,
    embeds_ptr,  # [T, H]
    slab_ptr,  # [R, L, H]
    slab_stride,
    H: tl.constexpr,
    BH: tl.constexpr,
):
    b = tl.program_id(0)
    slot = tl.load(index_ptr + b)
    pos = tl.load(index_ptr + n + b)
    row = tl.load(index_ptr + 2 * n + b)
    for h0 in tl.static_range(0, H, BH):
        c = h0 + tl.arange(0, BH)
        m = c < H
        value = tl.load(embeds_ptr + row * H + c, mask=m)
        tl.store(slab_ptr + slot * slab_stride + pos * H + c, value.to(slab_ptr.dtype.element_ty), mask=m)


def settled_frames(meta: torch.Tensor, n: int, eager: torch.Tensor, step: torch.Tensor, embeds: torch.Tensor) -> None:
    """``embeds[row] = eager[slot] + step`` for each (slot, row) column of ``meta``, rounded like the torch ops."""
    H = eager.shape[-1]
    _settled_kernel[(n,)](meta, n, eager, step, embeds, H=H, BH=min(1024, triton.next_power_of_2(H)))


def record_rows(index: torch.Tensor, n: int, embeds: torch.Tensor, slab: torch.Tensor) -> None:
    """``slab[slot, pos] = embeds[row]`` for each (slot, pos, row) column of ``index``."""
    H = embeds.shape[-1]
    _record_kernel[(n,)](index, n, embeds, slab, slab.stride(0), H=H, BH=min(1024, triton.next_power_of_2(H)))
