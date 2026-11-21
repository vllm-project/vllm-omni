# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Fused RVQ filters, sampling state and feedback; native RNG/top-k/top-p semantics."""

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax


@triton.jit
def _prepare(
    L,
    OUT,
    SLOTS,
    POSITION,
    HISTORY,
    TEMP,
    PENALTY,
    WINDOW,
    BOOKS,
    MINIMUM,
    K,
    P,
    OUT_K,
    OUT_P,
    V: tl.constexpr,
    CB: tl.constexpr,
    EOA: tl.constexpr,
    RING: tl.constexpr,
    WIDTH: tl.constexpr,
    BV: tl.constexpr,
    BW: tl.constexpr,
):
    row = tl.program_id(0)
    request, book = row // 10, row % 10
    slot = tl.load(SLOTS + request)
    position = tl.load(POSITION + slot)
    ids = tl.arange(0, BV)
    logits = tl.load(L + row * V + ids, ids < V, other=-float("inf")).to(tl.float32)
    minimum = tl.load(MINIMUM + slot)
    blocked = (minimum > 0) & (position < minimum + book)
    logits = tl.where((ids < CB) | ((ids == EOA) & ~blocked), logits, -float("inf"))
    if WIDTH > 0:
        offsets = tl.arange(0, BW)
        window = tl.load(WINDOW + slot)
        books = tl.load(BOOKS + slot)
        history = tl.load(
            HISTORY + (slot * 10 + book) * RING + (position - 1 - offsets + RING) % RING,
            (offsets < WIDTH) & (offsets < tl.minimum(position, window)) & (book < books),
            other=-1,
        )
        valid = (
            (offsets < WIDTH)
            & (offsets < tl.minimum(position, window))
            & (book < books)
            & (history >= 0)
            & (history < CB)
        )
        # Invalid history goes into bin zero, then is subtracted; no undefined
        # out-of-range histogram indices and no [B,10,window,vocab] stencil.
        counts = tl.histogram(tl.where(valid, history, 0).to(tl.int32), BV)
        counts -= tl.where(ids == 0, tl.sum((~valid).to(tl.int32), 0), 0)
        penalty = tl.load(PENALTY + slot)
        adjusted = tl.where(logits > 0, tl.div_rn(logits, penalty), logits * penalty)
        logits = tl.where(counts > 0, adjusted, logits)
    temperature = tl.load(TEMP + slot)
    logits = tl.div_rn(logits, tl.where(temperature > 0, temperature, 1.0))
    tl.store(OUT + row * V + ids, logits, ids < V)
    tl.store(OUT_K + row, tl.load(K + slot))
    tl.store(OUT_P + row, tl.load(P + slot))


@triton.jit
def _draw(L, SLOTS, POSITION, TEMP, SEED, MIN_P, OUT, V: tl.constexpr, BV: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(SLOTS + row // 10)
    ids = tl.arange(0, BV)
    logits = tl.load(L + row * V + ids, ids < V, other=-float("inf"))
    probs = tl.exp(logits - tl.max(logits, 0))
    probs = tl.div_rn(probs, tl.sum(probs, 0))
    probs = tl.where(probs < tl.load(MIN_P + slot) * tl.max(probs, 0), 0.0, probs)
    total = tl.sum(probs, 0)
    fallback = tl.argmax(logits, 0)
    probs = tl.where(total > 0, tl.div_rn(probs, tl.maximum(total, 1.0e-8)), (ids == fallback).to(tl.float32))
    log_probs = tl.log(probs)
    _, code = gumbel_noised_argmax(
        log_probs,
        ids,
        ids < V,
        tl.load(SEED + slot),
        tl.load(POSITION + slot) * 10 + row % 10,
        tl.load(TEMP + slot),
        IS_DRAFTING=False,
        USE_FP64=False,
        APPLY_TEMPERATURE=False,
    )
    tl.store(OUT + row, code)


def sample_rvq(logits, slots, state, settings, history, width, codebook_size, eoa_id):
    batch, books, vocab = logits.shape
    filtered = torch.empty((batch * books, vocab), device=logits.device, dtype=torch.float32)
    k = torch.empty(batch * books, device=logits.device, dtype=torch.int32)
    p = torch.empty(batch * books, device=logits.device, dtype=torch.float32)
    _prepare[(batch * books,)](
        logits,
        filtered,
        slots,
        state.position,
        history,
        settings.temperature.gpu,
        settings.repetition_penalty.gpu,
        settings.repetition_window.gpu,
        settings.repetition_codebooks.gpu,
        settings.min_frames.gpu,
        settings.top_k.gpu,
        settings.top_p.gpu,
        k,
        p,
        vocab,
        codebook_size,
        eoa_id,
        history.shape[-1],
        width,
        triton.next_power_of_2(vocab),
        triton.next_power_of_2(max(1, width)),
        enable_fp_fusion=False,
    )
    filtered = apply_top_k_top_p(filtered, k, p, has_topp_only_rows=True)
    codes = torch.empty((batch, books), device=logits.device, dtype=torch.int64)
    _draw[(batch * books,)](
        filtered,
        slots,
        state.position,
        settings.temperature.gpu,
        settings.seed.gpu,
        settings.min_p.gpu,
        codes,
        vocab,
        triton.next_power_of_2(vocab),
        enable_fp_fusion=False,
    )
    return codes


@triton.jit
def _accept(
    SLOTS,
    SAMPLED,
    ACTIVE,
    POSITION,
    EOA_FRAME,
    REMAINING,
    PREFIX,
    IGNORE,
    BOOK0,
    CODES,
    LOGICAL,
    LIVE,
    FINISHED,
    EOA: tl.constexpr,
    PAD: tl.constexpr,
    HAS_ACTIVE: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(SLOTS + row)
    columns = tl.arange(0, 16)
    position = tl.load(POSITION + slot)
    previous = tl.load(EOA_FRAME + slot)
    remaining = tl.load(REMAINING + slot)
    active = tl.full((), True, tl.int1)
    if HAS_ACTIVE:
        active = tl.load(ACTIVE + row) > 0
    active &= ~((previous >= 0) & (remaining == 0))
    started = tl.load(PREFIX + slot) | (position >= columns)
    raw = tl.load(SAMPLED + row * 10 + columns, columns < 10, other=PAD)
    codes = tl.where(started & active, raw, PAD)
    detected = (codes == EOA) & ~tl.load(IGNORE + slot) & (~tl.load(BOOK0 + slot) | (columns == 0)) & (columns < 10)
    last = tl.max(tl.where(detected, columns, -1), 0)
    first = active & (previous < 0) & (last >= 0)
    frame = tl.where(first, tl.maximum(0, position - last), previous)
    remaining = tl.where(first, 11, remaining)
    remaining = tl.where(active & (frame >= 0), tl.maximum(remaining - 1, 0), remaining)
    finished = (frame >= 0) & (remaining == 0)
    tl.store(POSITION + slot, position + active.to(tl.int64))
    tl.store(EOA_FRAME + slot, frame)
    tl.store(REMAINING + slot, remaining)
    tl.store(CODES + row * 10 + columns, codes, columns < 10)
    tl.store(LOGICAL + row, tl.where(finished, 2, 1))
    tl.store(LIVE + row, active)
    tl.store(FINISHED + row, finished)


def accept_rvq(state, slots, sampled_codes, active):
    batch = slots.numel()
    codes = torch.empty((batch, 10), dtype=torch.int64, device=slots.device)
    logical = torch.empty((batch, 1), dtype=torch.int32, device=slots.device)
    live = torch.empty(batch, dtype=torch.bool, device=slots.device)
    finished = torch.empty_like(live)
    _accept[(batch,)](
        slots,
        sampled_codes,
        active,
        state.position,
        state.eoa_frame,
        state.drain_remaining,
        state.has_audio_prefix,
        state.ignore_eos,
        state.eoa_codebook0,
        codes,
        logical,
        live,
        finished,
        state.eoa_id,
        state.pad_id,
        active is not None,
    )
    return codes, logical, live, finished


@triton.jit
def _feedback(
    SLOTS,
    CODES,
    ACTIVE,
    POSITION,
    HISTORY,
    NEXT,
    E0,
    E1,
    E2,
    E3,
    E4,
    E5,
    E6,
    E7,
    E8,
    E9,
    ET,
    D: tl.constexpr,
    BD: tl.constexpr,
    RING: tl.constexpr,
    TEXT_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(SLOTS + row)
    active = tl.load(ACTIVE + row)
    columns = tl.arange(0, 16)
    codes = tl.load(CODES + row * 10 + columns, columns < 10, other=0)
    position = tl.load(POSITION + slot) - active.to(tl.int64)
    tl.store(HISTORY + (slot * 10 + columns) * RING + position % RING, codes, (columns < 10) & active)
    dims = tl.arange(0, BD)
    first = tl.sum(tl.where(columns == 0, codes, 0), 0)
    value = tl.load(E0 + first * D + dims, dims < D, other=0).to(tl.float32)
    weights = (E1, E2, E3, E4, E5, E6, E7, E8, E9)
    # Training sums one codebook at a time in embedding dtype (including BF16).
    for book in tl.static_range(1, 10):
        code = tl.sum(tl.where(columns == book, codes, 0), 0)
        part = tl.load(weights[book - 1] + code * D + dims, dims < D, other=0).to(tl.float32)
        value = (value + part).to(E0.dtype.element_ty).to(tl.float32)
    text = tl.load(ET + TEXT_PAD * D + dims, dims < D, other=0).to(tl.float32)
    value = (value + text).to(E0.dtype.element_ty)
    tl.store(NEXT + slot * D + dims, value, (dims < D) & active)


def commit_feedback(model, slots, step, state, history, next_embeddings):
    weights = [e.weight for e in model.multi_embedder.embedders]
    _feedback[(slots.numel(),)](
        slots,
        step.codes,
        step.active,
        state.position,
        history,
        next_embeddings,
        *weights,
        next_embeddings.shape[1],
        triton.next_power_of_2(next_embeddings.shape[1]),
        history.shape[-1],
        model.config.text_vocab,
    )
