# ruff: noqa: N803
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused per-step Higgs kernels for the MRV2 decode loop.

Each replaces a chain of small eager launches that ran on every model step
(terminal-row restore, prefill reset and audio-tail validation before dense
sampling; token and feedback embeddings on pure decode steps).
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _step_tail_guard_kernel(
    ids_ptr,
    qsl_ptr,
    has_codes_ptr,
    done_ptr,
    delay_ptr,
    eoc_ptr,
    error_ptr,
    num_rows,
    audio_id,
    eos_id,
    HAS_QSL: tl.constexpr,
    PROMPT_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)
    mask = rows < num_rows
    if HAS_QSL:
        start = tl.load(qsl_ptr + rows, mask=mask, other=0)
        end = tl.load(qsl_ptr + rows + 1, mask=mask, other=1)
        valid = end > 0
        tail = tl.load(ids_ptr + tl.maximum(end - 1, 0), mask=mask, other=audio_id)
        prefill = mask & ((end - start) > 1)
    else:
        valid = mask
        tail = tl.load(ids_ptr + rows, mask=mask, other=audio_id)
        prefill = rows < 0
    done = tl.load(done_ptr + rows, mask=mask, other=0) != 0
    has = tl.load(has_codes_ptr + rows, mask=mask, other=0) != 0
    if PROMPT_MODE:
        # An in-flight EOS row after retirement is terminal, never a new stream.
        done = done | (tail == eos_id)
    # New prefill rows start a fresh delay-pattern stream.
    done = done & ~prefill
    has = has & ~prefill
    tl.store(done_ptr + rows, done, mask=mask)
    tl.store(has_codes_ptr + rows, has, mask=mask)
    tl.store(delay_ptr + rows, 0, mask=prefill)
    tl.store(eoc_ptr + rows, -1, mask=prefill)
    ok = (tail == audio_id) & valid
    if PROMPT_MODE:
        ok = ok | has | done
    tl.store(error_ptr, tl.max((mask & ~ok).to(tl.int32), axis=0))


def step_tail_guard(
    ids: torch.Tensor,
    qsl: torch.Tensor | None,
    has_codes: torch.Tensor,
    done: torch.Tensor,
    delay: torch.Tensor,
    eoc: torch.Tensor,
    error: torch.Tensor,
    *,
    audio_id: int,
    eos_id: int,
    prompt_mode: bool,
) -> None:
    """Fused terminal-row restore, prefill reset and audio-tail validation.

    One launch replaces the index/compare/reset chains that ran before every
    dense sample, for pure decode and mixed prefill steps alike; ``error``
    becomes nonzero if any row is not an audio row. ``qsl`` (query start
    locations) selects each row's last token; ``None`` means one token per row.
    """
    rows = int(has_codes.numel())
    _step_tail_guard_kernel[(1,)](
        ids,
        ids if qsl is None else qsl,
        has_codes,
        done,
        delay,
        eoc,
        error,
        rows,
        audio_id,
        eos_id,
        HAS_QSL=qsl is not None,
        PROMPT_MODE=prompt_mode,
        BLOCK=max(16, triton.next_power_of_2(rows)),
    )


@triton.jit
def _decode_embeddings_kernel(
    out_ptr,
    ids_ptr,
    text_weight_ptr,
    audio_weight_ptr,
    codes_ptr,
    has_codes_ptr,
    num_rows,
    hidden,
    audio_vocab,
    BOOKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    out = out_ptr + row.to(tl.int64) * hidden + cols
    if row >= num_rows:
        # Graph padding rows own no request and must stay zero.
        tl.store(out, tl.zeros((BLOCK,), dtype=out_ptr.dtype.element_ty), mask=mask)
        return
    if tl.load(has_codes_ptr + row) != 0:
        # Audio feedback: sum of the previous frame's codebook embeddings,
        # accumulated in float32 and rounded once like ``embedding(...).sum()``.
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for book in range(BOOKS):
            code = tl.load(codes_ptr + row * BOOKS + book).to(tl.int64) + book * audio_vocab
            acc += tl.load(audio_weight_ptr + code * hidden + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        token = tl.maximum(tl.load(ids_ptr + row), 0).to(tl.int64)
        tl.store(out, tl.load(text_weight_ptr + token * hidden + cols, mask=mask), mask=mask)


def decode_embeddings(
    out: torch.Tensor,
    ids: torch.Tensor,
    text_weight: torch.Tensor,
    audio_weight: torch.Tensor,
    codes: torch.Tensor,
    has_codes: torch.Tensor,
    padded_rows: int,
    audio_vocab: int,
) -> None:
    """One-token-per-row input embeddings with audio feedback, into ``out[:padded_rows]``."""
    rows, hidden = int(ids.numel()), int(out.shape[1])
    books = int(codes.shape[1])
    block = 1024
    _decode_embeddings_kernel[(padded_rows, triton.cdiv(hidden, block))](
        out, ids, text_weight, audio_weight, codes, has_codes, rows, hidden, audio_vocab, BOOKS=books, BLOCK=block
    )
