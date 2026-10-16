# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803 - Triton constexpr parameters use kernel-style capitals.
"""Prefill embedding rows of several Talker requests in one launch.

A new cached CustomVoice prompt is ``prefix + (table[text] + pad_row) +
eos_row + tail`` (rows of a constant table and the projected text table); a
later prefill chunk copies rows of the prompt stored by the first one; rows of
the scheduled span past the prompt's end take the tts pad row (row 0 of the
constant table). Each request is one descriptor row and the kernel derives
every row's sources and destinations from it, so the host does no per-row
work: a new prompt goes to the step's packed prompt buffer, and the scheduled
span to the runner's ``inputs_embeds`` (whose token ids become the codec pad
id). Sums are taken in fp32 and rounded once, as the eager ``a + b`` of two
bf16 rows is.
"""

from __future__ import annotations

import numpy as np
import torch
from vllm.triton_utils import tl, triton

# Descriptor columns (int64), in order:
#   0 prompt rows, 1 first prompt row of the scheduled span, 2 scheduled tokens,
#   3 first inputs_embeds row of the span, 4 first packed row of a new prompt
#   (-1 for a stored prompt), 5 device address of a stored [rows, H] prompt,
#   6 prefix start, 7 prefix length, 8 pad row, 9 eos row, 10 tail row
#   (constant-table rows), 11 index of the first text id in the ids array.
DESC_WIDTH = 12


@triton.jit
def _prefill_rows_kernel(
    desc_ptr,
    ids_ptr,
    const_ptr,
    table_ptr,
    packed_ptr,
    embeds_ptr,
    embeds_stride,
    input_ids_ptr,
    pad_id,
    hidden,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1).to(tl.int64)
    d = desc_ptr + req * 12  # DESC_WIDTH
    n = tl.load(d + 0)
    offset = tl.load(d + 1)
    span = tl.load(d + 2)
    start = tl.load(d + 3)
    packed_first = tl.load(d + 4)
    in_span = (j >= offset) & (j < offset + span)
    to_packed = (packed_first >= 0) & (j < n)
    if in_span | to_packed:
        dtype = embeds_ptr.dtype.element_ty
        prefix_len = tl.load(d + 7)
        is_text = (packed_first >= 0) & (j >= prefix_len) & (j < n - 2)
        text_id = tl.load(ids_ptr + tl.load(d + 11) + j - prefix_len, mask=is_text, other=0)
        if (packed_first < 0) & (j < n):
            a_ptr = tl.cast(tl.load(d + 5), tl.pointer_type(dtype)) + j * hidden
        else:
            const_row = tl.where(
                j >= n,
                0,
                tl.where(
                    j < prefix_len,
                    tl.load(d + 6) + j,
                    tl.where(j < n - 2, tl.load(d + 8), tl.where(j == n - 2, tl.load(d + 9), tl.load(d + 10))),
                ),
            )
            a_ptr = const_ptr + const_row * hidden
        for block in range(0, hidden, BLOCK):
            offs = block + tl.arange(0, BLOCK)
            mask = offs < hidden
            value = tl.load(a_ptr + offs, mask=mask).to(tl.float32)
            if is_text:
                value += tl.load(table_ptr + text_id * hidden + offs, mask=mask).to(tl.float32)
            out = value.to(dtype)
            if to_packed:
                tl.store(packed_ptr + (packed_first + j) * hidden + offs, out, mask=mask)
            if in_span:
                tl.store(embeds_ptr + (start + j - offset) * embeds_stride + offs, out, mask=mask)
        if in_span:
            tl.store(input_ids_ptr + start + j - offset, pad_id)


class PrefillRows:
    """Host-side descriptors for :func:`launch_prefill_rows`: no per-row work."""

    def __init__(self) -> None:
        self.desc: list[int] = []
        self.ids: list[np.ndarray] = []
        self.num_ids = 0
        self.num_packed = 0
        self.max_rows = 0

    def add_new_prompt(
        self,
        rows: tuple[int, int, int, int, int],
        ids: np.ndarray,
        offset: int,
        span: int,
        start: int,
    ) -> tuple[int, int]:
        """A cached CustomVoice prompt; returns its ``(first packed row, rows)``."""
        prefix_start, prefix_len, pad_row, eos_row, tail_row = rows
        n = prefix_len + ids.size - 8 + 2
        first = self.num_packed
        # The assistant template wraps the text in 3 leading and 5 trailing tokens.
        self.desc += (
            n,
            offset,
            span,
            start,
            first,
            0,
            prefix_start,
            prefix_len,
            pad_row,
            eos_row,
            tail_row,
            self.num_ids + 3,
        )
        self.ids.append(ids)
        self.num_ids += ids.size
        self.num_packed += n
        self.max_rows = max(self.max_rows, n, offset + span)
        return first, n

    def add_stored_prompt(self, address: int, n: int, offset: int, span: int, start: int) -> None:
        """Rows ``[offset, offset + span)`` of a stored ``n``-row prompt."""
        self.desc += (n, offset, span, start, -1, address, 0, 0, 0, 0, 0, 0)
        self.max_rows = max(self.max_rows, offset + span)

    def array(self) -> np.ndarray:
        """Descriptors, then the text ids, as one int64 array."""
        return np.concatenate([np.asarray(self.desc, dtype=np.int64), *self.ids])


def launch_prefill_rows(
    rows: PrefillRows,
    staged: torch.Tensor,
    const: torch.Tensor,
    table: torch.Tensor,
    packed: torch.Tensor,
    embeds: torch.Tensor,
    input_ids: torch.Tensor,
    pad_id: int,
) -> None:
    """Run the requests of ``rows``; ``staged`` is ``rows.array()`` readable by the device."""
    num_reqs = len(rows.desc) // DESC_WIDTH
    hidden = embeds.shape[-1]
    _prefill_rows_kernel[(num_reqs, rows.max_rows)](
        staged,
        staged[num_reqs * DESC_WIDTH :],
        const,
        table,
        packed,
        embeds,
        embeds.stride(0),
        input_ids,
        pad_id,
        hidden,
        BLOCK=1024,
    )
