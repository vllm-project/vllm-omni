# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused movement of model-owned delay state when a request batch changes."""

import torch
from vllm.triton_utils import tl
from vllm.triton_utils import triton as tr


@tr.jit(do_not_specialize=["count"])
def _move(
    indices,
    source_codes,
    source_has,
    source_delay,
    source_eoc,
    source_done,
    dest_codes,
    dest_has,
    dest_delay,
    dest_eoc,
    dest_done,
    count,
    books: tl.constexpr,
    gather: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    if gather:
        encoded = tl.load(indices + row)
        fresh = encoded < 0
        src = tl.where(fresh, -encoded - 1, encoded)
        dst = row
    else:
        src = tl.load(indices + row)
        dst = tl.load(indices + count + row)
        fresh = False
    codes = tl.load(source_codes + src * books + col, col < books, 0)
    has = tl.load(source_has + src)
    delay = tl.load(source_delay + src)
    eoc = tl.load(source_eoc + src)
    done = tl.load(source_done + src)
    tl.store(dest_codes + dst * books + col, tl.where(fresh, 0, codes), col < books)
    tl.store(dest_has + dst, tl.where(fresh, 0, has))
    tl.store(dest_delay + dst, tl.where(fresh, 0, delay))
    tl.store(dest_eoc + dst, tl.where(fresh, -1, eoc))
    tl.store(dest_done + dst, tl.where(fresh, 0, done))


def sync_state(model, previous_rows, pool_rows, current_rows):
    # One owned pinned upload, ordered on the decode stream. Separate scatter
    # and gather kernels provide a global ordering point for moved pool rows.
    device = model._decode_has_codes.device
    packed = torch.tensor(previous_rows + pool_rows + current_rows, dtype=torch.int64, pin_memory=True).to(
        device, non_blocking=True
    )
    names = ("last_codes", "has_codes", "delay_count", "eoc_countdown", "generation_done")
    decode = [getattr(model, "_decode_" + name) for name in names]
    pool = [getattr(model, "_state_pool_" + name) for name in names]
    n = len(previous_rows)
    block = tr.next_power_of_2(model.num_codebooks)
    if n:
        _move[(n,)](packed, *decode, *pool, n, model.num_codebooks, False, block)
    if current_rows:
        _move[(len(current_rows),)](
            packed[2 * n :], *pool, *decode, len(current_rows), model.num_codebooks, True, block
        )
