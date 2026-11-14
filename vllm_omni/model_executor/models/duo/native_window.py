# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Relocate retained rows of an unpadded single-session native FA3 paged cache."""

from dataclasses import replace

import torch


def scheduler_trim(session, prefix, count):
    end = prefix + count
    if not (0 < prefix < end <= session.num_prompt_tokens <= session.num_computed_tokens):
        raise ValueError("Native Realization window cut lies outside the committed prompt")
    retained = []
    for mm in session.mm_features:
        a, b = mm.mm_position.offset, mm.mm_position.offset + mm.mm_position.length
        if b <= prefix:
            retained.append(mm)
        elif a >= end:
            retained.append(replace(mm, mm_position=replace(mm.mm_position, offset=a - count)))
        elif not (prefix <= a and b <= end):
            raise ValueError("Native Realization chunk cut splits an encoded video")
    session.mm_features = retained
    del session.prompt_token_ids[prefix:end]
    del session._all_token_ids[prefix:end]
    session.num_prompt_tokens -= count
    session.num_computed_tokens -= count
    # APC is disabled. Drop stale hashes; upstream regenerates them on append.
    session.block_hashes.clear()


def move_rows(cache, table, prefix, count, old, page_size):
    # vLLM FlashAttention packed cache: [physical page, KV head, row, K|V].
    rows = torch.arange(prefix + count, old, device=table.device)
    dest = rows - count
    view = cache.transpose(1, 2)
    view[table[dest // page_size], dest % page_size] = view[table[rows // page_size], rows % page_size]


def apply_window(runner, scheduler_output):
    for new in scheduler_output.scheduled_new_reqs:
        meta = (getattr(new, "model_intermediate_buffer", None) or {}).get("meta", {})
        shift = meta.get("realization_rope_shift", 0)
        if not shift:
            continue
        prefix = meta["realization_prefix"]
        state = runner.requests[new.req_id]
        state.mrope_positions[:, prefix:] += shift
        state.mrope_position_delta += shift
        if drop := meta.get("realization_drop"):
            old = new.num_computed_tokens + drop
            table = torch.as_tensor(new.block_ids[0], dtype=torch.long, device=runner.device)
            for cache in runner.kv_caches:
                move_rows(cache, table, prefix, drop, old, cache.size(2))
