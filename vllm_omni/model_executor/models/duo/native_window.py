# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Relocate retained rows of an unpadded single-session native FA3 paged cache."""

import torch


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
