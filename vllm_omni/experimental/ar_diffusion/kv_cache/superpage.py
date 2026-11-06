# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pack baseline-visible boundary rows of manager-owned superpages."""

import torch

from .paged import chunk_window_skipped_tokens


def pack_visible_rows(ctx, first_position: dict[int, int], device: torch.device) -> tuple[list[int], int]:
    from .paged_attention import _to_device_async

    cache = ctx.kv_cache
    page, unit = ctx.block_size, cache.visibility_block_size
    history = ctx.adapter.absolute_num_computed_tokens
    end = history + ctx.seq_len
    sink_end = -(-ctx.sink_tokens // unit) * unit
    skipped = chunk_window_skipped_tokens(
        history,
        chunk_size=ctx.chunk_size,
        sliding_window=cache.spec.sliding_window,
        sink_chunks=cache.spec.sink_chunks,
        reset_at_boundary=cache.spec.reset_at_boundary,
    )
    recent = max(sink_end, max(0, end - ctx.window_tokens) // unit * unit, (ctx.sink_tokens + skipped) // unit * unit)
    ranges = ((0, min(sink_end, end)), (recent, end))
    signature = (tuple(first_position.items()), ranges, ctx.adapter.compacted_tokens)
    previous = getattr(cache, "superpage_plan", None)
    if previous is not None and previous[0] == signature:
        _, pages, length, ctx.private_slots, ctx.superpage_fragment_tokens = previous
        return pages, length

    pages, fragments = [], []
    sink_page_end = -(-ctx.sink_tokens // page) * page
    length = 0
    for block, first in sorted(first_position.items(), key=lambda item: item[1]):
        absolute = first + (ctx.adapter.compacted_tokens if first >= sink_page_end else 0)
        spans = [(max(lo, absolute), min(hi, absolute + page)) for lo, hi in ranges]
        spans = [(lo, hi) for lo, hi in spans if hi > lo]
        count = sum(hi - lo for lo, hi in spans)
        length += count
        if count == page:
            pages.append(block)
        else:
            for lo, hi in spans:
                fragments.extend(range(block * page + lo - absolute, block * page + hi - absolute))
    capacity = cache.superpage_private_pages * page
    if len(fragments) > capacity:
        raise RuntimeError(f"superpage boundary bound violated: {len(fragments)} > {capacity}")
    private_base = cache.num_blocks_total - cache.superpage_private_pages
    pages += list(range(private_base, private_base + -(-len(fragments) // page)))
    ctx.private_slots = _to_device_async(
        torch.tensor(fragments + [cache.null_block_id * page] * (capacity - len(fragments)), dtype=torch.long), device
    )
    ctx.superpage_fragment_tokens = len(fragments)
    cache.superpage_plan = (signature, pages, length, ctx.private_slots, len(fragments))
    return pages, length
