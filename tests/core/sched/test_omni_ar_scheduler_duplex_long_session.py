# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Duplex Stage 0 sessions that outlive the model's sliding window (RFC #7389 W6).

Every test drives PersonaPlex-shaped sessions (one placeholder per 80 ms frame,
one greedy token per frame) through a real ``OmniARScheduler`` and KV cache
manager with a sliding-window spec like the one Helium's per-layer window gets.
"""

from __future__ import annotations

import pytest

from tests.helpers.omni_scheduler import (
    DUPLEX_BLOCK_SIZE,
    append_duplex_frame,
    join_duplex_sessions,
    make_engine_duplex_scheduler,
)
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WINDOW = 3000
# A Stage 0 context larger than the window, so a session can run past it.
MAX_MODEL_LEN = 4096
# Pages one session can hold: the window, plus the page a new token opens.
WINDOW_PAGES = -(-WINDOW // DUPLEX_BLOCK_SIZE) + 1


def _blocks(sched: OmniARScheduler, request_id: str) -> list:
    (manager,) = sched.kv_cache_manager.coordinator.single_type_managers
    return list(manager.req_to_blocks[request_id])


def _live_pages(sched: OmniARScheduler, request_id: str) -> int:
    null_block = sched.kv_cache_manager.block_pool.null_block
    return sum(block is not null_block for block in _blocks(sched, request_id))


def test_sliding_window_kv_is_freed_behind_a_session_past_its_window(tmp_path) -> None:
    """The check W6 depends on, on the pinned vLLM: a duplex session whose KV
    spec is a ``SlidingWindowSpec`` has the pages behind its window freed as it
    appends, so its KV stays flat past the window. (That Helium's per-layer
    window gets this spec is checked on GPU by the Helium parity test.) The
    placeholder prompt still grows by one token per frame."""
    sched = make_engine_duplex_scheduler(tmp_path, max_model_len=MAX_MODEL_LEN, sliding_window=WINDOW)
    join_duplex_sessions(sched, {"s": WINDOW - 100})

    free_pages = []
    for _ in range(1000):
        sampled_at, _ = append_duplex_frame(sched)
        assert "s" in sampled_at
        assert _live_pages(sched, "s") <= WINDOW_PAGES
        if sampled_at["s"] > WINDOW + DUPLEX_BLOCK_SIZE:
            free_pages.append(sched.kv_cache_manager.block_pool.get_num_free_blocks())

    assert sched.requests["s"].num_prompt_tokens == WINDOW - 100 + 1000
    # Each new page past the window replaces one freed behind it.
    assert max(free_pages) - min(free_pages) <= 1
