# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Duplex Stage 0 sessions that outlive the model's sliding window (RFC #7389 W6).

Every test drives PersonaPlex-shaped sessions (one placeholder per 80 ms frame,
one greedy token per frame) through a real ``OmniARScheduler`` and KV cache
manager with a sliding-window spec like the one Helium's per-layer window gets.
"""

from __future__ import annotations

import pytest
from vllm.v1.engine import FinishReason
from vllm.v1.request import Request, RequestStatus

from tests.helpers.duplex_scheduler import (
    DUPLEX_BLOCK_SIZE,
    append_duplex_frame,
    duplex_append,
    engine_step,
    join_duplex_sessions,
    make_engine_duplex_scheduler,
)
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WINDOW = 3000
# The duplex profile's Stage 0 context: the window plus room to rebase.
MAX_MODEL_LEN = 4096
# The RoPE table (positions the model can rotate) of a profile serving calls
# well past the context: about two hours of 80 ms frames.
MAX_POSITIONS = 90_000
REBASE = {OmniARScheduler.STREAMING_POSITION_REBASE_KEY: True}
# Pages one session can hold: the window, plus the page a new token opens.
WINDOW_PAGES = -(-WINDOW // DUPLEX_BLOCK_SIZE) + 1


def _make_scheduler(tmp_path, *, max_model_len: int = MAX_MODEL_LEN) -> OmniARScheduler:
    return make_engine_duplex_scheduler(
        tmp_path,
        max_model_len=max_model_len,
        sliding_window=WINDOW,
        max_position_embeddings=MAX_POSITIONS,
    )


def _opt_in(append: Request) -> Request:
    """Carry the duplex runtime config a plugin that opts into the rebase sends."""
    append.model_intermediate_buffer = {"duplex": {"data_plane": True, "runtime_config": dict(REBASE)}}
    return append


def _join_rebased(sched: OmniARScheduler, joins: dict[str, int]) -> None:
    for request_id, num_tokens in joins.items():
        sched.add_request(_opt_in(duplex_append(request_id, num_tokens)))
    pending = set(joins)
    while pending:
        sampled_at, _ = engine_step(sched)
        pending -= set(sampled_at)


def _append_frame(sched: OmniARScheduler, *rebased: str):
    """Every live session appends one frame; the ``rebased`` ones opt in."""
    for request_id, request in list(sched.requests.items()):
        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            append = duplex_append(request_id, 1)
            sched.add_request(_opt_in(append) if request_id in rebased else append)
    return engine_step(sched)


def _blocks(sched: OmniARScheduler, request_id: str) -> list:
    (manager,) = sched.kv_cache_manager.coordinator.single_type_managers
    return list(manager.req_to_blocks[request_id])


def _live_pages(sched: OmniARScheduler, request_id: str) -> int:
    null_block = sched.kv_cache_manager.block_pool.null_block
    return sum(block is not null_block for block in _blocks(sched, request_id))


def _position_base(sched: OmniARScheduler, request_id: str) -> int:
    return int(getattr(sched.requests[request_id], "_omni_position_base", 0))


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


def test_rebased_session_keeps_prompt_and_kv_bounded_past_max_model_len(tmp_path) -> None:
    """With the plugin's opt-in, a session runs past max_model_len: its prompt
    is cut behind the window, the cut moves into its position base, and the
    absolute position (base + prompt) still advances by one per frame. A session
    on the same engine without the opt-in ends at max_model_len as before."""
    sched = _make_scheduler(tmp_path)
    _join_rebased(sched, {"long": WINDOW - 10})
    join_duplex_sessions(sched, {"plain": WINDOW - 10})

    absolute = WINDOW - 10
    plain_error = None
    free_pages = []
    for _ in range(2 * MAX_MODEL_LEN):
        sampled_at, outputs = _append_frame(sched, "long")
        if "plain" in outputs and outputs["plain"].finish_reason == FinishReason.ERROR:
            plain_error = outputs["plain"].stop_reason
        assert "long" in sampled_at, outputs.get("long")
        absolute += 1
        session = sched.requests["long"]
        base = _position_base(sched, "long")
        assert base + sampled_at["long"] == absolute
        assert base % OmniARScheduler.STREAMING_REBASE_ALIGN_TOKENS == 0
        assert session.num_prompt_tokens + 1 < MAX_MODEL_LEN
        assert session.model_intermediate_buffer["duplex"]["position_base"] == base
        assert _live_pages(sched, "long") <= WINDOW_PAGES
        assert len(_blocks(sched, "long")) <= -(-MAX_MODEL_LEN // DUPLEX_BLOCK_SIZE)
        if "plain" not in sched.requests:
            free_pages.append(sched.kv_cache_manager.block_pool.get_num_free_blocks())

    assert absolute > 2 * MAX_MODEL_LEN
    assert _position_base(sched, "long") > MAX_MODEL_LEN
    assert plain_error is not None and f"max_model_len {MAX_MODEL_LEN}" in plain_error
    assert max(free_pages) - min(free_pages) <= 1


def test_rebase_keeps_every_in_window_token_on_its_page(tmp_path, monkeypatch) -> None:
    """The cut is a whole number of pages taken off the front of the block
    table, so every token still inside the window resolves to the page it was
    written to, and the worker gets the trimmed prompt and table together."""
    sched = _make_scheduler(tmp_path)
    _join_rebased(sched, {"s": WINDOW - 10})
    while True:
        before = _blocks(sched, "s")
        prompt_before = sched.requests["s"].num_prompt_tokens
        base_before = _position_base(sched, "s")
        _append_frame(sched, "s")
        cut = _position_base(sched, "s") - base_before
        if cut:
            break

    after = _blocks(sched, "s")
    shift = cut // DUPLEX_BLOCK_SIZE
    # Pages that existed before the append keep their order past the cut.
    assert after[: len(before) - shift] == before[shift:]
    # Pages behind the cut went back to the pool (or were already the null page).
    assert all(block.ref_cnt == 0 or block.is_null for block in before[:shift])
    assert sched.requests["s"].num_prompt_tokens == prompt_before + 1 - cut
    # The cut leaves the whole window plus the new frame resident.
    assert sched.requests["s"].num_prompt_tokens >= WINDOW


def test_rebased_session_fails_cleanly_at_max_position_embeddings(tmp_path) -> None:
    """The RoPE table caps a rebased call: the last frame samples exactly at
    max_position_embeddings, and the next append finishes the session with an
    explicit context error instead of indexing past the table."""
    sched = make_engine_duplex_scheduler(
        tmp_path,
        max_model_len=1024,
        sliding_window=64,
        max_position_embeddings=1500,
    )
    _join_rebased(sched, {"s": 50})

    last_absolute = None
    for _ in range(1500):
        sampled_at, outputs = _append_frame(sched, "s")
        if "s" not in sched.requests:
            break
        last_absolute = _position_base(sched, "s") + sampled_at["s"]
    else:
        pytest.fail("the session never reached max_position_embeddings")

    # Well past max_model_len (1024): the RoPE table, not the context, ended it.
    assert last_absolute == 1499
    error = outputs["s"]
    assert error.finish_reason == FinishReason.ERROR
    assert error.stop_reason.startswith(OmniARScheduler.STREAMING_CONTEXT_OVERFLOW_STOP_REASON)
    assert "grow to 1500 tokens" in error.stop_reason
    assert "max_position_embeddings 1500" in error.stop_reason


def test_rebase_waits_while_tokens_are_in_flight(tmp_path) -> None:
    """A step still in flight may read pages the cut would renumber, so the cut
    waits for a later append; the slack keeps that append within the context."""
    sched = _make_scheduler(tmp_path)
    _join_rebased(sched, {"s": WINDOW + 600})
    session = sched.requests["s"]
    session.num_in_flight_tokens = 1
    assert sched._trim_streaming_prompt_behind_window(session) == 0
    session.num_in_flight_tokens = 0
    assert sched._trim_streaming_prompt_behind_window(session) > 0


def test_rebase_needs_room_past_the_window(tmp_path) -> None:
    """A context no larger than the window leaves nothing to rebase into, so the
    session ends at max_model_len as without the opt-in."""
    sched = _make_scheduler(tmp_path, max_model_len=WINDOW)
    _join_rebased(sched, {"s": WINDOW - 5})
    for _ in range(10):
        _, outputs = _append_frame(sched, "s")
        if "s" not in sched.requests:
            break
    assert outputs["s"].finish_reason == FinishReason.ERROR
    assert f"max_model_len {WINDOW}" in outputs["s"].stop_reason
