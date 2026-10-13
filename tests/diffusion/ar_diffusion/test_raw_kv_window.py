# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Raw positional-transform-on-read storage against the real vLLM pool."""

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.raw import RawKVWindow
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

FRAMES = 5
HISTORY = 5
HEADS = 2
WIDTH = 16


def make_state(*, area=6, dtype=torch.bfloat16, layers=3):
    # Thirty rows per five-frame chunk straddle sixteen-slot physical blocks.
    # Exercise the same compacting manager and per-session pool as the runner.
    config = ARDiffusionKVConfig(enable=True, chunk_size=area, window_chunks=FRAMES * HISTORY)
    pool = ARDiffusionKVCache(
        config,
        num_layers=layers,
        num_kv_heads=HEADS,
        head_size=WIDTH,
        dtype=dtype,
        block_size=16,
        max_model_len=4096,
        available_bytes=1 << 24,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=2,
        frames_per_block=FRAMES,
        max_scratch_tokens_per_branch=FRAMES * area,
        device=torch.device("cpu"),
    )
    state = ARDiffusionKVState(pool, "first", {"main": pool.begin_request("first-main")}, num_layers=layers)
    return pool, state


def kv_for(pool, layer, chunk):
    rows = FRAMES * pool.spec.chunk_size
    base = torch.arange(rows * HEADS * WIDTH, dtype=torch.float32).reshape(1, rows, HEADS, WIDTH)
    # Distinguishable chunks/layers, with distinguishable K and V.
    key = (base / base.numel() + chunk * 4 + layer).to(pool.key_cache(layer))
    value = (base / base.numel() - chunk * 4 - layer - 1).to(pool.value_cache(layer))
    return key, value


def window(state):
    return RawKVWindow(state, "main", frames_per_chunk=FRAMES, history_chunks=HISTORY)


@pytest.mark.parametrize("area", [6, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_raw_window_preserves_exact_five_chunk_history_through_eviction(area, dtype):
    pool, state = make_state(area=area, dtype=dtype)
    expected: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
    try:
        for chunk in range(13):
            raw = window(state)  # The pipeline creates a fresh view per forward.
            for layer in range(state.num_layers):
                history = raw.history(layer, chunk)
                assert [item[0] for item in history] == list(range(max(0, chunk - HISTORY), chunk))
                for previous, key, value in history:
                    assert key.dtype == value.dtype == dtype
                    assert key.device == value.device == pool.key_cache(layer).device
                    torch.testing.assert_close(key, expected[layer, previous][0], atol=0, rtol=0)
                    torch.testing.assert_close(value, expected[layer, previous][1], atol=0, rtol=0)
                current = kv_for(pool, layer, chunk)
                expected[layer, chunk] = tuple(t.clone() for t in current)
                raw.stage(layer, chunk, *current)
                # Uncommitted current rows never enter the visible history.
                assert len(raw.history(layer, chunk)) == min(chunk, HISTORY)
            raw.commit_chunk(chunk)
            adapter = state.adapter("main")
            assert adapter.completed_chunks == (chunk + 1) * FRAMES
            assert state._committed["main"] == (chunk + 1) * FRAMES * area
            assert len(window(state).history(0, chunk + 1)) == min(chunk + 1, HISTORY)
            # Metadata compaction bounds the actual backing block table.
            max_blocks = (HISTORY * FRAMES * area + 15) // 16 + 2
            assert len(pool.block_table(adapter)) <= max_blocks
    finally:
        state.close()


def test_raw_history_is_a_copy_and_two_sessions_do_not_alias():
    pool, first = make_state()
    second = ARDiffusionKVState(pool, "second", {"main": pool.begin_request("second-main")}, num_layers=3)
    try:
        for state, chunk_value in ((first, 1), (second, 7)):
            raw = window(state)
            for layer in range(state.num_layers):
                raw.stage(layer, 0, *kv_for(pool, layer, chunk_value))
            raw.commit_chunk(0)
        returned = window(first).history(0, 1)
        returned[0][1].fill_(99)
        returned[0][2].fill_(99)
        for state, expected in ((first, 1), (second, 7)):
            _, key, value = window(state).history(0, 1)[0]
            expected_k, expected_v = kv_for(pool, 0, expected)
            torch.testing.assert_close(key, expected_k, atol=0, rtol=0)
            torch.testing.assert_close(value, expected_v, atol=0, rtol=0)
    finally:
        first.close()
        second.close()


def test_duplicate_partial_and_out_of_order_commits_never_advance_timeline():
    pool, state = make_state()
    raw = window(state)
    try:
        with pytest.raises(ValueError, match="committed"):
            raw.history(0, 1)
        raw.stage(0, 0, *kv_for(pool, 0, 0))
        with pytest.raises(ValueError, match="already staged"):
            raw.stage(0, 0, *kv_for(pool, 0, 0))
        with pytest.raises(RuntimeError, match="every layer"):
            raw.commit_chunk(0)
        assert state.adapter("main").completed_chunks == 0
        assert raw.history(0, 0) == []
        for layer in (1, 2):
            raw.stage(layer, 0, *kv_for(pool, layer, 0))
        raw.commit_chunk(0)
        with pytest.raises(ValueError, match="committed"):
            raw.commit_chunk(0)
        assert state.adapter("main").completed_chunks == FRAMES
    finally:
        state.close()


def test_abort_does_not_publish_partial_layers_and_session_close_reclaims_pool():
    pool, state = make_state()
    free_before = pool.manager.block_pool.get_num_free_blocks()
    raw = window(state)
    raw.stage(0, 0, *kv_for(pool, 0, 0))
    raw.abort()
    assert state.adapter("main").completed_chunks == 0
    assert raw.history(0, 0) == []
    # A partially written committing context cannot be replaced/retried. The
    # runner closes a failed session before reusing its managed pool storage.
    with pytest.raises(RuntimeError, match="before.*committed"):
        window(state).stage(0, 0, *kv_for(pool, 0, 0))
    state.close()
    assert pool.manager.block_pool.get_num_free_blocks() == free_before
    with pytest.raises(RuntimeError, match="closed"):
        raw.history(0, 0)


@pytest.mark.parametrize("invalid", ["key_shape", "value_shape", "layer", "dtype", "device"])
def test_bad_raw_write_fails_before_allocating_or_modifying_pool(invalid):
    pool, state = make_state()
    raw = window(state)
    key, value = kv_for(pool, 0, 0)
    layer = 0
    if invalid == "key_shape":
        key = key[:, :-1]
    elif invalid == "value_shape":
        value = value[:, :-1]
    elif invalid == "layer":
        layer = 3
    elif invalid == "dtype":
        key = key.float()
    else:
        value = torch.empty_like(value, device="meta")
    free_before = pool.manager.block_pool.get_num_free_blocks()
    try:
        with pytest.raises(ValueError):
            raw.stage(layer, 0, key, value)
        assert state.adapter("main").completed_chunks == 0
        assert state._paged_pending["main"] is None
        assert pool.manager.block_pool.get_num_free_blocks() == free_before
    finally:
        state.close()
