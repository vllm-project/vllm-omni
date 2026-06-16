# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-equivalence tests: DreamZeroStateAdapter vs bespoke DreamZeroState.

These drive both state objects directly with tiny CPU tensors -- no model, no
GPU -- and assert the adapter (backed by the SessionMemoryManager) stores and
returns exactly what the bespoke DreamZeroState does. This gates routing
DreamZero through the shared session memory manager (RFC #4480).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.memory import SessionMemoryManager
from vllm_omni.diffusion.models.dreamzero.state_dreamzero import (
    FRAMES_PER_CHUNK,
    DreamZeroState,
)
from vllm_omni.diffusion.models.dreamzero.state_dreamzero_adapter import DreamZeroStateAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BATCH, LAYERS, HEADS, HEAD_DIM = 1, 2, 2, 4
DTYPE, DEVICE = torch.float32, torch.device("cpu")


def _adapter() -> DreamZeroStateAdapter:
    return DreamZeroStateAdapter("session-0", SessionMemoryManager())


def _create_both() -> tuple[DreamZeroState, DreamZeroStateAdapter]:
    bespoke = DreamZeroState()
    adapter = _adapter()
    for state in (bespoke, adapter):
        state.create_kv_caches(BATCH, DTYPE, DEVICE, LAYERS, HEADS, HEAD_DIM)
    return bespoke, adapter


def _assert_kv_equal(bespoke: DreamZeroState, adapter: DreamZeroStateAdapter, is_negative: bool) -> None:
    expected = bespoke.get_kv_caches(is_negative)
    actual = adapter.get_kv_caches(is_negative)
    assert len(expected) == len(actual) == LAYERS
    for exp, act in zip(expected, actual):
        torch.testing.assert_close(exp, act)


def test_create_kv_caches_shapes_match() -> None:
    bespoke, adapter = _create_both()
    for is_negative in (False, True):
        for tensor in adapter.get_kv_caches(is_negative):
            assert tensor.shape == (2, BATCH, 0, HEADS, HEAD_DIM)
        _assert_kv_equal(bespoke, adapter, is_negative)


def test_update_kv_cache_equivalent_pos_and_neg() -> None:
    bespoke, adapter = _create_both()
    for seq in (3, 7):
        for is_negative in (False, True):
            for layer in range(LAYERS):
                kv = torch.randn(2, BATCH, seq, HEADS, HEAD_DIM)
                # Same tensor object into both paths.
                bespoke.update_kv_cache(layer, kv, is_negative=is_negative)
                adapter.update_kv_cache(layer, kv, is_negative=is_negative)
            _assert_kv_equal(bespoke, adapter, is_negative)


def test_update_kv_cache_clones_source() -> None:
    # Both paths must clone on store so a later mutation of the source tensor
    # does not corrupt the cache.
    bespoke, adapter = _create_both()
    kv = torch.randn(2, BATCH, 5, HEADS, HEAD_DIM)
    bespoke.update_kv_cache(0, kv, is_negative=False)
    adapter.update_kv_cache(0, kv, is_negative=False)
    kv.add_(1.0)  # mutate source in place
    torch.testing.assert_close(bespoke.get_kv_caches(False)[0], adapter.get_kv_caches(False)[0])
    # And neither should equal the mutated source.
    assert not torch.allclose(adapter.get_kv_caches(False)[0], kv)


def test_crossattn_caches_equivalent() -> None:
    bespoke, adapter = _create_both()
    for is_negative in (False, True):
        b_caches = bespoke.get_crossattn_caches(is_negative)
        a_caches = adapter.get_crossattn_caches(is_negative)
        assert len(b_caches) == len(a_caches) == LAYERS
        for layer in range(LAYERS):
            assert b_caches[layer]["is_init"] is False
            assert a_caches[layer]["is_init"] is False
            # Populate in place (as the model does on first forward).
            k, v = torch.randn(2, 3), torch.randn(2, 3)
            for caches in (b_caches, a_caches):
                caches[layer]["is_init"] = True
                caches[layer]["k"] = k
                caches[layer]["v"] = v
        # Re-fetch and confirm the mutation persisted identically.
        for layer in range(LAYERS):
            a_again = adapter.get_crossattn_caches(is_negative)[layer]
            b_again = bespoke.get_crossattn_caches(is_negative)[layer]
            assert a_again["is_init"] == b_again["is_init"] is True
            torch.testing.assert_close(a_again["k"], b_again["k"])
            torch.testing.assert_close(a_again["v"], b_again["v"])


def test_accumulate_frames_and_call_count_match() -> None:
    bespoke, adapter = _create_both()
    rng = np.random.default_rng(0)
    for _ in range(2 * FRAMES_PER_CHUNK + 1):
        frame = rng.integers(0, 255, size=(5, 6, 3), dtype=np.uint8)
        b_out = bespoke.accumulate_frames(frame)
        a_out = adapter.accumulate_frames(frame)
        np.testing.assert_array_equal(b_out, a_out)
        assert bespoke.call_count == adapter.call_count
        assert len(bespoke.stitched_buffer) == len(adapter.stitched_buffer)
    # Deque rollover capped at FRAMES_PER_CHUNK.
    assert len(adapter.stitched_buffer) == FRAMES_PER_CHUNK


def test_accumulate_frames_4d_extend_matches() -> None:
    bespoke, adapter = _create_both()
    clip = np.random.default_rng(1).integers(0, 255, size=(3, 5, 6, 3), dtype=np.uint8)
    np.testing.assert_array_equal(bespoke.accumulate_frames(clip), adapter.accumulate_frames(clip))
    assert len(bespoke.stitched_buffer) == len(adapter.stitched_buffer)


def test_reset_clears_state_and_should_reset_matches() -> None:
    bespoke, adapter = _create_both()
    tokens = torch.tensor([1, 2, 3])
    # language unset -> both want reset.
    assert bespoke.should_reset(tokens, 4, -1) is adapter.should_reset(tokens, 4, -1) is True

    for state in (bespoke, adapter):
        state.language = tokens
        state.current_start_frame = 2
    # same language -> no reset; both agree.
    assert bespoke.should_reset(tokens, 4, -1) == adapter.should_reset(tokens, 4, -1) is False
    # language change -> reset.
    other = torch.tensor([9, 9])
    assert bespoke.should_reset(other, 4, -1) == adapter.should_reset(other, 4, -1) is True
    # local_attn_size exceeded -> reset.
    assert bespoke.should_reset(tokens, 4, 1) == adapter.should_reset(tokens, 4, 1) is True

    adapter.reset()
    assert adapter.language is None
    assert adapter.current_start_frame == 0
    assert adapter.call_count == 0
    with pytest.raises(RuntimeError):
        adapter.get_kv_caches(False)


def test_metadata_fields_persist_across_adapter_instances() -> None:
    # A fresh adapter for the same session must see prior metadata (the manager
    # is the single source of truth).
    manager = SessionMemoryManager()
    first = DreamZeroStateAdapter("s", manager)
    first.create_kv_caches(BATCH, DTYPE, DEVICE, LAYERS, HEADS, HEAD_DIM)
    first.current_start_frame = 5
    first.clip_feas = torch.ones(2, 2)

    second = DreamZeroStateAdapter("s", manager)
    assert second.current_start_frame == 5
    torch.testing.assert_close(second.clip_feas, torch.ones(2, 2))
    # KV created via `first` is visible via `second`.
    assert len(second.get_kv_caches(False)) == LAYERS


def test_session_lru_caps_retained_sessions() -> None:
    manager = SessionMemoryManager(max_sessions=3)
    for i in range(5):
        DreamZeroStateAdapter(f"s{i}", manager)
    assert len(manager) == 3
    # Oldest two evicted; newest three retained.
    assert "s0" not in manager
    assert "s1" not in manager
    for i in (2, 3, 4):
        assert f"s{i}" in manager


def test_cfg_branches_isolated() -> None:
    _, adapter = _create_both()
    pos = torch.randn(2, BATCH, 4, HEADS, HEAD_DIM)
    neg = torch.randn(2, BATCH, 4, HEADS, HEAD_DIM)
    for layer in range(LAYERS):
        adapter.update_kv_cache(layer, pos, is_negative=False)
        adapter.update_kv_cache(layer, neg, is_negative=True)
    for layer in range(LAYERS):
        torch.testing.assert_close(adapter.get_kv_caches(False)[layer], pos)
        torch.testing.assert_close(adapter.get_kv_caches(True)[layer], neg)
        assert not torch.allclose(pos, neg)
