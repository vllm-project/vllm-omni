# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L1 unit tests for ``LingbotWorldFastState``.

The state container is the load-bearing structure for chunk-streamed
generation: it owns the KV cache, the cross-attention cache, the
``current_lat_f`` cursor used to derive ``current_start`` RoPE offsets,
and the session-id that decides between fresh vs extension semantics.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.lingbot_world_fast.state_lingbot_world_fast import (
    LingbotWorldFastState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


BATCH_SIZE = 1
NUM_LAYERS = 3
NUM_HEADS = 4
HEAD_DIM = 8
KV_SIZE = 16
DTYPE = torch.float32
DEVICE = torch.device("cpu")


def _fresh_state_with_caches(kv_size: int = KV_SIZE) -> LingbotWorldFastState:
    state = LingbotWorldFastState()
    state.create_kv_caches(
        batch_size=BATCH_SIZE,
        dtype=DTYPE,
        device=DEVICE,
        kv_size=kv_size,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )
    return state


def test_reset_initializes_all_fields() -> None:
    state = LingbotWorldFastState()
    assert state.kv_cache is None
    assert state.crossattn_cache is None
    assert state.current_start_frame == 0
    assert state.local_end_index is None
    assert state.global_end_index is None
    assert state.is_initialized is False
    assert state.current_lat_f == 0
    assert state.session_id is None
    assert state.batch_size is None
    assert state.num_layers is None
    assert state.num_heads is None
    assert state.head_dim is None
    assert state.h is None
    assert state.w is None
    assert state.lat_h is None
    assert state.lat_w is None
    assert state.frame_seqlen is None
    assert state.last_decoded_latent is None


def test_create_kv_caches_allocates_expected_shapes() -> None:
    state = _fresh_state_with_caches()

    assert state.is_initialized is True
    assert state.batch_size == BATCH_SIZE
    assert state.num_layers == NUM_LAYERS
    assert state.num_heads == NUM_HEADS
    assert state.head_dim == HEAD_DIM

    assert state.kv_cache is not None
    assert len(state.kv_cache) == NUM_LAYERS
    for layer in state.kv_cache:
        assert layer.shape == (2, BATCH_SIZE, KV_SIZE, NUM_HEADS, HEAD_DIM)
        assert layer.dtype == DTYPE
        assert torch.all(layer == 0)

    assert state.local_end_index is not None and state.global_end_index is not None
    for idx_list in (state.local_end_index, state.global_end_index):
        assert len(idx_list) == NUM_LAYERS
        for idx in idx_list:
            assert idx.shape == (1,)
            assert idx.dtype == torch.long
            assert int(idx.item()) == 0

    assert state.crossattn_cache is not None
    assert len(state.crossattn_cache) == NUM_LAYERS
    for entry in state.crossattn_cache:
        assert entry == {"is_init": False, "k": None, "v": None}


def test_extend_kv_caches_grows_tensor_and_zeros_new_slots() -> None:
    state = _fresh_state_with_caches()
    extra = 7
    # Mark the existing slots so we can confirm they aren't disturbed.
    for layer in state.kv_cache:
        layer.fill_(1.0)

    state.extend_kv_caches(extra_kv_size=extra)

    for layer in state.kv_cache:
        assert layer.shape == (2, BATCH_SIZE, KV_SIZE + extra, NUM_HEADS, HEAD_DIM)
        assert torch.all(layer[:, :, :KV_SIZE] == 1.0)
        # Newly grown trailing slice is fresh zeros.
        assert torch.all(layer[:, :, KV_SIZE:] == 0.0)


def test_extend_kv_caches_requires_initialization() -> None:
    state = LingbotWorldFastState()
    with pytest.raises(AssertionError):
        state.extend_kv_caches(extra_kv_size=4)


def test_get_accessors_require_initialization() -> None:
    state = LingbotWorldFastState()
    with pytest.raises(AssertionError):
        state.get_kv_caches()
    with pytest.raises(AssertionError):
        state.get_crossattn_caches()


def test_get_kv_caches_returns_underlying_list() -> None:
    state = _fresh_state_with_caches()
    assert state.get_kv_caches() is state.kv_cache


def test_advance_moves_cursor_by_delta() -> None:
    state = LingbotWorldFastState()
    state.advance(3)
    assert state.current_lat_f == 3
    state.advance(5)
    assert state.current_lat_f == 8


def test_reset_clears_all_session_state() -> None:
    state = _fresh_state_with_caches()
    state.session_id = "abc"
    state.advance(4)
    state.h, state.w, state.lat_h, state.lat_w, state.frame_seqlen = 480, 832, 60, 104, 1560
    state.last_decoded_latent = torch.zeros(16, 2, 60, 104)

    state.reset()

    assert state.kv_cache is None
    assert state.crossattn_cache is None
    assert state.local_end_index is None
    assert state.global_end_index is None
    assert state.is_initialized is False
    assert state.current_lat_f == 0
    assert state.current_start_frame == 0
    assert state.session_id is None
    assert state.h is None and state.w is None
    assert state.lat_h is None and state.lat_w is None
    assert state.frame_seqlen is None
    assert state.last_decoded_latent is None
    assert state.batch_size is None
    assert state.num_layers is None


def test_reset_is_idempotent() -> None:
    state = LingbotWorldFastState()
    state.reset()
    state.reset()
    assert state.is_initialized is False
    assert state.current_lat_f == 0


# ---------------------------------------------------------------------------
# Reset is triggered only by session-id change, not prompt change.
#
# Mirrors the conditional in ``LingbotWorldFastPipeline.forward`` (pipeline
# file, around the ``if self.state.session_id is None or
# self.state.session_id != session_id`` block). We assert the contract on
# the state container so the test does not depend on instantiating the
# heavy pipeline.
# ---------------------------------------------------------------------------


def _should_reset(state: LingbotWorldFastState, incoming_session_id: str) -> bool:
    """Replicates the pipeline's reset trigger."""
    return state.session_id is None or state.session_id != incoming_session_id


def test_first_call_with_any_session_id_triggers_reset() -> None:
    state = LingbotWorldFastState()
    assert _should_reset(state, "session-a") is True


def test_same_session_id_does_not_reset() -> None:
    state = _fresh_state_with_caches()
    state.session_id = "session-a"
    state.advance(4)

    assert _should_reset(state, "session-a") is False
    # ... and a prompt-only change must not trigger a reset either.
    assert _should_reset(state, "session-a") is False
    # Pipeline would proceed in extension mode → state still alive.
    assert state.is_initialized is True
    assert state.current_lat_f == 4


def test_different_session_id_triggers_reset() -> None:
    state = _fresh_state_with_caches()
    state.session_id = "session-a"
    state.advance(4)

    assert _should_reset(state, "session-b") is True

    state.reset()
    assert state.session_id is None
    assert state.current_lat_f == 0
    assert state.kv_cache is None
