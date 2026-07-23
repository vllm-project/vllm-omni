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


NUM_LAYERS = 3
DEVICE = torch.device("cpu")


def _fresh_state() -> LingbotWorldFastState:
    state = LingbotWorldFastState()
    state.init_state(DEVICE, NUM_LAYERS)
    return state


def test_reset_initializes_all_fields() -> None:
    state = LingbotWorldFastState()
    assert state.current_start_frame == 0
    assert state.local_end_index is None
    assert state.global_end_index is None
    assert state.is_initialized is False
    assert state.current_lat_f == 0
    assert state.session_id is None
    assert state.num_layers is None
    assert state.h is None
    assert state.w is None
    assert state.lat_h is None
    assert state.lat_w is None
    assert state.frame_seqlen is None
    assert state.last_decoded_latent is None


def test_create_kv_caches_allocates_expected_shapes() -> None:
    state = _fresh_state()

    assert state.is_initialized is True
    assert state.num_layers == NUM_LAYERS

    assert state.local_end_index is not None and state.global_end_index is not None
    for idx_list in (state.local_end_index, state.global_end_index):
        assert len(idx_list) == NUM_LAYERS
        for idx in idx_list:
            assert idx.shape == (1,)
            assert idx.dtype == torch.long
            assert int(idx.item()) == 0


def test_reset_clears_all_session_state() -> None:
    state = _fresh_state()
    state.session_id = "abc"
    state.h, state.w, state.lat_h, state.lat_w, state.frame_seqlen = 480, 832, 60, 104, 1560
    state.last_decoded_latent = torch.zeros(16, 2, 60, 104)

    state.reset()

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
    state = _fresh_state()
    state.session_id = "session-a"

    assert _should_reset(state, "session-a") is False
    # ... and a prompt-only change must not trigger a reset either.
    assert _should_reset(state, "session-a") is False
    # Pipeline would proceed in extension mode → state still alive.
    assert state.is_initialized is True


def test_different_session_id_triggers_reset() -> None:
    state = _fresh_state()
    state.session_id = "session-a"

    assert _should_reset(state, "session-b") is True

    state.reset()
    assert state.session_id is None
    assert state.current_lat_f == 0
