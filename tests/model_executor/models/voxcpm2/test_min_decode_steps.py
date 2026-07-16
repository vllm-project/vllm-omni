# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the VoxCPM2 per-request minimum decode-step stop guard.

The guard mirrors the native VoxCPM ``min_len`` check: below a per-request
minimum number of decode steps, the learned stop head must not end the
request — on any of the three stop-decision paths (cached-logits check,
batched audio-collect precompute, and ``compute_logits``).
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_STOP_LOGITS = [[-1.0, 4.0]]  # decisively "stop" if the guard is not active


@functools.lru_cache(maxsize=1)
def _voxcpm2_talker_mod():
    """Defer talker import (pulls vLLM model_executor) until first use."""
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        VoxCPM2TalkerForConditionalGeneration,
        _RequestState,
        _VoxCPM2RuntimeConfig,
    )

    return VoxCPM2TalkerForConditionalGeneration, _RequestState, _VoxCPM2RuntimeConfig


def _make_bare_talker():
    VoxCPM2TalkerForConditionalGeneration, _, _ = _voxcpm2_talker_mod()
    talker = VoxCPM2TalkerForConditionalGeneration.__new__(VoxCPM2TalkerForConditionalGeneration)
    talker.config = SimpleNamespace(vocab_size=4)
    talker._active_states = {}
    talker._results_queue = []
    talker._audio_emit_every = 1
    talker._vae_decode_every = 1
    talker._enable_delayed_audio_copy = False
    return talker


def test_runtime_config_guard_disabled_by_default():
    _, _, RuntimeConfig = _voxcpm2_talker_mod()
    cfg = RuntimeConfig()
    assert cfg.min_decode_steps_per_text_token == 0.0
    assert cfg.min_decode_steps_floor == 0


def test_cached_logits_stop_gated_below_min_decode_steps():
    Talker, RState, _ = _voxcpm2_talker_mod()
    state = RState(
        request_id="req",
        precomputed_stop_logits=torch.tensor(_STOP_LOGITS),
        min_decode_steps=10,
    )
    state.decode_step_count = 5

    assert Talker._should_stop_from_cached_logits(state) is False
    assert state.precomputed_is_stopping is False
    assert state.is_stopping is False


def test_cached_logits_stop_honored_at_min_decode_steps():
    Talker, RState, _ = _voxcpm2_talker_mod()
    state = RState(
        request_id="req",
        precomputed_stop_logits=torch.tensor(_STOP_LOGITS),
        min_decode_steps=10,
    )
    state.decode_step_count = 10

    assert Talker._should_stop_from_cached_logits(state) is True
    assert state.is_stopping is True


def test_audio_collect_precompute_gated_below_min_decode_steps():
    _, RState, _ = _voxcpm2_talker_mod()
    talker = _make_bare_talker()
    talker._audio_emit_every = 2  # keep the sparse-audio precompute path active
    state = RState(
        request_id="req",
        precomputed_stop_logits=torch.tensor(_STOP_LOGITS),
        min_decode_steps=10,
    )
    state.decode_step_count = 5

    talker._precompute_stop_flags_for_audio_collect([state])

    assert state.precomputed_is_stopping is False
    assert state.is_stopping is False


def test_compute_logits_forces_continue_below_min_decode_steps():
    _, RState, _ = _voxcpm2_talker_mod()
    talker = _make_bare_talker()
    state = RState(
        request_id="req",
        precomputed_stop_logits=torch.tensor(_STOP_LOGITS),
        min_decode_steps=10,
    )
    state.decode_step_count = 5
    talker._active_states["req"] = state
    talker._results_queue = [("req", state.precomputed_stop_logits)]

    logits = talker.compute_logits(torch.zeros(1, 1))

    assert logits[0, 0] == 1.0
    assert logits[0, 1] == float("-inf")
    assert state.is_stopping is False
    assert state.precomputed_stop_logits is None
    assert state.precomputed_is_stopping is None


def test_compute_logits_passes_stop_logits_at_min_decode_steps():
    _, RState, _ = _voxcpm2_talker_mod()
    talker = _make_bare_talker()
    state = RState(
        request_id="req",
        precomputed_stop_logits=torch.tensor(_STOP_LOGITS),
        min_decode_steps=10,
    )
    state.decode_step_count = 10
    talker._active_states["req"] = state
    talker._results_queue = [("req", state.precomputed_stop_logits)]

    logits = talker.compute_logits(torch.zeros(1, 1))

    assert logits[0, 1] > logits[0, 0]
