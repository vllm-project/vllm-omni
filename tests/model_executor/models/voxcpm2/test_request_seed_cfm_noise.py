# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-level seeds must control the VoxCPM2 CFM noise stream (#7658).

``/v1/audio/speech`` folds ``request.seed`` into ``SamplingParams.seed``, which
reaches vLLM's sampler but never the CFM (flow-matching) noise draws: every
noise site used the global RNG, so identical text + seed returned different
audio. A seeded request now carries a dedicated ``torch.Generator`` in its
``_RequestState``; these tests pin the noise semantics without instantiating
the full model.
"""

from __future__ import annotations

import functools

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

torch = pytest.importorskip("torch")


@functools.lru_cache(maxsize=1)
def _talker_cls():
    """Defer talker import (pulls vLLM model_executor) until first use."""
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        VoxCPM2TalkerForConditionalGeneration,
    )

    return VoxCPM2TalkerForConditionalGeneration


def _make_talker(deterministic_cfm_noise: bool = False):
    cls = _talker_cls()
    talker = cls.__new__(cls)
    talker._deterministic_cfm_noise = deterministic_cfm_noise
    return talker


def _make_state(request_id: str = "req-1", seed: int | None = 42):
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import _RequestState

    state = _RequestState(request_id=request_id)
    if seed is not None:
        state.cfm_generator = torch.Generator(device="cpu")
        state.cfm_generator.manual_seed(seed)
    return state


def test_seeded_noise_is_deterministic_across_requests():
    talker = _make_talker()

    a1 = torch.empty(1, 128, 5)
    a2 = torch.empty(1, 128, 5)
    talker._fill_deterministic_cfm_noise_for_state(_make_state(seed=7), a1)
    talker._fill_deterministic_cfm_noise_for_state(_make_state(seed=7), a2)

    assert torch.equal(a1, a2)


def test_seed_value_changes_the_noise_stream():
    talker = _make_talker()

    a = torch.empty(1, 128, 5)
    b = torch.empty(1, 128, 5)
    talker._fill_deterministic_cfm_noise_for_state(_make_state(seed=7), a)
    talker._fill_deterministic_cfm_noise_for_state(_make_state(seed=8), b)

    assert not torch.equal(a, b)


def test_seeded_noise_advances_with_each_draw():
    talker = _make_talker()
    state = _make_state(seed=7)

    first = torch.empty(1, 128, 5)
    second = torch.empty(1, 128, 5)
    talker._fill_deterministic_cfm_noise_for_state(state, first)
    talker._fill_deterministic_cfm_noise_for_state(state, second)

    assert not torch.equal(first, second)


def test_seeded_noise_is_batch_position_independent():
    """A seeded row draws the same bytes whether it sits alone or mid-batch."""
    talker = _make_talker()

    solo = torch.empty(1, 128, 5)
    talker._fill_deterministic_cfm_noise_for_state(_make_state(seed=7), solo)

    batch = torch.empty(3, 128, 5)
    talker._fill_deterministic_cfm_noise_for_state(_make_state(seed=7), batch[1:2])

    assert torch.equal(solo, batch[1:2])


def test_has_deterministic_noise_prefers_request_seed_over_flag():
    unflagged = _make_talker(deterministic_cfm_noise=False)
    assert unflagged._has_deterministic_cfm_noise(_make_state(seed=1))
    assert not unflagged._has_deterministic_cfm_noise(_make_state(seed=None))

    flagged = _make_talker(deterministic_cfm_noise=True)
    assert flagged._has_deterministic_cfm_noise(_make_state(seed=None))


def test_request_seed_wins_over_replay_hash():
    """The per-request generator keys noise off the seed, not the request id."""
    talker = _make_talker(deterministic_cfm_noise=True)

    r1 = torch.empty(1, 128, 5)
    r2 = torch.empty(1, 128, 5)
    talker._fill_deterministic_cfm_noise_for_state(_make_state(request_id="111", seed=7), r1)
    talker._fill_deterministic_cfm_noise_for_state(_make_state(request_id="222", seed=7), r2)

    assert torch.equal(r1, r2)
