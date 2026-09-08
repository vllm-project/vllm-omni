# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the dots.tts talker's stop-signal pairing and
per-request state lifecycle.

Pure-CPU tests on a bare talker (``__new__`` + the handful of attributes
the methods under test read) — no weights, no GPU, no engine.  Mirrors
tests/model_executor/models/voxcpm2/test_talker_state_eviction.py.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


torch = pytest.importorskip("torch")

_VOCAB_SIZE = 151672


@functools.lru_cache(maxsize=1)
def _dots_tts_talker_mod():
    """Defer talker import (pulls vLLM model_executor) until first use."""
    from vllm_omni.model_executor.models.dots_tts.dots_tts_talker import (
        DotsTTSForConditionalGeneration,
        _RequestState,
    )

    return DotsTTSForConditionalGeneration, _RequestState


def _make_bare_talker():
    DotsTTSForConditionalGeneration, _ = _dots_tts_talker_mod()
    talker = DotsTTSForConditionalGeneration.__new__(DotsTTSForConditionalGeneration)
    talker.config = SimpleNamespace(vocab_size=_VOCAB_SIZE)
    talker._active_states = {}
    talker._pending_requests = []
    talker._results_queue = []
    talker._audio_queue = []
    talker._deferred_cleanup_ids = set()
    talker._beta_trace = False
    return talker


def _add_state(talker, req_id: str, *, is_stopping: bool = False):
    _, _RequestState = _dots_tts_talker_mod()
    state = _RequestState(request_id=req_id)
    state.prefill_completed = True
    state.is_stopping = is_stopping
    talker._active_states[req_id] = state
    return state


def _hidden(bsz: int) -> torch.Tensor:
    return torch.zeros(bsz, 1536, dtype=torch.float32)


class TestComputeLogitsPairing:
    """compute_logits pairs _results_queue entries with batch rows
    positionally, so every request must push exactly one entry per step —
    including prefills, which push a (req_id, None) placeholder."""

    def test_mixed_batch_prefill_placeholder_keeps_rows_aligned(self) -> None:
        talker = _make_bare_talker()
        _add_state(talker, "prefill-req")
        decode_state = _add_state(talker, "decode-req", is_stopping=True)
        decode_state.precomputed_stop_logits = torch.tensor([[0.1, 0.9]])

        # Row 0 is the prefill (None placeholder), row 1 the stopping decode.
        talker._results_queue = [
            ("prefill-req", None),
            ("decode-req", torch.tensor([[0.1, 0.9]])),
        ]
        logits = talker.compute_logits(_hidden(2))

        assert logits.shape == (2, _VOCAB_SIZE)
        # Prefill row: forced continue.
        assert logits[0, 0].item() == 1.0
        assert logits[0, 1].item() == float("-inf")
        # Stopping decode row: stop slot wins.
        assert logits[1, 1].item() == 1.0
        assert logits[1, 0].item() == 0.0
        # Everything else stays masked.
        assert logits[0, 2].item() == float("-inf")
        assert logits[1, 2].item() == float("-inf")
        # Queue drained.
        assert talker._results_queue == []

    def test_below_threshold_stop_logits_force_continue(self) -> None:
        """Raw (continue, stop) softmax must not reach the sampler: a
        prob_stop in (0.5, 0.8] would flip greedy argmax even though
        is_stopping stayed False (the 0.8-threshold contract)."""
        talker = _make_bare_talker()
        _add_state(talker, "req", is_stopping=False)
        talker._results_queue = [("req", torch.tensor([[0.3, 0.7]]))]

        logits = talker.compute_logits(_hidden(1))

        assert logits[0, 0].item() == 1.0
        assert logits[0, 1].item() == float("-inf")

    def test_empty_queue_defaults_to_all_continue(self) -> None:
        talker = _make_bare_talker()
        logits = talker.compute_logits(_hidden(3))

        assert torch.all(logits[:, 0] == 1.0)
        assert torch.all(logits[:, 1] == float("-inf"))


class TestStateEviction:
    """on_requests_finished fires BEFORE forward(), so eviction must be
    deferred to _flush_deferred_cleanup at the end of the step — the
    finishing request's audio still drains through the current forward."""

    def test_finished_request_survives_until_flush(self) -> None:
        talker = _make_bare_talker()
        _add_state(talker, "req-a")
        _add_state(talker, "req-b")

        talker.on_requests_finished(["req-a"])
        assert "req-a" in talker._active_states  # not evicted yet

        talker._flush_deferred_cleanup()
        assert "req-a" not in talker._active_states
        assert "req-b" in talker._active_states
        assert talker._deferred_cleanup_ids == set()

    def test_unknown_request_id_is_ignored(self) -> None:
        talker = _make_bare_talker()
        talker.on_requests_finished(["never-seen"])
        talker._flush_deferred_cleanup()
        assert talker._active_states == {}
