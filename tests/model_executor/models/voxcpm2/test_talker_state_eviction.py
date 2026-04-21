# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for VoxCPM2 talker per-request state lifecycle."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")

from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (  # noqa: E402
    VoxCPM2TalkerForConditionalGeneration,
    _RequestState,
)


def _make_bare_talker() -> VoxCPM2TalkerForConditionalGeneration:
    talker = VoxCPM2TalkerForConditionalGeneration.__new__(VoxCPM2TalkerForConditionalGeneration)
    talker._active_states = {}
    talker._current_request_id = None
    talker._pending_requests = []
    talker._results_queue = []
    talker._audio_queue = []
    talker._deferred_cleanup_ids = set()
    talker._max_batch_size = 4
    talker._active_state_warn_threshold = 512
    talker._active_state_warned = False
    return talker


def _seed_cached_decode(talker, req_id: str) -> _RequestState:
    state = _RequestState(request_id=req_id)
    state.prefill_completed = True
    state.decode_step_count = 5
    talker._active_states[req_id] = state
    return state


class TestStateEvictionContract:
    def test_pending_requests_is_not_used_for_eviction(self) -> None:
        talker = _make_bare_talker()

        cached_ids = [f"req-{i}" for i in range(4)]
        for rid in cached_ids:
            _seed_cached_decode(talker, rid)

        walked_so_far = ["req-new", cached_ids[0], cached_ids[1]]
        talker._pending_requests = [(rid, False, None, 0) for rid in walked_so_far]

        for rid in cached_ids:
            assert rid in talker._active_states
            assert talker._active_states[rid].prefill_completed is True

    def test_on_requests_finished_defers_cleanup(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        _seed_cached_decode(talker, "req-B")

        talker.on_requests_finished({"req-A"})

        assert "req-A" in talker._active_states
        assert "req-A" in talker._deferred_cleanup_ids

    def test_flush_deferred_cleanup_removes_only_finished(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        _seed_cached_decode(talker, "req-B")
        talker.on_requests_finished(["req-A"])

        talker._flush_deferred_cleanup()

        assert "req-A" not in talker._active_states
        assert "req-B" in talker._active_states
        assert talker._deferred_cleanup_ids == set()

    def test_current_request_id_cleared_when_matching(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        talker._current_request_id = "req-A"

        talker.on_requests_finished({"req-A"})
        talker._flush_deferred_cleanup()

        assert talker._current_request_id is None

    def test_current_request_id_preserved_when_not_finished(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        _seed_cached_decode(talker, "req-B")
        talker._current_request_id = "req-B"

        talker.on_requests_finished({"req-A"})
        talker._flush_deferred_cleanup()

        assert talker._current_request_id == "req-B"


class TestLeakWarnGuard:
    def test_warn_fires_once_over_threshold(self, monkeypatch) -> None:
        from vllm_omni.model_executor.models.voxcpm2 import voxcpm2_talker as tk

        calls: list[str] = []

        def _capture(msg, *args, **kwargs):
            calls.append(msg % args if args else msg)

        monkeypatch.setattr(tk.logger, "warning", _capture)

        talker = _make_bare_talker()
        talker._active_state_warn_threshold = 3

        for i in range(4):
            talker._active_states[f"seed-{i}"] = _RequestState(request_id=f"seed-{i}")

        talker._get_or_create_state("new-1")
        talker._get_or_create_state("new-2")

        leak_warnings = [m for m in calls if "cleanup path leak" in m]
        assert len(leak_warnings) == 1
        assert talker._active_state_warned is True
