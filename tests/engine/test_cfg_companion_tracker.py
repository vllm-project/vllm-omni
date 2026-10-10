# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_register_companion_and_cleanup():
    tracker = CfgCompanionTracker()

    tracker.register_companion("req1", "cfg_text", "req1__cfg_text")
    tracker.register_companion("req1", "cfg_img", "req1__cfg_img")

    assert tracker.is_companion("req1__cfg_text")
    assert tracker.get_companion_request_ids("req1") == {
        "cfg_text": "req1__cfg_text",
        "cfg_img": "req1__cfg_img",
    }

    removed = tracker.cleanup_parent("req1")

    assert sorted(removed) == ["req1__cfg_img", "req1__cfg_text"]
    assert not tracker.is_companion("req1__cfg_text")
    assert tracker.get_companion_request_ids("req1") == {}


def test_abort_parent_expands_to_companions_and_cleans_up_deferred_parent():
    tracker = CfgCompanionTracker()
    tracker.register_companion("req1", "cfg_text", "req1__cfg_text")
    tracker.defer_parent("req1", {"out": 1}, stage_id=0)

    aborted = tracker.abort_parents(["req1"])

    assert sorted(aborted) == ["req1", "req1__cfg_text"]
    assert not tracker.is_companion("req1__cfg_text")
    assert tracker.pop_pending_parent("req1") is None


def test_abort_companion_does_not_expand_to_parent():
    tracker = CfgCompanionTracker()
    tracker.register_companion("req1", "cfg_text", "req1__cfg_text")

    aborted = tracker.abort_parents(["req1__cfg_text"])

    assert aborted == ["req1__cfg_text"]


def test_companion_completion_flushes_deferred_parent():
    tracker = CfgCompanionTracker()
    tracker.register_companion("req1", "cfg_text", "req1__cfg_text")
    tracker.defer_parent("req1", {"out": 123}, stage_id=0)

    assert not tracker.all_companions_done("req1")
    assert tracker.on_companion_completed("req1__cfg_text") == "req1"
    assert tracker.all_companions_done("req1")

    popped = tracker.pop_pending_parent("req1")
    assert popped is not None
    assert popped["engine_outputs"] == {"out": 123}
    assert popped["stage_id"] == 0


def test_companion_completion_without_registered_parent_asserts():
    tracker = CfgCompanionTracker()
    tracker._companion_ids.add("req1__cfg_text")
    tracker._companion_to_parent["req1__cfg_text"] = "req1"

    with pytest.raises(AssertionError, match="completed before parent req1 was registered"):
        tracker.on_companion_completed("req1__cfg_text")


def test_redefer_preserves_original_deadline_and_updates_output():
    tracker = CfgCompanionTracker(clock=lambda: 100.0)
    tracker.register_companion("req1", "negative", "req1-negative")

    tracker.defer_parent("req1", {"attempt": 1}, stage_id=0)
    tracker.defer_parent("req1", {"attempt": 2}, stage_id=1)

    assert tracker.pop_expired_parents(10.0, now=109.999) == []
    expired = tracker.pop_expired_parents(10.0, now=110.0)
    assert expired == [
        (
            "req1",
            {
                "engine_outputs": {"attempt": 2},
                "stage_id": 1,
                "deferred_at": 100.0,
            },
        )
    ]
    assert tracker.pop_expired_parents(10.0, now=120.0) == []


@pytest.mark.parametrize("timeout", [0.0, float("nan"), float("inf")])
def test_pop_expired_parents_rejects_non_positive_or_non_finite_timeout(timeout: float):
    tracker = CfgCompanionTracker()

    with pytest.raises(ValueError, match="timeout_s must be finite and > 0"):
        tracker.pop_expired_parents(timeout)
