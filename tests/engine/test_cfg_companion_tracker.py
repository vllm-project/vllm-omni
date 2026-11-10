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


def test_defer_records_one_deadline_across_redeferrals():
    now = [10.0]
    tracker = CfgCompanionTracker(clock=lambda: now[0])
    tracker.register_companion("req1", "cfg_text", "req1__cfg_text")

    tracker.defer_parent("req1", {"out": 1}, stage_id=0)
    now[0] = 15.0
    tracker.defer_parent("req1", {"out": 2}, stage_id=1)

    assert tracker.poll_expired(10.0, now=19.9) == []
    expired = tracker.poll_expired(10.0, now=20.0)

    assert len(expired) == 1
    parent_id, pending = expired[0]
    assert parent_id == "req1"
    assert pending["engine_outputs"] == {"out": 2}
    assert pending["stage_id"] == 1
    assert tracker.poll_expired(10.0, now=21.0) == []


def test_terminal_claim_is_idempotent_until_cleanup():
    tracker = CfgCompanionTracker()
    tracker.register_companion("req1", "cfg_text", "req1__cfg_text")

    assert tracker.claim_terminal("req1") is True
    assert tracker.claim_terminal("req1") is False
    assert tracker.is_terminal("req1")
    assert tracker.is_request_terminal("req1__cfg_text")
    assert tracker.register_companion("req1", "late", "req1__late") is False
    assert not tracker.is_companion("req1__late")

    tracker.cleanup_parent("req1")
    assert tracker.is_request_terminal("req1__cfg_text")
    tracker.finalize_terminal("req1")
    assert not tracker.is_terminal("req1")
    assert not tracker.is_request_terminal("req1__cfg_text")


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("inf"), float("nan")])
def test_poll_expired_requires_positive_finite_timeout(timeout_s):
    with pytest.raises(ValueError, match="timeout_s must be finite and > 0"):
        CfgCompanionTracker().poll_expired(timeout_s)
