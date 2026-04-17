import pytest

from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

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


def test_attach_cfg_request_ids_clones_diffusion_params():
    tracker = CfgCompanionTracker()
    tracker.register_companion("req1", "cfg_text", "req1__cfg_text")

    params = OmniDiffusionSamplingParams()
    updated = tracker.attach_cfg_request_ids("req1", params)

    assert updated is not params
    assert params.cfg_kv_request_ids is None
    assert updated.cfg_kv_request_ids == {"cfg_text": "req1__cfg_text"}


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
