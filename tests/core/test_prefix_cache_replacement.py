# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replacement isolates consumer progress without discarding producer writes."""

import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Event as ThreadEvent
from types import SimpleNamespace

import pytest
import torch

from tests.core.test_prefix_cache import HIDDEN, make_manager, plan_fetch
from tests.core.test_prefix_cache_delivery import consume, positions, step
from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheEventKind as Kind,
)
from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheRequestEvent as Event,
)
from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheRequestOwner as Owner,
)
from vllm_omni.core.prefix_cache.adapter import PrefixCacheSchedulerAdapter, PrefixCacheStep
from vllm_omni.core.prefix_cache.interface import ModelCachePolicy, OmniPrefixCacheUnmatchError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def cache():
    manager, view = make_manager(policy=ModelCachePolicy(deferred_keys=frozenset({"codes"})))
    yield manager, view
    manager.shutdown()


def event(kind, generation=0, *, hit=None, blocks=(), admission=1):
    return Event(
        "a",
        kind,
        owner=Owner(admission, generation),
        lookup_complete=hit is not None,
        hit_end=hit or 0,
        block_ids=(tuple(blocks),) if blocks else (),
    )


def save(manager, view, start, end, blocks, *, codes=False):
    view.order = ["a"]
    view.req_blocks["a"] = blocks
    view.computed["a"] = start
    layout = PrefixCacheSchedulerAdapter().build_write_layout(view, num_scheduled_tokens={"a": end - start})
    rows = torch.arange(start, end, dtype=torch.float32).unsqueeze(1).expand(-1, HIDDEN).clone()
    return manager.save_outputs(
        rows,
        {"codes": rows[:, :1]} if codes else {},
        num_tokens_unpadded=end - start,
        num_tokens_padded=end - start,
        write_layout=layout,
    )


def test_replacement_does_not_inherit_old_delivered_64(cache):
    manager, view = cache
    manager.new_step_starts([event(Kind.STARTED, hit=0)])
    consume(manager, save(manager, view, 0, 64, list(range(16))))
    old = manager._request_progress["a"]

    manager.new_step_starts([event(Kind.REPLACED, 1, hit=32, blocks=range(8))], num_scheduled_tokens={"a": 2})
    fresh = manager._request_progress["a"]
    assert fresh is not old
    assert old.retired and old.delivered_upto == {"output_builder": 64}
    assert fresh.owner == Owner(1, 1)
    assert fresh.delivered_upto == {}
    assert positions(consume(manager, save(manager, view, 32, 34, list(range(9))))) == list(range(34))
    assert fresh.delivered_upto == {"output_builder": 34}
    assert old.delivered_upto == {"output_builder": 64}


def test_pending_replacement_resets_all_progress_and_discards_old_delivery(cache):
    manager, view = cache
    manager.new_step_starts([event(Kind.STARTED, hit=0)])
    raw = manager.materialize(save(manager, view, 0, 8, [0, 1], codes=True), ["a"])
    old_delivery = manager.delivery_view(raw, ["a"], consumer="output_builder")
    manager.ack_delivery(old_delivery)
    old = manager._request_progress["a"]
    old_task = manager._request_tasks.deferred["a"]
    assert old.computed_upto == old.saved_upto == 8
    assert old.delivered_upto == {"output_builder": 8}

    manager.new_step_starts([event(Kind.REPLACED, 1)])
    fresh = manager._request_progress["a"]
    assert old.retired
    assert fresh.owner == Owner(1, 1)
    assert fresh.lookup is None
    assert fresh.computed_upto == fresh.saved_upto == 0
    assert fresh.delivered_upto == {}
    assert "a" not in manager._request_tasks.deferred
    assert manager.commit_output(raw, lambda live: None, delivery=old_delivery) == frozenset()
    assert fresh.computed_upto == fresh.saved_upto == 0
    assert fresh.delivered_upto == {}
    # Detaching the append task does not invalidate its captured physical rows.
    assert old_task.tid in manager._join_finished_tids


@pytest.mark.parametrize("hit", [0, 8])
def test_pending_lookup_is_completed_by_first_extension(cache, hit):
    manager, view = cache
    consume(manager, step(manager, view, "producer", 0, 8, [0, 1], kind=Kind.STARTED), "producer")
    manager.new_step_starts([event(Kind.REPLACED, 1)])
    fresh = manager._request_progress["a"]
    assert fresh.lookup is None
    manager.new_step_starts([event(Kind.EXTENDED, 1, hit=hit, blocks=[0, 1])], num_scheduled_tokens={"a": 2})
    assert manager._request_progress["a"] is fresh
    assert fresh.lookup is not None
    assert fresh.lookup[0] == hit
    assert positions(consume(manager, save(manager, view, hit, hit + 2, [0, 1, 2]))) == list(range(hit + 2))


def test_completed_zero_lookup_is_not_later_reinterpreted_as_pending(cache):
    manager, _ = cache
    manager.new_step_starts([event(Kind.REPLACED, 1, hit=0)])
    fresh = manager._request_progress["a"]
    manager.new_step_starts([event(Kind.EXTENDED, 1, hit=8, blocks=[0, 1])])
    assert fresh.lookup[0] == 0
    assert not manager._hit_spans


def test_duplicate_and_stale_controls_preserve_current_progress(cache):
    manager, view = cache
    manager.new_step_starts([event(Kind.REPLACED, 2, hit=0)])
    consume(manager, save(manager, view, 0, 4, [0]))
    fresh = manager._request_progress["a"]
    manager.new_step_starts(
        [event(Kind.REPLACED, 2, hit=0), event(Kind.REPLACED, 1), event(Kind.EXTENDED, 1, hit=4, blocks=[0])]
    )
    assert manager._request_progress["a"] is fresh
    assert fresh.computed_upto == 4
    assert fresh.delivered_upto == {"output_builder": 4}
    assert not manager._hit_spans
    with pytest.raises(OmniPrefixCacheUnmatchError, match="conflicting.*lookup"):
        manager.new_step_starts([event(Kind.REPLACED, 2, hit=4, blocks=[0])])


def test_replacement_retires_deferred_task_outside_lock_and_preserves_rows(cache, monkeypatch):
    manager, view = cache
    manager.new_step_starts([event(Kind.STARTED, hit=0)])
    consume(manager, save(manager, view, 0, 8, [0, 1], codes=True))
    old_task = manager._request_tasks.deferred["a"]
    escalated = []
    original = manager._controller.escalate

    def escalate(tids):
        assert not manager._state_lock.locked()
        escalated.extend(tids)
        original(tids)

    monkeypatch.setattr(manager._controller, "escalate", escalate)
    manager.new_step_starts([event(Kind.REPLACED, 1, hit=0)])
    assert old_task.tid in escalated
    assert "a" not in manager._request_tasks.deferred
    consume(manager, save(manager, view, 0, 4, [2], codes=True))
    assert manager._request_tasks.deferred["a"] is not old_task
    assert plan_fetch(manager, torch.arange(8), "codes", req_id="a")[:, 0].tolist() == list(range(8))
    # A -> B -> A: retirement is about the consumer, not cache producer validity.
    manager.new_step_starts([event(Kind.REPLACED, 2, hit=8, blocks=[0, 1])], num_scheduled_tokens={"a": 2})
    assert positions(consume(manager, save(manager, view, 8, 10, [0, 1, 3], codes=True))) == list(range(10))


def test_late_ack_updates_only_captured_old_progress(cache):
    manager, view = cache
    manager.new_step_starts([event(Kind.STARTED, hit=0)])
    raw = manager.materialize(save(manager, view, 0, 4, [0]), ["a"])
    delivery = manager.delivery_view(raw, ["a"], consumer="output_builder")
    old = raw._progress["a"]
    manager.new_step_starts([event(Kind.REPLACED, 1)])
    manager.ack_delivery(delivery)
    assert old.delivered_upto == {"output_builder": 4}
    assert manager._request_progress["a"].delivered_upto == {}


def test_resume_preserves_owner_and_progress(cache):
    manager, view = cache
    manager.new_step_starts([event(Kind.REPLACED, 1, hit=0)])
    consume(manager, save(manager, view, 0, 8, [0, 1]))
    fresh = manager._request_progress["a"]
    manager.new_step_starts([event(Kind.RESUMED, 1, hit=4, blocks=[0])])
    assert manager._request_progress["a"] is fresh
    assert not fresh.retired
    assert positions(consume(manager, save(manager, view, 4, 10, [0, 1, 2]))) == [8, 9]


def test_new_admission_retires_old_owner_and_rejects_old_controls(cache):
    manager, _ = cache
    manager.new_step_starts([event(Kind.REPLACED, 8)])
    old = manager._request_progress["a"]
    manager.new_step_starts([event(Kind.STARTED, hit=0, admission=2)])
    fresh = manager._request_progress["a"]
    assert fresh is not old and old.retired
    manager.new_step_starts([event(Kind.REPLACED, 9), event(Kind.FINISHED, 8)])
    assert manager._request_progress["a"] is fresh


def test_replayed_step_cannot_reactivate_a_finished_owner(cache):
    manager, _ = cache
    replacement = PrefixCacheStep((event(Kind.REPLACED, 1),), (), sequence=7)
    manager.new_step_starts(replacement)
    manager.new_step_starts(PrefixCacheStep((event(Kind.FINISHED, 1),), (), sequence=8))
    manager.new_step_starts(replacement)
    assert "a" not in manager._request_progress


@pytest.mark.parametrize("old_finished", [False, True])
def test_terminal_of_undispatched_replacement_retires_old_progress(cache, old_finished):
    manager, _ = cache
    manager.new_step_starts([event(Kind.STARTED, hit=0)])
    old = manager._request_progress["a"]
    if old_finished:
        manager.new_step_starts([event(Kind.FINISHED)])
    manager.new_step_starts([event(Kind.FINISHED, 1)])
    assert old.retired
    assert "a" not in manager._request_progress


def test_commit_filters_replacement_that_arrives_during_build(cache):
    manager, view = cache
    manager.new_step_starts([event(Kind.STARTED, hit=0)])
    raw = manager.materialize(save(manager, view, 0, 4, [0]), ["a"])
    delivery = manager.delivery_view(raw, ["a"], consumer="output_builder")
    build_started, finish_build = ThreadEvent(), ThreadEvent()
    handed_off: list[frozenset[str]] = []

    def build_and_commit():
        build_started.set()
        assert finish_build.wait(5)
        return manager.commit_output(raw, handed_off.append, delivery=delivery)

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(build_and_commit)
        try:
            assert build_started.wait(5)
            manager.new_step_starts([event(Kind.REPLACED, 1)])
        finally:
            finish_build.set()
        assert future.result(timeout=5) == frozenset()
    assert handed_off == [frozenset()]
    assert raw._progress["a"].delivered_upto == {}
    assert manager._request_progress["a"].delivered_upto == {}


def test_mixed_commit_preserves_current_payload_and_batch_accounting(cache):
    from vllm_omni.core.prefix_cache.interface import StageCacheOutputs

    manager, _ = cache
    manager.new_step_starts([event(Kind.STARTED, hit=0), Event("b", Kind.STARTED, owner=Owner(2))])
    raw = StageCacheOutputs(
        hidden_states={"a": torch.zeros(2, 1), "b": torch.ones(2, 1)},
        mm_outputs={},
        token_ranges={"a": (0, 2), "b": (0, 2)},
        scheduled_token_ranges={"a": (0, 2), "b": (0, 2)},
        _progress=dict(manager._request_progress),
    )
    delivery = manager.delivery_view(raw, ["a", "b"], consumer="output_builder")
    manager.new_step_starts([event(Kind.REPLACED, 1)])
    batch = SimpleNamespace(req_ids=["a", "b"], sampled_token_ids=[[1], [2]], payloads=["old", "current"])

    def handoff(live):
        assert manager._state_lock.locked()
        batch.payloads = [value if req_id in live else None for req_id, value in zip(batch.req_ids, batch.payloads)]

    assert manager.commit_output(raw, handoff, delivery=delivery) == frozenset({"b"})
    assert batch.req_ids == ["a", "b"]
    assert batch.sampled_token_ids == [[1], [2]]
    assert batch.payloads == [None, "current"]
    assert raw._progress["a"].delivered_upto == {}
    assert raw._progress["b"].delivered_upto == {"output_builder": 2}


def test_commit_failure_does_not_acknowledge(cache):
    manager, view = cache
    manager.new_step_starts([event(Kind.STARTED, hit=0)])
    raw = manager.materialize(save(manager, view, 0, 4, [0]), ["a"])
    delivery = manager.delivery_view(raw, ["a"], consumer="output_builder")

    def fail(_live):
        raise RuntimeError("handoff failed")

    with pytest.raises(RuntimeError, match="handoff failed"):
        manager.commit_output(raw, fail, delivery=delivery)
    assert raw._progress["a"].delivered_upto == {}


def test_normal_final_output_can_commit_after_terminal_cleanup(cache):
    manager, view = cache
    manager.new_step_starts([event(Kind.STARTED, hit=0)])
    raw = manager.materialize(save(manager, view, 0, 4, [0]), ["a"])
    delivery = manager.delivery_view(raw, ["a"], consumer="output_builder")
    manager.new_step_starts([event(Kind.FINISHED)])
    handed_off: list[frozenset[str]] = []
    assert manager.commit_output(raw, handed_off.append, delivery=delivery) == frozenset({"a"})
    assert handed_off == [frozenset({"a"})]
    assert raw._progress["a"].delivered_upto == {"output_builder": 4}


@pytest.mark.parametrize("new_admission_finished", [False, True])
def test_finished_output_is_retired_when_request_id_is_reused(cache, new_admission_finished):
    manager, view = cache
    manager.new_step_starts([event(Kind.STARTED, hit=0)])
    raw = manager.materialize(save(manager, view, 0, 4, [0]), ["a"])
    delivery = manager.delivery_view(raw, ["a"], consumer="output_builder")
    manager.new_step_starts([event(Kind.FINISHED)])
    manager.new_step_starts([event(Kind.STARTED, hit=0, admission=2)])
    if new_admission_finished:
        manager.new_step_starts([event(Kind.FINISHED, admission=2)])
    assert manager.commit_output(raw, lambda _live: None, delivery=delivery) == frozenset()
    assert raw._progress["a"].retired
    assert raw._progress["a"].delivered_upto == {}


def test_finished_owner_index_does_not_keep_completed_requests_alive(cache):
    manager, view = cache
    manager.new_step_starts([event(Kind.STARTED, hit=0)])
    raw = manager.materialize(save(manager, view, 0, 4, [0]), ["a"])
    manager.new_step_starts([event(Kind.FINISHED)])
    assert manager._finished_progress["a"] is raw._progress["a"]
    del raw
    gc.collect()
    assert not manager._finished_progress


def test_metadata_only_commit_checks_owner_without_advancing_token_delivery(cache):
    manager, view = cache
    manager.new_step_starts([event(Kind.STARTED, hit=0)])
    raw = manager.materialize(save(manager, view, 0, 4, [0]), ["a"])
    handed_off: list[frozenset[str]] = []
    assert manager.commit_output(raw, handed_off.append) == frozenset({"a"})
    assert raw._progress["a"].delivered_upto == {}
