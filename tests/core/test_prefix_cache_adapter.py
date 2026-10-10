# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.request import Request

from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheEventKind,
    PrefixCacheRequestEvent,
    PrefixCacheRequestOwner,
    PrefixCacheSchedulerAdapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeView:
    def batch_req_ids(self):
        return ["b", "a"]

    def token_range(self, req_id, num_scheduled):
        start = {"b": 12, "a": 7}[req_id]
        return start, start + num_scheduled

    def step_slots_cpu(self, req_ids, num_scheduled):
        import torch

        return torch.tensor([20, 21, 7], dtype=torch.long)


def output(*, new=(), resumed=(), finished=(), aborted=(), cached=None):
    cached = cached or {}
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=SimpleNamespace(
            resumed_req_ids=set(resumed),
            req_ids=list(cached.get("req_ids", resumed)),
            num_computed_tokens=list(cached.get("num_computed_tokens", ())),
            new_block_ids=list(cached.get("new_block_ids", ())),
            num_output_tokens=list(cached.get("num_output_tokens", ())),
        ),
        finished_req_ids=set(finished),
        aborted_req_ids=set(aborted),
    )


def test_scheduler_events_are_explicit_and_immutable():
    adapter = PrefixCacheSchedulerAdapter()
    first = adapter.translate_scheduler_output(
        output(new=[SimpleNamespace(req_id="a", num_computed_tokens=4, block_ids=[[2]])])
    )
    assert first[0] == PrefixCacheRequestEvent("a", PrefixCacheEventKind.STARTED, 0, 4, ((2,),))
    with pytest.raises(AttributeError):
        first[0].req_id = "b"

    second = adapter.translate_scheduler_output(output(new=[SimpleNamespace(req_id="a", num_computed_tokens=0)]))
    assert second[0].kind is PrefixCacheEventKind.EXTENDED


def test_resume_and_abort_require_explicit_sources():
    adapter = PrefixCacheSchedulerAdapter()
    events = adapter.translate_scheduler_output(output(resumed={"r"}, finished={"f"}, aborted={"a"}))
    assert [(e.req_id, e.kind) for e in events] == [
        ("r", PrefixCacheEventKind.RESUMED),
        ("a", PrefixCacheEventKind.ABORTED),
        ("f", PrefixCacheEventKind.FINISHED),
    ]
    events = adapter.translate_scheduler_output(output(finished={"x"}))
    assert events[0].kind is PrefixCacheEventKind.FINISHED


def test_missing_abort_side_channel_never_infers_aborted():
    adapter = PrefixCacheSchedulerAdapter()
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=None,
        finished_req_ids={"finished"},
        num_scheduled_tokens={},
    )
    events = adapter.translate_scheduler_output(scheduler_output)
    assert [(event.req_id, event.kind) for event in events] == [("finished", PrefixCacheEventKind.FINISHED)]


def test_resumed_event_snapshots_cached_request_payload():
    adapter = PrefixCacheSchedulerAdapter()
    events = adapter.translate_scheduler_output(
        output(
            resumed={"r"},
            cached={
                "req_ids": ["other", "r"],
                "num_computed_tokens": [3, 8],
                "new_block_ids": [[[1]], [[4, 5]]],
                "num_output_tokens": [1, 6],
            },
        )
    )
    event = events[0]
    assert (event.kind, event.req_id, event.hit_start, event.hit_end) == (
        PrefixCacheEventKind.RESUMED,
        "r",
        0,
        8,
    )
    assert event.block_ids == ((4, 5),)
    assert event.scheduled_tokens == 0
    assert event.num_output_tokens == 6


def test_resumed_block_snapshot_is_immutable():
    blocks = [[7, 8]]
    adapter = PrefixCacheSchedulerAdapter()
    event = adapter.translate_scheduler_output(
        output(
            resumed={"r"},
            cached={"req_ids": ["r"], "new_block_ids": [blocks], "num_computed_tokens": [8]},
        )
    )[0]
    blocks[0][0] = 99
    assert event.block_ids == ((7, 8),)


def test_same_id_terminal_and_new_is_started():
    adapter = PrefixCacheSchedulerAdapter()
    adapter.translate_scheduler_output(output(new=[SimpleNamespace(req_id="r")]))
    events = adapter.translate_scheduler_output(output(new=[SimpleNamespace(req_id="r")], finished={"r"}))
    assert [event.kind for event in events] == [PrefixCacheEventKind.STARTED, PrefixCacheEventKind.FINISHED]
    events = adapter.translate_scheduler_output(output(new=[SimpleNamespace(req_id="r")]))
    assert [event.kind for event in events] == [PrefixCacheEventKind.EXTENDED]


def test_write_layout_uses_post_order_batch_and_slots():
    layout = PrefixCacheSchedulerAdapter().build_write_layout(FakeView(), num_scheduled_tokens={"b": 2, "a": 1})
    assert layout.total_rows == 3
    assert [(w.req_id, w.row_start, w.row_end) for w in layout.writes] == [
        ("b", 0, 2),
        ("a", 2, 3),
    ]
    assert torch.equal(layout.slots_cpu, torch.tensor([20, 21, 7]))
    with pytest.raises(AttributeError):
        layout.writes = ()


def test_write_layout_keeps_request_positions_separate_from_batch_rows():
    layout = PrefixCacheSchedulerAdapter().build_write_layout(FakeView(), num_scheduled_tokens={"b": 2, "a": 1})
    assert [(w.token_start, w.token_end) for w in layout.writes] == [(12, 14), (7, 8)]
    assert [(w.row_start, w.row_end) for w in layout.writes] == [(0, 2), (2, 3)]


@pytest.mark.parametrize("scheduled", [{}, {"a": 1}])
def test_write_layout_keeps_zero_token_requests_without_advancing_rows(scheduled, mocker):
    view = FakeView()
    slots = torch.arange(sum(scheduled.values()))
    slot_mapping = mocker.patch.object(view, "step_slots_cpu", return_value=slots)

    layout = PrefixCacheSchedulerAdapter().build_write_layout(view, num_scheduled_tokens=scheduled)

    assert layout.total_rows == len(slots)
    assert layout.slots_cpu is slots
    assert [(w.req_id, w.row_start, w.row_end, w.token_start, w.token_end) for w in layout.writes] == [
        ("b", 0, 0, 12, 12),
        ("a", 0, len(slots), 7, 7 + len(slots)),
    ]
    slot_mapping.assert_called_once_with(["b", "a"], scheduled)


def test_same_id_terminal_and_new_preserves_new_observation_for_extension():
    adapter = PrefixCacheSchedulerAdapter()
    adapter.translate_scheduler_output(output(new=[SimpleNamespace(req_id="r")]))
    adapter.translate_scheduler_output(output(new=[SimpleNamespace(req_id="r")], finished={"r"}))
    events = adapter.translate_scheduler_output(output(new=[SimpleNamespace(req_id="r", num_computed_tokens=8)]))
    assert events[0].kind is PrefixCacheEventKind.EXTENDED


def test_replacement_control_precedes_first_delayed_lookup():
    adapter = PrefixCacheSchedulerAdapter()
    owner = PrefixCacheRequestOwner(1, 2)
    control = PrefixCacheRequestEvent("a", PrefixCacheEventKind.REPLACED, owner=owner, lookup_complete=False)
    control_only = output()
    control_only.prefix_cache_replacements = (control,)
    control_only.prefix_cache_step_sequence = 1
    step = adapter.translate_step(control_only)
    assert step.sequence == 1
    assert step.events == (control,)
    assert not step.scheduled_tokens

    lookup = output(new=[SimpleNamespace(req_id="a", num_computed_tokens=0, block_ids=[[2]])])
    lookup.prefix_cache_owners = {"a": owner}
    event = adapter.translate_scheduler_output(lookup)[0]
    assert event.kind is PrefixCacheEventKind.EXTENDED
    assert event.owner == owner
    assert event.lookup_complete and event.hit_end == 0


def test_cached_extension_carries_delayed_lookup_and_owner():
    adapter = PrefixCacheSchedulerAdapter()
    owner = PrefixCacheRequestOwner(1, 1)
    lookup = output(cached={"req_ids": ["a"], "num_computed_tokens": [8], "new_block_ids": [[[3, 4]]]})
    lookup.prefix_cache_owners = {"a": owner}
    event = adapter.translate_scheduler_output(lookup)[0]
    assert event.kind is PrefixCacheEventKind.EXTENDED
    assert event.owner == owner
    assert event.hit_end == 8 and event.block_ids == ((3, 4),)


def test_terminal_captures_old_admission_when_id_is_reused():
    adapter = PrefixCacheSchedulerAdapter()
    first = output(new=[SimpleNamespace(req_id="a")])
    first.prefix_cache_owners = {"a": PrefixCacheRequestOwner(1)}
    adapter.translate_step(first)
    reused = output(new=[SimpleNamespace(req_id="a")], finished=["a"])
    reused.prefix_cache_owners = {"a": PrefixCacheRequestOwner(2)}
    events = adapter.translate_step(reused).events
    assert events[0].owner == PrefixCacheRequestOwner(2)
    assert events[-1].owner == PrefixCacheRequestOwner(1)


def test_replayed_scheduler_step_does_not_rewrite_adapter_observations():
    adapter = PrefixCacheSchedulerAdapter()
    first = output(new=[SimpleNamespace(req_id="a")])
    first.prefix_cache_step_sequence = 1
    first.prefix_cache_owners = {"a": PrefixCacheRequestOwner(1)}
    adapter.translate_step(first)
    last = output(finished=["a"])
    last.prefix_cache_step_sequence = 2
    adapter.translate_step(last)
    assert adapter.translate_step(first).events == ()
    assert adapter.translate_scheduler_output(first)[0].kind is PrefixCacheEventKind.STARTED


def test_stale_lookup_and_terminal_do_not_rewind_adapter_owner():
    adapter = PrefixCacheSchedulerAdapter()
    current = PrefixCacheRequestOwner(2, 1)
    admitted = output(new=[SimpleNamespace(req_id="a")])
    admitted.prefix_cache_owners = {"a": current}
    adapter.translate_step(admitted)
    stale = output(new=[SimpleNamespace(req_id="a")], finished=["a"])
    stale.prefix_cache_owners = {"a": PrefixCacheRequestOwner(1)}
    stale.prefix_cache_terminal_owners = {"a": PrefixCacheRequestOwner(1)}
    events = adapter.translate_step(stale).events
    assert all(event.kind is PrefixCacheEventKind.FINISHED for event in events)
    assert adapter._observed_owners["a"] == current
    assert adapter.translate_step(admitted).events[0].kind is PrefixCacheEventKind.EXTENDED


@pytest.mark.parametrize("owner", [None, PrefixCacheRequestOwner(1)])
def test_ownerless_cached_step_preserves_admission_until_terminal(owner):
    adapter = PrefixCacheSchedulerAdapter()
    request = Request(
        request_id="a",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
    )
    admitted = output(new=[NewRequestData.from_request(request, block_ids=())])
    admitted.prefix_cache_owners = {"a": owner} if owner is not None else {}
    assert adapter.translate_step(admitted).events[0].kind is PrefixCacheEventKind.STARTED
    cached = output(cached={"req_ids": ["a"], "num_computed_tokens": [4]})
    assert adapter.translate_step(cached).events[0].kind is PrefixCacheEventKind.EXTENDED
    assert adapter.translate_step(admitted).events[0].kind is PrefixCacheEventKind.EXTENDED
    terminal = adapter.translate_step(output(finished=["a"])).events[0]
    assert terminal.kind is PrefixCacheEventKind.FINISHED
    assert terminal.owner == owner
    assert adapter.translate_step(admitted).events[0].kind is PrefixCacheEventKind.STARTED
