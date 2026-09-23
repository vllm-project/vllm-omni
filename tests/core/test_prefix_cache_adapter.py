from types import SimpleNamespace

import pytest

from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheEventKind,
    PrefixCacheRequestEvent,
    PrefixCacheSchedulerAdapter,
)


class FakeView:
    def batch_req_ids(self):
        return ["b", "a"]

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


@pytest.mark.parametrize("field", ["num_computed_tokens", "new_block_ids", "num_output_tokens"])
def test_cached_request_parallel_fields_must_have_one_entry_per_request(field):
    values = {
        "req_ids": ["r1", "r2"],
        "num_computed_tokens": [8, 8],
        "new_block_ids": [[[1, 2]], [[3, 4]]],
        "num_output_tokens": [0, 0],
    }
    values[field] = values[field][:1]
    with pytest.raises(ValueError, match=field):
        PrefixCacheSchedulerAdapter().translate_scheduler_output(
            output(resumed={"r1"}, cached=values)
        )


def test_resumed_request_must_have_cached_payload():
    with pytest.raises(ValueError, match="missing from req_ids"):
        PrefixCacheSchedulerAdapter().translate_scheduler_output(
            output(resumed={"missing"}, cached={"req_ids": ["other"]})
        )


def test_resumed_request_ids_are_normalized_before_membership_check():
    events = PrefixCacheSchedulerAdapter().translate_scheduler_output(
        output(
            resumed={7},
            cached={
                "req_ids": ["7"],
                "num_computed_tokens": [8],
                "new_block_ids": [[[1, 2]]],
                "num_output_tokens": [0],
            },
        )
    )
    assert events[0].req_id == "7"


def test_same_id_terminal_and_new_is_started():
    adapter = PrefixCacheSchedulerAdapter()
    adapter.translate_scheduler_output(output(new=[SimpleNamespace(req_id="r")]))
    events = adapter.translate_scheduler_output(
        output(new=[SimpleNamespace(req_id="r")], finished={"r"})
    )
    assert [event.kind for event in events] == [PrefixCacheEventKind.STARTED, PrefixCacheEventKind.FINISHED]


def test_write_layout_uses_post_order_batch_and_slots():
    layout = PrefixCacheSchedulerAdapter().build_write_layout(
        FakeView(), num_scheduled_tokens={"b": 2, "a": 1}
    )
    assert layout.total_rows == 3
    assert [(w.req_id, w.row_start, w.row_end, w.slots) for w in layout.writes] == [
        ("b", 0, 2, (20, 21)),
        ("a", 2, 3, (7,)),
    ]
    with pytest.raises(AttributeError):
        layout.writes = ()


def test_write_layout_rejects_slot_count_mismatch():
    class BrokenView(FakeView):
        def step_slots_cpu(self, req_ids, num_scheduled):
            import torch

            return torch.tensor([20], dtype=torch.long)

    with pytest.raises(ValueError, match="slot count"):
        PrefixCacheSchedulerAdapter().build_write_layout(
            BrokenView(), num_scheduled_tokens={"b": 2, "a": 1}
        )
