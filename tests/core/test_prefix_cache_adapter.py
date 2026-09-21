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


def output(*, new=(), resumed=(), finished=(), aborted=()):
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set(resumed)),
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
