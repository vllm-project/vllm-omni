# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

try:  # pragma: no cover - namespace shim for vllm-less development
    import vllm  # noqa: F401
except ModuleNotFoundError:
    _root = Path(__file__).resolve().parents[2]
    for _pkg in ("vllm_omni", "vllm_omni.core", "vllm_omni.utils"):
        if _pkg not in sys.modules:
            _module = ModuleType(_pkg)
            _module.__path__ = [str(_root / _pkg.replace(".", "/"))]
            sys.modules[_pkg] = _module

from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheEventKind,
    PrefixCacheRequestEvent,
    PrefixCacheSchedulerAdapter,
)
from vllm_omni.core.prefix_cache.interface import OmniPrefixCacheUnmatchError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeView:
    def batch_req_ids(self):
        return ["b", "a"]

    def step_slots_cpu(self, req_ids, num_scheduled):
        return torch.tensor([20, 21, 7], dtype=torch.long)


def output(*, new=(), resumed=(), finished=(), aborted=(), cached=None, scheduled=None):
    cached = cached or {}
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=SimpleNamespace(
            resumed_req_ids=set(resumed),
            req_ids=list(cached.get("req_ids", sorted(resumed))),
            num_computed_tokens=list(cached.get("num_computed_tokens", ())),
            new_block_ids=list(cached.get("new_block_ids", ())),
            num_output_tokens=list(cached.get("num_output_tokens", ())),
        ),
        finished_req_ids=set(finished),
        aborted_req_ids=set(aborted),
        num_scheduled_tokens=dict(scheduled or {}),
    )


def test_scheduler_events_are_explicit_and_immutable():
    adapter = PrefixCacheSchedulerAdapter()
    first = adapter.translate_step(
        output(new=[SimpleNamespace(req_id="a", num_computed_tokens=4, block_ids=[[2]])])
    ).events
    assert first[0] == PrefixCacheRequestEvent("a", PrefixCacheEventKind.STARTED, hit_end=4, block_ids=((2,),))
    with pytest.raises(AttributeError):
        first[0].req_id = "b"

    second = adapter.translate_step(output(new=[SimpleNamespace(req_id="a", num_computed_tokens=0)]))
    assert second.events == ()
    assert second.extended_req_ids == ("a",)


def test_resume_and_abort_require_explicit_sources():
    adapter = PrefixCacheSchedulerAdapter()
    events = adapter.translate_step(output(resumed={"r"}, finished={"f"}, aborted={"a"})).events
    assert [(e.req_id, e.kind) for e in events] == [
        ("a", PrefixCacheEventKind.ABORTED),
        ("f", PrefixCacheEventKind.FINISHED),
        ("r", PrefixCacheEventKind.RESUMED),
    ]
    events = adapter.translate_step(output(finished={"x"})).events
    assert events[0].kind is PrefixCacheEventKind.FINISHED


def test_missing_abort_side_channel_never_infers_aborted():
    adapter = PrefixCacheSchedulerAdapter()
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=None,
        finished_req_ids={"finished"},
        num_scheduled_tokens={},
    )
    events = adapter.translate_step(scheduler_output).events
    assert [(event.req_id, event.kind) for event in events] == [("finished", PrefixCacheEventKind.FINISHED)]


def test_resumed_event_snapshots_cached_request_payload():
    adapter = PrefixCacheSchedulerAdapter()
    events = adapter.translate_step(
        output(
            resumed={"r"},
            cached={
                "req_ids": ["other", "r"],
                "num_computed_tokens": [3, 8],
                "new_block_ids": [[[1]], [[4, 5]]],
                "num_output_tokens": [1, 6],
            },
        )
    ).events
    event = events[0]
    assert (event.kind, event.req_id, event.hit_end) == (PrefixCacheEventKind.RESUMED, "r", 8)
    assert event.block_ids == ((4, 5),)
    assert event.scheduled_tokens == 0
    assert event.num_output_tokens == 6


def test_resumed_block_snapshot_is_immutable():
    blocks = [[7, 8]]
    adapter = PrefixCacheSchedulerAdapter()
    event = adapter.translate_step(
        output(
            resumed={"r"},
            cached={"req_ids": ["r"], "new_block_ids": [blocks], "num_computed_tokens": [8]},
        )
    ).events[0]
    blocks[0][0] = 99
    assert event.block_ids == ((7, 8),)


@pytest.mark.parametrize("terminal", ["finished", "aborted"])
def test_same_id_terminal_and_new_is_started(terminal):
    adapter = PrefixCacheSchedulerAdapter()
    adapter.translate_step(output(new=[SimpleNamespace(req_id="r")]))
    events = adapter.translate_step(output(new=[SimpleNamespace(req_id="r")], **{terminal: {"r"}})).events
    terminal_kind = PrefixCacheEventKind.FINISHED if terminal == "finished" else PrefixCacheEventKind.ABORTED
    assert [event.kind for event in events] == [terminal_kind, PrefixCacheEventKind.STARTED]
    step = adapter.translate_step(output(new=[SimpleNamespace(req_id="r")]))
    assert step.events == () and step.extended_req_ids == ("r",)


@pytest.mark.parametrize("count", [1, 128], ids=["decode", "chunked-prefill"])
def test_regular_scheduled_requests_emit_extended(count):
    adapter = PrefixCacheSchedulerAdapter()
    adapter.translate_step(output(new=[SimpleNamespace(req_id="r")], scheduled={"r": 256}))
    step = adapter.translate_step(output(cached={"req_ids": ["r"]}, scheduled={"r": count}))
    assert step.events == () and step.extended_req_ids == ("r",)
    assert step.scheduled_tokens == (("r", count),)


def test_only_scheduled_live_requests_emit_extended():
    adapter = PrefixCacheSchedulerAdapter()
    adapter.translate_step(output(new=[SimpleNamespace(req_id="a"), SimpleNamespace(req_id="b")]))
    step = adapter.translate_step(output(cached={"req_ids": ["a", "b"]}, scheduled={"b": 2, "a": 0}))
    assert step.events == () and step.extended_req_ids == ("b",)
    idle = adapter.translate_step(output())
    assert idle.events == () and idle.extended_req_ids == ()


def test_resumed_order_follows_scheduler_and_has_no_duplicate_extended():
    adapter = PrefixCacheSchedulerAdapter()
    adapter.translate_step(output(new=[SimpleNamespace(req_id=r) for r in ("a", "b", "running")]))
    step = adapter.translate_step(
        output(
            resumed={"a", "b"},
            cached={"req_ids": ["b", "running", "a"], "num_computed_tokens": [8, 4, 12]},
            scheduled={"a": 1, "b": 2, "running": 3},
        )
    )
    assert [(e.req_id, e.kind, e.scheduled_tokens) for e in step.events] == [
        ("b", PrefixCacheEventKind.RESUMED, 2),
        ("a", PrefixCacheEventKind.RESUMED, 1),
    ]
    assert step.extended_req_ids == ("running",)
    assert [e.hit_end for e in step.events] == [8, 12]


def test_new_continuation_is_not_duplicated_by_scheduled_tokens():
    adapter = PrefixCacheSchedulerAdapter()
    adapter.translate_step(output(new=[SimpleNamespace(req_id="r")]))
    step = adapter.translate_step(output(new=[SimpleNamespace(req_id="r")], scheduled={"r": 4}))
    assert step.events == () and step.extended_req_ids == ("r",)


def test_finished_ids_are_retired_before_classification():
    adapter = PrefixCacheSchedulerAdapter()
    adapter.translate_step(output(new=[SimpleNamespace(req_id="r")]))
    step = adapter.translate_step(output(finished={"r"}, scheduled={"r": 1}))
    assert [e.kind for e in step.events] == [PrefixCacheEventKind.FINISHED]
    assert step.extended_req_ids == ()
    step = adapter.translate_step(output(new=[SimpleNamespace(req_id="r")]))
    assert [e.kind for e in step.events] == [PrefixCacheEventKind.STARTED]
    assert step.extended_req_ids == ()


def test_abort_ids_are_normalized_before_membership_check():
    events = PrefixCacheSchedulerAdapter().translate_step(output(finished={7}, aborted={7}))
    assert events.events == (PrefixCacheRequestEvent("7", PrefixCacheEventKind.ABORTED),)


def test_step_snapshots_token_counts_and_new_hit_blocks():
    blocks = [[2, 3]]
    scheduler_output = output(
        new=[SimpleNamespace(req_id="r", num_computed_tokens=8, block_ids=blocks)], scheduled={"r": 2}
    )
    step = PrefixCacheSchedulerAdapter().translate_step(scheduler_output)
    scheduler_output.num_scheduled_tokens["r"] = 99
    blocks[0][0] = 100
    assert step.scheduled_tokens == (("r", 2),)
    assert step.events[0].block_ids == ((2, 3),)
    with pytest.raises(AttributeError):
        step.scheduled_tokens = ()


def test_no_hit_does_not_copy_unused_block_table():
    class UnusedBlocks:
        def __bool__(self):
            raise AssertionError("no-hit block table must not be inspected")

    step = PrefixCacheSchedulerAdapter().translate_step(
        output(new=[SimpleNamespace(req_id="r", num_computed_tokens=0, block_ids=UnusedBlocks())])
    )
    assert step.events[0].block_ids == ()


def test_request_id_fallback_and_missing_id_error():
    adapter = PrefixCacheSchedulerAdapter()
    assert adapter.translate_step(output(new=[SimpleNamespace(request_id="r")])).events[0].req_id == "r"
    with pytest.raises(OmniPrefixCacheUnmatchError, match="no req_id/request_id"):
        adapter.translate_step(output(new=[SimpleNamespace()]))


def test_write_layout_uses_post_order_batch_and_slots():
    layout = PrefixCacheSchedulerAdapter().build_write_layout(FakeView(), num_scheduled_tokens={"b": 2, "a": 1})
    assert layout.total_rows == 3
    assert [(w.req_id, w.row_start, w.row_end) for w in layout.writes] == [("b", 0, 2), ("a", 2, 3)]
    assert torch.equal(layout.slots_cpu, torch.tensor([20, 21, 7]))
    with pytest.raises(AttributeError):
        layout.writes = ()


def test_write_layout_keeps_single_fresh_slot_tensor():
    slots = torch.tensor([20, 21])
    view = SimpleNamespace(batch_req_ids=lambda: ["b", "a"], step_slots_cpu=lambda ids, counts: slots)
    layout = PrefixCacheSchedulerAdapter().build_write_layout(view, num_scheduled_tokens={"b": 2})
    assert layout.slots_cpu is slots
    assert [(w.row_start, w.row_end) for w in layout.writes] == [(0, 2), (2, 2)]


def test_empty_write_layout():
    view = SimpleNamespace(
        batch_req_ids=lambda: [], step_slots_cpu=lambda ids, counts: torch.empty(0, dtype=torch.long)
    )
    layout = PrefixCacheSchedulerAdapter().build_write_layout(view, num_scheduled_tokens={})
    assert layout.total_rows == 0 and layout.writes == () and layout.slots_cpu.numel() == 0


def test_negative_scheduled_tokens_fail_at_layout_boundary():
    with pytest.raises(OmniPrefixCacheUnmatchError, match="negative scheduled token count"):
        PrefixCacheSchedulerAdapter().build_write_layout(FakeView(), num_scheduled_tokens={"b": -1})


def test_real_vllm_scheduler_contract():
    """Fails on upstream field drift; skipped only when vLLM is not installed."""
    sched = pytest.importorskip("vllm.v1.core.sched.output")
    adapter = PrefixCacheSchedulerAdapter()
    new = sched.NewRequestData(
        req_id="r",
        prompt_token_ids=list(range(12)),
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([2, 3, 4],),
        num_computed_tokens=8,
        lora_request=None,
    )

    def real_output(*, new_reqs=(), cached=None, scheduled=None, finished=()):
        counts = dict(scheduled or {})
        return sched.SchedulerOutput(
            scheduled_new_reqs=list(new_reqs),
            scheduled_cached_reqs=cached if cached is not None else sched.CachedRequestData.make_empty(),
            num_scheduled_tokens=counts,
            total_num_scheduled_tokens=sum(counts.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(finished),
            free_encoder_mm_hashes=[],
        )

    step = adapter.translate_step(real_output(new_reqs=[new], scheduled={"r": 4}))
    assert step.events == (
        PrefixCacheRequestEvent(
            "r", PrefixCacheEventKind.STARTED, hit_end=8, block_ids=((2, 3, 4),), scheduled_tokens=4
        ),
    )
    cached = sched.CachedRequestData(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[12],
        num_output_tokens=[1],
    )
    step = adapter.translate_step(real_output(cached=cached, scheduled={"r": 1}))
    assert step.events == () and step.extended_req_ids == ("r",)
    cached.resumed_req_ids = {"r"}
    cached.num_computed_tokens = [8]
    cached.new_block_ids = [([5, 6, 7],)]
    step = adapter.translate_step(real_output(cached=cached, scheduled={"r": 1}))
    assert step.events[0].kind is PrefixCacheEventKind.RESUMED
    assert step.events[0].block_ids == ((5, 6, 7),) and step.events[0].hit_end == 8
    step = adapter.translate_step(real_output(new_reqs=[new], scheduled={"r": 4}, finished={"r"}))
    assert [e.kind for e in step.events] == [PrefixCacheEventKind.FINISHED, PrefixCacheEventKind.STARTED]
    step = adapter.translate_step(real_output(cached=cached, scheduled={"r": 1}))
    assert step.events[0].kind is PrefixCacheEventKind.RESUMED


def test_extended_ids_are_deduplicated_immutable_and_need_no_payload():
    class UnusedBlocks:
        def __bool__(self):
            raise AssertionError("continuation block table must not be inspected")

    adapter = PrefixCacheSchedulerAdapter()
    adapter.translate_step(output(new=[SimpleNamespace(req_id="r")]))
    data = SimpleNamespace(req_id="r", num_computed_tokens=8, block_ids=UnusedBlocks())
    scheduler_output = output(new=[data, data], scheduled={"r": 4})
    step = adapter.translate_step(scheduler_output)
    scheduler_output.scheduled_new_reqs.clear()
    scheduler_output.num_scheduled_tokens["r"] = 99
    assert step.events == ()
    assert step.extended_req_ids == ("r",)
    assert step.scheduled_tokens == (("r", 4),)
    with pytest.raises(AttributeError):
        step.extended_req_ids = ()


@pytest.mark.parametrize("count", [-1, -512])
@pytest.mark.parametrize("hit_end", [0, 8])
def test_negative_scheduled_tokens_fail_at_step_boundary(count, hit_end):
    scheduler_output = output(
        new=[SimpleNamespace(req_id="r", num_computed_tokens=hit_end, block_ids=[[0, 1]])],
        scheduled={"r": count},
    )
    with pytest.raises(OmniPrefixCacheUnmatchError, match=f"negative scheduled token count for req r: {count}"):
        PrefixCacheSchedulerAdapter().translate_step(scheduler_output)
