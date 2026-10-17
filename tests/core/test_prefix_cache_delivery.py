# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Position-based cache recovery and local output-builder delivery."""

import pytest
import torch

from tests.core.test_prefix_cache import HIDDEN, make_manager, plan_fetch
from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheEventKind as Kind,
)
from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheRequestEvent as Event,
)
from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheSchedulerAdapter,
    PrefixCacheStep,
)
from vllm_omni.core.prefix_cache.interface import HIDDEN_KEY

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def step(mgr, view, req_id, start, end, blocks, *, kind=None, hit=0, finished=(), mm=None):
    view.order = [req_id]
    view.req_blocks[req_id] = blocks
    view.computed[req_id] = start
    events = [Event(r, Kind.FINISHED) for r in finished]
    if kind is not None:
        events.append(Event(req_id, kind, hit_end=hit, block_ids=(tuple(blocks),)))
    mgr.new_step_starts(PrefixCacheStep(tuple(events), ((req_id, end - start),)))
    layout = PrefixCacheSchedulerAdapter().build_write_layout(view, num_scheduled_tokens={req_id: end - start})
    rows = torch.arange(start, end, dtype=torch.float32).unsqueeze(1).expand(-1, HIDDEN).clone()
    return mgr.save_outputs(
        rows, mm or {}, num_tokens_unpadded=end - start, num_tokens_padded=end - start, write_layout=layout
    )


def consume(mgr, sid, req_id="a"):
    raw = mgr.materialize(sid, [req_id])
    delivery = mgr.delivery_view(raw, [req_id])
    mgr.ack_delivery(delivery)
    return delivery


def positions(out, req_id="a"):
    return out.hidden_states[req_id][:, 0].tolist()


def test_resume_growing_hit_recovers_only_undelivered_positions():
    mgr, view = make_manager()
    consume(mgr, step(mgr, view, "a", 0, 6, [0, 1], kind=Kind.STARTED))
    consume(mgr, step(mgr, view, "peer", 0, 12, [0, 1, 2], kind=Kind.STARTED), "peer")
    sid = step(mgr, view, "a", 12, 14, [0, 1, 2, 3], kind=Kind.RESUMED, hit=12)
    out = consume(mgr, sid)
    assert positions(out) == list(range(6, 14))
    assert out.token_ranges["a"] == (6, 14)


def test_resume_shrinking_hit_saves_replay_but_delivers_only_new_suffix():
    mgr, view = make_manager()
    first = consume(mgr, step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED))
    replay = consume(mgr, step(mgr, view, "a", 4, 6, [0, 3, 4], kind=Kind.RESUMED, hit=4))
    assert positions(replay) == []
    progress = mgr._request_progress["a"]
    assert progress.delivered_upto == 8
    tail = consume(mgr, step(mgr, view, "a", 6, 10, [0, 3, 4]))
    assert positions(first) + positions(replay) + positions(tail) == list(range(10))
    saved = plan_fetch(mgr, view.slots_for("a", 4, 10), HIDDEN_KEY, req_id="a")
    assert saved[:, 0].tolist() == list(range(4, 10))


def test_materialization_without_handoff_does_not_advance_delivery():
    mgr, view = make_manager()
    sid = step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED)
    raw = mgr.materialize(sid, ["a"])
    first = mgr.delivery_view(raw, ["a"])
    retry = mgr.delivery_view(raw, ["a"])
    assert positions(first) == positions(retry) == list(range(8))
    assert not mgr._request_progress["a"].delivered_upto
    mgr.ack_delivery(first)
    assert mgr._request_progress["a"].delivered_upto == 8


def test_discard_is_not_delivery_and_resumed_hit_recovers_it():
    mgr, view = make_manager()
    mgr.discard_step(step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED))
    out = consume(mgr, step(mgr, view, "a", 8, 10, [0, 1, 2], kind=Kind.RESUMED, hit=8))
    assert positions(out) == list(range(10))


def test_late_delivery_uses_old_request_progress_after_id_reuse():
    mgr, view = make_manager()
    raw = mgr.materialize(step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED), ["a"])
    old = mgr.delivery_view(raw, ["a"])
    sid = step(mgr, view, "a", 0, 2, [2], kind=Kind.STARTED, finished=["a"])
    mgr.ack_delivery(old)
    assert not mgr._request_progress["a"].delivered_upto
    assert positions(consume(mgr, sid)) == [0, 1]


def test_resume_snapshots_previous_step_handoff():
    mgr, view = make_manager(staging_depth=3)
    prior = mgr.materialize(step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED), ["a"])
    mgr.ack_delivery(mgr.delivery_view(prior, ["a"]))
    sid = step(mgr, view, "a", 4, 10, [0, 1, 2], kind=Kind.RESUMED, hit=4)
    assert positions(consume(mgr, sid)) == [8, 9]


def test_extension_keeps_delivery_progress():
    mgr, view = make_manager()
    consume(mgr, step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED))
    out = consume(mgr, step(mgr, view, "a", 8, 10, [0, 1, 2], kind=Kind.EXTENDED, hit=8))
    assert positions(out) == [8, 9]


def test_cached_mm_is_clipped_but_global_metadata_is_preserved():
    mgr, view = make_manager()

    def codes(start, end):
        return torch.arange(start, end, dtype=torch.float32).reshape(-1, 1)

    consume(mgr, step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED, mm={"codes": codes(0, 8)}))
    raw = mgr.materialize(
        step(
            mgr,
            view,
            "a",
            4,
            10,
            [0, 1, 2],
            kind=Kind.RESUMED,
            hit=4,
            mm={"codes": codes(4, 10), "metadata": torch.tensor([7])},
        ),
        ["a"],
    )
    out = mgr.delivery_view(raw, ["a"])
    assert out.mm_outputs["codes"]["a"][:, 0].tolist() == [8, 9]
    assert out.mm_outputs["metadata"]["a"].item() == 7


def test_delivery_clips_current_only_fields_relative_to_scheduled_origin():
    from vllm_omni.core.prefix_cache.interface import PrefixCacheRequestProgress, StageCacheOutputs

    mgr, _ = make_manager()
    progress = PrefixCacheRequestProgress(delivered_upto=13)
    raw = StageCacheOutputs(
        hidden_states={"a": torch.arange(6, 14).reshape(-1, 1)},
        mm_outputs={
            "cached": {"a": torch.arange(6, 14).reshape(-1, 1)},
            "current": {"a": torch.tensor([[12], [13]])},
            "global": {"a": torch.tensor([[90], [91]])},
        },
        token_ranges={"a": (6, 14)},
        scheduled_token_ranges={"a": (12, 14)},
        cached_mm_keys=frozenset({"cached"}),
        token_mm_keys=frozenset({"cached", "current"}),
        _progress={"a": progress},
        delivery_starts={"a": 13},
    )
    view = mgr.delivery_view(raw, ["a"])
    assert view.mm_outputs["cached"]["a"].tolist() == [[13]]
    assert view.mm_outputs["current"]["a"].tolist() == [[13]]
    assert view.mm_outputs["global"]["a"].tolist() == [[90], [91]]


def test_delivery_acknowledges_only_selected_requests():
    from vllm_omni.core.prefix_cache.interface import PrefixCacheRequestProgress, StageCacheOutputs

    mgr, _ = make_manager()
    a, b = PrefixCacheRequestProgress(), PrefixCacheRequestProgress()
    raw = StageCacheOutputs(
        hidden_states={"a": torch.zeros(2, 1), "b": torch.zeros(2, 1)},
        mm_outputs={},
        token_ranges={"a": (0, 2), "b": (0, 2)},
        scheduled_token_ranges={"a": (0, 2), "b": (0, 2)},
        _progress={"a": a, "b": b},
        delivery_starts={"a": 0, "b": 0},
    )
    mgr.ack_delivery(mgr.delivery_view(raw, ["b"]))
    assert a.delivered_upto == 0
    assert b.delivered_upto == 2


@pytest.mark.parametrize("terminal", [Kind.FINISHED, Kind.ABORTED])
def test_terminal_event_releases_progress_without_losing_pending_output(terminal):
    mgr, view = make_manager()
    sid = step(mgr, view, "a", 0, 4, [0], kind=Kind.STARTED)
    mgr.new_step_starts(PrefixCacheStep((Event("a", terminal),), ()))
    assert "a" not in mgr._request_progress
    assert positions(consume(mgr, sid)) == list(range(4))


def test_repeated_resume_preserves_deferred_mm_positions():
    from vllm_omni.core.prefix_cache.interface import ModelCachePolicy

    mgr, view = make_manager(policy=ModelCachePolicy(deferred_keys=frozenset({"codes"})))
    emitted = []
    for start, end, kind, hit in [(0, 8, Kind.STARTED, 0), (4, 10, Kind.RESUMED, 4), (8, 12, Kind.RESUMED, 8)]:
        sid = step(
            mgr,
            view,
            "a",
            start,
            end,
            [0, 1, 2],
            kind=kind,
            hit=hit,
            mm={"codes": torch.arange(start, end, dtype=torch.float32).reshape(-1, 1)},
        )
        emitted.extend(consume(mgr, sid).mm_outputs["codes"]["a"][:, 0].tolist())
    assert emitted == list(range(12))
    mgr.new_step_starts(PrefixCacheStep((Event("a", Kind.FINISHED),), ()))
    saved = plan_fetch(mgr, view.slots_for("a", 0, 12), "codes", req_id="a")
    assert saved[:, 0].tolist() == list(range(12))


def test_full_payload_recovery_uses_emitted_per_key_boundary():
    mgr, view = make_manager()
    raw = mgr.materialize(step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED), ["a"])
    ends = {"hidden": 8, "codes": 4}
    mgr.record_full_payload_delivery(raw, "a", min(ends.values()))
    assert raw._progress["a"].delivered_upto == 4
    sid = step(mgr, view, "a", 8, 10, [0, 1, 2], kind=Kind.RESUMED, hit=8)
    recovered = mgr.materialize(sid, ["a"])
    assert recovered.token_ranges["a"] == (4, 10)
    ends["codes"] = 8
    assert raw._progress["a"].delivered_upto == 4


def test_late_full_payload_ack_preserves_new_id_progress():
    mgr, view = make_manager()
    raw = mgr.materialize(step(mgr, view, "a", 0, 4, [0], kind=Kind.STARTED), ["a"])
    sid = step(mgr, view, "a", 0, 2, [2], kind=Kind.STARTED, finished=["a"])
    mgr.record_full_payload_delivery(raw, "a", 4)
    assert mgr._request_progress["a"].delivered_upto == 0
    mgr.discard_step(sid)


def test_out_of_order_output_builders_preserve_both_step_ranges():
    mgr, view = make_manager(staging_depth=3)
    first = step(mgr, view, "a", 0, 4, [0, 1], kind=Kind.STARTED)
    second = step(mgr, view, "a", 4, 8, [0, 1])
    later = consume(mgr, second)
    earlier = consume(mgr, first)
    assert positions(earlier) == [0, 1, 2, 3]
    assert positions(later) == [4, 5, 6, 7]
    assert mgr._request_progress["a"].delivered_upto == 8
