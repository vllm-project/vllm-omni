# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Position-based cache recovery and scheduler-owned delivery."""

from types import SimpleNamespace
from typing import NamedTuple

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
from vllm_omni.utils.mm_outputs import slice_token_payload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class SavedStep(NamedTuple):
    step_id: int
    token_range: tuple[int, int]
    delivered_upto: int


def step(mgr, view, req_id, start, end, blocks, *, kind=None, hit=0, delivered=0, finished=(), mm=None):
    view.order = [req_id]
    view.req_blocks[req_id] = blocks
    view.computed[req_id] = start
    events = [Event(r, Kind.FINISHED) for r in finished]
    if kind is not None:
        events.append(
            Event(
                req_id,
                kind,
                hit_end=hit,
                block_ids=(tuple(blocks),),
                delivered_upto=delivered,
            )
        )
    mgr.new_step_starts(PrefixCacheStep(tuple(events), ((req_id, end - start),)))
    layout = PrefixCacheSchedulerAdapter().build_write_layout(view, num_scheduled_tokens={req_id: end - start})
    rows = torch.arange(start, end, dtype=torch.float32).unsqueeze(1).expand(-1, HIDDEN).clone()
    sid = mgr.save_outputs(
        rows, mm or {}, num_tokens_unpadded=end - start, num_tokens_padded=end - start, write_layout=layout
    )
    return SavedStep(sid, (start, end), delivered)


def consume(mgr, saved_step, req_id="a"):
    raw = mgr.materialize(saved_step.step_id, [req_id])
    payload = {key: values[req_id] for key, values in raw.mm_outputs.items()}
    if raw.hidden_states is not None:
        payload["hidden"] = raw.hidden_states[req_id]
    return slice_token_payload(
        payload,
        token_range=saved_step.token_range,
        delivered_upto=saved_step.delivered_upto,
        token_keys=raw.token_mm_keys | {"hidden"},
        cached_token_range=raw.token_ranges[req_id],
        cached_keys=raw.cached_mm_keys | {"hidden"},
    )


def positions(out):
    return out["hidden"][:, 0].tolist() if "hidden" in out else []


def test_resume_uses_scheduler_boundary_without_runner_ack():
    mgr, view = make_manager()
    consume(mgr, step(mgr, view, "a", 0, 12, [0, 1, 2], kind=Kind.STARTED))
    sid = step(mgr, view, "a", 12, 14, [0, 1, 2, 3], kind=Kind.RESUMED, hit=12, delivered=6)
    assert positions(consume(mgr, sid)) == list(range(6, 14))


def test_resume_growing_hit_recovers_only_undelivered_positions():
    mgr, view = make_manager()
    consume(mgr, step(mgr, view, "a", 0, 6, [0, 1], kind=Kind.STARTED))
    consume(mgr, step(mgr, view, "peer", 0, 12, [0, 1, 2], kind=Kind.STARTED), "peer")
    sid = step(mgr, view, "a", 12, 14, [0, 1, 2, 3], kind=Kind.RESUMED, hit=12, delivered=6)
    out = consume(mgr, sid)
    assert positions(out) == list(range(6, 14))


def test_resume_shrinking_hit_saves_replay_but_delivers_only_new_suffix():
    mgr, view = make_manager()
    first = consume(mgr, step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED))
    replay = consume(mgr, step(mgr, view, "a", 4, 6, [0, 3, 4], kind=Kind.RESUMED, hit=4, delivered=8))
    assert positions(replay) == []
    tail = consume(mgr, step(mgr, view, "a", 6, 10, [0, 3, 4], delivered=8))
    assert positions(first) + positions(replay) + positions(tail) == list(range(10))
    saved = plan_fetch(mgr, view.slots_for("a", 4, 10), HIDDEN_KEY, req_id="a")
    assert saved[:, 0].tolist() == list(range(4, 10))


def test_discard_is_not_delivery_and_resumed_hit_recovers_it():
    mgr, view = make_manager()
    mgr.discard_step(step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED).step_id)
    out = consume(mgr, step(mgr, view, "a", 8, 10, [0, 1, 2], kind=Kind.RESUMED, hit=8))
    assert positions(out) == list(range(10))


def test_late_delivery_uses_old_request_progress_after_id_reuse():
    mgr, view = make_manager(staging_depth=3)
    consume(mgr, step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED))
    old = step(mgr, view, "a", 8, 10, [0, 1, 2], delivered=8)
    new = step(mgr, view, "a", 0, 2, [3], kind=Kind.STARTED, finished=["a"])
    assert positions(consume(mgr, old)) == [8, 9]
    assert positions(consume(mgr, new)) == [0, 1]


def test_resume_snapshots_previous_step_handoff():
    mgr, view = make_manager(staging_depth=3)
    prior = consume(mgr, step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED))
    assert positions(prior) == list(range(8))
    sid = step(mgr, view, "a", 4, 10, [0, 1, 2], kind=Kind.RESUMED, hit=4, delivered=8)
    assert positions(consume(mgr, sid)) == [8, 9]


def test_extension_keeps_delivery_progress():
    mgr, view = make_manager()
    consume(mgr, step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED))
    out = consume(mgr, step(mgr, view, "a", 8, 10, [0, 1, 2], kind=Kind.EXTENDED, hit=8, delivered=8))
    assert positions(out) == [8, 9]


def test_cached_mm_is_clipped_but_global_metadata_is_preserved():
    mgr, view = make_manager()

    def codes(start, end):
        return torch.arange(start, end, dtype=torch.float32).reshape(-1, 1)

    consume(mgr, step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED, mm={"codes": codes(0, 8)}))
    out = consume(
        mgr,
        step(
            mgr,
            view,
            "a",
            4,
            10,
            [0, 1, 2],
            kind=Kind.RESUMED,
            hit=4,
            delivered=8,
            mm={"codes": codes(4, 10), "metadata": torch.tensor([7])},
        ),
    )
    assert out["codes"][:, 0].tolist() == [8, 9]
    assert out["metadata"].item() == 7


def test_delivery_clips_current_only_fields_relative_to_scheduled_origin():
    out = slice_token_payload(
        {
            "hidden": torch.arange(6, 14).reshape(-1, 1),
            "cached": torch.arange(6, 14).reshape(-1, 1),
            "current": torch.tensor([[12], [13]]),
            "global": torch.tensor([[90], [91]]),
        },
        token_range=(12, 14),
        delivered_upto=13,
        token_keys=frozenset({"hidden", "cached", "current"}),
        cached_token_range=(6, 14),
        cached_keys=frozenset({"hidden", "cached"}),
    )
    assert out["hidden"].tolist() == [[13]]
    assert out["cached"].tolist() == [[13]]
    assert out["current"].tolist() == [[13]]
    assert out["global"].tolist() == [[90], [91]]


@pytest.mark.parametrize("terminal", [Kind.FINISHED, Kind.ABORTED])
def test_terminal_event_preserves_pending_output_snapshot(terminal):
    mgr, view = make_manager()
    sid = step(mgr, view, "a", 0, 4, [0], kind=Kind.STARTED)
    mgr.new_step_starts(PrefixCacheStep((Event("a", terminal),), ()))
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
            delivered=len(emitted),
            mm={"codes": torch.arange(start, end, dtype=torch.float32).reshape(-1, 1)},
        )
        emitted.extend(consume(mgr, sid)["codes"][:, 0].tolist())
    assert emitted == list(range(12))
    mgr.new_step_starts(PrefixCacheStep((Event("a", Kind.FINISHED),), ()))
    saved = plan_fetch(mgr, view.slots_for("a", 0, 12), "codes", req_id="a")
    assert saved[:, 0].tolist() == list(range(12))


def test_out_of_order_output_builders_preserve_both_step_ranges():
    mgr, view = make_manager(staging_depth=3)
    first = step(mgr, view, "a", 0, 4, [0, 1], kind=Kind.STARTED)
    second = step(mgr, view, "a", 4, 8, [0, 1], delivered=4)
    later = consume(mgr, second)
    earlier = consume(mgr, first)
    assert positions(earlier) == [0, 1, 2, 3]
    assert positions(later) == [4, 5, 6, 7]


def test_prompt_replacement_resets_resume_floor():
    mgr, view = make_manager()
    consume(mgr, step(mgr, view, "a", 0, 8, [0, 1], kind=Kind.STARTED))
    consume(mgr, step(mgr, view, "a", 4, 5, [0, 1], kind=Kind.RESUMED, hit=4, delivered=8))
    sched = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="a", num_computed_tokens=0, block_ids=([2],))],
        scheduled_cached_reqs=None,
        num_scheduled_tokens={"a": 2},
        replaced_req_ids={"a"},
    )
    adapter = PrefixCacheSchedulerAdapter()
    adapter._observed_req_ids.add("a")
    event = adapter.translate_step(sched).events[0]
    assert event.kind is Kind.REPLACED
    sid = step(mgr, view, "a", 0, 2, [2], kind=event.kind)
    assert positions(consume(mgr, sid)) == [0, 1]
