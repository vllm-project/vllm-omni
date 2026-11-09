# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model Runner V2 prefix cache: adapter block ledger, delivery watermark, view."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.core.prefix_cache.adapter import PrefixCacheDelivery, PrefixCacheSchedulerAdapter
from vllm_omni.core.prefix_cache.group_view import MRv2FullAttentionGroupView, check_mrv2_prefix_cache_layout
from vllm_omni.core.prefix_cache.interface import HIDDEN_KEY, OmniPrefixCacheUnmatchError, PrefixCacheConfig
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_BLOCKS = 16
BLOCK_SIZE = 4
HIDDEN = 8
DTYPE = torch.float32


def sched_output(*, new=(), cached=(), resumed=(), finished=(), preempted=(), scheduled=None):
    """new: (req_id, num_computed, blocks[, session_id]); cached: (req_id, num_computed, new_blocks | None)."""
    new_reqs = []
    for entry in new:
        r, c, b = entry[0], entry[1], entry[2]
        ns = SimpleNamespace(req_id=r, num_computed_tokens=c, block_ids=[list(b)])
        if len(entry) > 3 and entry[3] is not None:
            ns.session_id = entry[3]
        new_reqs.append(ns)
    return SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=SimpleNamespace(
            resumed_req_ids=set(resumed),
            req_ids=[r for r, _, _ in cached],
            num_computed_tokens=[c for _, c, _ in cached],
            new_block_ids=[None if b is None else [list(b)] for _, _, b in cached],
            num_output_tokens=[],
        ),
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
        num_scheduled_tokens=dict(scheduled or {}),
    )


def input_batch(rows):
    """rows: (req_id, num_computed, num_scheduled) in batch order."""
    return SimpleNamespace(
        req_ids=[r for r, _, _ in rows],
        num_reqs=len(rows),
        num_scheduled_tokens=np.array([n for _, _, n in rows], dtype=np.int32),
        num_computed_tokens_np=np.array([c for _, c, _ in rows], dtype=np.int32),
    )


def rows_for(slots):
    return torch.tensor(slots, dtype=DTYPE).unsqueeze(1).expand(len(slots), HIDDEN)


class Harness:
    """One MRv2 stage: tracking adapter, ledger view and manager."""

    def __init__(self):
        cfg = PrefixCacheConfig(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
        self.mgr = OmniPrefixCacheManager(cfg, eager=True)
        self.adapter = PrefixCacheSchedulerAdapter(track_requests=True)
        self.view = MRv2FullAttentionGroupView(self.adapter, BLOCK_SIZE)

    def step(self, rows=(), **sched):
        """Translate, begin, and save one step; each row's hidden equals its slot id."""
        sched.setdefault("scheduled", {r: n for r, _, n in rows})
        step = self.adapter.translate_step(sched_output(**sched))
        self.mgr.new_step_starts(step)
        self.view.bind(input_batch(rows))
        try:
            layout = self.adapter.build_write_layout(self.view, num_scheduled_tokens=dict(step.scheduled_tokens))
        finally:
            self.view.bind(None)
        n = layout.total_rows
        hidden = layout.slots_cpu.to(DTYPE).unsqueeze(1).expand(n, HIDDEN).clone()
        step_id = self.mgr.save_outputs(hidden, {}, num_tokens_unpadded=n, num_tokens_padded=n, write_layout=layout)
        return step, step_id

    def close(self):
        self.mgr.shutdown()


@pytest.fixture
def harness():
    h = Harness()
    try:
        yield h
    finally:
        h.close()


def test_ledger_follows_scheduler_blocks():
    adapter = PrefixCacheSchedulerAdapter(track_requests=True)
    adapter.translate_step(sched_output(new=[("a", 0, [3, 5])], scheduled={"a": 8}))
    assert adapter.block_ids("a") == [3, 5]
    adapter.translate_step(sched_output(cached=[("a", 8, [7])], scheduled={"a": 1}))
    assert adapter.block_ids("a") == [3, 5, 7]
    adapter.translate_step(sched_output(cached=[("a", 9, None)], scheduled={"a": 1}))
    assert adapter.block_ids("a") == [3, 5, 7]

    adapter.translate_step(sched_output(preempted=["a"]))
    with pytest.raises(OmniPrefixCacheUnmatchError, match="no block ids"):
        adapter.block_ids("a")
    adapter.translate_step(sched_output(new=[("a", 8, [1, 2, 4])], scheduled={"a": 2}))
    assert adapter.block_ids("a") == [1, 2, 4]

    adapter.translate_step(sched_output(cached=[("a", 2, [6])], resumed=["a"], scheduled={"a": 1}))
    assert adapter.block_ids("a") == [6]

    adapter.translate_step(sched_output(finished=["a"]))
    with pytest.raises(OmniPrefixCacheUnmatchError):
        adapter.block_ids("a")


def test_untracked_adapter_keeps_no_ledger():
    adapter = PrefixCacheSchedulerAdapter()
    step = adapter.translate_step(sched_output(new=[("a", 0, [3])], scheduled={"a": 4}))
    assert step.deliveries == ()
    with pytest.raises(OmniPrefixCacheUnmatchError):
        adapter.block_ids("a")


def test_delivery_watermark_trims_overlap_and_replays_gap():
    adapter = PrefixCacheSchedulerAdapter(track_requests=True)
    assert adapter.translate_step(sched_output(new=[("a", 0, [0, 1])], scheduled={"a": 8})).deliveries == ()
    assert adapter.translate_step(sched_output(cached=[("a", 8, [2])], scheduled={"a": 1})).deliveries == ()

    adapter.translate_step(sched_output(preempted=["a"]))
    step = adapter.translate_step(sched_output(new=[("a", 4, [0, 1, 2])], scheduled={"a": 6}))
    assert step.deliveries == (PrefixCacheDelivery("a", trim=5),)
    assert step.extended_req_ids == ("a",)

    adapter.translate_step(sched_output(preempted=["a"]))
    step = adapter.translate_step(sched_output(new=[("a", 12, [0, 1, 2, 3])], scheduled={"a": 2}))
    assert step.deliveries == (PrefixCacheDelivery("a", read_start=10, read_end=12, block_ids=((0, 1, 2, 3),)),)

    # In-order steps produce nothing; a finished id starts over.
    assert adapter.translate_step(sched_output(cached=[("a", 14, None)], scheduled={"a": 1})).deliveries == ()
    adapter.translate_step(sched_output(finished=["a"]))
    assert adapter.translate_step(sched_output(new=[("a", 4, [0, 1])], scheduled={"a": 4})).deliveries == ()


def test_view_derives_slots_from_ledger_and_batch():
    adapter = PrefixCacheSchedulerAdapter(track_requests=True)
    adapter.translate_step(sched_output(new=[("a", 0, [3, 5]), ("b", 0, [9])], scheduled={"a": 6, "b": 2}))
    view = MRv2FullAttentionGroupView(adapter, BLOCK_SIZE)
    view.bind(input_batch([("b", 1, 2), ("a", 2, 5)]))
    assert view.batch_req_ids() == ["b", "a"]
    slots = view.step_slots_cpu(["b", "a"], {"b": 2, "a": 5})
    assert slots.tolist() == [37, 38, 14, 15, 20, 21, 22]


def test_view_refuses_disagreeing_or_unbound_state():
    adapter = PrefixCacheSchedulerAdapter(track_requests=True)
    adapter.translate_step(sched_output(new=[("a", 0, [3])], scheduled={"a": 4}))
    view = MRv2FullAttentionGroupView(adapter, BLOCK_SIZE)
    with pytest.raises(OmniPrefixCacheUnmatchError, match="no bound input batch"):
        view.batch_req_ids()
    view.bind(input_batch([("a", 0, 4)]))
    with pytest.raises(OmniPrefixCacheUnmatchError, match="disagrees"):
        view.step_slots_cpu(["a"], {"a": 3})
    view.bind(input_batch([("a", 2, 4)]))
    with pytest.raises(OmniPrefixCacheUnmatchError, match="past its 1 recorded blocks"):
        view.step_slots_cpu(["a"], {"a": 4})


def test_mrv2_layout_check():
    check_mrv2_prefix_cache_layout(block_sizes=[16], block_size=16, dcp_size=1)
    with pytest.raises(OmniPrefixCacheUnmatchError, match="decode context parallel"):
        check_mrv2_prefix_cache_layout(block_sizes=[16], block_size=16, dcp_size=2)
    with pytest.raises(OmniPrefixCacheUnmatchError, match="does not match"):
        check_mrv2_prefix_cache_layout(block_sizes=[32], block_size=16, dcp_size=1)


def test_miss_steps_need_no_rewrite(harness):
    _, s1 = harness.step([("a", 0, 8)], new=[("a", 0, [0, 1])])
    assert harness.mgr.step_rewrite_req_ids(s1) == frozenset()
    # Querying does not consume the step id.
    assert torch.equal(harness.mgr.materialize(s1, ["a"]).hidden_states["a"], rows_for(range(8)))
    with pytest.raises(OmniPrefixCacheUnmatchError, match="missing"):
        harness.mgr.step_rewrite_req_ids(s1)


def test_new_request_hit_is_rewritten(harness):
    harness.mgr.discard_step(harness.step([("a", 0, 8)], new=[("a", 0, [0, 1])])[1])
    harness.mgr.discard_step(harness.step(finished=["a"])[1])
    _, s3 = harness.step([("b", 8, 2)], new=[("b", 8, [0, 1, 2])])
    assert harness.mgr.step_rewrite_req_ids(s3) == {"b"}
    out = harness.mgr.materialize(s3, ["b"], include_uncached=False)
    assert torch.equal(out.hidden_states["b"], rows_for(range(10)))


def test_resume_below_watermark_trims_delivered_rows(harness):
    harness.mgr.discard_step(harness.step([("a", 0, 8)], new=[("a", 0, [0, 1])])[1])
    harness.mgr.discard_step(harness.step([("a", 8, 1)], cached=[("a", 8, [2])])[1])
    harness.mgr.discard_step(harness.step(preempted=["a"])[1])

    step, s4 = harness.step([("a", 8, 2)], new=[("a", 8, [0, 1, 2])])
    assert step.deliveries == (PrefixCacheDelivery("a", trim=1),)
    assert harness.mgr.step_rewrite_req_ids(s4) == {"a"}
    out = harness.mgr.materialize(s4, ["a"], include_uncached=False)
    assert torch.equal(out.hidden_states["a"], rows_for([9]))


def test_resume_above_watermark_replays_gap(harness):
    harness.mgr.discard_step(harness.step([("a", 0, 4)], new=[("a", 0, [0, 1])])[1])
    harness.mgr.discard_step(harness.step(preempted=["a"])[1])
    # Another request fills the shared prefix blocks, then leaves.
    harness.mgr.discard_step(harness.step([("b", 0, 8)], new=[("b", 0, [0, 1])])[1])
    harness.mgr.discard_step(harness.step(finished=["b"])[1])

    step, s5 = harness.step([("a", 8, 2)], new=[("a", 8, [0, 1, 2])])
    assert step.deliveries == (PrefixCacheDelivery("a", read_start=4, read_end=8, block_ids=((0, 1, 2),)),)
    assert harness.mgr.step_rewrite_req_ids(s5) == {"a"}
    out = harness.mgr.materialize(s5, ["a"], include_uncached=False)
    assert torch.equal(out.hidden_states["a"], rows_for(range(4, 10)))


def test_session_replay_hit_serves_only_requested_range(harness):
    """A5/B1: a session-served replay [16, 32) serves exactly its span.

    The session store covers [0, 32), but the downstream stage already
    has [0, 16) (the delivered watermark): the merged output must be
    the 16 replay rows plus this step's 4 rows — not the 32-row head
    prepended again. Decode steps do not advance the adapter's
    delivered watermark, which is what leaves the replay gap.
    """
    harness.mgr.discard_step(harness.step([("a", 0, 16)], new=[("a", 0, [0, 1, 2, 3, 4, 5, 6, 7], "S")])[1])
    harness.mgr.discard_step(harness.step([("a", 16, 16)])[1])
    assert harness.mgr._session_store.stored_tokens("S", HIDDEN_KEY) == 32
    harness.mgr.discard_step(harness.step(preempted=["a"])[1])

    step, s4 = harness.step([("a", 32, 4)], new=[("a", 32, [0, 1, 2, 3, 4, 5, 6, 7, 8])])
    assert step.deliveries == (
        PrefixCacheDelivery("a", read_start=16, read_end=32, block_ids=((0, 1, 2, 3, 4, 5, 6, 7, 8),)),
    )
    out = harness.mgr.materialize(s4, ["a"], include_uncached=False)
    assert torch.equal(out.hidden_states["a"], rows_for(range(16, 36)))
    # The step's own rows still append at their true position.
    assert harness.mgr._session_store.stored_tokens("S", HIDDEN_KEY) == 36
