# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eager prefix-cache save/materialize behavior used by Ascend backends.

The facade runs without CUDA streams and preserves byte-identical rows for
misses and hit merges. V1 runner-specific tuple wiring is retired.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.core.prefix_cache.adapter import PrefixCacheSchedulerAdapter
from vllm_omni.core.prefix_cache.interface import PrefixCacheConfig
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


NUM_BLOCKS = 16
BLOCK_SIZE = 4
HIDDEN = 8
DTYPE = torch.float32


class _FakeView:
    """Minimal group-view double (mirrors tests/core/test_prefix_cache)."""

    def __init__(self):
        self.block_size = BLOCK_SIZE
        self.num_blocks = NUM_BLOCKS
        self.req_blocks: dict[str, list[int]] = {}
        self.order: list[str] = []
        self.computed: dict[str, int] = {}

    def slots_for(self, req_id, token_start, token_end):
        blocks = self.req_blocks[req_id]
        return torch.tensor(
            [blocks[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE for p in range(token_start, token_end)],
            dtype=torch.long,
        )

    def batch_req_ids(self) -> list[str]:
        return list(self.order)

    def step_slots_cpu(self, req_ids, num_scheduled) -> torch.Tensor:
        parts = []
        for r in req_ids:
            n = int(num_scheduled.get(r, 0))
            if n > 0:
                start = self.computed.get(r, 0)
                parts.append(self.slots_for(r, start, start + n))
        return torch.cat(parts) if parts else torch.empty((0,), dtype=torch.long)


def _run_step(mgr, view, req_id, blocks, start_pos, sched, *, hit=0, finished=()):
    """One NPU-shaped step: new_step_starts -> save_outputs -> sid."""
    view.order = [req_id]
    view.req_blocks[req_id] = blocks
    view.computed[req_id] = start_pos
    slots = view.slots_for(req_id, start_pos, start_pos + sched)
    hidden = slots.to(DTYPE).unsqueeze(1).expand(sched, HIDDEN).clone()
    sched_out = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=req_id, num_computed_tokens=hit, block_ids=[list(blocks)]),
        ],
        finished_req_ids=set(finished),
        num_scheduled_tokens={req_id: sched},
    )
    adapter = getattr(mgr, "_test_adapter", None)
    if adapter is None:
        adapter = mgr._test_adapter = PrefixCacheSchedulerAdapter()
    step = adapter.translate_step(sched_out)
    mgr.new_step_starts(step)
    layout = adapter.build_write_layout(view, num_scheduled_tokens=dict(step.scheduled_tokens))
    return mgr.save_outputs(
        hidden,
        {},
        num_tokens_unpadded=sched,
        num_tokens_padded=sched,
        write_layout=layout,
    )


def _expected_rows(slots: torch.Tensor) -> torch.Tensor:
    return slots.to(DTYPE).unsqueeze(1).expand(slots.numel(), HIDDEN)


def _make_npu_mode_manager(monkeypatch) -> tuple[OmniPrefixCacheManager, _FakeView]:
    # The NPU condition: no CUDA -> the controller must auto-select eager
    # (submit() completes copy+scatter synchronously), exactly how
    # Ascend backends build the manager with the default eager=None.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    view = _FakeView()
    config = PrefixCacheConfig(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    return OmniPrefixCacheManager(config), view


def test_npu_mode_auto_selects_eager_and_roundtrips(monkeypatch):
    mgr, view = _make_npu_mode_manager(monkeypatch)
    assert mgr._controller._eager is True
    sid = _run_step(mgr, view, "a", [0, 1], 0, 8)
    outs = mgr.materialize(sid, ["a"])
    assert torch.equal(outs.hidden_states["a"], _expected_rows(view.slots_for("a", 0, 8)))


def test_npu_mode_hit_merges_cached_prefix(monkeypatch):
    """The hit-read path in eager mode: consumer's merged output must be
    [cached prefix rows + this step's rows], byte-identical."""
    mgr, view = _make_npu_mode_manager(monkeypatch)
    s1 = _run_step(mgr, view, "a", [0, 1], 0, 8)
    mgr.materialize(s1, ["a"])
    s2 = _run_step(mgr, view, "b", [0, 1, 2], 8, 4, hit=8, finished=["a"])
    merged = mgr.materialize(s2, ["b"]).hidden_states["b"]
    assert merged.shape == (12, HIDDEN)
    assert torch.equal(merged[:8], _expected_rows(view.slots_for("b", 0, 8)))
    assert torch.equal(merged[8:], _expected_rows(view.slots_for("b", 8, 12)))


def test_npu_mode_continuation_uses_adapter_observations(monkeypatch):
    mgr, view = _make_npu_mode_manager(monkeypatch)
    try:
        first = _run_step(mgr, view, "a", [0, 1], 0, 8)
        mgr.materialize(first, ["a"])
        second = _run_step(mgr, view, "a", [0, 1, 2], 8, 1, hit=8)
        assert not mgr._step_ctxs[second].hits
        rows = mgr.materialize(second, ["a"]).hidden_states["a"]
        assert torch.equal(rows, _expected_rows(view.slots_for("a", 8, 9)))
    finally:
        mgr.shutdown()
