# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for ``DTPSScheduler`` and ``DitLoadState``.

Covers:
- ``DitLoadState`` update / remove / snapshot round-trip
- ``DTPSScheduler.maybe_reorder_waiting`` 4-tier layering (L0→L1→L2→L3)
- Stale-snapshot fallback (no reorder when ``fresh=False``)
- Starving requests ordered by arrival_time (FCFS within L0)
- Blind in-flight eviction via ``evict_request``
- Corrupt-request fallback (classified as ar_only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from vllm_omni.core.sched.dit_load_state import DitLoadSnapshot, DitLoadState
from vllm_omni.core.sched.dtps_scheduler import DTPSScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Stub request
# ---------------------------------------------------------------------------


@dataclass
class _StubReq:
    request_id: str
    arrival_time: float = 0.0
    num_prompt_tokens: int = 0
    additional_information: Any = None


def _make_ar_only(req_id: str, *, arrival: float = 0.0, npt: int = 10) -> _StubReq:
    return _StubReq(
        request_id=req_id,
        arrival_time=arrival,
        num_prompt_tokens=npt,
        additional_information={"omni_final_stage_id": 0},
    )


def _make_downstream(req_id: str, *, arrival: float = 0.0, npt: int = 10) -> _StubReq:
    return _StubReq(
        request_id=req_id,
        arrival_time=arrival,
        num_prompt_tokens=npt,
        additional_information={"omni_final_stage_id": 1},
    )


def _fresh_snapshot(
    min_waiting: int = 0,
    num_replicas: int = 1,
    waiting_ids: frozenset[str] | None = None,
    running_ids: frozenset[str] | None = None,
) -> DitLoadSnapshot:
    return {
        "min_waiting": min_waiting,
        "max_waiting": min_waiting,
        "total_waiting": min_waiting,
        "total_running": 0,
        "num_replicas": num_replicas,
        "fresh": True,
        "waiting_ids": waiting_ids or frozenset(),
        "running_ids": running_ids or frozenset(),
    }


class _ListQueue:
    """Minimal list-backed queue with clear/extend for ``maybe_reorder_waiting``."""

    def __init__(self, reqs=None) -> None:
        self._reqs: list = list(reqs) if reqs else []

    def __iter__(self):
        return iter(self._reqs)

    def __len__(self) -> int:
        return len(self._reqs)

    def clear(self) -> None:
        self._reqs.clear()

    def extend(self, reqs) -> None:
        self._reqs.extend(reqs)

    @property
    def ids(self) -> list[str]:
        return [r.request_id for r in self._reqs]


# ---------------------------------------------------------------------------
# DitLoadState tests
# ---------------------------------------------------------------------------


class TestDitLoadState:
    def test_update_and_snapshot(self):
        state = DitLoadState()
        changed = state.update(1, 0, waiting=3, running=1, waiting_ids={"a", "b"}, running_ids={"c"})
        assert changed is True
        snap = state.snapshot()
        assert snap["fresh"] is True
        assert snap["min_waiting"] == 3
        assert snap["total_running"] == 1
        assert snap["num_replicas"] == 1
        assert snap["waiting_ids"] == frozenset({"a", "b"})

    def test_update_no_change(self):
        state = DitLoadState()
        state.update(1, 0, waiting=2, running=1)
        changed = state.update(1, 0, waiting=2, running=1)
        assert changed is False

    def test_remove(self):
        state = DitLoadState()
        state.update(1, 0, waiting=5, running=0)
        state.update(1, 1, waiting=0, running=2)
        state.remove(1, 1)
        snap = state.snapshot()
        assert snap["num_replicas"] == 1
        assert snap["min_waiting"] == 5

    def test_empty_snapshot(self):
        state = DitLoadState()
        snap = state.snapshot()
        assert snap["fresh"] is False
        assert snap["num_replicas"] == 0
        assert snap["min_waiting"] == 0

    def test_multi_replica_aggregate(self):
        state = DitLoadState()
        state.update(0, 0, waiting=4, running=1, waiting_ids={"x"})
        state.update(0, 1, waiting=1, running=0, waiting_ids={"y"})
        snap = state.snapshot()
        assert snap["min_waiting"] == 1
        assert snap["max_waiting"] == 4
        assert snap["total_waiting"] == 5
        assert snap["num_replicas"] == 2
        assert snap["waiting_ids"] == frozenset({"x", "y"})


# ---------------------------------------------------------------------------
# DTPSScheduler tests
# ---------------------------------------------------------------------------


class TestDTPSScheduler:
    def test_no_snapshot_no_reorder(self):
        """When no snapshot has been received, the queue stays in arrival order."""
        sched = DTPSScheduler(stage_id=0, dit_load_threshold=2)
        reqs = [
            _make_ar_only("a", arrival=1.0),
            _make_downstream("b", arrival=2.0),
        ]
        q = _ListQueue(reqs)
        sched.maybe_reorder_waiting(q, running=[])
        assert q.ids == ["a", "b"]

    def test_stale_snapshot_no_reorder(self):
        """When snapshot is stale (fresh=False), the queue stays in arrival order."""
        sched = DTPSScheduler(stage_id=0, dit_load_threshold=2)
        sched.update_dit_load(
            {
                "min_waiting": 0,
                "max_waiting": 0,
                "total_waiting": 0,
                "total_running": 0,
                "num_replicas": 1,
                "fresh": False,
                "waiting_ids": frozenset(),
                "running_ids": frozenset(),
            }
        )
        reqs = [_make_ar_only("a"), _make_downstream("b")]
        q = _ListQueue(reqs)
        sched.maybe_reorder_waiting(q, running=[])
        assert q.ids == ["a", "b"]

    def test_idle_dit_prioritizes_downstream(self):
        """DiT idle (min_waiting=0) → downstream requests admitted first."""
        sched = DTPSScheduler(stage_id=0, dit_load_threshold=2)
        sched.update_dit_load(_fresh_snapshot(min_waiting=0, num_replicas=1))
        reqs = [
            _make_ar_only("ar1", arrival=1.0),
            _make_downstream("ds1", arrival=2.0, npt=50),
            _make_downstream("ds2", arrival=3.0, npt=10),
        ]
        q = _ListQueue(reqs)
        sched.maybe_reorder_waiting(q, running=[])
        ids = q.ids
        assert ids[0] == "ds2"
        assert ids[1] == "ds1"
        assert "ar1" in ids[2:]

    def test_busy_dit_prioritizes_ar_only(self):
        """DiT busy (min_waiting >= threshold) → ar_only requests first."""
        sched = DTPSScheduler(stage_id=0, dit_load_threshold=2)
        sched.update_dit_load(_fresh_snapshot(min_waiting=3, num_replicas=1))
        reqs = [
            _make_downstream("ds1", arrival=1.0),
            _make_ar_only("ar1", arrival=2.0),
        ]
        q = _ListQueue(reqs)
        sched.maybe_reorder_waiting(q, running=[])
        assert q.ids[0] == "ar1"

    def test_starving_ordered_by_arrival(self):
        """L0 starving requests must be sorted by arrival_time (FCFS)."""
        sched = DTPSScheduler(stage_id=0, dit_load_threshold=2, aging_s=0.0)
        sched.update_dit_load(_fresh_snapshot(min_waiting=5, num_replicas=1))
        reqs = [
            _make_downstream("ds_late", arrival=10.0),
            _make_ar_only("ar_late", arrival=8.0),
            _make_ar_only("ar_early", arrival=5.0),
            _make_downstream("ds_early", arrival=3.0),
        ]
        q = _ListQueue(reqs)
        sched.maybe_reorder_waiting(q, running=[])
        ids = q.ids
        assert ids[0] == "ds_early"
        assert ids[1] == "ar_early"
        assert ids[2] == "ar_late"
        assert ids[3] == "ds_late"

    def test_evict_request_clears_inflight(self):
        """evict_request removes the id from the blind in-flight set."""
        sched = DTPSScheduler(stage_id=0, dit_load_threshold=2)
        sched.register_finished_downstream("req1")
        assert len(sched._dit_inflight_ids) == 1
        sched.evict_request("req1")
        assert len(sched._dit_inflight_ids) == 0

    def test_corrupt_request_classified_as_ar_only(self):
        """A request with un-deserializable additional_information falls back to ar_only."""
        sched = DTPSScheduler(stage_id=0, dit_load_threshold=2)

        class _BadBytes:
            def __iter__(self):
                raise TypeError("not iterable")

        req = _StubReq(
            request_id="bad",
            additional_information=_BadBytes(),
        )
        assert sched._request_is_ar_only(req) is True

    def test_no_additional_info_classified_as_ar_only(self):
        """A request with no additional_information falls back to ar_only."""
        sched = DTPSScheduler(stage_id=0, dit_load_threshold=2)
        req = _StubReq(request_id="nil", additional_information=None)
        assert sched._request_is_ar_only(req) is True

    def test_downstream_budget_cap(self):
        """Over-budget downstream requests are demoted below ar_only."""
        sched = DTPSScheduler(stage_id=0, dit_load_threshold=2)
        sched.update_dit_load(_fresh_snapshot(min_waiting=0, num_replicas=1))
        reqs = [
            _make_ar_only("ar1", arrival=1.0),
            _make_downstream("ds1", arrival=2.0, npt=10),
            _make_downstream("ds2", arrival=3.0, npt=20),
            _make_downstream("ds3", arrival=4.0, npt=30),
        ]
        q = _ListQueue(reqs)
        sched.maybe_reorder_waiting(q, running=[])
        ids = q.ids
        assert ids[0] == "ds1"
        assert ids[1] == "ds2"
        assert ids[2] == "ar1"
        assert ids[3] == "ds3"

    def test_aging_s_configurable(self):
        """aging_s can be overridden via constructor."""
        sched = DTPSScheduler(stage_id=0, dit_load_threshold=2, aging_s=50.0)
        assert sched.aging_s == 50.0
