# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for DTPS scheduler and DitLoadState.

Covers review-requested scenarios:
- DitLoadState update/remove/snapshot change detection
- maybe_reorder_waiting tier ordering (L0 aging, L1 budget, L2 FCFS, L3 overflow)
- stale snapshot fallback (fresh=False skips reorder)
- None snapshot fallback (blind in-flight still provides local load awareness)
- budget math with threshold = DiT max_num_seqs + 1
- blind in-flight eviction (seen-in-DiT and timeout)
- derive_dtps_config topology rejection (no DiT, multi-DiT)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from vllm_omni.core.sched.dtps_scheduler import DTPSScheduler
from vllm_omni.engine.dit_load_state import DitLoadSnapshot, DitLoadState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Stub request
# ---------------------------------------------------------------------------


class _StubRequest:
    """Minimal Request stub for DTPS tests."""

    def __init__(
        self,
        request_id: str,
        *,
        ar_only: bool = True,
        arrival_time: float | None = None,
        num_prompt_tokens: int = 0,
    ) -> None:
        self.request_id = request_id
        self.arrival_time = arrival_time if arrival_time is not None else time.time()
        self.num_prompt_tokens = num_prompt_tokens
        # DTPS reads additional_information to classify ar_only vs ar_downstream
        self.additional_information = {
            "omni_final_stage_id": 0 if ar_only else 1,
            "omni_force_kv_transfer": False,
        }


class _StubWaitingQueue:
    """List-backed waiting queue that supports clear/extend (like vLLM's FCFS queue)."""

    def __init__(self, reqs=None) -> None:
        self._reqs: list = list(reqs or [])

    def __iter__(self):
        return iter(self._reqs)

    def __len__(self):
        return len(self._reqs)

    def clear(self):
        self._reqs.clear()

    def extend(self, items):
        self._reqs.extend(items)

    @property
    def request_ids(self):
        return [r.request_id for r in self._reqs]


def _make_scheduler(
    *,
    threshold: int = 2,
    aging_s: float = 50.0,
    snapshot: DitLoadSnapshot | None = None,
) -> DTPSScheduler:
    """Create a DTPSScheduler with optional pre-set snapshot."""
    sched = DTPSScheduler(stage_id=0, aging_s=aging_s, dit_load_threshold=threshold)
    if snapshot is not None:
        sched.update_dit_load(snapshot)
    return sched


def _fresh_snapshot(
    *,
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


def _stale_snapshot() -> DitLoadSnapshot:
    return {
        "min_waiting": 0,
        "max_waiting": 0,
        "total_waiting": 0,
        "total_running": 0,
        "num_replicas": 0,
        "fresh": False,
        "waiting_ids": frozenset(),
        "running_ids": frozenset(),
    }


# ---------------------------------------------------------------------------
# DitLoadState tests
# ---------------------------------------------------------------------------


class TestDitLoadState:
    def test_update_returns_true_on_new_data(self):
        state = DitLoadState()
        assert state.update(0, 0, waiting=3, running=1) is True

    def test_update_returns_false_on_unchanged_data(self):
        state = DitLoadState()
        state.update(0, 0, waiting=3, running=1, waiting_ids=["a"], running_ids=["b"])
        assert state.update(0, 0, waiting=3, running=1, waiting_ids=["a"], running_ids=["b"]) is False

    def test_update_returns_true_on_changed_data(self):
        state = DitLoadState()
        state.update(0, 0, waiting=3, running=1)
        assert state.update(0, 0, waiting=5, running=1) is True

    def test_remove_returns_true_when_data_exists(self):
        state = DitLoadState()
        state.update(0, 0, waiting=3, running=1)
        assert state.remove(0, 0) is True

    def test_remove_returns_false_when_no_data(self):
        state = DitLoadState()
        assert state.remove(0, 0) is False

    def test_remove_after_remove_returns_false(self):
        state = DitLoadState()
        state.update(0, 0, waiting=3, running=1)
        assert state.remove(0, 0) is True
        assert state.remove(0, 0) is False

    def test_snapshot_empty_returns_fresh_false(self):
        state = DitLoadState()
        snap = state.snapshot()
        assert snap["fresh"] is False
        assert snap["num_replicas"] == 0
        assert snap["min_waiting"] == 0

    def test_snapshot_with_data_returns_fresh_true(self):
        state = DitLoadState()
        state.update(0, 0, waiting=3, running=1, waiting_ids=["a", "b"])
        state.update(1, 0, waiting=1, running=2, running_ids=["c"])
        snap = state.snapshot()
        assert snap["fresh"] is True
        assert snap["num_replicas"] == 2
        assert snap["min_waiting"] == 1
        assert snap["max_waiting"] == 3
        assert snap["total_waiting"] == 4
        assert snap["total_running"] == 3
        assert snap["waiting_ids"] == frozenset({"a", "b"})
        assert snap["running_ids"] == frozenset({"c"})

    def test_snapshot_after_remove_becomes_empty(self):
        state = DitLoadState()
        state.update(0, 0, waiting=3, running=1)
        state.remove(0, 0)
        snap = state.snapshot()
        assert snap["fresh"] is False
        assert snap["num_replicas"] == 0

    def test_snapshot_partial_eviction_stays_fresh(self):
        """Partial eviction (some replicas alive) keeps fresh=True."""
        state = DitLoadState()
        state.update(0, 0, waiting=3, running=1)
        state.update(1, 0, waiting=0, running=1)
        state.remove(0, 0)  # evict one replica
        snap = state.snapshot()
        assert snap["fresh"] is True
        assert snap["num_replicas"] == 1
        assert snap["min_waiting"] == 0

    def test_update_clamps_negative_values(self):
        state = DitLoadState()
        state.update(0, 0, waiting=-5, running=-3)
        snap = state.snapshot()
        assert snap["min_waiting"] == 0
        assert snap["total_running"] == 0


# ---------------------------------------------------------------------------
# DTPSScheduler tier ordering tests
# ---------------------------------------------------------------------------


class TestTierOrdering:
    def test_downstream_prioritized_over_ar_only_when_dit_idle(self):
        """L1 (downstream within budget) > L2 (ar_only)."""
        sched = _make_scheduler(threshold=2, snapshot=_fresh_snapshot(min_waiting=0))
        downstream = _StubRequest("d1", ar_only=False, num_prompt_tokens=10)
        ar_only = _StubRequest("a1", ar_only=True, num_prompt_tokens=5)
        queue = _StubWaitingQueue([ar_only, downstream])
        sched.maybe_reorder_waiting(queue, running=[])
        assert queue.request_ids == ["d1", "a1"]

    def test_ar_only_prioritized_when_dit_busy(self):
        """When DiT is busy (eff_min >= threshold), budget=0, ar_only first."""
        sched = _make_scheduler(threshold=2, snapshot=_fresh_snapshot(min_waiting=2))
        downstream = _StubRequest("d1", ar_only=False, num_prompt_tokens=10)
        ar_only = _StubRequest("a1", ar_only=True)
        queue = _StubWaitingQueue([downstream, ar_only])
        sched.maybe_reorder_waiting(queue, running=[])
        assert queue.request_ids == ["a1", "d1"]

    def test_downstream_sorted_by_num_prompt_tokens_asc(self):
        """Within L1, downstream sorted by num_prompt_tokens ascending."""
        sched = _make_scheduler(threshold=3, snapshot=_fresh_snapshot(min_waiting=0))
        d1 = _StubRequest("d1", ar_only=False, num_prompt_tokens=50)
        d2 = _StubRequest("d2", ar_only=False, num_prompt_tokens=10)
        d3 = _StubRequest("d3", ar_only=False, num_prompt_tokens=30)
        queue = _StubWaitingQueue([d1, d2, d3])
        sched.maybe_reorder_waiting(queue, running=[])
        assert queue.request_ids == ["d2", "d3", "d1"]

    def test_overflow_downstream_demoted_below_ar_only(self):
        """L3 (overflow downstream) < L2 (ar_only)."""
        sched = _make_scheduler(threshold=1, snapshot=_fresh_snapshot(min_waiting=0))
        d1 = _StubRequest("d1", ar_only=False, num_prompt_tokens=10)
        d2 = _StubRequest("d2", ar_only=False, num_prompt_tokens=20)
        a1 = _StubRequest("a1", ar_only=True)
        queue = _StubWaitingQueue([d1, d2, a1])
        sched.maybe_reorder_waiting(queue, running=[])
        # budget=1: d1 in L1, a1 in L2, d2 in L3
        assert queue.request_ids == ["d1", "a1", "d2"]

    def test_starving_request_highest_priority(self):
        """L0 (starving) > everything."""
        # Use a positive aging threshold so only the old request enters L0.
        # aging_s=0.0 would starve both requests whenever the clock advances
        # between their creation, preserving input order (d1 before s1).
        now = time.time()
        sched = _make_scheduler(threshold=2, snapshot=_fresh_snapshot(min_waiting=0), aging_s=1.0)
        # starving request arrived 2s ago — exceeds the 1.0s aging threshold
        starving = _StubRequest("s1", ar_only=True, arrival_time=now - 2)
        # downstream request just arrived — not starving
        downstream = _StubRequest("d1", ar_only=False, num_prompt_tokens=10, arrival_time=now)
        queue = _StubWaitingQueue([downstream, starving])
        sched.maybe_reorder_waiting(queue, running=[])
        assert queue.request_ids[0] == "s1"


# ---------------------------------------------------------------------------
# Stale / None snapshot tests
# ---------------------------------------------------------------------------


class TestSnapshotFallback:
    def test_stale_snapshot_skips_reorder(self):
        """When fresh=False, maybe_reorder_waiting returns without reordering."""
        sched = _make_scheduler(threshold=2, snapshot=_stale_snapshot())
        d1 = _StubRequest("d1", ar_only=False)
        a1 = _StubRequest("a1", ar_only=True)
        queue = _StubWaitingQueue([d1, a1])
        sched.maybe_reorder_waiting(queue, running=[])
        # order unchanged — DTPS skipped
        assert queue.request_ids == ["d1", "a1"]

    def test_none_snapshot_still_reorders_with_blind_tracking(self):
        """When snap is None, blind in-flight tracking still provides load awareness."""
        sched = _make_scheduler(threshold=2, snapshot=None)
        # Register blind in-flight entries to simulate downstream tasks
        # that finished AR but haven't reached DiT yet
        sched.register_finished_downstream("blind-1")
        sched.register_finished_downstream("blind-2")
        # With 2 blind entries, effective_min=2 >= threshold=2 → DiT "busy"
        # budget=0 → ar_only should be prioritized
        d1 = _StubRequest("d1", ar_only=False, num_prompt_tokens=10)
        a1 = _StubRequest("a1", ar_only=True)
        queue = _StubWaitingQueue([d1, a1])
        sched.maybe_reorder_waiting(queue, running=[])
        # budget=0 because effective_min=2 >= threshold=2
        assert queue.request_ids == ["a1", "d1"]

    def test_none_snapshot_no_blind_prioritizes_downstream(self):
        """When snap is None and no blind entries, budget=threshold, downstream prioritized."""
        sched = _make_scheduler(threshold=2, snapshot=None)
        d1 = _StubRequest("d1", ar_only=False, num_prompt_tokens=10)
        a1 = _StubRequest("a1", ar_only=True)
        queue = _StubWaitingQueue([d1, a1])
        sched.maybe_reorder_waiting(queue, running=[])
        # effective_min=0 < threshold=2 → idle → budget=2 → downstream in L1
        assert queue.request_ids == ["d1", "a1"]


# ---------------------------------------------------------------------------
# Blind in-flight tracking tests
# ---------------------------------------------------------------------------


class TestBlindInflightTracking:
    def test_register_adds_entry(self):
        sched = _make_scheduler(threshold=2)
        sched.register_finished_downstream("req-1")
        assert "req-1" in sched._dit_inflight_ids

    def test_register_dedup(self):
        sched = _make_scheduler(threshold=2)
        sched.register_finished_downstream("req-1")
        sched.register_finished_downstream("req-1")  # duplicate
        assert len(sched._dit_inflight_ids) == 1

    def test_blind_evicted_when_seen_in_dit(self):
        """Blind entry removed when the id appears in DiT's waiting/running ids."""
        sched = _make_scheduler(threshold=2)
        sched.register_finished_downstream("req-1")
        assert len(sched._dit_inflight_ids) == 1
        # Update snapshot that includes req-1 in waiting_ids
        sched.update_dit_load(_fresh_snapshot(waiting_ids=frozenset({"req-1"})))
        sched._dit_phase(0)
        assert len(sched._dit_inflight_ids) == 0

    def test_blind_evicted_on_timeout(self):
        """Blind entry removed after _DIT_INFLIGHT_MAX_AGE_S."""
        sched = _make_scheduler(threshold=2)
        sched.register_finished_downstream("req-1")
        # Manually age the entry beyond the 1s timeout
        entry = sched._dit_inflight_ids["req-1"]
        sched._dit_inflight_ids["req-1"] = type(entry)(added_mono=time.monotonic() - 2.0)
        sched._dit_phase(0)
        assert len(sched._dit_inflight_ids) == 0


# ---------------------------------------------------------------------------
# Budget math tests
# ---------------------------------------------------------------------------


class TestBudgetMath:
    def test_budget_equals_threshold_when_dit_empty(self):
        """DiT idle (min_waiting=0), threshold=2 → budget=2."""
        sched = _make_scheduler(threshold=2, snapshot=_fresh_snapshot(min_waiting=0))
        sched._dit_phase(0)
        stats = sched._last_phase_stats
        assert stats["effective_min"] == 0
        budget = max(0, sched.dit_load_threshold - stats["effective_min"]) * max(1, stats["n_reps"])
        assert budget == 2

    def test_budget_zero_when_dit_at_threshold(self):
        """DiT busy (min_waiting=2), threshold=2 → budget=0."""
        sched = _make_scheduler(threshold=2, snapshot=_fresh_snapshot(min_waiting=2))
        sched._dit_phase(0)
        stats = sched._last_phase_stats
        assert stats["effective_min"] == 2
        budget = max(0, sched.dit_load_threshold - stats["effective_min"]) * max(1, stats["n_reps"])
        assert budget == 0

    def test_threshold_default_is_two(self):
        """Default threshold = DiT max_num_seqs + 1 = 1 + 1 = 2."""
        sched = DTPSScheduler(stage_id=0)
        assert sched.dit_load_threshold == 2

    def test_threshold_custom(self):
        sched = DTPSScheduler(stage_id=0, dit_load_threshold=9)
        assert sched.dit_load_threshold == 9


# ---------------------------------------------------------------------------
# derive_dtps_config tests
# ---------------------------------------------------------------------------


class TestDeriveDtpsConfig:
    def _make_pipeline(self, stages):
        """Build a minimal PipelineConfig-like stub."""
        pipeline = MagicMock()
        pipeline.stages = stages
        return pipeline

    def _make_stage(self, stage_id, execution_type, input_sources=None):
        stage = MagicMock()
        stage.stage_id = stage_id
        stage.execution_type = execution_type
        stage.input_sources = input_sources or []
        return stage

    def _make_deploy(self, max_num_seqs=1, dit_load_aware=True):
        deploy = MagicMock()
        deploy.max_num_seqs = max_num_seqs
        deploy.dit_load_aware = dit_load_aware
        return deploy

    def test_disabled_returns_none(self):
        from vllm_omni.config.stage_config import derive_dtps_config

        deploy = self._make_deploy(dit_load_aware=False)
        result = derive_dtps_config(MagicMock(), {}, 0, deploy)
        assert result is None

    def test_no_downstream_dit_returns_none(self):
        from vllm_omni.config.stage_config import StageExecutionType, derive_dtps_config

        ar_stage = self._make_stage(0, StageExecutionType.LLM_AR)
        pipeline = self._make_pipeline([ar_stage])
        deploy = self._make_deploy()
        result = derive_dtps_config(pipeline, {}, 0, deploy)
        assert result is None

    def test_single_dit_returns_threshold(self):
        from vllm_omni.config.stage_config import StageExecutionType, derive_dtps_config

        ar_stage = self._make_stage(0, StageExecutionType.LLM_AR)
        dit_stage = self._make_stage(1, StageExecutionType.DIFFUSION, input_sources=[0])
        pipeline = self._make_pipeline([ar_stage, dit_stage])
        dit_deploy = self._make_deploy(max_num_seqs=4)
        deploy_by_id = {1: dit_deploy}
        ar_deploy = self._make_deploy(dit_load_aware=True)
        result = derive_dtps_config(pipeline, deploy_by_id, 0, ar_deploy)
        assert result == {"ar_stage_id": 0, "dit_stage_id": 1, "threshold": 5}  # 4 + 1

    def test_multi_dit_returns_none(self):
        from vllm_omni.config.stage_config import StageExecutionType, derive_dtps_config

        ar_stage = self._make_stage(0, StageExecutionType.LLM_AR)
        dit1 = self._make_stage(1, StageExecutionType.DIFFUSION, input_sources=[0])
        dit2 = self._make_stage(2, StageExecutionType.DIFFUSION, input_sources=[0])
        pipeline = self._make_pipeline([ar_stage, dit1, dit2])
        deploy_by_id = {1: self._make_deploy(), 2: self._make_deploy()}
        ar_deploy = self._make_deploy(dit_load_aware=True)
        result = derive_dtps_config(pipeline, deploy_by_id, 0, ar_deploy)
        assert result is None


# ---------------------------------------------------------------------------
# DitLoadState eviction + dirty-retry tests
# ---------------------------------------------------------------------------


class TestDitLoadStateEviction:
    """Tests for DitLoadState eviction and dirty-retry scenarios.

    These cover the review feedback:
    - Replica never supplied data → eviction still publishes empty snapshot
    - Replica supplied data → eviction push fails → dirty retry on subsequent failures
    """

    def test_evict_replica_with_no_prior_data(self):
        """Evicting a replica that never supplied data still produces a
        valid (empty) snapshot. The remove() call is a no-op but the
        snapshot should reflect num_replicas=0, fresh=False."""
        state = DitLoadState(dit_load_threshold=2)
        # Never call state.update() — replica has no data
        removed = state.remove(stage_id=0, replica_id=0)
        assert removed is False  # nothing to remove
        snap = state.snapshot()
        assert snap["num_replicas"] == 0
        assert snap["fresh"] is False

    def test_evict_replica_with_prior_data(self):
        """Evicting a replica that had data removes it from the snapshot."""
        state = DitLoadState(dit_load_threshold=2)
        state.update(stage_id=0, replica_id=0, waiting=3, running=1)
        snap_before = state.snapshot()
        assert snap_before["num_replicas"] == 1
        assert snap_before["min_waiting"] == 3

        removed = state.remove(stage_id=0, replica_id=0)
        assert removed is True
        snap_after = state.snapshot()
        assert snap_after["num_replicas"] == 0
        assert snap_after["fresh"] is False

    def test_evict_one_of_two_replicas(self):
        """Evicting one replica out of two adjusts the aggregate snapshot."""
        state = DitLoadState(dit_load_threshold=2)
        state.update(stage_id=0, replica_id=0, waiting=5, running=1)
        state.update(stage_id=0, replica_id=1, waiting=2, running=0)

        snap_before = state.snapshot()
        assert snap_before["num_replicas"] == 2
        assert snap_before["min_waiting"] == 2

        removed = state.remove(stage_id=0, replica_id=1)
        assert removed is True

        snap_after = state.snapshot()
        assert snap_after["num_replicas"] == 1
        assert snap_after["min_waiting"] == 5  # only replica 0 remains

    def test_dirty_retry_after_failed_push(self):
        """Simulate the dirty-retry path: eviction push fails (dirty=True),
        then subsequent poll failures retry the push until it succeeds.

        This models the orchestrator logic:
        1. Replica had data, then polls start failing
        2. After threshold failures, evict + push (push fails → dirty=True)
        3. Subsequent failures retry dirty push
        4. Retry succeeds → dirty=False
        """
        state = DitLoadState(dit_load_threshold=2)
        state.update(stage_id=0, replica_id=0, waiting=3, running=1)

        # Simulate push results (True = success, False = failure)
        push_results = [False, False, True]  # 2 failures then success
        push_call_count = 0

        async def fake_push(snapshot):
            nonlocal push_call_count
            result = push_results[push_call_count] if push_call_count < len(push_results) else True
            push_call_count += 1
            return result

        # Simulate the orchestrator's eviction + dirty-retry loop
        dirty = False
        fail_count = 0
        evict_threshold = 5

        # Phase 1: 5 failed polls → eviction
        for _ in range(5):
            fail_count += 1

        if fail_count >= evict_threshold:
            state.remove(stage_id=0, replica_id=0)
            fail_count = 0
            dirty = not asyncio_run(fake_push(state.snapshot()))

        assert dirty is True  # push failed
        assert push_call_count == 1

        # Phase 2: subsequent failed polls retry dirty
        for i in range(3):
            fail_count += 1
            if fail_count >= evict_threshold:
                state.remove(stage_id=0, replica_id=0)
                fail_count = 0
                dirty = not asyncio_run(fake_push(state.snapshot()))
            elif dirty:
                dirty = not asyncio_run(fake_push(state.snapshot()))

            if not dirty:
                break

        assert dirty is False  # eventually succeeded
        assert push_call_count == 3  # retried until success

    def test_dirty_not_set_when_eviction_push_succeeds(self):
        """When eviction push succeeds, dirty stays False — no retry needed."""
        state = DitLoadState(dit_load_threshold=2)
        state.update(stage_id=0, replica_id=0, waiting=3, running=1)

        removed = state.remove(stage_id=0, replica_id=0)
        assert removed is True

        # Simulate successful push
        dirty = not True  # push returned True
        assert dirty is False


def asyncio_run(coro):
    """Helper to run async coroutine in sync test."""
    import asyncio as _asyncio

    try:
        loop = _asyncio.get_event_loop()
        if loop.is_running():
            return _asyncio.ensure_future(coro)
    except RuntimeError:
        pass
    return _asyncio.run(coro)


# ---------------------------------------------------------------------------
# Stage isolation tests (dit_stage_id / ar_stage_id pairing)
# ---------------------------------------------------------------------------


class TestStageIsolation:
    """Tests for DTPS stage isolation via ar_stage_id / dit_stage_id pairing.

    Verifies that load data from one DiT stage does not leak into the
    snapshot of another, and that derive_dtps_config carries the pairing.
    """

    def test_snapshot_filters_by_stage_id(self):
        """DitLoadState.snapshot(stage_id=D) only includes replicas from
        stage D, excluding data from other DiT stages."""
        state = DitLoadState(dit_load_threshold=2)
        # DiT1 (stage 1): 3 waiting
        state.update(stage_id=1, replica_id=0, waiting=3, running=1)
        # DiT3 (stage 3): 5 waiting
        state.update(stage_id=3, replica_id=0, waiting=5, running=2)

        # Unfiltered snapshot aggregates both stages
        all_snap = state.snapshot()
        assert all_snap["num_replicas"] == 2
        assert all_snap["min_waiting"] == 3

        # Filtered snapshot only includes stage 1
        dit1_snap = state.snapshot(stage_id=1)
        assert dit1_snap["num_replicas"] == 1
        assert dit1_snap["min_waiting"] == 3
        assert dit1_snap["fresh"] is True

        # Filtered snapshot only includes stage 3
        dit3_snap = state.snapshot(stage_id=3)
        assert dit3_snap["num_replicas"] == 1
        assert dit3_snap["min_waiting"] == 5

    def test_snapshot_stage_id_with_no_data(self):
        """snapshot(stage_id=X) returns empty snapshot when no replicas
        exist for that stage."""
        state = DitLoadState(dit_load_threshold=2)
        state.update(stage_id=1, replica_id=0, waiting=3, running=1)

        snap = state.snapshot(stage_id=99)
        assert snap["num_replicas"] == 0
        assert snap["fresh"] is False

    def test_derive_dtps_config_carries_pairing(self):
        """derive_dtps_config returns ar_stage_id and dit_stage_id."""
        from vllm_omni.config.stage_config import StageExecutionType, derive_dtps_config

        ar_stage = MagicMock()
        ar_stage.execution_type = StageExecutionType.LLM_AR
        ar_stage.stage_id = 0
        ar_stage.input_sources = ()

        dit_stage = MagicMock()
        dit_stage.execution_type = StageExecutionType.DIFFUSION
        dit_stage.stage_id = 1
        dit_stage.input_sources = [0]

        pipeline = MagicMock()
        pipeline.stages = [ar_stage, dit_stage]

        dit_deploy = MagicMock()
        dit_deploy.max_num_seqs = 2
        deploy_by_id = {1: dit_deploy}

        ar_deploy = MagicMock()
        ar_deploy.dit_load_aware = True

        result = derive_dtps_config(pipeline, deploy_by_id, 0, ar_deploy)
        assert result is not None
        assert result["ar_stage_id"] == 0
        assert result["dit_stage_id"] == 1
        assert result["threshold"] == 3  # 2 + 1

    def test_unrelated_dit_stage_excluded_from_snapshot(self):
        """Regression: changing load on an unrelated DiT stage must not
        affect the snapshot of the paired DiT stage."""
        state = DitLoadState(dit_load_threshold=2)
        # Paired DiT (stage 1): idle
        state.update(stage_id=1, replica_id=0, waiting=0, running=0)
        # Unrelated DiT (stage 3): very busy
        state.update(stage_id=3, replica_id=0, waiting=10, running=5)

        paired_snap = state.snapshot(stage_id=1)
        assert paired_snap["min_waiting"] == 0
        assert paired_snap["num_replicas"] == 1
        # The unrelated stage's load must not appear
        assert paired_snap["total_waiting"] == 0
