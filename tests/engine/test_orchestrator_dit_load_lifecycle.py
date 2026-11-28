# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""DTPS DiT-load lifecycle: replica removal invalidates snapshots; load
feedback stays off the output-drain path.

Review ask (PR #9634, two P2s):

* Every common replica-removal path (membership detach and local
  dead-replica eviction) must drop the departed DiT replica's load entry and
  republish the pair snapshot — including when it is the *last* replica, so
  the AR scheduler never keeps scheduling on a stale ``fresh=True`` aggregate.
* Load collection/delivery runs in its own background loop, coalesced per
  stage pair, and never blocks diffusion output draining.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.messages import ShutdownRequestMessage
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_pool import StagePool

from .test_orchestrator import FakeStageClient, _build_harness, _build_stage_pools

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RecordingArClient(FakeStageClient):
    """AR replica that records utility pushes (``push_dit_load`` target)."""

    def __init__(self) -> None:
        super().__init__(final_output=True, final_output_type="text")
        self.utility_calls: list[tuple[str, dict[str, Any]]] = []

    async def call_utility_async(self, method: str, payload: dict[str, Any]) -> bool:
        self.utility_calls.append((method, dict(payload)))
        return True


class _FailingArClient(_RecordingArClient):
    """AR replica whose utility channel fails on demand (push_dit_load → False)."""

    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    async def call_utility_async(self, method: str, payload: dict[str, Any]) -> bool:
        self.utility_calls.append((method, dict(payload)))
        if self.fail:
            raise RuntimeError("utility channel down")
        return True


class _LoadDiTClient(FakeStageClient):
    """DiT replica answering load queries with a mutable queue depth."""

    def __init__(self, waiting: int = 0, running: int = 0) -> None:
        super().__init__(stage_type="diffusion", final_output=True, final_output_type="image")
        self.waiting = waiting
        self.running = running
        self.diffusion_polls = 0
        self.load_polls = 0

    async def get_dit_load_async(self) -> tuple[int, int, list[str], list[str]]:
        self.load_polls += 1
        return (self.waiting, self.running, [f"w-{i}" for i in range(self.waiting)], [])

    def get_diffusion_output_nowait(self):
        self.diffusion_polls += 1
        return super().get_diffusion_output_nowait()


class _SlowLoadDiTClient(_LoadDiTClient):
    """DiT replica whose load query stalls, but whose output path is healthy."""

    async def get_dit_load_async(self) -> tuple[int, int, list[str], list[str]]:
        self.load_polls += 1
        # Longer than the test's drain window, shorter than the loop's bounded
        # shutdown wait: with the old inline design this stall froze the drain
        # poller; now only the background cycle waits on it.
        await asyncio.sleep(2.5)
        return (0, 0, [], [])


def _build_idle_orchestrator(
    ar_client: FakeStageClient,
    dit_clients: list[_LoadDiTClient],
) -> tuple[Orchestrator, StagePool, StagePool]:
    """Orchestrator with DTPS pairs wired but no running event loop."""
    pools = _build_stage_pools([[ar_client], [dit_clients[0]]])
    if len(dit_clients) > 1:
        pools[1] = StagePool(1, list(dit_clients))
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=pools,
        dit_load_aware_config=[{"ar_stage_id": 0, "dit_stage_id": 1, "threshold": 2}],
    )
    return orchestrator, pools[0], pools[1]


def _last_payload(ar_client: _RecordingArClient) -> dict[str, Any]:
    assert ar_client.utility_calls, "no snapshot was pushed to the AR stage"
    method, payload = ar_client.utility_calls[-1]
    assert method == "omni_update_dit_load"
    return payload


async def test_replica_removed_callback_invalidates_dit_load() -> None:
    """Membership detach (last remote replica) drops the entry and republishes fresh=False."""
    ar_client = _RecordingArClient()
    dit_client = _LoadDiTClient(waiting=3, running=2)
    orchestrator, _, dit_pool = _build_idle_orchestrator(ar_client, [dit_client])

    state = orchestrator._dit_load_state
    assert state is not None
    assert state.update(1, 0, 3, 2, waiting_ids=["w-1", "w-2", "w-3"], running_ids=["r-1", "r-2"])

    # Mirror MembershipController.handle_unregister: detach the (single, last)
    # replica BEFORE the callback fires, so the pool is empty by then.
    dit_pool.evict_replica(0)
    assert dit_pool.stage_type is None

    await orchestrator._handle_replica_removed(1, 0)

    snapshot = state.snapshot(stage_id=1)
    assert snapshot["num_replicas"] == 0
    assert snapshot["fresh"] is False
    payload = _last_payload(ar_client)
    assert payload["fresh"] is False
    assert payload["num_replicas"] == 0


async def test_local_eviction_invalidates_dit_load() -> None:
    """Local dead-replica eviction shares the same invalidation path."""
    ar_client = _RecordingArClient()
    dit_client = _LoadDiTClient(waiting=1, running=1)
    orchestrator, ar_pool, dit_pool = _build_idle_orchestrator(ar_client, [dit_client])

    state = orchestrator._dit_load_state
    assert state is not None
    assert state.update(1, 0, 1, 1, waiting_ids=["w-1"], running_ids=["r-1"])

    await orchestrator._handle_dead_replica(1, 0, EngineDeadError("replica died"))

    assert dit_pool.live_replica_ids() == []
    assert state.snapshot(stage_id=1)["num_replicas"] == 0
    payload = _last_payload(ar_client)
    assert payload["fresh"] is False
    assert ar_pool.live_replica_ids() == [0]  # AR stage untouched by the DiT eviction


async def test_invalidated_snapshot_not_republished_for_non_dtps_stage() -> None:
    """Removing an AR (non-DiT) replica must not touch the DiT snapshot."""
    ar_client = _RecordingArClient()
    dit_client = _LoadDiTClient(waiting=2)
    orchestrator, _, _ = _build_idle_orchestrator(ar_client, [dit_client])

    state = orchestrator._dit_load_state
    assert state is not None
    assert state.update(1, 0, 2, 0)

    await orchestrator._handle_replica_removed(0, 0)

    assert state.snapshot(stage_id=1)["num_replicas"] == 1


async def test_cycle_coalesces_pair_into_single_push() -> None:
    """Two replicas changing in one cycle still produce exactly one pair push."""
    ar_client = _RecordingArClient()
    dit_clients = [_LoadDiTClient(waiting=1), _LoadDiTClient(waiting=2)]
    orchestrator, _, _ = _build_idle_orchestrator(ar_client, dit_clients)

    # Cycle 1: both replicas report loads → exactly one coalesced push.
    await orchestrator._poll_dit_load_cycle()
    assert len(ar_client.utility_calls) == 1
    payload = _last_payload(ar_client)
    assert payload["total_waiting"] == 3
    assert payload["num_replicas"] == 2

    # Cycle 2: nothing changed → no additional push.
    await orchestrator._poll_dit_load_cycle()
    assert len(ar_client.utility_calls) == 1

    # Cycle 3: one replica's depth changes → one more coalesced push.
    dit_clients[1].waiting = 5
    await orchestrator._poll_dit_load_cycle()
    assert len(ar_client.utility_calls) == 2
    payload = _last_payload(ar_client)
    assert payload["total_waiting"] == 6


async def test_failed_pair_push_retried_on_next_cycle() -> None:
    """A pair whose push failed is marked dirty and retried even with no change."""
    ar_client = _FailingArClient()
    dit_client = _LoadDiTClient(waiting=1)
    orchestrator, _, _ = _build_idle_orchestrator(ar_client, [dit_client])

    # Cycle 1: changed → push fails → pair marked dirty.
    await orchestrator._poll_dit_load_cycle()
    assert len(ar_client.utility_calls) == 1
    assert orchestrator._dit_load_dirty_pairs == {1}

    # Cycle 2: unchanged load, but the dirty pair is retried.
    await orchestrator._poll_dit_load_cycle()
    assert len(ar_client.utility_calls) == 2

    # A replica departure clears the entry and keeps retrying the push; once
    # the AR channel recovers, the pair leaves the dirty set.
    dit_client.waiting = 0
    await orchestrator._poll_dit_load_cycle()
    assert orchestrator._dit_load_dirty_pairs == {1}
    ar_client.fail = False
    dit_client.waiting = 2
    await orchestrator._poll_dit_load_cycle()
    assert orchestrator._dit_load_dirty_pairs == set()
    payload = _last_payload(ar_client)
    assert payload["total_waiting"] == 2
    assert payload["fresh"] is True


def test_slow_load_poll_does_not_block_output_drain() -> None:
    """Event-driven drain keeps polling outputs while the load query hangs.

    Load collection used to run inline in the diffusion poller, so one stuck
    load query froze output draining for the whole stage. It now lives in the
    background loop; the drain path must keep running regardless.
    """
    ar_client = _RecordingArClient()
    dit_client = _SlowLoadDiTClient()

    fixture = _build_harness(
        [ar_client, dit_client],
        dit_load_aware_config=[{"ar_stage_id": 0, "dit_stage_id": 1, "threshold": 2}],
    )
    try:
        deadline = time.monotonic() + 2.0
        while dit_client.diffusion_polls < 5:
            assert time.monotonic() < deadline, (
                f"output drain stalled while the load query hung (diffusion_polls={dit_client.diffusion_polls})"
            )
            time.sleep(0.01)
        assert dit_client.load_polls >= 1
    finally:
        fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
        fixture.thread.join(5)
    assert not fixture.thread.is_alive()
    fixture.result_future.result(timeout=0)
