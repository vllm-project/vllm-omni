# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from vllm_omni.distributed.omni_coordinator.messages import ReplicaStatus
from vllm_omni.engine.membership_controller import MembershipController

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeHub:
    def __init__(self, snapshots=None):
        self.snapshots = list(snapshots or [])
        self.closed = False

    def get_replica_list(self):
        if self.snapshots:
            return self.snapshots.pop(0)
        return SimpleNamespace(replicas=[])

    def close(self):
        self.closed = True


class FakePool:
    def __init__(self, stage_id: int):
        self.stage_id = stage_id
        self.clients = []
        self.added = []
        self.removed = []
        self.invalidated = []
        self.hub = None
        self.lb = None

    def attach_hub(self, hub):
        self.hub = hub

    def attach_load_balancer(self, lb):
        self.lb = lb

    def add_client(self, input_addr, client):
        self.added.append((input_addr, client))
        self.clients.append(client)
        return len(self.clients) - 1

    def invalidate_addr(self, input_addr):
        self.invalidated.append(input_addr)
        return ["req-1", "req-2"]

    def remove_client(self, input_addr):
        self.removed.append(input_addr)
        return SimpleNamespace(shutdown=lambda: self.removed.append("shutdown"))


def _snapshot(*replicas):
    return SimpleNamespace(replicas=list(replicas))


def _replica(stage_id: int, input_addr: str, status=ReplicaStatus.UP):
    return SimpleNamespace(stage_id=stage_id, input_addr=input_addr, status=status)


def _controller(monkeypatch, pool, hub, remote_replica_factory=None):
    import vllm_omni.engine.membership_controller as membership_mod

    monkeypatch.setattr(membership_mod, "OmniCoordClientForHub", lambda _addr: hub)
    if remote_replica_factory is None:

        def remote_replica_factory(stage_id, replica_id):
            return SimpleNamespace(client_addresses={"input_address": f"tcp://stage-{stage_id}-replica-{replica_id}"})

    return MembershipController(
        stage_pools=[pool],
        coordinator_pub_address="tcp://127.0.0.1:12345",
        load_balancer_factory=lambda: object(),
        remote_replica_factory=remote_replica_factory,
    )


@pytest.mark.asyncio
async def test_watch_replica_list_unregisters_disappeared_replicas(monkeypatch):
    pool = FakePool(stage_id=0)
    hub = FakeHub(
        snapshots=[
            _snapshot(_replica(0, "tcp://gone")),
            _snapshot(),
        ]
    )
    controller = _controller(monkeypatch, pool, hub)
    controller.WATCH_INTERVAL_S = 0
    unregistered = []

    async def _handle_unregister(stage_id, input_addr, *args, **kwargs):
        unregistered.append((stage_id, input_addr))
        controller._shutdown_event.set()

    controller.handle_unregister = _handle_unregister  # type: ignore[method-assign]

    await asyncio.wait_for(controller._watch_replica_list(), timeout=1)
    await controller.drain_tasks(timeout=1)

    assert unregistered == [(0, "tcp://gone")]


@pytest.mark.asyncio
async def test_shutdown_closes_hub_then_cancels_watcher(monkeypatch):
    pool = FakePool(stage_id=0)
    hub = FakeHub()
    controller = _controller(monkeypatch, pool, hub)
    watcher = asyncio.create_task(asyncio.sleep(10))
    controller._watcher_task = watcher

    controller.shutdown()
    await asyncio.sleep(0)

    assert hub.closed is True
    assert watcher.cancelled()


@pytest.mark.asyncio
async def test_drain_tasks_waits_for_membership_tasks(monkeypatch):
    pool = FakePool(stage_id=0)
    controller = _controller(monkeypatch, pool, FakeHub())
    completed = []

    async def _task():
        await asyncio.sleep(0)
        completed.append(True)

    controller._spawn_task(_task(), label="unit")
    await controller.drain_tasks(timeout=1)

    assert completed == [True]


@pytest.mark.asyncio
async def test_do_register_offloads_remote_factory(monkeypatch):
    pool = FakePool(stage_id=0)
    factory_thread_ids = []

    def _factory(stage_id, replica_id):
        factory_thread_ids.append(threading.get_ident())
        return SimpleNamespace(client_addresses={"input_address": f"tcp://stage-{stage_id}-replica-{replica_id}"})

    controller = _controller(monkeypatch, pool, FakeHub(), remote_replica_factory=_factory)

    await controller._do_register(0, 1)

    assert factory_thread_ids
    assert factory_thread_ids[0] != threading.get_ident()
    assert pool.added[0][0] == "tcp://stage-0-replica-1"
