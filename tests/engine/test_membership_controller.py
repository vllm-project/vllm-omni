# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from vllm_omni.distributed.omni_coordinator.messages import ReplicaStatus
from vllm_omni.engine.membership_controller import MembershipController
from vllm_omni.engine.stage_pool import StagePool

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


async def _wait_until(predicate, timeout=1):
    async def _poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(_poll(), timeout=timeout)


class FakePool:
    def __init__(self, stage_id: int):
        self.stage_id = stage_id
        self.clients = []
        self.added = []
        self.removed = []
        self.invalidated = []
        self.hub = None
        self.lb = None
        self.replica_ids = {}

    def attach_hub(self, hub):
        self.hub = hub

    def attach_load_balancer(self, lb):
        self.lb = lb

    def add_client(self, input_addr, client, *, replica_id=None):
        self.added.append((input_addr, client, replica_id))
        self.clients.append(client)
        resolved_replica_id = len(self.clients) - 1 if replica_id is None else replica_id
        self.replica_ids[input_addr] = resolved_replica_id
        return resolved_replica_id

    def invalidate_addr(self, input_addr):
        self.invalidated.append(input_addr)
        return ["req-1", "req-2"]

    def remove_client(self, input_addr):
        self.removed.append(input_addr)
        return SimpleNamespace(shutdown=lambda: self.removed.append("shutdown"))

    def get_client_by_addr(self, input_addr):
        return SimpleNamespace() if input_addr in self.replica_ids else None

    def get_replica_id_by_addr(self, input_addr):
        return self.replica_ids.get(input_addr)


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
    assert pool.added[0] == ("tcp://stage-0-replica-1", pool.clients[0], 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing-address", "add-client"])
async def test_do_register_shuts_down_rejected_client(monkeypatch, failure):
    shutdown_calls = 0

    class Client:
        client_addresses = {} if failure == "missing-address" else {"input_address": "tcp://rejected"}

        def shutdown(self):
            nonlocal shutdown_calls
            shutdown_calls += 1

    pool = FakePool(stage_id=0)
    if failure == "add-client":
        monkeypatch.setattr(pool, "add_client", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("reject")))
    controller = _controller(monkeypatch, pool, FakeHub(), remote_replica_factory=lambda *_args: Client())

    with pytest.raises(RuntimeError):
        await controller._do_register(0, 1)

    assert shutdown_calls == 1


@pytest.mark.asyncio
async def test_unregister_then_register_restores_coordinator_replica_slot(monkeypatch):
    class Client:
        def __init__(self, input_addr: str) -> None:
            self.client_addresses = {"input_address": input_addr}
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    old_clients = [Client("tcp://old-0"), Client("tcp://old-1")]
    replacement = Client("tcp://new-1")
    pool = StagePool(0, old_clients)

    def factory(stage_id: int, replica_id: int):
        assert (stage_id, replica_id) == (0, 1)
        return replacement

    controller = _controller(monkeypatch, pool, FakeHub(), remote_replica_factory=factory)
    removed_replicas = []

    async def cleanup(_request_ids):
        return None

    controller.install_unregister_handlers(
        output_queue=asyncio.Queue(),
        cleanup_callback=cleanup,
        replica_removed_callback=lambda stage_id, replica_id: removed_replicas.append((stage_id, replica_id)),
    )
    await controller.handle_unregister(0, "tcp://old-0")
    await controller.handle_unregister(0, "tcp://old-1")

    await controller.handle_register(0, 1)
    await controller.drain_tasks(timeout=1)

    assert pool.clients == [None, replacement]
    assert pool.get_replica_id_by_addr("tcp://new-1") == 1
    assert pool.get_replica_id_by_addr("tcp://old-1") is None
    assert pool.available_replica_ids() == [1]
    assert removed_replicas == [(0, 0), (0, 1)]


@pytest.mark.asyncio
async def test_handle_register_deduplicates_inflight_and_attached_replica(monkeypatch):
    pool = FakePool(stage_id=0)
    factory_started = threading.Event()
    release_factory = threading.Event()
    factory_calls = []

    def _factory(stage_id, replica_id):
        factory_calls.append((stage_id, replica_id))
        factory_started.set()
        assert release_factory.wait(timeout=1)
        return SimpleNamespace(
            client_addresses={"input_address": f"tcp://stage-{stage_id}-replica-{replica_id}"},
            shutdown=lambda: None,
        )

    controller = _controller(monkeypatch, pool, FakeHub(), remote_replica_factory=_factory)

    await asyncio.gather(
        controller.handle_register(0, 2),
        controller.handle_register(0, 2),
    )
    assert await asyncio.to_thread(factory_started.wait, 1)

    await controller.handle_register(0, 2)
    release_factory.set()
    await controller.drain_tasks(timeout=1)

    await controller.handle_register(0, 2)
    await controller.drain_tasks(timeout=1)

    assert factory_calls == [(0, 2)]
    assert len(pool.added) == 1


@pytest.mark.asyncio
async def test_handle_register_allows_retry_after_factory_failure(monkeypatch):
    pool = FakePool(stage_id=0)
    factory_calls = 0

    def _factory(stage_id, replica_id):
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            raise RuntimeError("injected registration failure")
        return SimpleNamespace(
            client_addresses={"input_address": f"tcp://stage-{stage_id}-replica-{replica_id}"},
            shutdown=lambda: None,
        )

    controller = _controller(monkeypatch, pool, FakeHub(), remote_replica_factory=_factory)

    await controller.handle_register(0, 3)
    await controller.drain_tasks(timeout=1)
    await controller.handle_register(0, 3)
    await controller.drain_tasks(timeout=1)

    assert factory_calls == 2
    assert len(pool.added) == 1


@pytest.mark.asyncio
async def test_unregister_allows_same_replica_to_register_again(monkeypatch):
    pool = FakePool(stage_id=0)
    factory_calls = 0

    def _factory(stage_id, replica_id):
        nonlocal factory_calls
        factory_calls += 1
        return SimpleNamespace(
            client_addresses={"input_address": f"tcp://stage-{stage_id}-replica-{replica_id}"},
            shutdown=lambda: None,
        )

    controller = _controller(monkeypatch, pool, FakeHub(), remote_replica_factory=_factory)

    await controller.handle_register(0, 4)
    await controller.drain_tasks(timeout=1)
    await controller.handle_unregister(0, "tcp://stage-0-replica-4")
    await controller.handle_register(0, 4)
    await controller.drain_tasks(timeout=1)

    assert factory_calls == 2
    assert len(pool.added) == 2


@pytest.mark.asyncio
async def test_handle_register_ignores_registration_after_shutdown(monkeypatch):
    pool = FakePool(stage_id=0)
    factory_calls = 0

    def _factory(stage_id, replica_id):
        nonlocal factory_calls
        factory_calls += 1
        return SimpleNamespace(
            client_addresses={"input_address": f"tcp://stage-{stage_id}-replica-{replica_id}"},
            shutdown=lambda: None,
        )

    controller = _controller(monkeypatch, pool, FakeHub(), remote_replica_factory=_factory)
    controller.shutdown()

    await controller.handle_register(0, 5)
    await controller.drain_tasks(timeout=1)

    assert factory_calls == 0
    assert pool.added == []


@pytest.mark.asyncio
async def test_shutdown_discards_client_from_inflight_registration(monkeypatch):
    pool = FakePool(stage_id=0)
    factory_started = threading.Event()
    release_factory = threading.Event()
    client_shutdown = threading.Event()

    def _factory(stage_id, replica_id):
        factory_started.set()
        assert release_factory.wait(timeout=1)
        return SimpleNamespace(
            client_addresses={"input_address": f"tcp://stage-{stage_id}-replica-{replica_id}"},
            shutdown=client_shutdown.set,
        )

    controller = _controller(monkeypatch, pool, FakeHub(), remote_replica_factory=_factory)

    await controller.handle_register(0, 6)
    assert await asyncio.to_thread(factory_started.wait, 1)
    controller.shutdown()
    release_factory.set()
    await controller.drain_tasks(timeout=1)

    assert client_shutdown.is_set()
    assert pool.added == []


@pytest.mark.asyncio
async def test_watcher_reconciles_replica_that_disappears_during_registration(monkeypatch):
    input_addr = "tcp://stage-0-replica-7"
    pool = StagePool(0, [])
    hub = FakeHub(
        snapshots=[
            _snapshot(_replica(0, input_addr)),
            _snapshot(),
        ]
    )
    factory_started = threading.Event()
    release_factory = threading.Event()
    client_shutdown = threading.Event()

    def _factory(stage_id, replica_id):
        assert (stage_id, replica_id) == (0, 7)
        factory_started.set()
        assert release_factory.wait(timeout=1)
        return SimpleNamespace(
            client_addresses={"input_address": input_addr},
            shutdown=client_shutdown.set,
        )

    controller = _controller(monkeypatch, pool, hub, remote_replica_factory=_factory)
    controller.WATCH_INTERVAL_S = 0
    original_handle_unregister = controller.handle_unregister
    first_unregister = asyncio.Event()
    second_unregister = asyncio.Event()
    unregister_calls = 0

    async def _track_unregister(stage_id, addr, *args, **kwargs):
        nonlocal unregister_calls
        await original_handle_unregister(stage_id, addr, *args, **kwargs)
        unregister_calls += 1
        if unregister_calls == 1:
            first_unregister.set()
        elif unregister_calls == 2:
            second_unregister.set()

    controller.handle_unregister = _track_unregister  # type: ignore[method-assign]

    await controller.handle_register(0, 7)
    assert await asyncio.to_thread(factory_started.wait, 1)
    watcher = controller.start()

    try:
        await asyncio.wait_for(first_unregister.wait(), timeout=1)
        assert pool.live_num_replicas == 0

        release_factory.set()
        await asyncio.wait_for(second_unregister.wait(), timeout=1)

        assert client_shutdown.is_set()
        assert pool.live_num_replicas == 0
        assert controller._attached_remote_replicas == set()
        assert controller._remote_replica_keys_by_input_addr == {}
    finally:
        release_factory.set()
        controller.shutdown()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
        await controller.drain_tasks(timeout=1)


@pytest.mark.asyncio
async def test_attached_replica_survives_until_hub_observes_it_up(monkeypatch):
    class EmptyHub:
        def __init__(self):
            self.polls = 0

        def get_replica_list(self):
            self.polls += 1
            return SimpleNamespace(replicas=[], timestamp=0.0)

        def close(self):
            pass

    input_addr = "tcp://stage-0-replica-8"
    pool = StagePool(0, [])
    shutdown_calls = []
    hub = EmptyHub()

    def _factory(stage_id, replica_id):
        return SimpleNamespace(
            client_addresses={"input_address": input_addr},
            shutdown=lambda: shutdown_calls.append((stage_id, replica_id)),
        )

    controller = _controller(monkeypatch, pool, hub, remote_replica_factory=_factory)
    controller.WATCH_INTERVAL_S = 0.001

    await controller.handle_register(0, 8)
    await controller.drain_tasks(timeout=1)
    assert pool.live_num_replicas == 1

    watcher = controller.start()
    try:
        await _wait_until(lambda: hub.polls >= 3)

        assert shutdown_calls == []
        assert pool.live_num_replicas == 1
    finally:
        controller.shutdown()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
        await controller.drain_tasks(timeout=1)


@pytest.mark.asyncio
async def test_up_observed_before_attach_does_not_qualify_attachment(monkeypatch):
    import vllm_omni.engine.membership_controller as membership_mod

    input_addr = "tcp://stage-0-replica-10"
    pool = StagePool(0, [])
    hub = FakeHub(
        snapshots=[
            _snapshot(_replica(0, input_addr)),
            _snapshot(),
        ]
    )
    shutdown_calls = []
    sleep_gates = asyncio.Queue()

    async def _controlled_sleep(_delay):
        gate = asyncio.Event()
        await sleep_gates.put(gate)
        await gate.wait()

    monkeypatch.setattr(membership_mod.asyncio, "sleep", _controlled_sleep)

    def _factory(stage_id, replica_id):
        return SimpleNamespace(
            client_addresses={"input_address": input_addr},
            shutdown=lambda: shutdown_calls.append((stage_id, replica_id)),
        )

    controller = _controller(monkeypatch, pool, hub, remote_replica_factory=_factory)
    watcher = controller.start()

    try:
        first_sleep = await asyncio.wait_for(sleep_gates.get(), timeout=1)
        await controller.handle_register(0, 10)
        await controller.drain_tasks(timeout=1)
        assert pool.live_num_replicas == 1

        first_sleep.set()
        await asyncio.wait_for(sleep_gates.get(), timeout=1)
        await controller.drain_tasks(timeout=1)

        assert shutdown_calls == []
        assert pool.live_num_replicas == 1
    finally:
        controller.shutdown()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
        await controller.drain_tasks(timeout=1)


@pytest.mark.asyncio
async def test_reattached_address_requires_a_fresh_up_observation(monkeypatch):
    class MutableHub:
        def __init__(self, input_addr):
            self.input_addr = input_addr
            self.up = True
            self.polls = 0

        def get_replica_list(self):
            self.polls += 1
            replicas = [_replica(0, self.input_addr)] if self.up else []
            return _snapshot(*replicas)

        def close(self):
            pass

    class Client:
        def __init__(self, input_addr):
            self.client_addresses = {"input_address": input_addr}
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    input_addr = "tcp://stage-0-replica-9"
    pool = StagePool(0, [])
    hub = MutableHub(input_addr)
    clients = []

    def _factory(_stage_id, _replica_id):
        client = Client(input_addr)
        clients.append(client)
        return client

    controller = _controller(monkeypatch, pool, hub, remote_replica_factory=_factory)
    controller.WATCH_INTERVAL_S = 0.001

    await controller.handle_register(0, 9)
    await controller.drain_tasks(timeout=1)
    watcher = controller.start()

    try:
        await _wait_until(lambda: hub.polls >= 2)
        hub.up = False
        await _wait_until(lambda: clients[0].shutdown_calls == 1)
        assert pool.live_num_replicas == 0

        await controller.handle_register(0, 9)
        await controller.drain_tasks(timeout=1)
        polls_after_reattach = hub.polls
        await _wait_until(lambda: hub.polls >= polls_after_reattach + 3)

        assert len(clients) == 2
        assert clients[1].shutdown_calls == 0
        assert pool.live_num_replicas == 1
    finally:
        controller.shutdown()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
        await controller.drain_tasks(timeout=1)
