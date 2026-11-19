# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading
import time
from collections.abc import Callable
from unittest.mock import Mock

import pytest
import zmq
from vllm.v1.utils import get_engine_client_zmq_addr

from vllm_omni.distributed.omni_coordinator import (
    OmniCoordClientForStage,
    OmniCoordinator,
    ReplicaStatus,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CoordEnv:
    """Sockets and clients owned by one test.

    The process-global ``zmq.Context.instance()`` must not be terminated here.
    ``Context.term()`` waits until every socket on that context is closed, and
    a failed test keeps its sockets alive through the traceback. The next
    test then blocks in ``term()`` with no timeout.
    """

    def __init__(self) -> None:
        self.ctx = zmq.Context()
        self.coordinators: list[OmniCoordinator] = []
        self.clients: list[OmniCoordClientForStage] = []
        self.sockets: list[zmq.Socket] = []

    def coordinator(self, heartbeat_timeout: float) -> OmniCoordinator:
        router_addr = get_engine_client_zmq_addr(local_only=False, host="127.0.0.1", port=0)
        pub_addr = get_engine_client_zmq_addr(local_only=False, host="127.0.0.1", port=0)
        coord = OmniCoordinator(
            router_zmq_addr=router_addr,
            pub_zmq_addr=pub_addr,
            heartbeat_timeout=heartbeat_timeout,
        )
        self.coordinators.append(coord)
        return coord

    def socket(self, socket_type: int) -> zmq.Socket:
        sock = self.ctx.socket(socket_type)
        sock.setsockopt(zmq.LINGER, 0)
        self.sockets.append(sock)
        return sock

    def sub(self, coordinator: OmniCoordinator) -> zmq.Socket:
        sock = self.socket(zmq.SUB)
        sock.connect(coordinator.pub_zmq_addr)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        return sock

    def stage_client(
        self,
        coordinator: OmniCoordinator,
        input_addr: str,
        output_addr: str,
        stage_id: int,
    ) -> OmniCoordClientForStage:
        client = OmniCoordClientForStage(coordinator.router_zmq_addr, input_addr, output_addr, stage_id)
        self.clients.append(client)
        return client

    def close(self) -> None:
        for client in self.clients:
            if not client._closed:
                client.close()
        for coord in self.coordinators:
            if not coord._closed:
                coord.close()
        for sock in self.sockets:
            if not sock.closed:
                sock.close(0)
        if not self.ctx.closed:
            self.ctx.term()


@pytest.fixture
def coord_env():
    env = _CoordEnv()
    try:
        yield env
    finally:
        env.close()


def _recv_replica_list(sub: zmq.Socket, timeout_ms: int = 2000) -> dict | None:
    """Receive ReplicaList JSON from SUB socket. Returns None on timeout."""
    sub.setsockopt(zmq.RCVTIMEO, timeout_ms)
    try:
        data = sub.recv()
        return json.loads(data.decode("utf-8"))
    except zmq.Again:
        return None


def _wait_for_replica_list(
    sub: zmq.Socket,
    expected_count: int,
    timeout: float = 3.0,
) -> dict | None:
    """Wait until received ReplicaList with expected_count active replicas."""
    start = time.time()
    while time.time() - start < timeout:
        msg = _recv_replica_list(sub, timeout_ms=500)
        if msg is not None and len(msg.get("replicas", [])) == expected_count:
            return msg
    return None


def _drain_sub_messages(sub: zmq.Socket, max_seconds: float = 0.4) -> None:
    """Drain queued SUB messages for a short window."""
    deadline = time.time() + max_seconds
    while time.time() < deadline:
        _recv_replica_list(sub, timeout_ms=50)


def _wait_until_subscribed(sub: zmq.Socket, coordinator: OmniCoordinator, timeout: float = 2.0) -> None:
    """Publish until SUB delivers one message, then drop that handshake.

    A fixed sleep does not prove the subscription is active. PUB drops
    messages sent before the peer subscribes, and the coordinator's keepalive
    can be several seconds later than this test is willing to wait.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        coordinator.publish_replica_list_update()
        if _recv_replica_list(sub, timeout_ms=50) is not None:
            _drain_sub_messages(sub, max_seconds=0.05)
            return
    raise AssertionError("timed out waiting for PUB/SUB subscription")


def test_omni_coordinator_wait_for_shutdown_unblocks_on_close(coord_env: _CoordEnv):
    coordinator = coord_env.coordinator(heartbeat_timeout=1000.0)

    waiter = threading.Thread(target=coordinator.wait_for_shutdown)
    waiter.start()
    time.sleep(0.05)
    assert waiter.is_alive()

    coordinator.close()
    waiter.join(timeout=1.0)
    assert not waiter.is_alive()


def test_omni_coordinator_pub_coalescing_on_rapid_queue_updates(coord_env: _CoordEnv):
    """Rapid updates should be coalesced into fewer PUB messages."""
    coordinator = coord_env.coordinator(heartbeat_timeout=1000.0)
    sub = coord_env.sub(coordinator)
    _wait_until_subscribed(sub, coordinator)

    client = coord_env.stage_client(coordinator, "tcp://stage:coalesce", "tcp://stage:coalesce-out", 0)

    # Wait for initial registration broadcast and clear any queued messages.
    msg = _wait_for_replica_list(sub, expected_count=1)
    assert msg is not None
    _drain_sub_messages(sub)

    # Burst many queue updates in a short period.
    update_count = 80
    for i in range(update_count):
        client.update_info(queue_length=i)

    # With publish_min_interval=0.1s, received messages over ~1s should be
    # much smaller than update_count (coalescing effect).
    window_s = 1.1
    deadline = time.time() + window_s
    recv_count = 0
    while time.time() < deadline:
        if _recv_replica_list(sub, timeout_ms=100) is not None:
            recv_count += 1

    assert recv_count < update_count // 2, (
        f"expected coalesced PUB traffic, got {recv_count} for {update_count} updates"
    )


def test_omni_coordinator_registration_broadcast(coord_env: _CoordEnv):
    """Verify that after multiple OmniCoordClientForStage replicas register,
    OmniCoordinator publishes a ReplicaList containing all registered replicas.
    """
    coordinator = coord_env.coordinator(heartbeat_timeout=1000.0)
    sub = coord_env.sub(coordinator)
    _wait_until_subscribed(sub, coordinator)

    # Create 3 stage clients; each auto-registers on init.
    coord_env.stage_client(coordinator, "tcp://stage:10001", "tcp://stage:10001-out", 0)
    coord_env.stage_client(coordinator, "tcp://stage:10002", "tcp://stage:10002-out", 0)
    coord_env.stage_client(coordinator, "tcp://stage:10003", "tcp://stage:10003-out", 1)

    msg = _wait_for_replica_list(sub, expected_count=3)
    assert msg is not None, "Expected ReplicaList with 3 replicas"
    assert len(msg["replicas"]) == 3
    assert isinstance(msg["timestamp"], (int, float))

    input_addrs = {rep["input_addr"] for rep in msg["replicas"]}
    assert "tcp://stage:10001" in input_addrs
    assert "tcp://stage:10002" in input_addrs
    assert "tcp://stage:10003" in input_addrs


def test_omni_coordinator_publish_does_not_drop_inflight_registration(coord_env: _CoordEnv):
    """A registration handled during an in-flight PUB flush must still be sent.

    The flush used to clear the pending flag after the snapshot. Replica
    events handled in that window were acknowledged and then never published
    until the next keepalive, which is longer than the registration wait.
    """
    coordinator = coord_env.coordinator(heartbeat_timeout=1000.0)
    sub = coord_env.sub(coordinator)
    _wait_until_subscribed(sub, coordinator)

    entered = threading.Event()
    release = threading.Event()
    stalls_left = {"n": 1}
    orig_publish: Callable[[], bool] = coordinator.publish_replica_list_update

    def _stall_once() -> bool:
        result = orig_publish()
        if stalls_left["n"] > 0:
            stalls_left["n"] -= 1
            entered.set()
            # Stay inside this flush until the test has observed the later
            # registrations. A short wait can resume the loop and clear the
            # flag before that observation.
            release.wait(timeout=5.0)
        return result

    coordinator.publish_replica_list_update = _stall_once  # type: ignore[method-assign]
    try:
        coord_env.stage_client(coordinator, "tcp://stage:10001", "tcp://stage:10001-out", 0)
        assert entered.wait(timeout=2.0), "publish did not start"

        coord_env.stage_client(coordinator, "tcp://stage:10002", "tcp://stage:10002-out", 0)
        coord_env.stage_client(coordinator, "tcp://stage:10003", "tcp://stage:10003-out", 1)
        deadline = time.time() + 2.0
        registered = 0
        pending = False
        while time.time() < deadline:
            with coordinator._lock:
                registered = len(coordinator._replicas)
            with coordinator._pending_lock:
                pending = coordinator._pending_broadcast
            if registered >= 3 and pending:
                break
            time.sleep(0.01)
        assert registered >= 3 and pending, (
            f"expected later registrations to be pending during publish, got replicas={registered} pending={pending}"
        )
    finally:
        release.set()

    msg = _wait_for_replica_list(sub, expected_count=3, timeout=1.5)
    assert msg is not None, "in-flight registrations were dropped until the next keepalive"
    input_addrs = {rep["input_addr"] for rep in msg["replicas"]}
    assert input_addrs == {"tcp://stage:10001", "tcp://stage:10002", "tcp://stage:10003"}


def test_omni_coordinator_retries_failed_publish_without_new_events(coord_env: _CoordEnv, monkeypatch):
    """Retry a failed send on the next tick without another event or keepalive."""
    coordinator = coord_env.coordinator(heartbeat_timeout=1000.0)
    # Drive the production loop synchronously after stopping its worker threads.
    coordinator._running = False
    coordinator._stop_event.set()
    coordinator._recv_thread.join(timeout=2.0)
    coordinator._periodic_thread.join(timeout=2.0)
    assert not coordinator._recv_thread.is_alive()
    assert not coordinator._periodic_thread.is_alive()
    coordinator._running = True

    publish = Mock(side_effect=[False, True])
    heartbeat_check = Mock()
    wait = Mock(side_effect=[False, True])
    monkeypatch.setattr(coordinator, "publish_replica_list_update", publish)
    monkeypatch.setattr(coordinator, "_check_heartbeat_timeouts", heartbeat_check)
    monkeypatch.setattr(coordinator._stop_event, "wait", wait)
    # The first tick schedules a keepalive; frozen time prevents another one.
    monkeypatch.setattr("vllm_omni.distributed.omni_coordinator.omni_coordinator.time", lambda: 100.0)

    coordinator._periodic_loop()

    assert publish.call_count == 2
    heartbeat_check.assert_called_once_with()
    assert wait.call_count == 2
    assert all(call.kwargs == {"timeout": coordinator._publish_min_interval} for call in wait.call_args_list)
    with coordinator._pending_lock:
        assert not coordinator._pending_broadcast


def test_omni_coordinator_heartbeat_timeout_handling(coord_env: _CoordEnv):
    """Verify that when a stage replica stops sending heartbeats,
    OmniCoordinator marks it as unhealthy and excludes it from the active list.
    """
    coordinator = coord_env.coordinator(heartbeat_timeout=5.0)
    sub = coord_env.sub(coordinator)
    _wait_until_subscribed(sub, coordinator)

    # A and B: real clients that send heartbeats every 5s.
    coord_env.stage_client(coordinator, "tcp://stage:a", "tcp://stage:a-out", 0)
    coord_env.stage_client(coordinator, "tcp://stage:b", "tcp://stage:b-out", 0)

    # C: raw DEALER that sends only registration, no heartbeat.
    dealer_c = coord_env.socket(zmq.DEALER)
    dealer_c.setsockopt(zmq.SNDTIMEO, 1000)
    dealer_c.connect(coordinator.router_zmq_addr)
    reg_event = {
        "input_addr": "tcp://stage:c",
        "output_addr": "tcp://stage:c-out",
        "stage_id": 0,
        "event_type": "update",
        "status": ReplicaStatus.UP.value,
        "queue_length": 0,
    }
    dealer_c.send(json.dumps(reg_event).encode("utf-8"))

    msg = _wait_for_replica_list(sub, expected_count=3)
    assert msg is not None, "Expected initial 3 replicas"
    assert len(msg["replicas"]) == 3

    # Wait for heartbeat timeout (timeout=5s, check interval ~2.5s).
    time.sleep(8.0)

    # Receive the update (C should be ERROR and excluded from active list).
    msg_after_timeout = _wait_for_replica_list(sub, expected_count=2, timeout=5.0)
    assert msg_after_timeout is not None, "Expected ReplicaList with 2 replicas after timeout"
    replicas = msg_after_timeout.get("replicas", [])
    input_addrs = {rep["input_addr"] for rep in replicas}

    assert "tcp://stage:a" in input_addrs
    assert "tcp://stage:b" in input_addrs
    assert "tcp://stage:c" not in input_addrs


def test_omni_coordinator_replica_shutdown_handling(coord_env: _CoordEnv):
    """Verify that when a stage replica sends status='down',
    OmniCoordinator removes it from the active list and broadcasts an updated list.
    """
    coordinator = coord_env.coordinator(heartbeat_timeout=1000.0)
    sub = coord_env.sub(coordinator)
    _wait_until_subscribed(sub, coordinator)

    client = coord_env.stage_client(coordinator, "tcp://stage:shutdown", "tcp://stage:shutdown-out", 0)

    msg = _wait_for_replica_list(sub, expected_count=1)
    assert msg is not None
    assert len(msg["replicas"]) == 1
    assert msg["replicas"][0]["input_addr"] == "tcp://stage:shutdown"

    # Send down status (simulating graceful shutdown).
    client.update_info(status=ReplicaStatus.DOWN)

    # Receive updated list (should have 0 active replicas).
    msg = _wait_for_replica_list(sub, expected_count=0)
    assert msg is not None
    assert len(msg["replicas"]) == 0
