# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time

import zmq
from vllm.v1.utils import get_engine_client_zmq_addr

from vllm_omni.distributed.omni_coordinator import (
    OmniCoordClientForStage,
    OmniCoordinator,
    StageStatus,
)


def _recv_instance_list(sub: zmq.Socket, timeout_ms: int = 2000) -> dict | None:
    """Receive InstanceList JSON from SUB socket. Returns None on timeout."""
    sub.setsockopt(zmq.RCVTIMEO, timeout_ms)
    try:
        data = sub.recv()
        return json.loads(data.decode("utf-8"))
    except zmq.Again:
        return None


def _wait_for_instance_list(
    sub: zmq.Socket,
    expected_count: int,
    timeout: float = 3.0,
) -> dict | None:
    """Wait until received InstanceList with expected_count active instances."""
    start = time.time()
    while time.time() - start < timeout:
        msg = _recv_instance_list(sub, timeout_ms=500)
        if msg is not None and len(msg.get("instances", [])) == expected_count:
            return msg
    return None


def test_omni_coordinator_registration_broadcast():
    """Verify that after multiple OmniCoordClientForStage instances register,
    OmniCoordinator publishes an InstanceList containing all registered instances.
    """
    router_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    pub_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    coordinator = OmniCoordinator(
        router_zmq_addr=router_addr,
        pub_zmq_addr=pub_addr,
        heartbeat_timeout=1000.0,
    )

    sub_ctx = zmq.Context.instance()
    sub = sub_ctx.socket(zmq.SUB)
    sub.connect(pub_addr)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    # ZMQ PUB/SUB slow-joiner: allow SUB to connect before clients register.
    time.sleep(0.3)

    # Create 3 stage clients; each auto-registers on init.
    clients = [
        OmniCoordClientForStage(router_addr, "tcp://stage:10001", "tcp://stage:10001-out", 0),
        OmniCoordClientForStage(router_addr, "tcp://stage:10002", "tcp://stage:10002-out", 0),
        OmniCoordClientForStage(router_addr, "tcp://stage:10003", "tcp://stage:10003-out", 1),
    ]

    msg = _wait_for_instance_list(sub, expected_count=3)
    assert msg is not None, "Expected InstanceList with 3 instances"
    assert len(msg["instances"]) == 3
    assert isinstance(msg["timestamp"], (int, float))

    input_addrs = {inst["input_addr"] for inst in msg["instances"]}
    assert "tcp://stage:10001" in input_addrs
    assert "tcp://stage:10002" in input_addrs
    assert "tcp://stage:10003" in input_addrs

    for c in clients:
        c.close()
    coordinator.close()
    sub.close(0)
    sub_ctx.term()


def test_omni_coordinator_heartbeat_timeout_handling():
    """Verify that when a stage instance stops sending heartbeats,
    OmniCoordinator marks it as unhealthy and excludes it from the active list.
    """
    router_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    pub_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    coordinator = OmniCoordinator(
        router_zmq_addr=router_addr,
        pub_zmq_addr=pub_addr,
        heartbeat_timeout=5.0,
    )

    sub_ctx = zmq.Context.instance()
    sub = sub_ctx.socket(zmq.SUB)
    sub.connect(pub_addr)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    time.sleep(0.3)

    # A and B: real clients that send heartbeats every 5s.
    client_a = OmniCoordClientForStage(router_addr, "tcp://stage:a", "tcp://stage:a-out", 0)
    client_b = OmniCoordClientForStage(router_addr, "tcp://stage:b", "tcp://stage:b-out", 0)

    # C: raw DEALER that sends only registration, no heartbeat.
    dealer_ctx = zmq.Context.instance()
    dealer_c = dealer_ctx.socket(zmq.DEALER)
    dealer_c.connect(router_addr)
    reg_event = {
        "input_addr": "tcp://stage:c",
        "output_addr": "tcp://stage:c-out",
        "stage_id": 0,
        "event_type": "update",
        "status": StageStatus.UP.value,
        "queue_length": 0,
    }
    dealer_c.send(json.dumps(reg_event).encode("utf-8"))

    msg = _wait_for_instance_list(sub, expected_count=3)
    assert msg is not None, "Expected initial 3 instances"
    assert len(msg["instances"]) == 3

    # Wait for heartbeat timeout (timeout=5s, check interval ~2.5s).
    time.sleep(8.0)

    # Receive the update (C should be ERROR and excluded from active list).
    msg_after_timeout = _wait_for_instance_list(sub, expected_count=2, timeout=5.0)
    assert msg_after_timeout is not None, "Expected InstanceList with 2 instances after timeout"
    instances = msg_after_timeout.get("instances", [])
    input_addrs = {inst["input_addr"] for inst in instances}

    assert "tcp://stage:a" in input_addrs
    assert "tcp://stage:b" in input_addrs
    assert "tcp://stage:c" not in input_addrs

    client_a.close()
    client_b.close()
    dealer_c.close(0)
    coordinator.close()
    sub.close(0)
    dealer_ctx.term()
    sub_ctx.term()


def test_omni_coordinator_instance_shutdown_handling():
    """Verify that when a stage instance sends status='down',
    OmniCoordinator removes it from the active list and broadcasts an updated list.
    """
    router_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    pub_addr = get_engine_client_zmq_addr(
        local_only=False,
        host="127.0.0.1",
        port=0,
    )
    coordinator = OmniCoordinator(
        router_zmq_addr=router_addr,
        pub_zmq_addr=pub_addr,
        heartbeat_timeout=1000.0,
    )

    sub_ctx = zmq.Context.instance()
    sub = sub_ctx.socket(zmq.SUB)
    sub.connect(pub_addr)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    time.sleep(0.3)  # PUB/SUB slow-joiner

    client = OmniCoordClientForStage(router_addr, "tcp://stage:shutdown", "tcp://stage:shutdown-out", 0)

    msg = _wait_for_instance_list(sub, expected_count=1)
    assert msg is not None
    assert len(msg["instances"]) == 1
    assert msg["instances"][0]["input_addr"] == "tcp://stage:shutdown"

    # Send down status (simulating graceful shutdown).
    client.update_info(status=StageStatus.DOWN)

    # Receive updated list (should have 0 active instances).
    msg = _wait_for_instance_list(sub, expected_count=0)
    assert msg is not None
    assert len(msg["instances"]) == 0

    client.close()
    coordinator.close()
    sub.close(0)
    sub_ctx.term()
