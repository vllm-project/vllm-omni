# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time

import pytest
import zmq

from vllm_omni.distributed.omni_coordinator import (
    OmniCoordClientForHub,
    ReplicaList,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _bind_pub() -> tuple[zmq.Context, zmq.Socket, str]:
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:*")
    endpoint = pub.getsockopt(zmq.LAST_ENDPOINT).decode("ascii")
    return ctx, pub, endpoint


def _wait_for_condition(cond, timeout: float = 2.0, interval: float = 0.01) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if cond():
            return True
        time.sleep(interval)
    return False


def test_hub_client_caches_replica_list_from_pub():
    """Verify OmniCoordClientForHub receives replica list updates from OmniCoordinator and caches for get_replica_list()."""
    ctx, pub, endpoint = _bind_pub()

    client = OmniCoordClientForHub(endpoint)
    # ZMQ PUB/SUB slow-joiner: allow SUB to finish connecting before first send
    time.sleep(0.2)

    now = time.time()
    replicas_payload = [
        {
            "input_addr": "tcp://stage:10001",
            "output_addr": "tcp://stage:10001-out",
            "stage_id": 0,
            "status": "up",
            "queue_length": 0,
            "last_heartbeat": now,
            "registered_at": now,
        },
        {
            "input_addr": "tcp://stage:10002",
            "output_addr": "tcp://stage:10002-out",
            "stage_id": 0,
            "status": "up",
            "queue_length": 1,
            "last_heartbeat": now,
            "registered_at": now,
        },
        {
            "input_addr": "tcp://stage:10003",
            "output_addr": "tcp://stage:10003-out",
            "stage_id": 1,
            "status": "error",
            "queue_length": 5,
            "last_heartbeat": now,
            "registered_at": now,
        },
    ]

    payload = {"replicas": replicas_payload, "timestamp": now}
    pub.send(json.dumps(payload).encode("utf-8"))

    assert _wait_for_condition(lambda: len(client.get_replica_list().replicas) == 3)

    rep_list = client.get_replica_list()
    assert isinstance(rep_list, ReplicaList)
    assert len(rep_list.replicas) == 3

    for src, rep in zip(replicas_payload, rep_list.replicas, strict=True):
        assert rep.input_addr == src["input_addr"]
        assert rep.output_addr == src["output_addr"]
        assert rep.stage_id == src["stage_id"]
        assert rep.status.value == src["status"]

    stage0 = client.get_replicas_for_stage(0)
    stage1 = client.get_replicas_for_stage(1)

    assert all(rep.stage_id == 0 for rep in stage0.replicas)
    assert all(rep.stage_id == 1 for rep in stage1.replicas)

    # Send an updated list with fewer replicas and verify cache refresh.
    updated_payload = {
        "replicas": replicas_payload[:2],
        "timestamp": now + 1.0,
    }
    pub.send(json.dumps(updated_payload).encode("utf-8"))

    assert _wait_for_condition(lambda: len(client.get_replica_list().replicas) == 2)
    updated_list = client.get_replica_list()
    assert len(updated_list.replicas) == 2

    client.close()
    pub.close(0)
    ctx.term()


def test_hub_client_close_closes_sub_socket():
    """Verify OmniCoordClientForHub.close() marks client as closed; second close raises."""
    ctx, pub, endpoint = _bind_pub()
    client = OmniCoordClientForHub(endpoint)
    client.close()

    with pytest.raises(RuntimeError, match="already closed"):
        client.close()

    pub.close(0)
    ctx.term()
