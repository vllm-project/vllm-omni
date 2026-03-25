# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time

import pytest
import zmq

from vllm_omni.distributed.omni_coordinator import (
    InstanceList,
    OmniCoordClientForHub,
)


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


def test_hub_client_caches_instance_list_from_pub():
    """Verify OmniCoordClientForHub receives instance list updates from OmniCoordinator and caches for get_instance_list()."""
    ctx, pub, endpoint = _bind_pub()

    client = OmniCoordClientForHub(endpoint)
    # ZMQ PUB/SUB slow-joiner: allow SUB to finish connecting before first send
    time.sleep(0.2)

    now = time.time()
    instances_payload = [
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

    payload = {"instances": instances_payload, "timestamp": now}
    pub.send(json.dumps(payload).encode("utf-8"))

    assert _wait_for_condition(lambda: len(client.get_instance_list().instances) == 3)

    inst_list = client.get_instance_list()
    assert isinstance(inst_list, InstanceList)
    assert len(inst_list.instances) == 3

    for src, inst in zip(instances_payload, inst_list.instances, strict=True):
        assert inst.input_addr == src["input_addr"]
        assert inst.output_addr == src["output_addr"]
        assert inst.stage_id == src["stage_id"]
        assert inst.status.value == src["status"]

    stage0 = client.get_instances_for_stage(0)
    stage1 = client.get_instances_for_stage(1)

    assert all(inst.stage_id == 0 for inst in stage0.instances)
    assert all(inst.stage_id == 1 for inst in stage1.instances)

    # Send an updated list with fewer instances and verify cache refresh.
    updated_payload = {
        "instances": instances_payload[:2],
        "timestamp": now + 1.0,
    }
    pub.send(json.dumps(updated_payload).encode("utf-8"))

    assert _wait_for_condition(lambda: len(client.get_instance_list().instances) == 2)
    updated_list = client.get_instance_list()
    assert len(updated_list.instances) == 2

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
