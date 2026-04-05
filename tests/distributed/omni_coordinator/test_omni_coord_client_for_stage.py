# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import zmq

from vllm_omni.distributed.omni_coordinator import (
    OmniCoordClientForStage,
    StageStatus,
)


def _bind_router() -> tuple[zmq.Context, zmq.Socket, str]:
    ctx = zmq.Context.instance()
    router = ctx.socket(zmq.ROUTER)
    router.bind("tcp://127.0.0.1:*")
    endpoint = router.getsockopt(zmq.LAST_ENDPOINT).decode("ascii")
    return ctx, router, endpoint


def _recv_event(router: zmq.Socket) -> dict:
    frames = router.recv_multipart()
    # ROUTER adds identity frame; the last frame is the payload.
    payload = frames[-1]
    return json.loads(payload.decode("utf-8"))


def test_stage_client_auto_register_on_init():
    """Verify OmniCoordClientForStage automatically sends initial registration/status-up event when created."""
    ctx, router, endpoint = _bind_router()

    input_addr = "tcp://stage:10001"
    output_addr = "tcp://stage:10001-out"
    stage_id = 0

    client = OmniCoordClientForStage(endpoint, input_addr, output_addr, stage_id)

    event = _recv_event(router)

    assert event["event_type"] == "update"
    assert event["status"] == StageStatus.UP.value
    assert event["stage_id"] == stage_id
    assert event["input_addr"] == input_addr
    assert event["output_addr"] == output_addr

    client.close()
    router.close(0)
    ctx.term()


def test_stage_client_update_info_sends_correct_event():
    """Verify OmniCoordClientForStage.update_info() sends status/load update events with expected fields."""
    ctx, router, endpoint = _bind_router()

    input_addr = "tcp://stage:10002"
    output_addr = "tcp://stage:10002-out"
    stage_id = 1

    client = OmniCoordClientForStage(endpoint, input_addr, output_addr, stage_id)

    # Discard initial registration event.
    _recv_event(router)

    client.update_info(status=StageStatus.ERROR)
    client.update_info(queue_length=10)

    first = _recv_event(router)
    second = _recv_event(router)

    assert first["status"] == StageStatus.ERROR.value
    assert first["stage_id"] == stage_id
    assert first["input_addr"] == input_addr
    assert first["output_addr"] == output_addr

    assert second["queue_length"] == 10
    assert second["stage_id"] == stage_id
    assert second["input_addr"] == input_addr
    assert second["output_addr"] == output_addr

    client.close()
    router.close(0)
    ctx.term()


def test_stage_client_close_sends_down_status():
    """Verify close() sends final status-down event before closing underlying socket."""
    ctx, router, endpoint = _bind_router()

    input_addr = "tcp://stage:10003"
    output_addr = "tcp://stage:10003-out"
    stage_id = 2

    client = OmniCoordClientForStage(endpoint, input_addr, output_addr, stage_id)

    # Discard initial registration event.
    _recv_event(router)

    client.close()

    event = _recv_event(router)
    assert event["status"] == StageStatus.DOWN.value
    assert event["stage_id"] == stage_id
    assert event["input_addr"] == input_addr
    assert event["output_addr"] == output_addr

    assert client._socket.closed  # DEALER socket no longer usable after close

    router.close(0)
    ctx.term()
