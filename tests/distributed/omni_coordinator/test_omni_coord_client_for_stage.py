# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading

import pytest
import zmq

from vllm_omni.distributed.omni_coordinator import (
    OmniCoordClientForStage,
    StageStatus,
)
from vllm_omni.distributed.omni_coordinator import (
    omni_coord_client_for_stage as stage_client_module,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _bind_router() -> tuple[zmq.Context, zmq.Socket, str]:
    ctx = zmq.Context.instance()
    router = ctx.socket(zmq.ROUTER)
    router.bind("tcp://127.0.0.1:*")
    endpoint = router.getsockopt(zmq.LAST_ENDPOINT).decode("ascii")
    return ctx, router, endpoint


def _recv_event(router: zmq.Socket, timeout_ms: int = 2000) -> dict:
    assert router.poll(timeout=timeout_ms) != 0, "Timed out waiting for coordinator event"
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


def test_stage_client_reconnects_after_send_failure(mocker):
    """Verify send failure path invokes reconnect before retrying send."""
    ctx, router, endpoint = _bind_router()

    client = OmniCoordClientForStage(
        endpoint,
        "tcp://stage:reconnect-in",
        "tcp://stage:reconnect-out",
        0,
    )

    # Discard initial registration event from the real socket.
    _recv_event(router)

    class _FlakySocket:
        def __init__(self):
            self.send_calls = 0
            self.closed = False

        def send(self, *_args, **_kwargs):
            self.send_calls += 1
            if self.send_calls == 1:
                raise RuntimeError("simulated send failure")

        def close(self, *_args, **_kwargs):
            self.closed = True

    flaky_socket = _FlakySocket()
    client._socket = flaky_socket
    client._reconnect = mocker.Mock(return_value=True)

    client.update_info(queue_length=1)

    client._reconnect.assert_called_once_with(max_retries=3)
    assert flaky_socket.send_calls == 2

    client.close()
    router.close(0)
    ctx.term()


def test_stage_client_raises_when_reconnect_fails(mocker):
    """Verify send failure is propagated when reconnect cannot recover."""
    ctx, router, endpoint = _bind_router()

    client = OmniCoordClientForStage(
        endpoint,
        "tcp://stage:reconnect-fail-in",
        "tcp://stage:reconnect-fail-out",
        0,
    )

    # Discard initial registration event from the real socket.
    _recv_event(router)

    class _AlwaysFailSocket:
        def send(self, *_args, **_kwargs):
            raise RuntimeError("simulated send failure")

        def close(self, *_args, **_kwargs):
            pass

    client._socket = _AlwaysFailSocket()
    client._reconnect = mocker.Mock(return_value=False)

    with pytest.raises(RuntimeError, match="simulated send failure"):
        client.update_info(queue_length=2)

    client._reconnect.assert_called_once_with(max_retries=3)
    client.close()
    router.close(0)
    ctx.term()


def test_stage_client_close_handles_runtime_error_in_final_update(mocker):
    """Verify close() still releases resources when final update raises RuntimeError."""
    ctx, router, endpoint = _bind_router()

    client = OmniCoordClientForStage(
        endpoint,
        "tcp://stage:close-runtime-in",
        "tcp://stage:close-runtime-out",
        0,
    )

    # Discard initial registration event from the real socket.
    _recv_event(router)

    client._send_event = mocker.Mock(side_effect=RuntimeError("simulated close-time failure"))
    client.close()

    assert client._closed
    assert client._socket.closed

    router.close(0)
    ctx.term()


def test_reconnect_respects_retry_limit(monkeypatch):
    """Verify _reconnect stops after max_retries on repeated failures."""
    attempts = {"connect": 0}

    class _FailSocket:
        def close(self, *_args, **_kwargs):
            pass

        def connect(self, *_args, **_kwargs):
            attempts["connect"] += 1
            raise zmq.ZMQError("simulated reconnect failure")

    class _FailContext:
        def socket(self, *_args, **_kwargs):
            return _FailSocket()

        def term(self):
            pass

    client = OmniCoordClientForStage.__new__(OmniCoordClientForStage)
    client._closed = False
    client._coord_zmq_addr = "tcp://127.0.0.1:9999"
    client._stop_event = threading.Event()
    client._send_lock = threading.RLock()
    client._socket = _FailSocket()
    client._ctx = _FailContext()

    monkeypatch.setattr(stage_client_module.zmq, "Context", lambda: _FailContext())
    monkeypatch.setattr(stage_client_module.time, "sleep", lambda *_args, **_kwargs: None)

    assert client._reconnect(max_retries=3, retry_interval=5.0) is False
    assert attempts["connect"] == 3


def test_heartbeat_loop_retries_after_transient_send_failure():
    """Verify heartbeat loop continues after one transient send failure."""

    class _FakeStopEvent:
        def __init__(self):
            self.wait_calls = 0
            self._set = False

        def wait(self, timeout=None):
            _ = timeout
            self.wait_calls += 1
            # Run two loop iterations, then stop.
            return self._set or self.wait_calls >= 3

        def is_set(self):
            return self._set

        def set(self):
            self._set = True

    client = OmniCoordClientForStage.__new__(OmniCoordClientForStage)
    client._closed = False
    client._heartbeat_interval = 0.0
    client._stop_event = _FakeStopEvent()

    calls = {"count": 0}

    def _fake_send(event_type):
        assert event_type == "heartbeat"
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient heartbeat failure")

    client._send_event = _fake_send

    client._heartbeat_loop()

    assert calls["count"] == 2


def test_update_info_rejected_while_closing():
    """Verify update_info is rejected once client enters closing state."""
    ctx, router, endpoint = _bind_router()

    client = OmniCoordClientForStage(
        endpoint,
        "tcp://stage:closing-in",
        "tcp://stage:closing-out",
        0,
    )
    _recv_event(router)

    client._closing = True
    with pytest.raises(RuntimeError, match="closing"):
        client.update_info(queue_length=3)

    client._closing = False
    client.close()
    router.close(0)
    ctx.term()
