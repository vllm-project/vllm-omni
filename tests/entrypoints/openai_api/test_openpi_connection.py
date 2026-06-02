import asyncio
import builtins
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm_omni.entrypoints.openpi import connection as openpi_connection
from vllm_omni.entrypoints.openpi.serving import PolicyServerConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeWebSocket:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent_bytes = []
        self.sent_texts = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def send_bytes(self, data):
        self.sent_bytes.append(data)

    async def send_text(self, data):
        self.sent_texts.append(data)

    async def receive(self):
        return self._messages.pop(0)

    async def close(self):
        self.closed = True


def _serving_mock():
    serving = MagicMock()
    serving.policy_server_config = PolicyServerConfig(
        {
            "image_resolution": (180, 320),
            "n_external_cameras": 2,
            "needs_wrist_camera": True,
            "needs_stereo_camera": False,
            "needs_session_id": True,
            "action_space": "joint_position",
        }
    )
    serving.infer = AsyncMock(return_value=[0.0])
    return serving


def test_pack_reports_clear_error_when_openpi_client_is_missing(monkeypatch):
    real_import = builtins.__import__

    def import_without_openpi_client(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openpi_client":
            raise ModuleNotFoundError("No module named 'openpi_client'", name="openpi_client")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_openpi_client)

    with pytest.raises(ImportError) as exc_info:
        openpi_connection._pack({"prompt": "pick up the object"})

    message = str(exc_info.value)
    assert "/v1/realtime/robot/openpi" in message
    assert "pip install openpi-client" in message


def test_pack_and_unpack_delegate_to_openpi_msgpack_numpy(monkeypatch):
    calls = []

    class FakeMsgpackNumpy:
        @staticmethod
        def packb(obj):
            calls.append(("packb", obj))
            return b"packed"

        @staticmethod
        def unpackb(data):
            calls.append(("unpackb", data))
            return {"unpacked": data}

    fake_openpi_client = types.ModuleType("openpi_client")
    fake_openpi_client.msgpack_numpy = FakeMsgpackNumpy
    monkeypatch.setitem(sys.modules, "openpi_client", fake_openpi_client)

    assert openpi_connection._pack({"x": 1}) == b"packed"
    assert openpi_connection._unpack(b"payload") == {"unpacked": b"payload"}
    assert calls == [
        ("packb", {"x": 1}),
        ("unpackb", b"payload"),
    ]


def test_handle_connection_returns_structured_error_for_invalid_payload(monkeypatch):
    monkeypatch.setattr(openpi_connection, "_pack", lambda obj: obj)
    monkeypatch.setattr(
        openpi_connection,
        "_unpack",
        lambda _data: (_ for _ in ()).throw(ValueError("bad payload traceback")),
    )

    websocket = FakeWebSocket(
        [
            {"type": "websocket.receive", "bytes": b"bad"},
            {"type": "websocket.disconnect"},
        ]
    )
    serving = MagicMock()
    connection = openpi_connection.RobotRealtimeConnection(websocket, serving)

    asyncio.run(connection.handle_connection())

    assert websocket.accepted is True
    assert websocket.sent_bytes[1] == {"type": "error", "message": "Invalid request payload"}
    assert "traceback" not in str(websocket.sent_bytes[1]).lower()
    assert websocket.sent_texts == []
    serving.infer.assert_not_called()
    serving.reset.assert_not_called()


def test_handle_connection_rejects_oversized_payload_before_unpack(monkeypatch):
    unpack_mock = MagicMock(side_effect=AssertionError("_unpack should not be called"))
    monkeypatch.setattr(openpi_connection, "_pack", lambda obj: obj)
    monkeypatch.setattr(openpi_connection, "_unpack", unpack_mock)
    monkeypatch.setattr(openpi_connection, "MAX_OPENPI_PAYLOAD_BYTES", 4)

    websocket = FakeWebSocket(
        [
            {"type": "websocket.receive", "bytes": b"too-large"},
            {"type": "websocket.disconnect"},
        ]
    )
    serving = MagicMock()
    connection = openpi_connection.RobotRealtimeConnection(websocket, serving)

    asyncio.run(connection.handle_connection())

    assert websocket.sent_bytes[1] == {"type": "error", "message": "Invalid request payload"}
    unpack_mock.assert_not_called()
    serving.infer.assert_not_called()
    serving.reset.assert_not_called()


def test_handle_connection_returns_structured_error_for_infer_exception(monkeypatch):
    monkeypatch.setattr(openpi_connection, "_pack", lambda obj: obj)
    monkeypatch.setattr(
        openpi_connection,
        "_unpack",
        lambda _data: {"prompt": "pick up the object"},
    )

    websocket = FakeWebSocket(
        [
            {"type": "websocket.receive", "bytes": b"request"},
            {"type": "websocket.disconnect"},
        ]
    )
    serving = MagicMock()
    serving.infer = AsyncMock(side_effect=RuntimeError("secret traceback text"))
    connection = openpi_connection.RobotRealtimeConnection(websocket, serving)

    asyncio.run(connection.handle_connection())

    assert websocket.sent_bytes[1] == {"type": "error", "message": "Internal inference error"}
    assert "secret traceback text" not in str(websocket.sent_bytes[1])
    assert websocket.sent_texts == []
    serving.infer.assert_awaited_once_with(
        {"prompt": "pick up the object"},
        session_id="default",
        reset=True,
    )


def test_handle_connection_closes_websocket_on_idle_timeout(monkeypatch):
    monkeypatch.setattr(openpi_connection, "_pack", lambda obj: obj)

    websocket = FakeWebSocket([])

    async def never_receives():
        await asyncio.sleep(1)

    websocket.receive = never_receives
    serving = MagicMock()
    serving.policy_server_config = PolicyServerConfig(
        {
            "image_resolution": (180, 320),
            "n_external_cameras": 2,
            "needs_wrist_camera": True,
            "needs_stereo_camera": False,
            "needs_session_id": True,
            "action_space": "joint_position",
        }
    )
    connection = openpi_connection.RobotRealtimeConnection(
        websocket,
        serving,
        idle_timeout=0.01,
    )

    asyncio.run(connection.handle_connection())

    assert websocket.accepted is True
    assert websocket.sent_bytes[0]["action_space"] == "joint_position"
    assert websocket.closed is True
    assert websocket.sent_texts == []
    serving.infer.assert_not_called()


def test_handle_connection_keeps_session_state_per_websocket(monkeypatch):
    monkeypatch.setattr(openpi_connection, "_pack", lambda obj: obj)
    requests = {
        b"a1": {"prompt": "first", "session_id": "session-a"},
        b"a2": {"prompt": "second", "session_id": "session-a"},
        b"b1": {"prompt": "other", "session_id": "session-b"},
    }
    monkeypatch.setattr(openpi_connection, "_unpack", lambda data: dict(requests[data]))
    serving = _serving_mock()

    websocket_a = FakeWebSocket(
        [
            {"type": "websocket.receive", "bytes": b"a1"},
            {"type": "websocket.receive", "bytes": b"a2"},
            {"type": "websocket.disconnect"},
        ]
    )
    websocket_b = FakeWebSocket(
        [
            {"type": "websocket.receive", "bytes": b"b1"},
            {"type": "websocket.disconnect"},
        ]
    )

    asyncio.run(openpi_connection.RobotRealtimeConnection(websocket_a, serving).handle_connection())
    asyncio.run(openpi_connection.RobotRealtimeConnection(websocket_b, serving).handle_connection())

    calls = serving.infer.await_args_list
    assert calls[0].kwargs == {"session_id": "session-a", "reset": True}
    assert calls[1].kwargs == {"session_id": "session-a", "reset": False}
    assert calls[2].kwargs == {"session_id": "session-b", "reset": True}


def test_handle_connection_reset_endpoint_resets_next_infer(monkeypatch):
    monkeypatch.setattr(openpi_connection, "_pack", lambda obj: obj)
    requests = {
        b"a1": {"prompt": "first", "session_id": "session-a"},
        b"reset": {"endpoint": "reset"},
        b"a2": {"prompt": "second", "session_id": "session-a"},
    }
    monkeypatch.setattr(openpi_connection, "_unpack", lambda data: dict(requests[data]))
    serving = _serving_mock()
    websocket = FakeWebSocket(
        [
            {"type": "websocket.receive", "bytes": b"a1"},
            {"type": "websocket.receive", "bytes": b"reset"},
            {"type": "websocket.receive", "bytes": b"a2"},
            {"type": "websocket.disconnect"},
        ]
    )

    asyncio.run(openpi_connection.RobotRealtimeConnection(websocket, serving).handle_connection())

    assert [call.kwargs["reset"] for call in serving.infer.await_args_list] == [True, True]
    serving.reset.assert_called_once_with({})
    assert websocket.sent_bytes[2] == {"status": "reset successful"}
    assert websocket.sent_texts == []
