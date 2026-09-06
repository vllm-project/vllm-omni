import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
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


def test_pack_and_unpack_round_trip_numpy_values():
    payload = {
        "image": np.arange(6, dtype=np.uint8).reshape(2, 3),
        "action": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "scalar": np.float32(3.5),
        "nested": [{"done": np.bool_(True)}],
    }

    decoded = openpi_connection._unpack(openpi_connection._pack(payload))

    np.testing.assert_array_equal(decoded["image"], payload["image"])
    np.testing.assert_allclose(decoded["action"], payload["action"])
    assert decoded["image"].dtype == np.uint8
    assert decoded["action"].dtype == np.float32
    assert decoded["scalar"] == np.float32(3.5)
    assert decoded["nested"][0]["done"] == np.bool_(True)


def test_unpack_accepts_msgpack_numpy_marker_dicts():
    action = np.asarray([[1.0, 2.0]], dtype=np.float32)
    payload = {
        b"actions": {
            b"nd": True,
            b"type": action.dtype.str,
            b"kind": action.dtype.kind,
            b"shape": action.shape,
            b"data": action.tobytes(),
        }
    }

    decoded = openpi_connection._unpack_numpy(payload)

    np.testing.assert_allclose(decoded[b"actions"], action)
    assert decoded[b"actions"].flags.writeable is True
    decoded[b"actions"][:, -1] = 0.0
    np.testing.assert_allclose(decoded[b"actions"], np.asarray([[1.0, 0.0]], dtype=np.float32))


def test_unpack_accepts_openpi_client_ndarray_markers():
    image = np.arange(6, dtype=np.uint8).reshape(2, 3)
    payload = {
        "observation/exterior_image_0_left": {
            b"__ndarray__": True,
            b"dtype": image.dtype.str,
            b"shape": image.shape,
            b"data": image.tobytes(),
        }
    }

    decoded = openpi_connection._unpack_numpy(payload)

    np.testing.assert_array_equal(decoded["observation/exterior_image_0_left"], image)
    assert decoded["observation/exterior_image_0_left"].dtype == np.uint8


def test_unpack_openpi_client_packed_observation():
    openpi_msgpack = pytest.importorskip("openpi_client.msgpack_numpy")
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    obs = {
        "observation/exterior_image_0_left": image,
        "observation/joint_position": np.zeros(7, dtype=np.float32),
    }

    decoded = openpi_connection._unpack(openpi_msgpack.packb(obs))

    np.testing.assert_array_equal(decoded["observation/exterior_image_0_left"], image)
    np.testing.assert_allclose(decoded["observation/joint_position"], obs["observation/joint_position"])


def test_pack_uses_openpi_client_ndarray_markers():
    openpi_msgpack = pytest.importorskip("openpi_client.msgpack_numpy")
    actions = np.asarray([[1.0, 2.0]], dtype=np.float32)

    packed = openpi_connection._pack(actions)
    decoded = openpi_msgpack.unpackb(packed)

    np.testing.assert_allclose(decoded, actions)


def test_unpack_leaves_user_dict_without_numpy_kind_marker_unchanged():
    payload = {
        "metadata": {
            "nd": True,
            "type": "<f4",
            "data": b"user payload",
        }
    }

    decoded = openpi_connection._unpack_numpy(payload)

    assert decoded == payload


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


def test_handle_connection_can_disable_idle_timeout(monkeypatch):
    monkeypatch.setattr(openpi_connection, "_pack", lambda obj: obj)

    async def fail_wait_for(*_args, **_kwargs):
        pytest.fail("asyncio.wait_for should not be used when the idle timeout is disabled")

    monkeypatch.setattr(openpi_connection.asyncio, "wait_for", fail_wait_for)
    websocket = FakeWebSocket([{"type": "websocket.disconnect"}])
    serving = _serving_mock()

    asyncio.run(
        openpi_connection.RobotRealtimeConnection(
            websocket,
            serving,
            idle_timeout=None,
        ).handle_connection()
    )

    assert websocket.accepted is True
    assert websocket.closed is False
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


def test_handle_connection_keeps_interleaved_parallel_session_state(monkeypatch):
    monkeypatch.setattr(openpi_connection, "_pack", lambda obj: obj)
    requests = {
        b"a1": {"prompt": "first-a", "session_id": "session-a"},
        b"b1": {"prompt": "first-b", "session_id": "session-b"},
        b"a2": {"prompt": "second-a", "session_id": "session-a"},
        b"b2": {"prompt": "second-b", "session_id": "session-b"},
    }
    monkeypatch.setattr(openpi_connection, "_unpack", lambda data: dict(requests[data]))
    serving = _serving_mock()
    websocket = FakeWebSocket(
        [{"type": "websocket.receive", "bytes": key} for key in (b"a1", b"b1", b"a2", b"b2")]
        + [{"type": "websocket.disconnect"}]
    )

    asyncio.run(openpi_connection.RobotRealtimeConnection(websocket, serving).handle_connection())

    assert [call.kwargs for call in serving.infer.await_args_list] == [
        {"session_id": "session-a", "reset": True},
        {"session_id": "session-b", "reset": True},
        {"session_id": "session-a", "reset": False},
        {"session_id": "session-b", "reset": False},
    ]


def test_handle_connection_bounds_tracked_sessions_without_resetting_live_ones(monkeypatch):
    """A client may hold one socket across many episodes, minting a fresh
    session id per episode and never sending ``reset`` (RoboLab does exactly
    this). Session tracking must stay bounded, and evicting finished episodes
    must not disturb the sessions that are still running.
    """
    monkeypatch.setattr(openpi_connection, "_pack", lambda obj: obj)
    monkeypatch.setattr(openpi_connection, "MAX_TRACKED_SESSIONS", 8)

    # One long-lived session interleaved with far more short episodes than the cap.
    episodes = 40
    keys = []
    requests = {}
    for episode in range(episodes):
        for session_id in (f"episode-{episode}", "live-env"):
            key = f"{session_id}@{episode}".encode()
            keys.append(key)
            requests[key] = {"prompt": "pick", "session_id": session_id}

    monkeypatch.setattr(openpi_connection, "_unpack", lambda data: dict(requests[data]))
    serving = _serving_mock()
    websocket = FakeWebSocket(
        [{"type": "websocket.receive", "bytes": key} for key in keys] + [{"type": "websocket.disconnect"}]
    )
    connection = openpi_connection.RobotRealtimeConnection(websocket, serving)

    asyncio.run(connection.handle_connection())

    assert len(connection._seen_sessions) <= 8

    resets_by_session: dict[str, list[bool]] = {}
    for call in serving.infer.await_args_list:
        resets_by_session.setdefault(call.kwargs["session_id"], []).append(call.kwargs["reset"])

    # Each new episode resets exactly once, on its first observation.
    for episode in range(episodes):
        assert resets_by_session[f"episode-{episode}"] == [True]
    # The continuously active session is never evicted, so it resets only once
    # despite ``episodes`` other sessions churning past the cap.
    assert resets_by_session["live-env"] == [True] + [False] * (episodes - 1)


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
