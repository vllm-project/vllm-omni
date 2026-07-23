# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L1 protocol-validation tests for ``/v1/realtime/world/camera``."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from tests.diffusion.models.lingbot_world_fast.conftest import make_dummy_camera_inputs
from vllm_omni.entrypoints.openai.realtime.world.camera_connection import (
    DEFAULT_FRAMES_PER_CHUNK,
    WorldCameraRealtimeConnection,
)
from vllm_omni.entrypoints.openai.realtime.world.camera_serving import (
    CameraServerConfig,
    ServingRealtimeWorldCamera,
)

# The endpoint's wire codec is provided by the optional openpi-client dep.
msgpack_numpy = pytest.importorskip("openpi_client.msgpack_numpy")

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


class MockWebSocket:
    def __init__(self, incoming: Iterable[dict[str, Any]] | None = None) -> None:
        self._incoming: list[dict[str, Any]] = list(incoming or [])
        self._idx = 0
        self.sent_bytes: list[bytes] = []
        self.sent_text: list[str] = []
        self.accepted = False
        self.closed = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict[str, Any]:
        if self._idx >= len(self._incoming):
            return {"type": "websocket.disconnect"}
        msg = self._incoming[self._idx]
        self._idx += 1
        return msg

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def send_text(self, data: str) -> None:
        self.sent_text.append(data)

    async def close(self) -> None:
        self.closed = True


def _bytes_frame(payload: Any) -> dict[str, Any]:
    return {"type": "websocket.receive", "bytes": msgpack_numpy.packb(payload)}


def _raw_bytes_frame(data: bytes) -> dict[str, Any]:
    return {"type": "websocket.receive", "bytes": data}


class _AsyncIter:
    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class FakeResult:
    def __init__(self, frames: np.ndarray) -> None:
        self.images = [frames]


class FakeEngineClient:
    """Stand-in for ``AsyncOmni`` engine client.

    Captures the ``generate(...)`` arguments and yields a single fake result
    so the connection's framing logic can be exercised without a real engine.
    """

    def __init__(self, frames: np.ndarray | None = None) -> None:
        if frames is None:
            # Default: CHUNK_FRAMES*(1+1/2) RGB frames so we exercise the chunk split.
            frames = np.zeros((DEFAULT_FRAMES_PER_CHUNK * 3 // 2, 16, 16, 3), dtype=np.uint8)
        self._frames = frames
        self.calls: list[dict[str, Any]] = []
        self.fail_with: Exception | None = None
        # Attributes consulted by ``CameraServerConfig.from_model_config``.
        self.model_config = {"pipeline": "lingbot_world_fast", "resolution": [480, 832], "fps": 16}

    def generate(self, *, prompt, request_id, sampling_params_list):
        self.calls.append(
            {
                "prompt": prompt,
                "request_id": request_id,
                "sampling_params_list": sampling_params_list,
            }
        )
        if self.fail_with is not None:
            raise self.fail_with
        return _AsyncIter([FakeResult(self._frames)])


def _make_serving(engine_client: FakeEngineClient | None = None) -> ServingRealtimeWorldCamera:
    return ServingRealtimeWorldCamera(engine_client=engine_client or FakeEngineClient(), model_name="lingbot")


# ---------------------------------------------------------------------------
# CameraServerConfig
# ---------------------------------------------------------------------------


def test_camera_server_config_round_trip_through_dict() -> None:
    cfg = CameraServerConfig.from_model_config({"pipeline": "lingbot", "fps": 16})
    out = cfg.to_dict()
    assert isinstance(out, dict)
    assert out["pipeline"] == "lingbot"
    assert out["fps"] == 16


# ---------------------------------------------------------------------------
# msgpack-numpy round-trip
# ---------------------------------------------------------------------------


def test_msgpack_camera_payload_round_trip() -> None:
    camera = make_dummy_camera_inputs(num_frames=8)
    payload = {
        "image": np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8),
        "prompt": "walk forward",
        "camera": camera,
        "session_id": "sess-1",
        "extra_body": {"height": 240, "width": 416, "num_frames": 25, "fps": 16},
    }
    packed = msgpack_numpy.packb(payload)
    decoded = msgpack_numpy.unpackb(packed)

    assert decoded["prompt"] == "walk forward"
    assert decoded["session_id"] == "sess-1"
    assert decoded["extra_body"] == payload["extra_body"]

    image_out = decoded["image"]
    assert isinstance(image_out, np.ndarray)
    assert image_out.shape == payload["image"].shape
    assert image_out.dtype == payload["image"].dtype
    np.testing.assert_array_equal(image_out, payload["image"])

    for key in ("intrinsics", "poses"):
        arr_in = camera[key]
        arr_out = decoded["camera"][key]
        assert arr_out.dtype == arr_in.dtype
        assert arr_out.shape == arr_in.shape
        np.testing.assert_array_equal(arr_out, arr_in)


# ---------------------------------------------------------------------------
# Connection-level framing
# ---------------------------------------------------------------------------


def test_handshake_sends_camera_server_config_on_connect() -> None:
    serving = _make_serving()
    ws = MockWebSocket(incoming=[])  # client disconnects immediately after handshake
    conn = WorldCameraRealtimeConnection(ws, serving)
    asyncio.run(conn.handle_connection())

    assert ws.accepted is True
    assert len(ws.sent_bytes) == 1
    handshake = msgpack_numpy.unpackb(ws.sent_bytes[0])
    assert isinstance(handshake, dict)
    assert handshake["pipeline"] == "lingbot_world_fast"


def test_invalid_msgpack_returns_error_frame_and_keeps_connection_open() -> None:
    serving = _make_serving()
    ws = MockWebSocket(
        incoming=[
            _raw_bytes_frame(b"\x99not-msgpack"),  # malformed
            _bytes_frame({"endpoint": "reset"}),
        ]
    )
    conn = WorldCameraRealtimeConnection(ws, serving)
    asyncio.run(conn.handle_connection())

    # First sent message is the handshake, next is the error frame, then the
    # "reset successful" text reply — proving the connection stayed open.
    assert len(ws.sent_bytes) >= 2
    error = msgpack_numpy.unpackb(ws.sent_bytes[1])
    assert error == {"type": "error", "message": "Invalid request payload"}
    assert ws.sent_text == ["reset successful"]


def test_non_dict_payload_is_rejected_with_error_frame() -> None:
    serving = _make_serving()
    ws = MockWebSocket(incoming=[_bytes_frame([1, 2, 3])])  # list, not dict
    conn = WorldCameraRealtimeConnection(ws, serving)
    asyncio.run(conn.handle_connection())

    assert len(ws.sent_bytes) >= 2
    error = msgpack_numpy.unpackb(ws.sent_bytes[1])
    assert error["type"] == "error"


def test_reset_endpoint_clears_session_and_returns_text_ack() -> None:
    engine_client = FakeEngineClient()
    serving = ServingRealtimeWorldCamera(engine_client=engine_client, model_name="lingbot")
    # Pre-populate as if a prior session were active.
    serving._current_session_id = "session-a"

    ws = MockWebSocket(incoming=[_bytes_frame({"endpoint": "reset"})])
    conn = WorldCameraRealtimeConnection(ws, serving)
    asyncio.run(conn.handle_connection())

    assert serving._current_session_id is None
    assert ws.sent_text == ["reset successful"]


def test_infer_frames_are_chunked() -> None:
    num_frames = DEFAULT_FRAMES_PER_CHUNK * 3 // 2

    frames = np.arange(num_frames * 4 * 4 * 3, dtype=np.uint8).reshape(num_frames, 4, 4, 3)
    engine_client = FakeEngineClient(frames=frames)
    serving = ServingRealtimeWorldCamera(engine_client=engine_client, model_name="lingbot")

    request = {
        "prompt": "p",
        "camera": make_dummy_camera_inputs(num_frames=6),
        "session_id": "s1",
        "extra_body": {"num_frames": 6, "height": 16, "width": 16, "fps": 16},
        "image": np.zeros((16, 16, 3), dtype=np.uint8),
    }
    ws = MockWebSocket(incoming=[_bytes_frame(request)])
    conn = WorldCameraRealtimeConnection(ws, serving)
    asyncio.run(conn.handle_connection())

    # Drop the handshake; the remaining sent_bytes are frame chunks.
    chunks = [msgpack_numpy.unpackb(b) for b in ws.sent_bytes[1:]]
    assert [c["type"] for c in chunks] == ["frame", "frame"]
    assert [c["index"] for c in chunks] == [0, 1]
    assert {c["total"] for c in chunks} == {2}

    assert len(chunks[0]["video"]) == DEFAULT_FRAMES_PER_CHUNK
    assert len(chunks[1]["video"]) == num_frames - DEFAULT_FRAMES_PER_CHUNK

    for chunk in chunks:
        assert chunk["video"][0].shape == (4, 4, 3)


def test_session_id_churn_flips_current_session_id() -> None:
    engine_client = FakeEngineClient()
    serving = ServingRealtimeWorldCamera(engine_client=engine_client, model_name="lingbot")

    base_obs = {
        "prompt": "p",
        "camera": make_dummy_camera_inputs(num_frames=4),
        "image": np.zeros((16, 16, 3), dtype=np.uint8),
        "extra_body": {"num_frames": 4, "height": 16, "width": 16, "fps": 16},
    }
    obs_a = {**base_obs, "session_id": "session-a"}
    obs_b = {**base_obs, "session_id": "session-b"}

    ws = MockWebSocket(incoming=[_bytes_frame(obs_a), _bytes_frame(obs_b)])
    conn = WorldCameraRealtimeConnection(ws, serving)
    asyncio.run(conn.handle_connection())

    assert serving._current_session_id == "session-b"
    assert len(engine_client.calls) == 2
    # Each engine call observes the active session id via extra_args.
    seen_session_ids = [call["sampling_params_list"][0].extra_args["session_id"] for call in engine_client.calls]
    assert seen_session_ids == ["session-a", "session-b"]


def test_engine_failure_surfaces_as_error_frame_not_close() -> None:
    engine_client = FakeEngineClient()
    engine_client.fail_with = RuntimeError("kaboom")
    serving = ServingRealtimeWorldCamera(engine_client=engine_client, model_name="lingbot")

    request = {
        "prompt": "p",
        "camera": make_dummy_camera_inputs(num_frames=4),
        "session_id": "s1",
        "image": np.zeros((16, 16, 3), dtype=np.uint8),
        "extra_body": {"num_frames": 4, "height": 16, "width": 16, "fps": 16},
    }
    ws = MockWebSocket(incoming=[_bytes_frame(request)])
    conn = WorldCameraRealtimeConnection(ws, serving)
    asyncio.run(conn.handle_connection())

    error = msgpack_numpy.unpackb(ws.sent_bytes[-1])
    assert error == {"type": "error", "message": "Internal inference error"}


# ---------------------------------------------------------------------------
# Required-field validation: a missing ``camera`` propagates to the pipeline
# layer's ValueError. At the serving layer we exercise this by giving the
# fake engine a side-effect that raises like the pipeline would, then assert
# the connection responds with an error frame and keeps running.
# ---------------------------------------------------------------------------


def test_missing_camera_surfaces_as_error_frame() -> None:
    engine_client = FakeEngineClient()
    # Pipeline's actual ValueError text — useful to keep this in sync.
    engine_client.fail_with = ValueError("A path to camera positions must be passed to this model")
    serving = ServingRealtimeWorldCamera(engine_client=engine_client, model_name="lingbot")

    request = {
        "prompt": "p",
        "session_id": "s1",
        "image": np.zeros((16, 16, 3), dtype=np.uint8),
        "extra_body": {"num_frames": 4, "height": 16, "width": 16, "fps": 16},
    }
    ws = MockWebSocket(incoming=[_bytes_frame(request)])
    conn = WorldCameraRealtimeConnection(ws, serving)
    asyncio.run(conn.handle_connection())

    error = msgpack_numpy.unpackb(ws.sent_bytes[-1])
    assert error["type"] == "error"


# ---------------------------------------------------------------------------
# Dtype/rank guard: the wire codec preserves bit patterns, so a malformed
# camera entry (float64 vs float32, wrong rank) passes through the connection
# unchanged. The pipeline-layer assertions are exercised in the L2 offline
# test; here we just confirm the codec doesn't silently coerce.
# ---------------------------------------------------------------------------


def test_msgpack_does_not_silently_coerce_camera_dtypes() -> None:
    payload = {
        "intrinsics": np.eye(3, dtype=np.float64),  # wrong dtype on purpose
        "poses": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))[None],  # extra leading dim
    }
    decoded = msgpack_numpy.unpackb(msgpack_numpy.packb(payload))
    assert decoded["intrinsics"].dtype == np.float64
    assert decoded["poses"].shape == (1, 2, 4, 4)


# ---------------------------------------------------------------------------
# Suppress event-loop teardown noise in some environments
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _silence_runtime_warnings(recwarn):  # noqa: PT004
    yield
    with contextlib.suppress(Exception):
        recwarn.clear()
