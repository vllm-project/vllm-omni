# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L2 online smoke for ``/v1/realtime/world/camera``."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.realtime.world.camera_connection import (
    DEFAULT_FRAMES_PER_CHUNK,
    WorldCameraRealtimeConnection,
)
from vllm_omni.entrypoints.openai.realtime.world.camera_serving import ServingRealtimeWorldCamera

msgpack_numpy = pytest.importorskip("openpi_client.msgpack_numpy")

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# Test plumbing
# ---------------------------------------------------------------------------


class _MockWebSocket:
    """ASGI-shaped mock matching ``WorldCameraRealtimeConnection``'s call sites.

    ``receive`` is the lowest-level ASGI hook the connection uses (it pulls
    ``{"type": "websocket.receive", "bytes": ...}`` dicts directly, not the
    higher-level ``receive_bytes``). After the scripted messages are
    exhausted, ``receive`` returns a disconnect frame so the connection's
    main loop exits cleanly instead of timing out.
    """

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


class _FakeAsyncIter:
    """Async iterable for the canned engine output."""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


class _FakeResult:
    """Stand-in for ``OmniRequestOutput`` — only ``.images`` is consulted."""

    def __init__(self, frames: np.ndarray) -> None:
        self.images = [frames]


class _FakeEngineClient:
    """Stand-in for ``AsyncOmni``: records calls, returns a per-call frame buffer.

    The connection's framing logic calls ``generate(...)`` once per ``infer``
    request. Tests pre-load ``self.queued_frames`` with one buffer per
    expected call, in order.
    """

    def __init__(self, queued_frames: list[np.ndarray]) -> None:
        self.queued_frames = list(queued_frames)
        self.calls: list[dict[str, Any]] = []
        # Attributes consulted by ``CameraServerConfig.from_model_config``.
        self.model_config = {"pipeline": "lingbot_world_fast", "resolution": [480, 832], "fps": 16}

    def generate(self, *, prompt, request_id, sampling_params_list):
        self.calls.append(
            {
                "prompt": prompt,
                "request_id": request_id,
                "session_id": sampling_params_list[0].extra_args.get("session_id"),
            }
        )
        if not self.queued_frames:
            raise AssertionError("FakeEngineClient ran out of queued frames")
        frames = self.queued_frames.pop(0)
        return _FakeAsyncIter([_FakeResult(frames)])


def _pack_frame(payload: Any) -> dict[str, Any]:
    return {"type": "websocket.receive", "bytes": msgpack_numpy.packb(payload)}


def _camera_payload(num_frames: int) -> dict[str, np.ndarray]:
    return {
        "intrinsics": np.eye(3, dtype=np.float32),
        "poses": np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1)),
    }


def _infer_req(*, session_id: str, num_frames: int, include_image: bool) -> dict[str, Any]:
    req: dict[str, Any] = {
        "prompt": "walk along the Great Wall of China",
        "camera": _camera_payload(num_frames),
        "session_id": session_id,
        "extra_body": {"num_frames": num_frames, "height": 480, "width": 832, "fps": 16},
    }
    if include_image:
        req["image"] = np.zeros((480, 832, 3), dtype=np.uint8)
    return req


# ---------------------------------------------------------------------------
# Lifecycle smoke test
# ---------------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.cpu
def test_camera_session_lifecycle_handshake_infer_reset_infer() -> None:
    """End-to-end client session: handshake → infer → reset → infer (new
    session_id). Mirrors what ``examples/online_serving/lingbot_world_fast/openai_client.py``
    does on the wire, minus the actual model."""
    # Distinct, non-divisible-by-CHUNK_FRAMES buffer sizes so both calls
    # exercise the boundary case (final chunk shorter than CHUNK_FRAMES) and
    # the fill-value lets us prove chunks aren't swapped between requests.
    first_frames = np.full((DEFAULT_FRAMES_PER_CHUNK * 2 + 1, 8, 8, 3), 3, dtype=np.uint8)
    second_frames = np.full((DEFAULT_FRAMES_PER_CHUNK + 1, 8, 8, 3), 7, dtype=np.uint8)

    engine_client = _FakeEngineClient(queued_frames=[first_frames, second_frames])
    serving = ServingRealtimeWorldCamera(engine_client=engine_client, model_name="lingbot")

    # First infer uses ``4N+1`` to mimic the openai_client's fresh-call shape;
    # second infer uses ``4N`` for the extension shape (modelled on the
    # client's branch at ``openai_client.py:84``).
    fresh_req = _infer_req(session_id="session-1", num_frames=25, include_image=True)
    ext_req = _infer_req(session_id="session-2", num_frames=24, include_image=False)

    ws = _MockWebSocket(
        incoming=[
            _pack_frame(fresh_req),
            _pack_frame({"endpoint": "reset"}),
            _pack_frame(ext_req),
        ]
    )
    conn = WorldCameraRealtimeConnection(ws, serving)
    asyncio.run(conn.handle_connection())

    # ---------------- Handshake ------------------
    assert ws.accepted is True, "Connection must accept() before sending the handshake."
    assert len(ws.sent_bytes) >= 1
    handshake = msgpack_numpy.unpackb(ws.sent_bytes[0])
    assert isinstance(handshake, dict) and handshake, "Handshake must be a non-empty msgpack dict."
    assert handshake.get("pipeline") == "lingbot_world_fast"

    # ---------------- Frame chunks ---------------
    decoded = [msgpack_numpy.unpackb(b) for b in ws.sent_bytes[1:]]
    print(f"DECODED: {decoded}")
    frame_chunks = [d for d in decoded if isinstance(d, dict) and d.get("type") == "frame"]

    first_total = (len(first_frames) + DEFAULT_FRAMES_PER_CHUNK - 1) // DEFAULT_FRAMES_PER_CHUNK
    second_total = (len(second_frames) + DEFAULT_FRAMES_PER_CHUNK - 1) // DEFAULT_FRAMES_PER_CHUNK

    assert len(frame_chunks) == first_total + second_total, (
        f"Expected {first_total} + {second_total} frame chunks, got {len(frame_chunks)}."
    )

    for chunk in frame_chunks:
        assert chunk.keys() >= {"type", "index", "total", "video"}
        video = chunk["video"]
        for frame in video:
            assert frame.dtype == np.float32
            assert (frame >= 0).all() and (frame <= 1).all()
            assert frame.ndim == 3  # [h, w, 3]
            assert frame.shape[-1] == 3

    # Send order is first-call chunks, then second-call chunks. Index runs
    # 0..total-1 per request, and the per-chunk fill-value proves the chunker
    # didn't leak state between requests.
    for i, chunk in enumerate(frame_chunks[:first_total]):
        assert chunk["index"] == i
        assert chunk["total"] == first_total
    for i, chunk in enumerate(frame_chunks[first_total:]):
        assert chunk["index"] == i
        assert chunk["total"] == second_total

    # ---------------- reset ----------------------
    assert "reset successful" in ws.sent_text, "Reset endpoint must reply with a text ack."
    # ``ServingRealtimeWorldCamera.reset`` wipes ``_current_session_id``;
    # the third request then sets it to ``session-2``.
    assert serving._current_session_id == "session-2"

    # ---------------- Engine call accounting -----
    # Exactly two ``generate`` invocations; the reset request must not call
    # the engine.
    assert len(engine_client.calls) == 2
    assert [c["session_id"] for c in engine_client.calls] == ["session-1", "session-2"]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.cpu
def test_camera_session_handshake_does_not_repeat_within_connection() -> None:
    """The handshake is sent **once** at connect, even if many infer/reset
    operations follow. Other diffusion clients depend on this invariant to
    avoid double-initialising their config."""
    first_frames = np.full((DEFAULT_FRAMES_PER_CHUNK + 1, 4, 4, 3), 1, dtype=np.uint8)
    second_frames = np.full((DEFAULT_FRAMES_PER_CHUNK + 2, 4, 4, 3), 2, dtype=np.uint8)
    engine_client = _FakeEngineClient(queued_frames=[first_frames, second_frames])
    serving = ServingRealtimeWorldCamera(engine_client=engine_client, model_name="lingbot")

    ws = _MockWebSocket(
        incoming=[
            _pack_frame(_infer_req(session_id="s", num_frames=25, include_image=True)),
            _pack_frame({"endpoint": "reset"}),
            _pack_frame(_infer_req(session_id="s", num_frames=24, include_image=False)),
        ]
    )
    conn = WorldCameraRealtimeConnection(ws, serving)
    asyncio.run(conn.handle_connection())

    # Exactly one msgpack-encoded non-frame, non-error dict in the entire
    # outbound stream: the handshake.
    handshakes = []
    for b in ws.sent_bytes:
        decoded = msgpack_numpy.unpackb(b)
        if isinstance(decoded, dict) and decoded.get("type") not in ("frame", "error"):
            handshakes.append(decoded)
    assert len(handshakes) == 1, f"Expected exactly one handshake, got {len(handshakes)}: {handshakes}"
