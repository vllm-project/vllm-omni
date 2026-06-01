# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
import pytest
from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vllm_omni.entrypoints.openai.realtime.world.camera_serving import CameraServerConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


mock_model_config = {"model_name": "/m/foo", "image_width": 832, "image_height": 480, "extra_args": {"foo": "bar"}}


def test_from_model_config_loads_correctly():
    cfg = CameraServerConfig.from_model_config(mock_model_config)

    assert cfg.to_dict() == mock_model_config


def test_msgpack_roundtrip():
    cfg = CameraServerConfig.from_model_config(mock_model_config)
    encoded = msgspec.msgpack.encode(cfg)
    decoded = msgspec.msgpack.decode(encoded, type=CameraServerConfig)
    assert decoded == cfg


def _build_camera_app(*, supports: bool, cfg: CameraServerConfig | None):
    """Build a minimal FastAPI app that mirrors the api_server.py handler."""
    app = FastAPI()

    @app.websocket("/v1/realtime/world/camera")
    async def realtime_world_camera(websocket: WebSocket):
        await websocket.accept()
        if cfg is None or not supports:
            await websocket.send_json(
                {"type": "error", "error": "Camera realtime API is not available", "code": "unsupported"}
            )
            await websocket.close()
            return
        await websocket.send_bytes(msgspec.msgpack.encode(cfg))
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            return

    return app


class TestRealtimeWorldCameraEndpoint:
    def test_sends_msgpack_config_on_connect(self):
        cfg = CameraServerConfig.from_model_config(mock_model_config)
        app = _build_camera_app(supports=True, cfg=cfg)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/world/camera") as ws:
                payload = ws.receive_bytes()
                decoded = msgspec.msgpack.decode(payload, type=CameraServerConfig)
                assert decoded == cfg

    def test_keeps_socket_open_after_initial_send(self):
        cfg = CameraServerConfig.from_model_config(mock_model_config)
        app = _build_camera_app(supports=True, cfg=cfg)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/world/camera") as ws:
                ws.receive_bytes()
                # Client-initiated messages are accepted (currently ignored).
                ws.send_text("ping")
                # Closing from the client side must not raise on the server.

    def test_unsupported_path_sends_error_and_closes(self):
        app = _build_camera_app(supports=False, cfg=None)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime/world/camera") as ws:
                err = ws.receive_json()
                assert err["type"] == "error"
                assert err["code"] == "unsupported"
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_bytes()
