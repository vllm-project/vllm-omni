# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vllm_omni.experimental.fullduplex.mage_vl import MageVLDuplexAdapter
from vllm_omni.experimental.fullduplex.mage_vl.serving import backend as backend_module
from vllm_omni.experimental.fullduplex.mage_vl.serving.server import (
    MageVLServingConfig,
    create_app,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_mage_vl_codec_backend_requires_preprocessor(monkeypatch):
    monkeypatch.setattr(backend_module.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="cv-preinfer"):
        backend_module.MageVLTransformersBackend(video_backend="codec")


def test_mage_vl_frames_backend_does_not_require_preprocessor(monkeypatch):
    def unexpected_lookup(_name):
        raise AssertionError("frames backend must not inspect codec dependencies")

    monkeypatch.setattr(backend_module.shutil, "which", unexpected_lookup)
    backend = backend_module.MageVLTransformersBackend(video_backend="frames")

    assert backend.video_backend == "frames"


def _factory():
    async def gate(_session, windows):
        return {
            "should_respond": windows[-1].segment_id == "goal",
            "event_id": "goal-1",
            "score": 0.9,
        }

    async def generate(_session, windows, query, _decision):
        yield f"{query or 'proactive'}:{windows[-1].segment_id}"

    return MageVLDuplexAdapter(gate=gate, generate=generate, window_size=1)


def test_mage_vl_health_ready_and_models():
    with TestClient(create_app(_factory)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready", "active_sessions": 0}
        assert client.get("/v1/models").json()["data"][0]["id"] == "microsoft/Mage-VL"


def test_mage_vl_websocket_runs_gate_and_generation():
    with TestClient(create_app(_factory)) as client:
        with client.websocket_connect("/v1/mage-vl/duplex?session_id=test-session") as websocket:
            created = websocket.receive_json()
            assert created["type"] == "session.created"
            assert created["session_id"] == "test-session"
            websocket.send_json(
                {
                    "type": "input.append",
                    "modality": "video",
                    "data": {"segment_id": "goal", "frames": ["frame-1"]},
                }
            )
            assert websocket.receive_json()["type"] == "response.created"
            delta = websocket.receive_json()
            assert delta == {
                "type": "response.delta",
                "response_index": 1,
                "modality": "text",
                "data": "proactive:goal",
            }
            assert websocket.receive_json()["type"] == "response.done"
            websocket.send_json({"type": "close"})


def test_mage_vl_websocket_requires_bearer_token():
    app = create_app(_factory, MageVLServingConfig(auth_token="secret"))
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as denied:
            with client.websocket_connect("/v1/mage-vl/duplex"):
                pass
        assert denied.value.code == 4401

        with client.websocket_connect(
            "/v1/mage-vl/duplex",
            headers={"Authorization": "Bearer secret"},
        ) as websocket:
            assert websocket.receive_json()["type"] == "session.created"
            websocket.send_json({"type": "close"})


def test_mage_vl_rejects_duplicate_session_and_releases_disconnect():
    app = create_app(_factory, MageVLServingConfig(max_sessions=1))
    with TestClient(app) as client:
        with client.websocket_connect("/v1/mage-vl/duplex?session_id=one") as first:
            assert first.receive_json()["session_id"] == "one"
            with pytest.raises(WebSocketDisconnect) as denied:
                with client.websocket_connect("/v1/mage-vl/duplex?session_id=two") as second:
                    second.receive_json()
            assert denied.value.code == 4429
            first.send_json({"type": "close"})

        with client.websocket_connect("/v1/mage-vl/duplex?session_id=two") as second:
            assert second.receive_json()["session_id"] == "two"
            second.send_json({"type": "close"})


def test_mage_vl_rejects_malformed_event():
    with TestClient(create_app(_factory)) as client:
        with client.websocket_connect("/v1/mage-vl/duplex") as websocket:
            websocket.receive_json()
            websocket.send_text("[]")
            event = websocket.receive_json()
            assert event["type"] == "error"
            assert "JSON object" in event["message"]


def test_mage_vl_cancel_is_not_blocked_by_slow_gate():
    async def slow_gate(_session, _windows):
        await asyncio.sleep(2)
        return {"should_respond": True, "event_id": "late"}

    def factory():
        return MageVLDuplexAdapter(gate=slow_gate, window_size=1)

    with TestClient(create_app(factory)) as client:
        with client.websocket_connect("/v1/mage-vl/duplex") as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "input.append",
                    "modality": "video",
                    "data": {"segment_id": "slow", "frames": ["frame"]},
                }
            )
            started = time.monotonic()
            websocket.send_json({"type": "response.cancel"})
            event = websocket.receive_json()
            assert time.monotonic() - started < 0.5
            assert event["type"] == "response.cancelled"
            websocket.send_json({"type": "close"})
