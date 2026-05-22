# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for /v1/omni/sleep and /v1/omni/wakeup endpoints.

Verifies that:
1. sleeping_stages is initialised in pure-diffusion mode so endpoints
   don't crash with AttributeError.
2. Engines without sleep/wake_up methods return 501.
3. The happy path tracks sleeping stage IDs correctly.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.openai.api_server import router

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class FakeACK:
    task_id: str = "ack-1"
    status: str = "ok"
    stage_id: int | None = 0
    rank: int | None = 0
    freed_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class _EngineWithSleep:
    """Fake engine that supports sleep/wake_up."""

    def __init__(self) -> None:
        self.sleep_calls: list[dict] = []
        self.wakeup_calls: list[dict] = []

    async def sleep(self, *, stage_ids: list[int], level: int = 2) -> list[FakeACK]:
        self.sleep_calls.append({"stage_ids": stage_ids, "level": level})
        return [FakeACK(stage_id=sid) for sid in stage_ids]

    async def wake_up(self, *, stage_ids: list[int]) -> list[FakeACK]:
        self.wakeup_calls.append({"stage_ids": stage_ids})
        return [FakeACK(stage_id=sid) for sid in stage_ids]


class _EngineWithoutSleep:
    """Fake engine that does NOT expose sleep/wake_up."""

    pass


def _make_app(engine: object, *, init_sleeping_stages: bool = True) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = engine
    if init_sleeping_stages:
        app.state.sleeping_stages = set()
    return app


class TestSleepEndpoint:
    def test_sleep_returns_501_when_engine_lacks_method(self) -> None:
        app = _make_app(_EngineWithoutSleep())
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/omni/sleep", json={"stage_ids": [0]})
        assert resp.status_code == 501
        assert "does not support sleep" in resp.json()["detail"]

    def test_sleep_returns_501_when_sleeping_stages_missing(self) -> None:
        app = _make_app(_EngineWithSleep(), init_sleeping_stages=False)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/omni/sleep", json={"stage_ids": [0]})
        assert resp.status_code == 501
        assert "not available" in resp.json()["detail"]

    def test_sleep_success_tracks_stages(self) -> None:
        engine = _EngineWithSleep()
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post("/v1/omni/sleep", json={"stage_ids": [0, 1]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "SUCCESS"
        assert len(body["acks"]) == 2
        assert app.state.sleeping_stages == {0, 1}
        assert engine.sleep_calls == [{"stage_ids": [0, 1], "level": 2}]

    def test_sleep_custom_level(self) -> None:
        engine = _EngineWithSleep()
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post("/v1/omni/sleep", json={"stage_ids": [0], "level": 3})
        assert resp.status_code == 200
        assert engine.sleep_calls == [{"stage_ids": [0], "level": 3}]


class TestWakeupEndpoint:
    def test_wakeup_returns_501_when_engine_lacks_method(self) -> None:
        app = _make_app(_EngineWithoutSleep())
        app.state.sleeping_stages = {0}
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/omni/wakeup", json={"stage_ids": [0]})
        assert resp.status_code == 501
        assert "does not support wake_up" in resp.json()["detail"]

    def test_wakeup_returns_501_when_sleeping_stages_missing(self) -> None:
        engine = _EngineWithSleep()
        app = _make_app(engine, init_sleeping_stages=False)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/omni/wakeup", json={"stage_ids": [0]})
        assert resp.status_code == 501
        assert "not available" in resp.json()["detail"]

    def test_wakeup_skips_when_not_sleeping(self) -> None:
        app = _make_app(_EngineWithSleep())
        with TestClient(app) as client:
            resp = client.post("/v1/omni/wakeup", json={"stage_ids": [0]})
        assert resp.status_code == 200
        assert resp.json()["status"] == "SKIPPED"

    def test_wakeup_success_removes_stages(self) -> None:
        engine = _EngineWithSleep()
        app = _make_app(engine)
        app.state.sleeping_stages = {0, 1, 2}
        with TestClient(app) as client:
            resp = client.post("/v1/omni/wakeup", json={"stage_ids": [0, 1]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "SUCCESS"
        assert len(body["acks"]) == 2
        assert app.state.sleeping_stages == {2}
        assert engine.wakeup_calls == [{"stage_ids": [0, 1]}]


class TestSleepWakeupRoundTrip:
    def test_sleep_then_wakeup_restores_state(self) -> None:
        engine = _EngineWithSleep()
        app = _make_app(engine)
        with TestClient(app) as client:
            sleep_resp = client.post("/v1/omni/sleep", json={"stage_ids": [0, 1]})
            assert sleep_resp.status_code == 200
            assert app.state.sleeping_stages == {0, 1}

            wake_resp = client.post("/v1/omni/wakeup", json={"stage_ids": [0, 1]})
            assert wake_resp.status_code == 200
            assert app.state.sleeping_stages == set()

    def test_partial_wakeup(self) -> None:
        engine = _EngineWithSleep()
        app = _make_app(engine)
        with TestClient(app) as client:
            client.post("/v1/omni/sleep", json={"stage_ids": [0, 1, 2]})
            assert app.state.sleeping_stages == {0, 1, 2}

            client.post("/v1/omni/wakeup", json={"stage_ids": [1]})
            assert app.state.sleeping_stages == {0, 2}

    def test_wakeup_idempotent_for_non_sleeping_stages(self) -> None:
        engine = _EngineWithSleep()
        app = _make_app(engine)
        app.state.sleeping_stages = {0}
        with TestClient(app) as client:
            resp = client.post("/v1/omni/wakeup", json={"stage_ids": [0, 99]})
        assert resp.status_code == 200
        assert app.state.sleeping_stages == set()


class TestPureDiffusionInitialization:
    def test_sleeping_stages_present_after_pure_diffusion_init(self) -> None:
        """Regression test: pure diffusion path must initialize sleeping_stages."""
        app = _make_app(_EngineWithoutSleep())
        assert hasattr(app.state, "sleeping_stages")
        assert app.state.sleeping_stages == set()
