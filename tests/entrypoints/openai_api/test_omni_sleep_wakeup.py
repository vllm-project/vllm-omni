# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclasses.dataclass
class FakeAck:
    stage_id: int
    result: str = "ok"


def _make_app(engine_client):
    from vllm_omni.entrypoints.openai.api_server import router

    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = engine_client
    app.state.sleeping_stages = set()
    return app


@pytest.fixture
def sleep_capable_engine(mocker):
    engine = mocker.MagicMock()
    engine.sleep = mocker.AsyncMock(return_value=[FakeAck(stage_id=0), FakeAck(stage_id=1)])
    engine.wake_up = mocker.AsyncMock(return_value=[FakeAck(stage_id=0), FakeAck(stage_id=1)])
    return engine


@pytest.fixture
def sleep_incapable_engine(mocker):
    return mocker.MagicMock(spec=[])  # no 'sleep' or 'wake_up' attributes


# ---------------------------------------------------------------------------
# /v1/omni/sleep
# ---------------------------------------------------------------------------


def test_sleep_success(sleep_capable_engine):
    app = _make_app(sleep_capable_engine)
    client = TestClient(app)

    response = client.post("/v1/omni/sleep", json={"stage_ids": [0, 1], "level": 2})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert [ack["stage_id"] for ack in data["acks"]] == [0, 1]
    assert app.state.sleeping_stages == {0, 1}
    sleep_capable_engine.sleep.assert_awaited_once_with(stage_ids=[0, 1], level=2)


def test_sleep_default_level(sleep_capable_engine):
    app = _make_app(sleep_capable_engine)
    client = TestClient(app)

    response = client.post("/v1/omni/sleep", json={"stage_ids": [0]})

    assert response.status_code == 200
    sleep_capable_engine.sleep.assert_awaited_once_with(stage_ids=[0], level=2)


def test_sleep_empty_stage_ids(sleep_capable_engine, mocker):
    """Empty stage_ids: engine is still called and returns SUCCESS with no acks."""
    sleep_capable_engine.sleep = mocker.AsyncMock(return_value=[])
    app = _make_app(sleep_capable_engine)
    client = TestClient(app)

    response = client.post("/v1/omni/sleep", json={"stage_ids": [], "level": 2})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["acks"] == []
    assert app.state.sleeping_stages == set()
    sleep_capable_engine.sleep.assert_awaited_once_with(stage_ids=[], level=2)


def test_sleep_updates_sleeping_set(sleep_capable_engine):
    app = _make_app(sleep_capable_engine)
    app.state.sleeping_stages = {5}  # pre-existing entry
    client = TestClient(app)

    client.post("/v1/omni/sleep", json={"stage_ids": [0, 1], "level": 2})

    assert app.state.sleeping_stages == {5, 0, 1}


def test_sleep_engine_not_support(sleep_incapable_engine):
    app = _make_app(sleep_incapable_engine)
    client = TestClient(app)

    response = client.post("/v1/omni/sleep", json={"stage_ids": [0], "level": 1})

    assert response.status_code == 501
    assert "sleep" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /v1/omni/wakeup
# ---------------------------------------------------------------------------


def test_wakeup_success(sleep_capable_engine):
    app = _make_app(sleep_capable_engine)
    app.state.sleeping_stages = {0, 1}
    client = TestClient(app)

    response = client.post("/v1/omni/wakeup", json={"stage_ids": [0, 1]})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert [ack["stage_id"] for ack in data["acks"]] == [0, 1]
    assert app.state.sleeping_stages == set()
    sleep_capable_engine.wake_up.assert_awaited_once_with(stage_ids=[0, 1])


def test_wakeup_skipped_when_not_sleeping(sleep_capable_engine):
    app = _make_app(sleep_capable_engine)
    app.state.sleeping_stages = set()  # nothing sleeping
    client = TestClient(app)

    response = client.post("/v1/omni/wakeup", json={"stage_ids": [0, 1]})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SKIPPED"
    sleep_capable_engine.wake_up.assert_not_awaited()


def test_wakeup_partial_sleeping(sleep_capable_engine):
    """Only stage 0 is sleeping; stage 1 is not. Should still proceed (partial match)."""
    app = _make_app(sleep_capable_engine)
    app.state.sleeping_stages = {0}
    client = TestClient(app)

    response = client.post("/v1/omni/wakeup", json={"stage_ids": [0, 1]})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert 0 not in app.state.sleeping_stages


def test_wakeup_empty_stage_ids(sleep_capable_engine):
    """Empty stage_ids: any() on empty iterable is False → SKIPPED, engine not called."""
    app = _make_app(sleep_capable_engine)
    app.state.sleeping_stages = {0, 1}
    client = TestClient(app)

    response = client.post("/v1/omni/wakeup", json={"stage_ids": []})

    assert response.status_code == 200
    assert response.json()["status"] == "SKIPPED"
    sleep_capable_engine.wake_up.assert_not_awaited()


def test_wakeup_engine_not_support(sleep_incapable_engine):
    app = _make_app(sleep_incapable_engine)
    app.state.sleeping_stages = {0}
    client = TestClient(app)

    response = client.post("/v1/omni/wakeup", json={"stage_ids": [0]})

    assert response.status_code == 501
    assert "wake_up" in response.json()["detail"]


def test_wakeup_removes_only_requested_stages(sleep_capable_engine):
    app = _make_app(sleep_capable_engine)
    app.state.sleeping_stages = {0, 1, 2}
    client = TestClient(app)

    client.post("/v1/omni/wakeup", json={"stage_ids": [0]})

    assert app.state.sleeping_stages == {1, 2}


def test_sleep_then_wakeup_roundtrip(sleep_capable_engine):
    app = _make_app(sleep_capable_engine)
    client = TestClient(app)

    sleep_resp = client.post("/v1/omni/sleep", json={"stage_ids": [0, 1], "level": 2})
    assert sleep_resp.status_code == 200
    assert app.state.sleeping_stages == {0, 1}

    wakeup_resp = client.post("/v1/omni/wakeup", json={"stage_ids": [0, 1]})
    assert wakeup_resp.status_code == 200
    assert app.state.sleeping_stages == set()


# ---------------------------------------------------------------------------
# pure-diffusion branch: omni_init_app_state initialises sleeping_stages
# ---------------------------------------------------------------------------


@pytest.fixture
def pure_diffusion_engine(mocker):
    engine = mocker.MagicMock()
    engine.stage_configs = [{"stage_type": "diffusion"}]
    engine.sleep = mocker.AsyncMock(return_value=[FakeAck(stage_id=0)])
    engine.wake_up = mocker.AsyncMock(return_value=[FakeAck(stage_id=0)])
    # Remove attributes that would make _get_vllm_config return a config
    del engine.get_vllm_config
    del engine.vllm_config
    return engine


@pytest.fixture
def pure_diffusion_app(pure_diffusion_engine, mocker):
    """App whose state was initialised via omni_init_app_state (pure diffusion path)."""
    import asyncio
    from argparse import Namespace

    from vllm_omni.entrypoints.openai.api_server import omni_init_app_state, router

    args = Namespace(
        served_model_name=None,
        model="fake-diffusion-model",
        enable_log_requests=False,
        disable_log_stats=True,
        max_log_len=0,
        enable_server_load_tracking=False,
    )

    app = FastAPI()
    app.include_router(router)

    mocker.patch(
        "vllm_omni.entrypoints.openai.api_server.OmniOpenAIServingChat"
    ).for_diffusion.return_value = mocker.MagicMock()
    mocker.patch(
        "vllm_omni.entrypoints.openai.api_server.OmniOpenAIServingVideo"
    ).for_diffusion.return_value = mocker.MagicMock()
    mocker.patch(
        "vllm_omni.entrypoints.openai.api_server.OmniOpenAIServingSpeech"
    ).for_diffusion.return_value = mocker.MagicMock()
    mocker.patch("vllm_omni.entrypoints.openai.api_server._DiffusionServingModels")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(omni_init_app_state(pure_diffusion_engine, app.state, args))
    finally:
        loop.close()

    return app


def test_pure_diffusion_sleep_success(pure_diffusion_app, pure_diffusion_engine):
    client = TestClient(pure_diffusion_app)

    response = client.post("/v1/omni/sleep", json={"stage_ids": [0], "level": 2})

    assert response.status_code == 200
    assert response.json()["status"] == "SUCCESS"
    assert pure_diffusion_app.state.sleeping_stages == {0}
    pure_diffusion_engine.sleep.assert_awaited_once_with(stage_ids=[0], level=2)


def test_pure_diffusion_wakeup_skipped(pure_diffusion_app, pure_diffusion_engine):
    # sleeping_stages is empty after init — nothing is sleeping
    client = TestClient(pure_diffusion_app)

    response = client.post("/v1/omni/wakeup", json={"stage_ids": [0]})

    assert response.status_code == 200
    assert response.json()["status"] == "SKIPPED"
    pure_diffusion_engine.wake_up.assert_not_awaited()


def test_pure_diffusion_sleep_then_wakeup(pure_diffusion_app, pure_diffusion_engine):
    client = TestClient(pure_diffusion_app)

    sleep_resp = client.post("/v1/omni/sleep", json={"stage_ids": [0], "level": 2})
    assert sleep_resp.status_code == 200
    assert pure_diffusion_app.state.sleeping_stages == {0}

    wakeup_resp = client.post("/v1/omni/wakeup", json={"stage_ids": [0]})
    assert wakeup_resp.status_code == 200
    assert pure_diffusion_app.state.sleeping_stages == set()
