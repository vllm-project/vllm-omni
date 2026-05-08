# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from unittest.mock import AsyncMock, MagicMock

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
def sleep_capable_engine():
    engine = MagicMock()
    engine.sleep = AsyncMock(return_value=[FakeAck(stage_id=0), FakeAck(stage_id=1)])
    engine.wake_up = AsyncMock(return_value=[FakeAck(stage_id=0), FakeAck(stage_id=1)])
    return engine


@pytest.fixture
def sleep_incapable_engine():
    engine = MagicMock(spec=[])  # no 'sleep' or 'wake_up' attributes
    return engine


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
    assert len(data["acks"]) == 2
    # sleeping_stages should be updated
    assert app.state.sleeping_stages == {0, 1}
    sleep_capable_engine.sleep.assert_awaited_once_with(stage_ids=[0, 1], level=2)


def test_sleep_default_level(sleep_capable_engine):
    app = _make_app(sleep_capable_engine)
    client = TestClient(app)

    response = client.post("/v1/omni/sleep", json={"stage_ids": [0]})

    assert response.status_code == 200
    sleep_capable_engine.sleep.assert_awaited_once_with(stage_ids=[0], level=2)


def test_sleep_updates_sleeping_set(sleep_capable_engine):
    app = _make_app(sleep_capable_engine)
    app.state.sleeping_stages = {5}  # pre-existing entry
    client = TestClient(app)

    client.post("/v1/omni/sleep", json={"stage_ids": [0, 1], "level": 2})

    assert app.state.sleeping_stages == {5, 0, 1}


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
    assert len(data["acks"]) == 2
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
    # Only stage 0 was in sleeping_stages so only it gets removed
    assert 0 not in app.state.sleeping_stages


def test_wakeup_removes_only_requested_stages(sleep_capable_engine):
    app = _make_app(sleep_capable_engine)
    app.state.sleeping_stages = {0, 1, 2}
    client = TestClient(app)

    client.post("/v1/omni/wakeup", json={"stage_ids": [0]})

    # stage 1 and 2 remain sleeping
    assert app.state.sleeping_stages == {1, 2}


def _make_pure_diffusion_engine(sleep_capable: bool = True):
    """Return a mock engine that omni_init_app_state recognises as pure-diffusion.

    Requirements checked by omni_init_app_state:
      - has stage_configs with exactly one entry whose stage_type == "diffusion"
      - has no vllm_config / get_vllm_config  (so _get_vllm_config returns None)
    """
    engine = MagicMock()
    engine.stage_configs = [{"stage_type": "diffusion"}]
    # Remove attributes that would make _get_vllm_config return a config
    del engine.get_vllm_config
    del engine.vllm_config
    if sleep_capable:
        engine.sleep = AsyncMock(return_value=[FakeAck(stage_id=0)])
        engine.wake_up = AsyncMock(return_value=[FakeAck(stage_id=0)])
    else:
        # Remove sleep/wake_up so hasattr returns False
        engine_no_sleep = MagicMock(spec=["stage_configs"])
        engine_no_sleep.stage_configs = [{"stage_type": "diffusion"}]
        return engine_no_sleep
    return engine


def _make_pure_diffusion_args():
    from argparse import Namespace

    return Namespace(
        served_model_name=None,
        model="fake-diffusion-model",
        enable_log_requests=False,
        disable_log_stats=True,
        max_log_len=0,
        enable_server_load_tracking=False,
    )


@pytest.fixture
def pure_diffusion_engine():
    return _make_pure_diffusion_engine(sleep_capable=True)


@pytest.fixture
def pure_diffusion_incapable_engine():
    return _make_pure_diffusion_engine(sleep_capable=False)


@pytest.fixture
def pure_diffusion_args():
    return _make_pure_diffusion_args()


@pytest.fixture
def pure_diffusion_app(pure_diffusion_engine, pure_diffusion_args):
    """App whose state was initialised via omni_init_app_state (pure diffusion path)."""
    import asyncio
    from unittest.mock import patch

    from fastapi import FastAPI

    from vllm_omni.entrypoints.openai.api_server import omni_init_app_state, router

    app = FastAPI()
    app.include_router(router)

    with (
        patch("vllm_omni.entrypoints.openai.api_server.OmniOpenAIServingChat") as mock_chat,
        patch("vllm_omni.entrypoints.openai.api_server.OmniOpenAIServingVideo") as mock_video,
        patch("vllm_omni.entrypoints.openai.api_server.OmniOpenAIServingSpeech") as mock_speech,
        patch("vllm_omni.entrypoints.openai.api_server._DiffusionServingModels"),
    ):
        mock_chat.for_diffusion.return_value = MagicMock()
        mock_video.for_diffusion.return_value = MagicMock()
        mock_speech.for_diffusion.return_value = MagicMock()
        asyncio.get_event_loop().run_until_complete(
            omni_init_app_state(pure_diffusion_engine, app.state, pure_diffusion_args)
        )

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
