# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.openai import api_server

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("path", "engine_method"),
    [
        ("/start_profile", "start_profile"),
        ("/stop_profile", "stop_profile"),
    ],
)
@pytest.mark.parametrize("rpc_results", [None, [None], [[None]]])
def test_profiler_endpoint_forwards_selected_stages(path: str, engine_method: str, rpc_results) -> None:
    engine_client = SimpleNamespace(
        start_profile=AsyncMock(return_value=rpc_results),
        stop_profile=AsyncMock(return_value=rpc_results),
    )

    app = FastAPI()
    app.include_router(api_server.profiler_router)
    app.state.engine_client = engine_client

    response = TestClient(app).post(path, json={"stages": [1]})

    assert response.status_code == 200
    assert response.json() == {"status": "SUCCESS"}
    getattr(engine_client, engine_method).assert_awaited_once_with(stages=[1])
    if engine_method == "start_profile":
        engine_client.stop_profile.assert_not_awaited()


@pytest.mark.parametrize("action", ["start", "stop"])
def test_profiler_endpoint_reports_failed_stage_rpc(action: str, mocker) -> None:
    engine_client = mocker.AsyncMock(spec=api_server.AsyncOmni)
    engine_client.stop_profile.return_value = None
    detail = f"Could not {action} profiler: bad trace directory"
    getattr(engine_client, f"{action}_profile").return_value = [
        {"supported": False, "error": detail},
    ]

    app = FastAPI()
    app.include_router(api_server.profiler_router)
    app.state.engine_client = engine_client

    response = TestClient(app).post(f"/{action}_profile", json={"stages": [0]})

    assert response.status_code == 500
    assert response.json() == {"detail": f"Failed to {action} profiler: {detail}"}
    getattr(engine_client, f"{action}_profile").assert_awaited_once_with(stages=[0])


@pytest.mark.parametrize("cleanup", ["success", "exception", "rpc_error"])
def test_partial_profiler_start_attempts_cleanup_and_preserves_error(cleanup: str, caplog, mocker) -> None:
    engine_client = mocker.AsyncMock(spec=api_server.AsyncOmni)
    active_stages: set[int] = set()
    start_error = "stage 1: bad trace directory"

    async def start_profile(*, stages):
        active_stages.add(0)
        return [None, {"supported": False, "error": start_error}]

    async def stop_profile(*, stages):
        if cleanup == "exception":
            raise RuntimeError("cleanup failed")
        if cleanup == "rpc_error":
            return [{"supported": False, "error": "cleanup failed"}]
        active_stages.clear()
        return [None, None]

    engine_client.start_profile.side_effect = start_profile
    engine_client.stop_profile.side_effect = stop_profile
    app = FastAPI()
    app.include_router(api_server.profiler_router)
    app.state.engine_client = engine_client

    response = TestClient(app).post("/start_profile", json={"stages": [0, 1]})

    assert response.status_code == 500
    assert response.json() == {"detail": f"Failed to start profiler: {start_error}"}
    engine_client.stop_profile.assert_awaited_once_with(stages=[0, 1])
    if cleanup == "success":
        assert not active_stages
    else:
        assert "cleanup failed" in caplog.text


@pytest.mark.parametrize("failed_stage", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {
            "profile_prefix": "benchmark",
            "delay_iterations": 2,
            "max_iterations": 3,
            "profiler_kwargs": {"record_shapes": False},
        },
        {"profile_prefix": "", "delay_iterations": 0, "max_iterations": 0, "profiler_kwargs": {}},
    ],
)
def test_profiler_session_overrides_preserve_rpc_error_handling(overrides, failed_stage, mocker) -> None:
    engine_client = mocker.AsyncMock(spec=api_server.AsyncOmni)
    detail = "stage 1: bad trace directory"
    engine_client.start_profile.return_value = [None, {"supported": False, "error": detail}] if failed_stage else [None]
    engine_client.stop_profile.return_value = [None]
    app = FastAPI()
    app.include_router(api_server.profiler_router)
    app.state.engine_client = engine_client

    response = TestClient(app).post("/start_profile", json={"stages": [0, 1], **overrides})

    engine_client.start_profile.assert_awaited_once_with(stages=[0, 1], **overrides)
    if failed_stage:
        assert response.status_code == 500
        assert response.json() == {"detail": f"Failed to start profiler: {detail}"}
        engine_client.stop_profile.assert_awaited_once_with(stages=[0, 1])
    else:
        assert response.status_code == 200
        assert response.json() == {"status": "SUCCESS"}
        engine_client.stop_profile.assert_not_awaited()
