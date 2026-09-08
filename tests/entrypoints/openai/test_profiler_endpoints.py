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
def test_profiler_endpoint_forwards_selected_stages(path: str, engine_method: str) -> None:
    engine_client = SimpleNamespace(
        start_profile=AsyncMock(),
        stop_profile=AsyncMock(),
    )

    app = FastAPI()
    app.include_router(api_server.profiler_router)
    app.state.engine_client = engine_client

    response = TestClient(app).post(path, json={"stages": [1]})

    assert response.status_code == 200
    assert response.json() == {"status": "SUCCESS"}
    getattr(engine_client, engine_method).assert_awaited_once_with(stages=[1])
