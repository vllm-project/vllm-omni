# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.serve.weight_transfer_api import router

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(params=[("init_weight_transfer_engine", "init_info"), ("update_weights", "update_info")])
def endpoint(request):
    method, field = request.param
    call = AsyncMock()
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = SimpleNamespace(**{method: call})
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, f"/{method}", field, call


@pytest.mark.parametrize("body", ["[]", "null", '"hello"', "42", "true"])
def test_non_object_json_returns_bad_request(endpoint, body):
    client, path, _, call = endpoint
    response = client.post(path, content=body, headers={"Content-Type": "application/json"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Request body must be a JSON object"
    call.assert_not_awaited()


@pytest.mark.parametrize("body", ["{", "{}"])
def test_invalid_json_or_missing_field_returns_bad_request(endpoint, body):
    client, path, _, call = endpoint
    response = client.post(path, content=body, headers={"Content-Type": "application/json"})

    assert response.status_code == 400
    call.assert_not_awaited()


def test_valid_object_forwards_backend_payload(endpoint):
    client, path, field, call = endpoint
    payload: dict[str, object] = (
        {"packed": False} if field == "init_info" else {"names": ["transformer.weight"], "shapes": [[2, 2]]}
    )
    response = client.post(path, json={field: payload})

    assert response.status_code == 200
    call.assert_awaited_once_with(payload)
