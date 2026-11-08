# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the public runtime LoRA inventory route."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.openai.api_server import router

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _client(engine: object) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = engine
    return TestClient(app, raise_server_exceptions=False)


def test_lora_inventory_returns_stage_and_replica_records() -> None:
    async def get_lora_inventory() -> list[dict[str, object]]:
        return [
            {
                "stage_id": 0,
                "model_stage": "thinker",
                "replicas": [
                    {
                        "replica_id": 0,
                        "adapters": [{"id": 7, "name": "style-a"}],
                    }
                ],
            },
            {
                "stage_id": 1,
                "model_stage": "denoise",
                "replicas": [{"replica_id": 0, "adapters": []}],
            },
        ]

    with _client(SimpleNamespace(get_lora_inventory=get_lora_inventory)) as client:
        response = client.get("/v1/omni/loras")

    assert response.status_code == 200
    assert response.json() == {
        "object": "omni.lora_inventory",
        "data": [
            {
                "stage_id": 0,
                "model_stage": "thinker",
                "replicas": [
                    {
                        "replica_id": 0,
                        "adapters": [{"id": 7, "name": "style-a"}],
                    }
                ],
            },
            {
                "stage_id": 1,
                "model_stage": "denoise",
                "replicas": [{"replica_id": 0, "adapters": []}],
            },
        ],
    }


def test_lora_inventory_without_inventory_capability_returns_service_unavailable() -> None:
    with _client(SimpleNamespace()) as client:
        response = client.get("/v1/omni/loras")

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "LoRA inventory is temporarily unavailable."


def test_lora_inventory_query_failure_is_isolated() -> None:
    async def get_lora_inventory() -> list[dict[str, object]]:
        raise RuntimeError("worker failure")

    engine = SimpleNamespace(get_lora_inventory=get_lora_inventory, generate=object())
    with _client(engine) as client:
        response = client.get("/v1/omni/loras")

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "LoRA inventory is temporarily unavailable."
