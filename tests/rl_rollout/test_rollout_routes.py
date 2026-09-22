# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Route-level tests for RL rollout serving endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.protocol.rollout import (
    ErrorObject,
    RolloutStepResponse,
    SessionMetadata,
)
from vllm_omni.entrypoints.openai.rollout_session import (
    RolloutSessionClosedError,
    RolloutSessionNotFoundError,
)


class _FakeRolloutServing:
    def __init__(self, close_error: Exception | None = None, step_error_code: str | None = None) -> None:
        self.close_error = close_error
        self.step_error_code = step_error_code
        self.step_request = None

    async def step(self, session_id, body):
        self.step_request = body
        error = None
        if self.step_error_code is not None:
            error = ErrorObject(
                code=self.step_error_code,
                message=f"{self.step_error_code}: {session_id}",
                step_id=body.step_id,
                committed_step_id=-1,
            )
        return RolloutStepResponse(
            step_id=body.step_id,
            next_observation=None,
            model_metadata=SessionMetadata(
                latency_ms=0.0,
                steps_generated=0,
                context_length=0,
                committed_step_id=-1,
            ),
            error=error,
        )

    async def close_session(self, session_id):
        if self.close_error is not None:
            raise self.close_error


def _client(serving: _FakeRolloutServing | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    if serving is not None:
        app.state.rl_rollout_serving = serving
    return TestClient(app)


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_step_route_rejects_base64_image_payload():
    serving = _FakeRolloutServing()
    client = _client(serving)

    response = client.post(
        "/v1/realtime/sessions/s1/step",
        json={
            "step_id": 0,
            "observation": {"images": {"front": "base64-frames"}},
            "action": {},
        },
    )

    assert response.status_code == 422
    assert serving.step_request is None


def test_step_route_accepts_nested_image_payload():
    serving = _FakeRolloutServing()
    client = _client(serving)

    response = client.post(
        "/v1/realtime/sessions/s1/step",
        json={
            "step_id": 0,
            "observation": {"images": {"observation/exterior_image_1_left": [[[0, 0, 0]]]}},
            "action": {},
        },
    )

    assert response.status_code == 200
    assert serving.step_request.observation.images == {"observation/exterior_image_1_left": [[[0, 0, 0]]]}


def test_step_route_requires_action():
    client = _client(_FakeRolloutServing())

    response = client.post(
        "/v1/realtime/sessions/s1/step",
        json={"step_id": 0, "observation": {}},
    )

    assert response.status_code == 422


def test_step_route_maps_missing_session_to_404():
    client = _client(_FakeRolloutServing(step_error_code="session_not_found"))

    response = client.post(
        "/v1/realtime/sessions/s1/step",
        json={"step_id": 0, "observation": {}, "action": {}},
    )

    assert response.status_code == 404


def test_step_route_maps_closed_session_to_410():
    client = _client(_FakeRolloutServing(step_error_code="session_closed"))

    response = client.post(
        "/v1/realtime/sessions/s1/step",
        json={"step_id": 0, "observation": {}, "action": {}},
    )

    assert response.status_code == 410


def test_step_route_maps_invalid_request_to_400():
    client = _client(_FakeRolloutServing(step_error_code="invalid_request"))

    response = client.post(
        "/v1/realtime/sessions/s1/step",
        json={"step_id": 0, "observation": {}, "action": {}},
    )

    assert response.status_code == 400


def test_close_route_maps_missing_session_to_404():
    client = _client(_FakeRolloutServing(close_error=RolloutSessionNotFoundError("s1")))

    response = client.post("/v1/realtime/sessions/s1/close")

    assert response.status_code == 404


def test_close_route_is_idempotent_for_closed_session():
    client = _client(_FakeRolloutServing(close_error=RolloutSessionClosedError("s1")))

    response = client.post("/v1/realtime/sessions/s1/close")

    assert response.status_code == 200
    assert response.json()["closed"] is True


def test_rollout_routes_return_501_when_serving_is_unavailable():
    client = _client()

    response = client.post("/v1/realtime/sessions/s1/close")

    assert response.status_code == 501
