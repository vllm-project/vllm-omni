# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route-level tests for RL rollout serving endpoints."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.protocol.rollout import (
    RolloutStepResponse,
    SessionMetadata,
)
from vllm_omni.entrypoints.openai.rollout_session import (
    RolloutSessionClosedError,
    RolloutSessionNotFoundError,
)


class _FakeRolloutServing:

    def __init__(self, close_error: Exception | None = None) -> None:
        self.close_error = close_error
        self.step_request = None

    async def step(self, session_id, body):
        self.step_request = body
        return RolloutStepResponse(
            step_id=body.step_id,
            next_observation=None,
            model_metadata=SessionMetadata(
                latency_ms=0.0,
                steps_generated=0,
                session_context_length=0,
                committed_step_id=-1,
            ),
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


def test_step_route_accepts_base64_image_payload():
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

    assert response.status_code == 200
    assert serving.step_request.observation.images == {"front": "base64-frames"}


def test_step_route_requires_action():
    client = _client(_FakeRolloutServing())

    response = client.post(
        "/v1/realtime/sessions/s1/step",
        json={"step_id": 0, "observation": {}},
    )

    assert response.status_code == 422


def test_close_route_maps_missing_session_to_404():
    client = _client(_FakeRolloutServing(close_error=RolloutSessionNotFoundError("s1")))

    response = client.post("/v1/realtime/sessions/s1/close")

    assert response.status_code == 404


def test_close_route_maps_closed_session_to_410():
    client = _client(_FakeRolloutServing(close_error=RolloutSessionClosedError("s1")))

    response = client.post("/v1/realtime/sessions/s1/close")

    assert response.status_code == 410


def test_rollout_routes_return_501_when_serving_is_unavailable():
    client = _client()

    response = client.post("/v1/realtime/sessions/s1/close")

    assert response.status_code == 501
