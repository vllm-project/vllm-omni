# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for endpoint restrictions logic."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_omni.config.endpoint_policy import (
    EndpointRestriction,
    OmniServingCapability,
    shutdown_unsupported_routes,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SERVER_ROUTE = "/v1/completions"
REJECTION_REASON = "This route is banned!"


def _make_app_with_server_route():
    """Create a FastAPI app with a route to reject."""
    app = FastAPI()

    @app.post(SERVER_ROUTE)
    async def existing_handler():
        """Returns a response; we want to disable this."""
        return {"ok": True}

    return app


def test_restricted_completions_returns_400():
    """Ensure that simple route rejection works."""
    app = _make_app_with_server_route()
    restrictions = (EndpointRestriction(OmniServingCapability.COMPLETIONS, REJECTION_REASON),)
    shutdown_unsupported_routes(app, restrictions)

    client = TestClient(app)
    resp = client.post(SERVER_ROUTE, json={"prompt": "hello", "model": "m"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["message"] == REJECTION_REASON
    assert body["error"]["type"] == "BadRequestError"


def test_unrestricted_completions_not_blocked():
    """Ensure that if we don't shutdown any routes, the route stays."""
    app = _make_app_with_server_route()
    shutdown_unsupported_routes(app, ())

    client = TestClient(app)
    resp = client.post(SERVER_ROUTE)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_route_rejection_is_idempotent():
    """Ensure that calling rejection twice doesn't break things."""
    app = _make_app_with_server_route()
    restrictions = (EndpointRestriction(OmniServingCapability.COMPLETIONS, REJECTION_REASON),)
    shutdown_unsupported_routes(app, restrictions)
    shutdown_unsupported_routes(app, restrictions)

    client = TestClient(app)
    resp = client.post(SERVER_ROUTE, json={"prompt": "hello", "model": "m"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["message"] == REJECTION_REASON
    assert body["error"]["type"] == "BadRequestError"


def test_invalid_route_rejection():
    """Ensure that a restriction won't crash if a route doesn't exist."""
    # NOTE: This test is for safety since the route organization is still messy.
    # Since the restriction needs a valid OmniServingCapability, its implied the route
    # is a valid vLLM route, but was already removed from the app by earlier processing.
    app = FastAPI()
    restrictions = (EndpointRestriction(OmniServingCapability.COMPLETIONS, REJECTION_REASON),)
    shutdown_unsupported_routes(app, restrictions)

    client = TestClient(app)
    resp = client.post(SERVER_ROUTE, json={"prompt": "hello", "model": "m"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["message"] == REJECTION_REASON
    assert body["error"]["type"] == "BadRequestError"
