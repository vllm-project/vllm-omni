"""Tests for sleep/wake HTTP endpoint registration in vLLM-Omni.

Verifies that the sleep router is conditionally attached to the FastAPI app
based on ``--enable-sleep-mode``, without requiring ``VLLM_SERVER_DEV_MODE``.
"""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.routing import Route

from vllm_omni.entrypoints.sleep_api import include_sleep_router_if_enabled, sleep_router

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SLEEP_PATHS = {"/sleep", "/wake_up", "/is_sleeping"}


def _get_route_paths(app: FastAPI) -> set[str]:
    """Extract all route paths from a FastAPI application."""
    return {r.path for r in app.routes if isinstance(r, Route)}


def _build_app_with_sleep(enable_sleep: bool) -> FastAPI:
    """Build a minimal FastAPI app that mirrors the omni server's
    sleep-router registration logic."""
    app = FastAPI()
    args = argparse.Namespace(enable_sleep_mode=enable_sleep)
    include_sleep_router_if_enabled(app, args)

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSleepRouterRegistration:
    """Verify sleep/wake routes are registered iff enable_sleep_mode=True."""

    def test_sleep_routes_registered_when_enabled(self):
        app = _build_app_with_sleep(enable_sleep=True)
        paths = _get_route_paths(app)
        for p in SLEEP_PATHS:
            assert p in paths, f"Expected {p} in app routes"

    def test_sleep_routes_absent_when_disabled(self):
        app = _build_app_with_sleep(enable_sleep=False)
        paths = _get_route_paths(app)
        for p in SLEEP_PATHS:
            assert p not in paths, f"Did not expect {p} in app routes"

    def test_no_duplicate_routes_when_already_registered(self):
        """If upstream DEV_MODE already registered the router, we should
        not add duplicate routes."""
        app = FastAPI()
        app.include_router(sleep_router)  # simulate DEV_MODE registration
        count_before = sum(1 for r in app.routes if isinstance(r, Route) and r.path in SLEEP_PATHS)

        include_sleep_router_if_enabled(app, argparse.Namespace(enable_sleep_mode=True))

        count_after = sum(1 for r in app.routes if isinstance(r, Route) and r.path in SLEEP_PATHS)
        assert count_after == count_before, "Sleep routes were duplicated"


class TestSleepEndpoints:
    """Test the actual HTTP endpoint behaviour using a test client."""

    @pytest.fixture()
    def client(self):
        app = _build_app_with_sleep(enable_sleep=True)

        # Provide a mock engine_client on app.state
        mock_engine = AsyncMock()
        mock_engine.sleep = AsyncMock(return_value=None)
        mock_engine.wake_up = AsyncMock(return_value=None)
        mock_engine.is_sleeping = AsyncMock(return_value=False)
        app.state.engine_client = mock_engine

        return TestClient(app)

    def test_is_sleeping_returns_json(self, client: TestClient):
        resp = client.get("/is_sleeping")
        assert resp.status_code == 200
        assert resp.json() == {"is_sleeping": False}

    def test_sleep_returns_200(self, client: TestClient):
        resp = client.post("/sleep?level=1")
        assert resp.status_code == 200

    def test_wake_up_returns_200(self, client: TestClient):
        resp = client.post("/wake_up")
        assert resp.status_code == 200

    def test_sleep_passes_level(self, client: TestClient):
        client.post("/sleep?level=2")
        engine = client.app.state.engine_client
        engine.sleep.assert_awaited_once_with(level=2, mode="abort")

    def test_sleep_passes_mode(self, client: TestClient):
        client.post("/sleep?level=2&mode=wait")
        engine = client.app.state.engine_client
        engine.sleep.assert_awaited_once_with(level=2, mode="wait")

    def test_wake_up_passes_tags(self, client: TestClient):
        client.post("/wake_up?tags=weights")
        engine = client.app.state.engine_client
        engine.wake_up.assert_awaited_once_with(tags=["weights"])

    def test_wake_up_no_tags_passes_none(self, client: TestClient):
        client.post("/wake_up")
        engine = client.app.state.engine_client
        engine.wake_up.assert_awaited_once_with(tags=None)
