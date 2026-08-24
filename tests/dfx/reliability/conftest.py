"""Pytest fixtures for reliability tests."""

from __future__ import annotations

from typing import Any

import pytest

from tests.dfx.reliability.helpers import FaultInjector


@pytest.fixture
def fault_injector(request: pytest.FixtureRequest) -> FaultInjector:
    """Indirect only: ``request.param`` must be a ``FaultInjector`` callable."""
    return request.param


@pytest.fixture
def omni_server_after_fault(omni_server: Any, fault_injector: FaultInjector):
    """After ``omni_server`` is up, run ``fault_injector(omni_server)``, then yield the server."""
    fault_injector(omni_server)
    yield omni_server


@pytest.fixture
def omni_server_after_fault_function(omni_server_function: Any, fault_injector: FaultInjector):
    """Inject fault after function-scoped server startup, then yield server."""
    fault_injector(omni_server_function)
    yield omni_server_function
