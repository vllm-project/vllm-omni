"""Pytest fixtures for reliability tests.

``ps -ef`` transcripts (optional): set ``RELIABILITY_PS_EF`` (case-insensitive):

- unset / ``0`` / ``off`` — disabled (default; keeps CI logs small)
- ``session`` / ``1`` / ``on`` — dump at pytest session start and finish
- ``each`` / ``test`` — dump before and after **every** test node. **before** runs
  before other fixtures (e.g. before ``omni_server_function`` starts a server);
  **after** runs last (after server teardown), and is the transcript that reflects
  post-scenario process state.
- ``all`` — session + per-test

POSIX only; on Windows, dumps print a skip line.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.dfx.reliability.helpers import (
    FaultInjector,
    dump_ps_ef_transcript,
    reliability_ps_ef_active_modes,
)


def pytest_sessionstart(session: pytest.Session) -> None:
    if "session" in reliability_ps_ef_active_modes():
        dump_ps_ef_transcript("pytest session start (reliability)")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if "session" in reliability_ps_ef_active_modes():
        dump_ps_ef_transcript(f"pytest session finish (reliability) exitstatus={exitstatus}")


@pytest.fixture(autouse=True)
def _reliability_ps_ef_around_test(request: pytest.FixtureRequest) -> Any:
    if "each" in reliability_ps_ef_active_modes():
        dump_ps_ef_transcript(f"before test: {request.node.nodeid}")
    yield
    if "each" in reliability_ps_ef_active_modes():
        dump_ps_ef_transcript(f"after test: {request.node.nodeid}")


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
