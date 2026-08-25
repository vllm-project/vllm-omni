# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Runtime fixtures (OmniRunner / OmniServer). Imports are deferred to fixture time.

Loading ``tests.helpers.runtime`` at plugin import time (before session fixtures)
pulls in vLLM/vllm_omni too early and breaks initialization order vs the legacy
monolithic conftest. Defer imports until fixtures run so session fixtures in
``tests.helpers.fixtures.config`` run first. Server/runner helpers live in
``tests.helpers.runtime``; request clients live in ``tests.helpers.client``.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.helpers.runtime import OmniRunner, OmniServer

omni_fixture_lock = threading.Lock()


@pytest.fixture(scope="function")
def omni_server_function(
    request: pytest.FixtureRequest,
    run_level: str,
) -> Generator[OmniServer, Any, None]:
    from tests.helpers.runtime import iter_omni_server

    yield from iter_omni_server(request, run_level, omni_fixture_lock)


@pytest.fixture(scope="module")
def omni_server(request: pytest.FixtureRequest, run_level: str) -> Generator[OmniServer, Any, None]:
    """Start vLLM-Omni through the standard or stage-CLI launcher.

    The fixture stays module-scoped because multi-stage initialization is costly.
    The ``use_stage_cli`` flag on ``OmniServerParams`` routes the setup through the
    stage-CLI harness while still reusing the same fixture grouping semantics.
    """
    from tests.helpers.runtime import iter_omni_server

    yield from iter_omni_server(request, run_level, omni_fixture_lock)


@pytest.fixture
def online_client(request: pytest.FixtureRequest, run_level: str):
    """Resolve ``omni_server`` lazily so parametrized server fixtures work like upstream."""
    from tests.helpers.client import OnlineOmniClient

    server = request.getfixturevalue("omni_server")
    return OnlineOmniClient(
        host=server.host,
        port=server.port,
        api_key="EMPTY",
        run_level=run_level,
        log_stats=server.log_stats,
    )


@pytest.fixture
def openai_client(online_client):
    """Historical alias for :func:`online_client`."""
    return online_client


@pytest.fixture
def online_client_function(request: pytest.FixtureRequest, run_level: str):
    """Resolve ``omni_server_function`` lazily for function-scoped reliability tests."""
    from tests.helpers.client import OnlineOmniClient

    server = request.getfixturevalue("omni_server_function")
    return OnlineOmniClient(
        host=server.host,
        port=server.port,
        api_key="EMPTY",
        run_level=run_level,
        log_stats=server.log_stats,
    )


@pytest.fixture
def openai_client_function(online_client_function):
    """Historical alias for :func:`online_client_function`."""
    return online_client_function


@pytest.fixture(scope="function")
def omni_runner_function(
    request: pytest.FixtureRequest,
    run_level: str,
) -> Generator[OmniRunner, Any, None]:
    """Function-scoped :class:`~tests.helpers.runtime.OmniRunner` (cf. :func:`omni_server_function`).

    Tears down the runner after each test so the next test does not share engine
    state with a module-scoped :func:`omni_runner`.
    """
    from tests.helpers.runtime import iter_omni_runner

    yield from iter_omni_runner(request, run_level, omni_fixture_lock)


@pytest.fixture(scope="module")
def omni_runner(request: pytest.FixtureRequest, run_level: str) -> Generator[OmniRunner, Any, None]:
    """Module-scoped :class:`~tests.helpers.runtime.OmniRunner` (cf. :func:`omni_server`).

    Reuses one runner for the whole module to amortize multi-stage init cost.
    """
    from tests.helpers.runtime import iter_omni_runner

    yield from iter_omni_runner(request, run_level, omni_fixture_lock)


@pytest.fixture
def offline_client_function(omni_runner_function: OmniRunner):
    """Resolve :class:`~tests.helpers.client.OfflineOmniClient` for :func:`omni_runner_function`."""
    from tests.helpers.client import OfflineOmniClient

    return OfflineOmniClient(omni_runner_function)


@pytest.fixture
def omni_runner_handler_function(offline_client_function):
    """Historical alias for :func:`offline_client_function`."""
    return offline_client_function


@pytest.fixture
def offline_client(omni_runner: OmniRunner):
    from tests.helpers.client import OfflineOmniClient

    return OfflineOmniClient(omni_runner)


@pytest.fixture
def omni_runner_handler(offline_client):
    """Historical alias for :func:`offline_client`."""
    return offline_client
