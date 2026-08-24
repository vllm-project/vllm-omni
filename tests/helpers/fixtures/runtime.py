"""Runtime fixtures (OmniRunner / OmniServer). Imports are deferred to fixture time.

Loading ``tests.helpers.runtime`` at plugin import time (before session fixtures)
pulls in vLLM/vllm_omni too early and breaks initialization order vs the legacy
monolithic conftest. Defer imports until fixtures run so ``default_env`` /
``default_vllm_config`` run first. Implementation helpers live in
``tests.helpers.runtime`` (``iter_omni_server`` / ``iter_omni_runner``).
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
    model_prefix: str,
) -> Generator[OmniServer, Any, None]:
    from tests.helpers.runtime import iter_omni_server

    yield from iter_omni_server(request, run_level, model_prefix, omni_fixture_lock)


@pytest.fixture(scope="module")
def omni_server(request: pytest.FixtureRequest, run_level: str, model_prefix: str) -> Generator[OmniServer, Any, None]:
    """Start vLLM-Omni through the standard or stage-CLI launcher.

    The fixture stays module-scoped because multi-stage initialization is costly.
    The ``use_stage_cli`` flag on ``OmniServerParams`` routes the setup through the
    stage-CLI harness while still reusing the same fixture grouping semantics.
    """
    from tests.helpers.runtime import iter_omni_server

    yield from iter_omni_server(request, run_level, model_prefix, omni_fixture_lock)


@pytest.fixture
def openai_client(request: pytest.FixtureRequest, run_level: str):
    """Resolve ``omni_server`` lazily so parametrized server fixtures work like upstream."""
    from tests.helpers.runtime import OpenAIClientHandler

    server = request.getfixturevalue("omni_server")
    return OpenAIClientHandler(
        host=server.host,
        port=server.port,
        api_key="EMPTY",
        run_level=run_level,
        log_stats=server.log_stats,
    )


@pytest.fixture
def openai_client_function(request: pytest.FixtureRequest, run_level: str):
    """Resolve ``omni_server_function`` lazily for function-scoped reliability tests."""
    from tests.helpers.runtime import OpenAIClientHandler

    server = request.getfixturevalue("omni_server_function")
    return OpenAIClientHandler(
        host=server.host,
        port=server.port,
        api_key="EMPTY",
        run_level=run_level,
        log_stats=server.log_stats,
    )


@pytest.fixture(scope="function")
def omni_runner_function(
    request: pytest.FixtureRequest,
    model_prefix: str,
    run_level: str,
) -> Generator[OmniRunner, Any, None]:
    """Function-scoped :class:`~tests.helpers.runtime.OmniRunner` (cf. :func:`omni_server_function`).

    Tears down the runner after each test so the next test does not share engine
    state with a module-scoped :func:`omni_runner`.
    """
    from tests.helpers.runtime import iter_omni_runner

    yield from iter_omni_runner(request, model_prefix, run_level, omni_fixture_lock)


@pytest.fixture(scope="module")
def omni_runner(request: pytest.FixtureRequest, model_prefix: str, run_level: str) -> Generator[OmniRunner, Any, None]:
    """Module-scoped :class:`~tests.helpers.runtime.OmniRunner` (cf. :func:`omni_server`).

    Reuses one runner for the whole module to amortize multi-stage init cost.
    """
    from tests.helpers.runtime import iter_omni_runner

    yield from iter_omni_runner(request, model_prefix, run_level, omni_fixture_lock)


@pytest.fixture
def omni_runner_handler_function(omni_runner_function: OmniRunner):
    """Resolve :class:`~tests.helpers.runtime.OmniRunnerHandler` for :func:`omni_runner_function`."""
    from tests.helpers.runtime import OmniRunnerHandler

    return OmniRunnerHandler(omni_runner_function)


@pytest.fixture
def omni_runner_handler(omni_runner: OmniRunner):
    from tests.helpers.runtime import OmniRunnerHandler

    return OmniRunnerHandler(omni_runner)
