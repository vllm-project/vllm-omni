"""Runtime fixtures (OmniRunner / OmniServer). Imports are deferred to fixture time.

Loading ``tests.helpers.runtime`` at plugin import time (before session fixtures)
pulls in vLLM/vllm_omni too early and breaks initialization order vs the legacy
monolithic conftest. Defer imports until fixtures run so ``default_env`` /
``default_vllm_config`` run first.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.helpers.runtime import OmniServer

omni_fixture_lock = threading.Lock()


@pytest.fixture(scope="function")
def omni_server_function(
    request: pytest.FixtureRequest,
    run_level: str,
    model_prefix: str,
) -> Generator[OmniServer, Any, None]:
    from tests.helpers.runtime import run_omni_server

    yield from run_omni_server(request, run_level, model_prefix, omni_fixture_lock)


@pytest.fixture(scope="module")
def omni_server(request: pytest.FixtureRequest, run_level: str, model_prefix: str) -> Generator[OmniServer, Any, None]:
    """Start vLLM-Omni through the standard or stage-CLI launcher.

    The fixture stays module-scoped because multi-stage initialization is costly.
    The ``use_stage_cli`` flag on ``OmniServerParams`` routes the setup through the
    stage-CLI harness while still reusing the same fixture grouping semantics.
    """
    from tests.helpers.runtime import run_omni_server

    yield from run_omni_server(request, run_level, model_prefix, omni_fixture_lock)


@pytest.fixture
def openai_client(request: pytest.FixtureRequest, run_level: str):
    """Resolve ``omni_server`` lazily so parametrized server fixtures work like upstream."""
    from tests.helpers.runtime import OpenAIClientHandler

    server = request.getfixturevalue("omni_server")
    return OpenAIClientHandler(host=server.host, port=server.port, api_key="EMPTY", run_level=run_level)


@pytest.fixture
def openai_client_function(request: pytest.FixtureRequest, run_level: str):
    """Resolve ``omni_server_function`` lazily for function-scoped reliability tests."""
    from tests.helpers.runtime import OpenAIClientHandler

    server = request.getfixturevalue("omni_server_function")
    return OpenAIClientHandler(host=server.host, port=server.port, api_key="EMPTY", run_level=run_level)


@pytest.fixture(scope="module")
def omni_runner(request: pytest.FixtureRequest, model_prefix: str):
    from tests.helpers.runtime import OmniRunner

    with omni_fixture_lock:
        model, stage_config_path = request.param
        model = model_prefix + model
        with OmniRunner(model, seed=42, stage_configs_path=stage_config_path) as runner:
            print("OmniRunner started successfully")
            yield runner
            print("OmniRunner stopping...")

        print("OmniRunner stopped")


@pytest.fixture
def omni_runner_handler(omni_runner: Any):
    from tests.helpers.runtime import OmniRunnerHandler

    return OmniRunnerHandler(omni_runner)
