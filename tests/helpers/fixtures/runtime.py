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


class _SharedMiniCPMoServer:
    """Reference-counted MiniCPM-o duplex server singleton.

    Combined with ``pytest_collection_modifyitems`` grouping (see
    ``tests/helpers/fixtures/minicpmo_grouping.py``), this keeps the
    server alive only while MiniCPM-o tests are running.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._refcount = 0
        self._server: Any = None
        self._cm: Any = None

    def acquire(self, run_level: str, model_prefix: str) -> Any:
        with self._lock:
            self._refcount += 1
            if self._server is not None:
                return self._server
        self._server = self._start(run_level, model_prefix)
        return self._server

    def release(self) -> None:
        with self._lock:
            self._refcount -= 1
            if self._refcount > 0:
                return
        self._stop()

    def _start(self, run_level: str, model_prefix: str) -> Any:
        from tests.helpers.minicpmo_4_5_duplex import DEPLOY_CONFIG, MODEL
        from tests.helpers.runtime import OmniServerParams, OmniServerStageCli
        from tests.helpers.stage_config import stage_config_path_for_run_level

        params = OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            use_stage_cli=True,
            server_args=["--trust-remote-code"],
        )
        model = model_prefix + params.model
        stage_config_path = stage_config_path_for_run_level(params.stage_config_path, run_level)
        server_args = list(params.server_args or [])
        server_args += ["--stage-init-timeout", "600", "--init-timeout", "900", "--log-stats"]
        server_args += ["--stage-configs-path", stage_config_path]
        cm = OmniServerStageCli(model, stage_config_path, server_args)
        self._cm = cm
        server = cm.__enter__()
        print("SharedMiniCPMoServer started")
        return server

    def _stop(self) -> None:
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            print("SharedMiniCPMoServer stopped")
        self._cm = None
        self._server = None


_minicpmo_shared = _SharedMiniCPMoServer()


@pytest.fixture(scope="module")
def minicpmo_duplex_server(
    run_level: str,
    model_prefix: str,
) -> Generator[OmniServer, Any, None]:
    """Module-scoped fixture backed by a reference-counted server singleton."""
    server = _minicpmo_shared.acquire(run_level, model_prefix)
    try:
        yield server
    finally:
        _minicpmo_shared.release()


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
