"""Shared fixtures for MiniCPM-o 4.5 E2E tests.

All tests in this directory share a single three-stage duplex server
per module, avoiding redundant weight loads across test files.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from typing import Any

import pytest

from tests.helpers.minicpmo_4_5_duplex import DEPLOY_CONFIG, MODEL
from tests.helpers.runtime import OmniServerParams

_SERVER_PARAMS = OmniServerParams(
    model=MODEL,
    stage_config_path=DEPLOY_CONFIG,
    use_stage_cli=True,
    server_args=["--trust-remote-code"],
)

_omni_fixture_lock = threading.Lock()


@pytest.fixture(scope="module")
def omni_server(
    request: pytest.FixtureRequest,
    run_level: str,
    model_prefix: str,
) -> Generator[Any, Any, None]:
    from tests.helpers.runtime import iter_omni_server

    request.param = _SERVER_PARAMS
    yield from iter_omni_server(request, run_level, model_prefix, _omni_fixture_lock)
