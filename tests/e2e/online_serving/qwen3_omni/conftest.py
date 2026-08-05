"""Shared fixtures for Qwen3-Omni duplex E2E tests."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEPLOY_CONFIG = get_deploy_config_path("qwen3_omni_moe_duplex_1gpu.yaml")

DUPLEX_SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            use_stage_cli=True,
        ),
        id="three-stage-duplex",
    )
]


@pytest.fixture(scope="module", params=DUPLEX_SERVER_PARAMS)
def omni_server(request, run_level: str, model_prefix: str) -> Generator[Any, Any, None]:
    from tests.helpers.fixtures.runtime import omni_fixture_lock
    from tests.helpers.runtime import iter_omni_server

    yield from iter_omni_server(request, run_level, model_prefix, omni_fixture_lock)
