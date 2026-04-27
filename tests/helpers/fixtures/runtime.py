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


def _core_model_stage_config_path_with_dummy_load_format(stage_config_path: str | None, run_level: str) -> str | None:
    """For ``core_model`` runs, patch every stage in the deploy YAML to ``load_format: dummy``.

    Matches ``omni_server`` and keeps multi-stage L2 tests fast without full weights.
    """
    if run_level != "core_model" or stage_config_path is None:
        return stage_config_path
    import yaml

    from tests.helpers.stage_config import modify_stage_config

    with open(stage_config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    new_schema_stages = cfg.get("stages")
    stage_key = "stages" if new_schema_stages is not None else "stage_args"
    update_path = "load_format" if new_schema_stages is not None else "engine_args.load_format"
    stage_entries = cfg.get(stage_key, [])
    stage_ids = [stage["stage_id"] for stage in stage_entries if "stage_id" in stage]
    return modify_stage_config(
        stage_config_path,
        updates={stage_key: {stage_id: {update_path: "dummy"} for stage_id in stage_ids}},
    )


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
def omni_runner(request: pytest.FixtureRequest, model_prefix: str, run_level: str):
    from tests.helpers.runtime import OmniRunner

    with omni_fixture_lock:
        param = request.param
        if not isinstance(param, (tuple, list)) or len(param) not in (2, 3):
            raise ValueError(
                "omni_runner param must be (model, stage_config_path) or "
                "(model, stage_config_path, extra_omni_kwargs_dict)"
            )
        if len(param) == 2:
            model, stage_config_path = param[0], param[1]
            extra_omni_kwargs: dict = {}
        else:
            model, stage_config_path, extra = param[0], param[1], param[2]
            extra_omni_kwargs = dict(extra) if extra is not None else {}
        stage_config_path = _core_model_stage_config_path_with_dummy_load_format(stage_config_path, run_level)
        model = model_prefix + model
        with OmniRunner(model, seed=42, stage_configs_path=stage_config_path, **extra_omni_kwargs) as runner:
            print("OmniRunner started successfully")
            yield runner
            print("OmniRunner stopping...")

        print("OmniRunner stopped")


@pytest.fixture
def omni_runner_handler(omni_runner: Any):
    from tests.helpers.runtime import OmniRunnerHandler

    return OmniRunnerHandler(omni_runner)
