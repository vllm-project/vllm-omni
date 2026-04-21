"""Pytest fixtures for reliability tests."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from tests.dfx.reliability.helpers import FaultInjector
from tests.helpers.fixtures.runtime import omni_fixture_lock as _omni_server_lock
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OmniServerStageCli,
    OpenAIClientHandler,
)
from tests.helpers.stage_config import modify_stage_config


@pytest.fixture
def fault_injector(request: pytest.FixtureRequest) -> FaultInjector:
    """Indirect only: ``request.param`` must be a ``FaultInjector`` callable."""
    return request.param


@pytest.fixture
def omni_server_after_fault(omni_server: Any, fault_injector: FaultInjector):
    """After ``omni_server`` is up, run ``fault_injector(omni_server)``, then yield the server."""
    fault_injector(omni_server)
    yield omni_server


@pytest.fixture(scope="function")
def omni_server_function(
    request: pytest.FixtureRequest,
    run_level: str,
    model_prefix: str,
):
    """Function-scoped Omni server fixture for reliability tests."""
    with _omni_server_lock:
        params: OmniServerParams = request.param
        model = model_prefix + params.model
        port = params.port
        stage_config_path = params.stage_config_path
        if run_level == "advanced_model" and stage_config_path is not None:
            with open(stage_config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            stage_ids = [stage["stage_id"] for stage in cfg.get("stage_args", []) if "stage_id" in stage]
            stage_config_path = modify_stage_config(
                stage_config_path,
                deletes={"stage_args": {stage_id: ["engine_args.load_format"] for stage_id in stage_ids}},
            )

        server_args = params.server_args or []
        if params.use_omni and params.stage_init_timeout is not None:
            server_args = [*server_args, "--stage-init-timeout", str(params.stage_init_timeout)]
        else:
            server_args = [*server_args, "--stage-init-timeout", "600"]
        if params.init_timeout is not None:
            server_args = [*server_args, "--init-timeout", str(params.init_timeout)]
        else:
            server_args = [*server_args, "--init-timeout", "900"]
        if params.use_stage_cli:
            if not params.use_omni:
                raise ValueError("omni_server with use_stage_cli=True requires use_omni=True")
            if stage_config_path is None:
                raise ValueError("omni_server with use_stage_cli=True requires a stage_config_path")

            with OmniServerStageCli(
                model,
                stage_config_path,
                server_args,
                port=port,
                env_dict=params.env_dict,
            ) as server:
                print("OmniServer started successfully")
                yield server
                print("OmniServer stopping...")
        else:
            if stage_config_path is not None:
                server_args += ["--stage-configs-path", stage_config_path]

            with (
                OmniServer(
                    model,
                    server_args,
                    port=port,
                    env_dict=params.env_dict,
                    use_omni=params.use_omni,
                )
                if port
                else OmniServer(
                    model,
                    server_args,
                    env_dict=params.env_dict,
                    use_omni=params.use_omni,
                )
            ) as server:
                print("OmniServer started successfully")
                yield server
                print("OmniServer stopping...")
        print("OmniServer stopped")


@pytest.fixture
def openai_client_function(omni_server_function: Any, run_level: str):
    """OpenAI client bound to function-scoped ``omni_server_function``."""
    return OpenAIClientHandler(
        host=omni_server_function.host,
        port=omni_server_function.port,
        api_key="EMPTY",
        run_level=run_level,
    )


@pytest.fixture
def omni_server_after_fault_function(omni_server_function: Any, fault_injector: FaultInjector):
    """Inject fault after function-scoped server startup, then yield server."""
    fault_injector(omni_server_function)
    yield omni_server_function
