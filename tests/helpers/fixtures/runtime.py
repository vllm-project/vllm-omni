"""Runtime fixtures (OmniRunner / OmniServer). Imports are deferred to fixture time.

Loading ``tests.helpers.runtime`` at plugin import time (before session fixtures)
pulls in vLLM/vllm_omni too early and breaks initialization order vs the legacy
monolithic conftest. Defer imports until fixtures run so ``default_env`` /
``default_vllm_config`` run first.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from typing import Any

import pytest
import yaml

from tests.helpers.stage_config import modify_stage_config

omni_fixture_lock = threading.Lock()


@pytest.fixture(scope="module")
def omni_server(request: pytest.FixtureRequest, run_level: str, model_prefix: str) -> Generator[Any, Any, None]:
    """Start vLLM-Omni through the standard or stage-CLI launcher (aligned with upstream ``tests/conftest``)."""
    from tests.helpers.runtime import OmniServer, OmniServerParams, OmniServerStageCli

    with omni_fixture_lock:
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
def openai_client(request: pytest.FixtureRequest, run_level: str):
    """Resolve ``omni_server`` lazily so parametrized server fixtures work like upstream."""
    from tests.helpers.runtime import OpenAIClientHandler

    server = request.getfixturevalue("omni_server")
    return OpenAIClientHandler(host=server.host, port=server.port, api_key="EMPTY", run_level=run_level)


@pytest.fixture(scope="module")
def omni_runner(request: pytest.FixtureRequest, model_prefix: str):
    from tests.helpers.runtime import OmniRunner

    with omni_fixture_lock:
        model, stage_config_path = request.param
        model = model_prefix + model
        with OmniRunner(model, seed=42, stage_configs_path=stage_config_path, stage_init_timeout=300) as runner:
            print("OmniRunner started successfully")
            yield runner
            print("OmniRunner stopping...")

        print("OmniRunner stopped")


@pytest.fixture
def omni_runner_handler(omni_runner: Any):
    from tests.helpers.runtime import OmniRunnerHandler

    return OmniRunnerHandler(omni_runner)
