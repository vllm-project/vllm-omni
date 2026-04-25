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

from tests.helpers.runtime import OmniServer
from tests.helpers.stage_config import modify_stage_config

omni_fixture_lock = threading.Lock()


def _core_model_stage_config_path_with_dummy_load_format(stage_config_path: str | None, run_level: str) -> str | None:
    """For ``core_model`` runs, patch every stage in the deploy YAML to ``load_format: dummy``.

    Matches ``omni_server`` and keeps multi-stage L2 tests fast without full weights.
    """
    if run_level != "core_model" or stage_config_path is None:
        return stage_config_path
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


@pytest.fixture(scope="module")
def omni_server(request: pytest.FixtureRequest, run_level: str, model_prefix: str) -> Generator[OmniServer, Any, None]:
    """Start vLLM-Omni through the standard or stage-CLI launcher.

    The fixture stays module-scoped because multi-stage initialization is costly.
    The ``use_stage_cli`` flag on ``OmniServerParams`` routes the setup through the
    stage-CLI harness while still reusing the same fixture grouping semantics.
    """
    with omni_fixture_lock:
        from tests.helpers.runtime import OmniServer, OmniServerParams, OmniServerStageCli

        params: OmniServerParams = request.param
        model = model_prefix + params.model
        port = params.port
        stage_config_path = _core_model_stage_config_path_with_dummy_load_format(params.stage_config_path, run_level)

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
            server_args += ["--stage-configs-path", stage_config_path]

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
