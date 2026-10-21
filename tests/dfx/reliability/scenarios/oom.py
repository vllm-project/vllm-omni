# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU OOM fault steps. The model supplies the probe and the memory config."""

from __future__ import annotations

from typing import Any

import pytest

from tests.dfx.conftest import assert_fault_exception, resolve_oom_device_spec
from tests.dfx.reliability.helpers import FaultModel, FaultScenario, inject_gpu_oom, stop_gpu_oom_hogs


def _deploy_config_path(server: Any) -> str | None:
    stage_config_path = getattr(server, "stage_config_path", None)
    if stage_config_path:
        return str(stage_config_path)
    args: list[str] = getattr(server, "serve_args", [])
    for index, arg in enumerate(args):
        if arg == "--deploy-config" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--deploy-config="):
            return arg.split("=", 1)[1]
    return None


def _start_hard_oom(model: FaultModel, server: Any) -> Any:
    config = model.hard_oom_config
    kwargs: dict[str, Any] = {
        "device": resolve_oom_device_spec(config, _deploy_config_path(server)),
        "target_mem_ratio": config["target_mem_ratio"],
        "hold_seconds": config["hold_seconds"],
        "startup_timeout_sec": config["startup_timeout_sec"],
        "strict": config["strict"],
    }
    if "max_free_bytes" in config:
        kwargs["max_free_bytes"] = config["max_free_bytes"]
    return inject_gpu_oom(**kwargs)


def _expect_probe_failure(model: FaultModel, probe: Any, fail_message: str, client: Any, server: Any) -> None:
    handle = _start_hard_oom(model, server)
    try:
        try:
            probe(client, server)
        except Exception as exc:
            assert_fault_exception(exc, model.fault_error_keywords)
        else:
            pytest.fail(fail_message)
    finally:
        stop_gpu_oom_hogs(handle)


class GpuOomErrorContract(FaultScenario):
    """Text chat and audio output both return HTTP 500 under OOM, and the API stays up."""

    server_set = "oom"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        model.run_error_contract(server, client)


class GpuOomLarge(FaultScenario):
    server_set = "oom"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        _expect_probe_failure(model, model.oom_large_probe, model.oom_large_fail_message, client, server)


class GpuOomConcurrent(FaultScenario):
    server_set = "oom"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        _expect_probe_failure(model, model.oom_concurrent_probe, model.oom_concurrent_fail_message, client, server)


class GpuOomRecovery(FaultScenario):
    server_set = "default"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        model.run_oom_recovery(server, client)
