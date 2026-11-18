# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""HunyuanImage reliability scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.dfx.reliability.helpers import (
    FaultModel,
    launch_scenario_params,
    scenario_params,
    sigterm_serve_root_and_assert_cleanup,
)
from tests.dfx.reliability.scenarios import (
    FullCrashAtLaunch,
    GpuOomAtLaunch,
    GpuOomUnderLoad,
    GracefulShutdown,
    PartialCrashAtExit,
    PartialCrashAtLaunch,
    ProcessKillServeRootNoLoad,
    ProcessKillServeRootWithLoad,
    ProcessKillTreeNoLoad,
    ProcessKillTreeWithLoad,
    ProcessKillWorkerNoLoad,
    ProcessKillWorkerWithLoad,
    ServeRootCrashAtLaunch,
)
from tests.dfx.reliability.scenarios.oom import _assert_image_success, run_oom_under_load
from tests.helpers.mark import hardware_marks
from vllm_omni.platforms import current_omni_platform

_IMAGE_FAST_FAIL_HTTP_TIMEOUT_SEC = 30
_DEPLOY_DIR = Path(__file__).resolve().parents[3] / "vllm_omni" / "deploy"


def _image_request(server: Any, *, size: str = "512x512", steps: int = 4) -> dict[str, Any]:
    return {
        "json": {
            "model": server.model,
            "prompt": "A simple red apple on a white background.",
            "size": size,
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": steps,
            "guidance_scale": 7.5,
            "seed": 42,
            "bot_task": "none",
            "use_system_prompt": "en_unified",
        },
        "timeout": 600,
    }


def _fast_fail_body(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": "post-fault fast-fail check",
        "size": "512x512",
        "n": 1,
        "response_format": "b64_json",
        "num_inference_steps": 4,
        "bot_task": "none",
        "use_system_prompt": "en_unified",
    }


def _reliability_server_params() -> dict[str, Any]:
    """Disable expert parallel only for ROCm fault-injection coverage."""
    params: dict[str, Any] = {
        "model": "tencent/HunyuanImage-3.0-Instruct",
        "stage_config_name": "hunyuan_image3_dit.yaml",
    }
    if current_omni_platform.is_rocm():
        params["update"] = {
            "stages": {
                0: {"parallel_config.enable_expert_parallel": False},
            }
        }
        # A cold or contended MI325 restart can still be loading the 32-shard
        # checkpoint after the 900-second fixture default. Keep the bounded
        # startup deadline, but leave enough headroom for observed progress.
        params["stage_init_timeout"] = 1800
        params["init_timeout"] = 2100
    return params


class HunyuanImage(FaultModel):
    model_id = "hunyuan_image"
    res = {"cuda": "H100", "rocm": "MI325"}
    num_cards = 4
    extra_marks = (pytest.mark.diffusion,)
    scenarios = [
        {
            "test_name": "hunyuan_image_dit_reliability_default",
            "server_params": _reliability_server_params(),
        }
    ]
    deploy_dir = _DEPLOY_DIR
    worker_pattern = "vLLM-Omni::"
    isolation_pattern = "vLLM-Omni::"
    hard_oom_config = {
        "target_mem_ratio": 0.95,
        "hold_seconds": 0,
        "startup_timeout_sec": 120,
        "strict": False,
        "max_free_bytes": 256 * 1024 * 1024,
    }
    process_kill_error_keywords = (
        "timeout",
        "did not complete within",
        "connection",
        "engine",
        "orchestrator",
        "dead",
        "internal",
        "500",
        "503",
    )

    def warmup(self, client: Any, server: Any) -> None:
        client.send_images_generations_http_request(_image_request(server))

    def submit_concurrent(self, client: Any, server: Any) -> Any:
        request = _image_request(server)
        request["timeout"] = 30
        return client.send_images_generations_http_request(request, err_code=(500, 503))

    def assert_fast_fail(self, client: Any, server: Any) -> None:
        client.send_images_generations_http_request(
            {"json": _fast_fail_body(server.model), "timeout": _IMAGE_FAST_FAIL_HTTP_TIMEOUT_SEC},
            err_code=500,
        )


MODEL = HunyuanImage()

pytestmark = [
    pytest.mark.slow,
    *MODEL.extra_marks,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


def _run_image_oom_at_load(server: Any, client: Any) -> None:
    # A larger spatial workload forces different allocation demand from preflight.
    run_oom_under_load(
        MODEL,
        server,
        client,
        warmup=_image_request(server, size="512x512", steps=2)["json"],
        payload=_image_request(server, size="1024x1024", steps=4)["json"],
        endpoint="/v1/images/generations",
        success_check=_assert_image_success,
    )


@scenario_params(MODEL, GpuOomUnderLoad)
def test_gpu_oom_at_load(omni_server_function, online_client_function) -> None:
    """One injection and large image request with actual OOM evidence.

    Health 200 after removal requires successful replay; 503/unreachable means
    restart is required, covered only by later successful serial cases.
    """
    _run_image_oom_at_load(omni_server_function, online_client_function)


@launch_scenario_params(MODEL, GpuOomAtLaunch)
def test_gpu_oom_at_launch(launch_params) -> None:
    """Exhaust GPU memory before startup. The failure is observable, and a later start serves."""
    GpuOomAtLaunch().run(MODEL, launch_params)


@launch_scenario_params(MODEL, PartialCrashAtLaunch)
def test_stage_crash_at_launch(launch_params) -> None:
    """Kill a stage before readiness; require failure and cleanup, with recovery covered by later serial cases."""
    PartialCrashAtLaunch().run(MODEL, launch_params)


@launch_scenario_params(MODEL, FullCrashAtLaunch)
def test_process_tree_crash_at_launch(launch_params) -> None:
    """Kill the startup process tree; later serial cases cover fresh startup and service.

    This case alone does not verify restart recovery.
    """
    FullCrashAtLaunch().run(MODEL, launch_params)


@launch_scenario_params(
    MODEL,
    ServeRootCrashAtLaunch,
    skip_reason="Worker survives root crash during startup: https://github.com/vllm-project/vllm-omni/issues/10398",
)
def test_serve_root_crash_at_launch(launch_params) -> None:
    """SIGKILL only root before readiness, after a diffusion worker appears.

    Require startup failure and orphan/resource cleanup before harness teardown.
    Later successful serial cases provide fresh-start recovery coverage.
    """
    ServeRootCrashAtLaunch().run(MODEL, launch_params)


@scenario_params(MODEL, PartialCrashAtExit)
def test_stage_crash_at_exit(omni_server_function, online_client_function) -> None:
    """Begin shutdown, kill one stage, and require timely cleanup.

    Later serial cases cover fresh startup and service; this case does not
    independently verify restart recovery.
    """
    PartialCrashAtExit().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, ProcessKillWorkerNoLoad)
def test_stage_crash_at_idle(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    """Kill a worker, check failed requests and health, then require SIGTERM cleanup."""
    ProcessKillWorkerNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)
    sigterm_serve_root_and_assert_cleanup(omni_server_function, scenario="stage_crash_at_idle")


@scenario_params(MODEL, ProcessKillWorkerWithLoad)
def test_stage_crash_at_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillWorkerWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillServeRootNoLoad)
def test_serve_root_crash_at_idle(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillServeRootNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillServeRootWithLoad)
def test_serve_root_crash_at_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillServeRootWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillTreeNoLoad)
def test_process_tree_crash_at_idle(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillTreeNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillTreeWithLoad)
def test_process_tree_crash_at_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillTreeWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, GracefulShutdown)
def test_graceful_shutdown(omni_server_function, online_client_function) -> None:
    """SIGTERM the healthy root; all children and resources must clear within 60s."""
    GracefulShutdown().run(MODEL, omni_server_function, online_client_function)
