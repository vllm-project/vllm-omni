# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""HunyuanImage reliability scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.dfx.reliability.helpers import FaultModel, scenario_params
from tests.dfx.reliability.scenarios import (
    GpuOomLarge,
    PartialStageIsolation,
    ProcessKillServeRootNoLoad,
    ProcessKillServeRootWithLoad,
    ProcessKillTreeNoLoad,
    ProcessKillTreeWithLoad,
    ProcessKillWorkerNoLoad,
    ProcessKillWorkerWithLoad,
)
from tests.helpers.mark import hardware_marks
from vllm_omni.platforms import current_omni_platform

_IMAGE_FAST_FAIL_HTTP_TIMEOUT_SEC = 30
_DEPLOY_DIR = Path(__file__).resolve().parents[3] / "vllm_omni" / "deploy"


def _image_request(server: Any) -> dict[str, Any]:
    return {
        "json": {
            "model": server.model,
            "prompt": "A simple red apple on a white background.",
            "size": "512x512",
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": 4,
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
    return params


FAULT_ERROR_KEYWORDS = (
    "oom",
    "out of memory",
    "cuda",
    "connection",
    "job failed",
    "unknown error",
    "internal server error",
    "500 server error",
)


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
        "startup_timeout_sec": 20,
        "strict": True,
    }
    fault_error_keywords = FAULT_ERROR_KEYWORDS
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
    oom_large_fail_message = "expected large /v1/images/generations request failure during GPU OOM injection"

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

    def oom_large_probe(self, client: Any, server: Any) -> None:
        client.send_images_generations_http_request(_image_request(server))


MODEL = HunyuanImage()

pytestmark = [
    pytest.mark.slow,
    *MODEL.extra_marks,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


@scenario_params(MODEL, GpuOomLarge)
def test_reliability_fault_gpu_oom_large_request_failure(omni_server_function, online_client_function) -> None:
    GpuOomLarge().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, ProcessKillWorkerNoLoad)
def test_reliability_fault_process_kill_worker_no_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    """After a quiet worker kill, the next request returns HTTP 500, then concurrent requests fail."""
    ProcessKillWorkerNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillWorkerWithLoad)
def test_reliability_fault_process_kill_worker_with_load_request_failure(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillWorkerWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillServeRootNoLoad)
def test_reliability_fault_process_kill_serve_root_no_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillServeRootNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillServeRootWithLoad)
def test_reliability_fault_process_kill_serve_root_with_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillServeRootWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillTreeNoLoad)
def test_reliability_fault_process_kill_tree_no_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillTreeNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillTreeWithLoad)
def test_reliability_fault_process_kill_tree_with_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillTreeWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, PartialStageIsolation)
def test_reliability_fault_partial_stage_isolation(omni_server_function, online_client_function) -> None:
    """Kill one stage or one worker. Requests bound to it fail, and one SIGTERM still exits the API."""
    PartialStageIsolation().run(MODEL, omni_server_function, online_client_function)
