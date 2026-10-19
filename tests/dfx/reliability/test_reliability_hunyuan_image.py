# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
HunyuanImage-3.0-Instruct reliability integration tests (DiT-only deploy).

Uses ``hunyuan_image3_dit.yaml`` (single diffusion stage, 4-GPU TP=4, no
AR/Mooncake connector) for simpler local bring-up. Scenarios follow
``docs/user_guide/fault_injection_reliability_matrix.md``.
"""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
from typing import Any

import pytest

from tests.dfx.conftest import (
    assert_fault_exception,
    create_reliability_omni_server_params,
    resolve_oom_device_spec,
)
from tests.dfx.reliability.helpers import (
    FaultInjector,
    GpuOomInjectionConfig,
    assert_no_server_tree_process_residual_and_gpu_release,
    assert_post_fault_api_unreachable,
    assert_post_fault_health_503,
    inject_gpu_oom,
    make_process_kill_fault_injector,
    make_server_root_kill_fault_injector,
    make_server_tree_kill_fault_injector,
    run_fault_injection_with_rate_load,
    stop_gpu_oom_hogs,
    worker_residual_timeout_after_kill_signal,
)
from tests.helpers.mark import hardware_test

RELIABILITY_SCENARIOS: list[dict[str, Any]] = [
    {
        "test_name": "hunyuan_image_dit_reliability_default",
        "server_params": {
            "model": "tencent/HunyuanImage-3.0-Instruct",
            "stage_config_name": "hunyuan_image3_dit.yaml",
        },
    },
]

DEPLOY_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "vllm_omni" / "deploy"
OOM_INJECTION_CONFIG: GpuOomInjectionConfig = {
    "target_mem_ratio": 0.95,
    "hold_seconds": 0,
    "startup_timeout_sec": 20,
    "strict": True,
}
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
PROCESS_KILL_ERROR_KEYWORDS = (
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
# Matrix: worker SIGTERM/SIGKILL; DiT-only HunyuanImage workers are vLLM-Omni::.
WORKER_KILL_PATTERNS = ("vLLM-Omni::",)
SERVE_SIGNAL_PARAMS = [
    pytest.param("SIGTERM", id="sigterm"),
    pytest.param("SIGINT", id="sigint"),
    pytest.param("SIGKILL", id="sigkill"),
]
TREE_SIGNAL_PARAMS = [
    pytest.param("SIGTERM", id="sigterm"),
    pytest.param("SIGKILL", id="sigkill"),
]
TREE_WITH_LOAD_SIGNAL_PARAMS = [
    pytest.param("SIGTERM", id="sigterm"),
    pytest.param("SIGKILL", id="sigkill"),
]
WORKER_SIGNAL_FAULT_PARAMS = [
    pytest.param(
        make_process_kill_fault_injector(
            grep_patterns=WORKER_KILL_PATTERNS,
            signal_name="SIGTERM",
            limit=1,
            post_kill_wait_seconds=2.0,
        ),
        id="runtime_process_chain_sigterm",
    ),
    pytest.param(
        make_process_kill_fault_injector(
            grep_patterns=WORKER_KILL_PATTERNS,
            signal_name="SIGKILL",
            limit=1,
            post_kill_wait_seconds=2.0,
        ),
        id="runtime_process_chain_sigkill",
    ),
]

HUNYUAN_PARAMS = create_reliability_omni_server_params(RELIABILITY_SCENARIOS, DEPLOY_CONFIGS_DIR)
INFLIGHT_INJECTION_REQUEST_RATE = 0.3
INFLIGHT_INJECTION_REQUEST_COUNT = 10
# Client read timeout. HTTP 500 must return before this; a timeout raises and fails the test.
_IMAGE_FAST_FAIL_HTTP_TIMEOUT_SEC = 30


def _create_image_request_config(omni_server: Any) -> dict[str, Any]:
    return {
        "json": {
            "model": omni_server.model,
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


def _create_image_fast_fail_request_config(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "size": "512x512",
        "n": 1,
        "response_format": "b64_json",
        "num_inference_steps": 4,
        "bot_task": "none",
        "use_system_prompt": "en_unified",
    }


def _assert_post_fault_image_fast_fail(online_client: Any, *, model: str) -> None:
    """New ``/v1/images/generations`` after a worker kill must return HTTP 500 within 30s.

    The API process is still listening. A timeout or a dropped connection raises and fails the test.
    """
    online_client.send_images_generations_http_request(
        {
            "json": _create_image_fast_fail_request_config(model, "post-fault fast-fail check"),
            "timeout": _IMAGE_FAST_FAIL_HTTP_TIMEOUT_SEC,
        },
        err_code=500,
    )


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=4)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_large_request_failure(
    omni_server_function,
    online_client_function,
) -> None:
    stage_config_path = getattr(omni_server_function, "stage_config_path", None)
    device_spec = resolve_oom_device_spec(OOM_INJECTION_CONFIG, stage_config_path)
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
    )
    try:
        try:
            online_client_function.send_images_generations_http_request(
                _create_image_request_config(omni_server_function)
            )
        except Exception as exc:
            assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
        else:
            pytest.fail("expected large /v1/images/generations request failure during GPU OOM injection")
    finally:
        stop_gpu_oom_hogs(handle)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=4)
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_request_failure(
    omni_server_function,
    online_client_function,
    fault_injector: FaultInjector,
) -> None:
    online_client_function.send_images_generations_http_request(_create_image_request_config(omni_server_function))
    fault_injector(omni_server_function)
    try:
        online_client_function.send_images_generations_http_request(_create_image_request_config(omni_server_function))
    except Exception as exc:
        assert_fault_exception(exc, PROCESS_KILL_ERROR_KEYWORDS)
    else:
        pytest.fail("expected /v1/images/generations request failure after process-kill injection")


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=4)
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_health_fast_fail_and_concurrent(
    omni_server_function,
    online_client_function,
    fault_injector: FaultInjector,
) -> None:
    online_client_function.send_images_generations_http_request(_create_image_request_config(omni_server_function))
    fault_injector(omni_server_function)
    _assert_post_fault_image_fast_fail(online_client_function, model=omni_server_function.model)

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                online_client_function.send_images_generations_http_request,
                _create_image_request_config(omni_server_function),
            )
            for _ in range(3)
        ]
        done, pending = concurrent.futures.wait(
            futures,
            timeout=40,
            return_when=concurrent.futures.ALL_COMPLETED,
        )

    elapsed = time.monotonic() - start
    assert not pending, f"[process_kill concurrent] some fault-time image requests hung: pending={len(pending)}"
    assert elapsed < 40, f"[process_kill concurrent] fault-time image request convergence is too slow: {elapsed:.2f}s"

    fault_observed = False
    for future in done:
        try:
            future.result()
        except Exception as exc:
            fault_observed = True
            assert_fault_exception(exc, PROCESS_KILL_ERROR_KEYWORDS)
    assert fault_observed, (
        "[process_kill concurrent] expected at least one /v1/images/generations request to fail after fault"
    )

    deadline = time.monotonic() + 20.0
    last_observation = ""
    final_health_status: int | None = None
    while time.monotonic() < deadline:
        try:
            resp = online_client_function.send_health_http_request({"timeout": 5}, err_code=(200, 503))[0]
            status = resp.status_code
            body = resp.error_message if resp.error_message is not None else resp.json_body
            last_observation = f"http={status}, body={str(body)[:200]!r}"
            final_health_status = status
            if status == 503:
                break
        except Exception as exc:  # noqa: BLE001
            last_observation = f"exception={exc!r}"
        time.sleep(0.5)
    assert final_health_status == 503, (
        "[process_kill health] expected /health 503 after fatal fault, "
        f"got status={final_health_status}, last_observation={last_observation}"
    )


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=4)
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_worker_with_load_request_failure(
    omni_server_function,
    online_client_function,
    fault_injector: FaultInjector,
) -> None:
    scenario = "kill_worker_with_load"
    online_client_function.send_images_generations_http_request(_create_image_request_config(omni_server_function))
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: online_client_function.send_images_generations_http_request(
            _create_image_request_config(omni_server_function)
        ),
        inject_fault=lambda: fault_injector(omni_server_function),
        num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
        request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
        completion_timeout_sec=120.0,
    )
    assert load_result["failure_observed"], (
        f"[{scenario}] expected at least one load request failure after fault; load_result={load_result}"
    )
    _assert_post_fault_image_fast_fail(online_client_function, model=omni_server_function.model)
    assert_post_fault_health_503(online_client_function, scenario=scenario)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=4)
@pytest.mark.parametrize("signal_name", SERVE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_serve_root_with_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    scenario = f"kill_serve_root_with_load_{signal_name.lower()}"
    injector = make_server_root_kill_fault_injector(signal_name=signal_name, post_kill_wait_seconds=2.0)
    online_client_function.send_images_generations_http_request(_create_image_request_config(omni_server_function))
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: online_client_function.send_images_generations_http_request(
            _create_image_request_config(omni_server_function)
        ),
        inject_fault=lambda: injector(omni_server_function),
        num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
        request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
        completion_timeout_sec=120.0,
    )
    assert load_result["failure_observed"], (
        f"[{scenario}] expected at least one load request failure after fault; load_result={load_result}"
    )
    assert_post_fault_api_unreachable(online_client_function, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=4)
@pytest.mark.parametrize("signal_name", SERVE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_serve_root_no_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    scenario = f"kill_serve_root_no_load_{signal_name.lower()}"
    injector = make_server_root_kill_fault_injector(signal_name=signal_name, post_kill_wait_seconds=2.0)
    online_client_function.send_images_generations_http_request(_create_image_request_config(omni_server_function))
    injector(omni_server_function)
    assert_post_fault_api_unreachable(online_client_function, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=4)
@pytest.mark.parametrize("signal_name", TREE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_tree_no_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    scenario = f"kill_serve_tree_no_load_{signal_name.lower()}"
    injector = make_server_tree_kill_fault_injector(
        signal_name=signal_name,
        post_kill_wait_seconds=2.0,
        inter_kill_wait_seconds=0.1,
    )
    online_client_function.send_images_generations_http_request(_create_image_request_config(omni_server_function))
    injector(omni_server_function)
    assert_post_fault_api_unreachable(online_client_function, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=4)
@pytest.mark.parametrize("signal_name", TREE_WITH_LOAD_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_tree_with_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    scenario = f"kill_serve_tree_with_load_{signal_name.lower()}"
    injector = make_server_tree_kill_fault_injector(
        signal_name=signal_name,
        post_kill_wait_seconds=2.0,
        inter_kill_wait_seconds=0.1,
    )
    online_client_function.send_images_generations_http_request(_create_image_request_config(omni_server_function))
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: online_client_function.send_images_generations_http_request(
            _create_image_request_config(omni_server_function)
        ),
        inject_fault=lambda: injector(omni_server_function),
        num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
        request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
        completion_timeout_sec=120.0,
    )
    assert load_result["failure_observed"], (
        f"[{scenario}] expected at least one load request failure after fault; load_result={load_result}"
    )
    assert_post_fault_api_unreachable(online_client_function, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )
