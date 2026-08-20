"""
HunyuanImage-3.0-Instruct reliability integration tests (DiT-only deploy).

Uses ``hunyuan_image3_dit.yaml`` (single diffusion stage, 4-GPU TP=4, no
AR/Mooncake connector) for simpler local bring-up. Scenarios follow
``docs/user_guide/fault_injection_reliability_matrix.md``.
"""

from __future__ import annotations

import concurrent.futures
import os
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
    assert_no_server_tree_process_residual_and_gpu_release,
    get_health_raw,
    inject_gpu_oom,
    make_process_kill_fault_injector,
    make_server_root_kill_fault_injector,
    make_server_tree_kill_fault_injector,
    post_json_raw,
    run_fault_injection_with_rate_load,
    stop_gpu_oom_hogs,
    worker_residual_timeout_after_kill_signal,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OpenAIClientHandler

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
OOM_INJECTION_CONFIG = {
    "target_mem_ratio": 0.95,
    "hold_seconds": 0,
    "startup_timeout_sec": 20,
    "strict": True,
}
FAULT_ERROR_KEYWORDS = (
    "oom",
    "out of memory",
    "cuda",
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


def _image_request_config(omni_server: Any) -> dict[str, Any]:
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


def _submit_image_generation(openai_client: OpenAIClientHandler, omni_server: Any) -> None:
    responses = openai_client.send_images_generations_http_request(_image_request_config(omni_server))
    resp = responses[0]
    if resp.status_code >= 400:
        raise RuntimeError(
            f"/v1/images/generations failed: status={resp.status_code}, body={getattr(resp, 'error_message', '')!r}"
        )


def _assert_post_fault_image_fast_fail(host: str, port: int, *, model: str, scenario: str) -> None:
    payload = {
        "model": model,
        "prompt": "post-fault fast-fail check",
        "size": "512x512",
        "n": 1,
        "response_format": "b64_json",
        "num_inference_steps": 4,
        "bot_task": "none",
        "use_system_prompt": "en_unified",
    }
    start = time.monotonic()
    try:
        status, body = post_json_raw(host, port, "/v1/images/generations", payload, timeout_sec=20)
        elapsed = time.monotonic() - start
        assert elapsed < 15, f"[{scenario} fast_fail] /v1/images/generations did not fail fast: {elapsed:.2f}s"
        assert status >= 500, (
            f"[{scenario} fast_fail] expected server-side error after fault, got status={status}, body={body[:200]!r}"
        )
    except Exception:  # noqa: BLE001
        elapsed = time.monotonic() - start
        assert elapsed < 15, f"[{scenario} fast_fail] exception was too slow after fault: {elapsed:.2f}s"


def _assert_post_fault_health_terminal(host: str, port: int, *, scenario: str) -> None:
    deadline = time.monotonic() + 20.0
    last_observation = ""
    while time.monotonic() < deadline:
        try:
            status, body = get_health_raw(host, port, timeout_sec=5)
            last_observation = f"http={status}, body={body[:200]!r}"
            if status == 503:
                return
        except Exception as exc:  # noqa: BLE001
            last_observation = f"exception={exc!r}"
            return
        time.sleep(0.5)
    pytest.fail(f"[{scenario} health] no terminal post-fault health observed: {last_observation}")


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.skip(reason="issue#4285")
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_image_large_request_failure(
    omni_server_function,
    openai_client_function,
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
            _submit_image_generation(openai_client_function, omni_server_function)
        except Exception as exc:
            assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
        else:
            pytest.fail("expected large /v1/images/generations request failure during GPU OOM injection")
    finally:
        stop_gpu_oom_hogs(handle)


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_request_failure(
    omni_server_after_fault_function,
    openai_client_function,
) -> None:
    try:
        _submit_image_generation(openai_client_function, omni_server_after_fault_function)
    except Exception as exc:
        assert_fault_exception(exc, PROCESS_KILL_ERROR_KEYWORDS)
    else:
        pytest.fail("expected /v1/images/generations request failure after process-kill injection")


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.skip(reason="issue#4522")
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_health_fast_fail_and_concurrent(
    omni_server_after_fault_function,
    openai_client_function,
) -> None:
    host = omni_server_after_fault_function.host
    port = omni_server_after_fault_function.port
    model = omni_server_after_fault_function.model

    payload = {
        "model": model,
        "prompt": "fast-fail check",
        "size": "512x512",
        "n": 1,
        "response_format": "b64_json",
        "num_inference_steps": 4,
        "bot_task": "none",
        "use_system_prompt": "en_unified",
    }
    start = time.monotonic()
    try:
        status, body = post_json_raw(host, port, "/v1/images/generations", payload, timeout_sec=20)
        elapsed = time.monotonic() - start
        assert elapsed < 15, (
            f"[process_kill fast_fail] /v1/images/generations did not fail fast after fault: {elapsed:.2f}s"
        )
        assert status >= 500, (
            "[process_kill fast_fail] expected server-side error after fatal fault, "
            f"got status={status}, body={body[:200]!r}"
        )
    except Exception:
        elapsed = time.monotonic() - start
        assert elapsed < 15, (
            f"[process_kill fast_fail] /v1/images/generations exception was too slow after fault: {elapsed:.2f}s"
        )

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(_submit_image_generation, openai_client_function, omni_server_after_fault_function)
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
            status, body = get_health_raw(host, port, timeout_sec=5)
            last_observation = f"http={status}, body={body[:200]!r}"
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
@pytest.mark.skip(reason="issue#4526")
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_worker_with_load_request_failure(
    omni_server_function,
    openai_client_function,
    fault_injector: FaultInjector,
) -> None:
    scenario = "kill_worker_with_load"
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: _submit_image_generation(openai_client_function, omni_server_function),
        inject_fault=lambda: fault_injector(omni_server_function),
        num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
        request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
        completion_timeout_sec=120.0,
    )
    assert load_result["failure_observed"], (
        f"[{scenario}] expected at least one load request failure after fault; load_result={load_result}"
    )
    host = omni_server_function.host
    port = omni_server_function.port
    _assert_post_fault_image_fast_fail(host, port, model=omni_server_function.model, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize("signal_name", SERVE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_serve_root_with_load_fast_fail_and_cleanup(
    omni_server_function,
    openai_client_function,
    signal_name: str,
) -> None:
    scenario = f"kill_serve_root_with_load_{signal_name.lower()}"
    injector = make_server_root_kill_fault_injector(signal_name=signal_name, post_kill_wait_seconds=2.0)
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: _submit_image_generation(openai_client_function, omni_server_function),
        inject_fault=lambda: injector(omni_server_function),
        num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
        request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
        completion_timeout_sec=120.0,
    )
    assert load_result["failure_observed"], (
        f"[{scenario}] expected at least one load request failure after fault; load_result={load_result}"
    )
    host = omni_server_function.host
    port = omni_server_function.port
    _assert_post_fault_image_fast_fail(host, port, model=omni_server_function.model, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize("signal_name", SERVE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_serve_root_no_load_fast_fail_and_cleanup(
    omni_server_function,
    signal_name: str,
) -> None:
    scenario = f"kill_serve_root_no_load_{signal_name.lower()}"
    injector = make_server_root_kill_fault_injector(signal_name=signal_name, post_kill_wait_seconds=2.0)
    injector(omni_server_function)
    host = omni_server_function.host
    port = omni_server_function.port
    _assert_post_fault_image_fast_fail(host, port, model=omni_server_function.model, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize("signal_name", TREE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_tree_no_load_fast_fail_and_cleanup(
    omni_server_function,
    signal_name: str,
) -> None:
    scenario = f"kill_serve_tree_no_load_{signal_name.lower()}"
    injector = make_server_tree_kill_fault_injector(
        signal_name=signal_name,
        post_kill_wait_seconds=2.0,
        inter_kill_wait_seconds=0.1,
    )
    injector(omni_server_function)
    host = omni_server_function.host
    port = omni_server_function.port
    _assert_post_fault_image_fast_fail(host, port, model=omni_server_function.model, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.skip(reason="issue#4526")
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize("signal_name", TREE_WITH_LOAD_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", HUNYUAN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_tree_with_load_fast_fail_and_cleanup(
    omni_server_function,
    openai_client_function,
    signal_name: str,
) -> None:
    scenario = f"kill_serve_tree_with_load_{signal_name.lower()}"
    injector = make_server_tree_kill_fault_injector(
        signal_name=signal_name,
        post_kill_wait_seconds=2.0,
        inter_kill_wait_seconds=0.1,
    )
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: _submit_image_generation(openai_client_function, omni_server_function),
        inject_fault=lambda: injector(omni_server_function),
        num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
        request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
        completion_timeout_sec=120.0,
    )
    assert load_result["failure_observed"], (
        f"[{scenario}] expected at least one load request failure after fault; load_result={load_result}"
    )
    host = omni_server_function.host
    port = omni_server_function.port
    _assert_post_fault_image_fast_fail(host, port, model=omni_server_function.model, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )
