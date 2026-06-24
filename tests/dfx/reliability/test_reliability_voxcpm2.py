"""
VoxCPM2 reliability integration tests.

Scenarios follow ``docs/user_guide/fault_injection_reliability_matrix.md``.
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
    extract_openai_error_contract_from_bytes,
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
from tests.helpers.media import load_test_audio_data_url
from tests.helpers.runtime import OpenAIClientHandler

RELIABILITY_SCENARIOS: list[dict[str, Any]] = [
    {
        "test_name": "voxcpm2_reliability_default",
        "server_params": {
            "model": "openbmb/VoxCPM2",
            "stage_config_name": "voxcpm2.yaml",
        },
    },
]

DEPLOY_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "vllm_omni" / "deploy"
OOM_INJECTION_CONFIG = {
    "target_mem_ratio": 0.96,
    "hold_seconds": 0,
    "startup_timeout_sec": 20,
    "strict": False,
}
FAULT_ERROR_KEYWORDS = (
    "the request failed",
    "oom",
    "out of memory",
    "cuda",
    "orchestrator",
    "timeout",
    "connection",
    "500",
    "503",
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
RUNTIME_WORKER_PATTERN = "VLLM::"
SERVE_SIGNAL_PARAMS = [
    pytest.param("SIGTERM", id="sigterm"),
    pytest.param("SIGINT", id="sigint"),
    pytest.param("SIGKILL", id="sigkill"),
]
SERVE_WITH_LOAD_SIGNAL_PARAMS = [
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
            grep_patterns=RUNTIME_WORKER_PATTERN,
            signal_name="SIGTERM",
            limit=1,
            post_kill_wait_seconds=2.0,
        ),
        id="runtime_process_chain_sigterm",
    ),
    pytest.param(
        make_process_kill_fault_injector(
            grep_patterns=RUNTIME_WORKER_PATTERN,
            signal_name="SIGKILL",
            limit=1,
            post_kill_wait_seconds=2.0,
        ),
        id="runtime_process_chain_sigkill",
    ),
]

VOXCPM2_PARAMS = create_reliability_omni_server_params(RELIABILITY_SCENARIOS, DEPLOY_CONFIGS_DIR)
INFLIGHT_INJECTION_REQUEST_RATE = 0.3
INFLIGHT_INJECTION_REQUEST_COUNT = 10
_SPEECH_FF_HTTP_TIMEOUT_SEC = 30
_SPEECH_FF_MAX_ELAPSED_SEC = 30


def _speech_request_config(omni_server: Any) -> dict[str, Any]:
    return {
        "model": omni_server.model,
        "input": "The weather is nice today, perfect for a walk in the park.",
        "stream": False,
        "timeout": 120.0,
        "response_format": "wav",
        "voice": "default",
    }


def _submit_speech_request(openai_client: OpenAIClientHandler, omni_server: Any) -> None:
    openai_client.send_audio_speech_request(_speech_request_config(omni_server), request_num=1)


def _oom_speech_request_config(omni_server: Any) -> dict[str, Any]:
    base = "The weather is nice today, perfect for a walk in the park. "
    return {
        "model": omni_server.model,
        "input": base * 80,
        "max_new_tokens": 4096,
        "ref_audio": load_test_audio_data_url("qwen3_tts/clone_2.wav"),
        "stream": False,
        "timeout": 300.0,
        "response_format": "wav",
        "voice": "default",
    }


def _submit_oom_speech_request(openai_client: OpenAIClientHandler, omni_server: Any) -> None:
    openai_client.send_audio_speech_request(_oom_speech_request_config(omni_server), request_num=1)


def _assert_post_fault_speech_fast_fail(host: str, port: int, *, model: str, scenario: str) -> None:
    payload = {
        "model": model,
        "input": "post-fault fast-fail check",
        "voice": "default",
    }
    start = time.monotonic()
    try:
        status, body = post_json_raw(host, port, "/v1/audio/speech", payload, timeout_sec=20)
        elapsed = time.monotonic() - start
        assert elapsed < 15, f"[{scenario} fast_fail] /v1/audio/speech did not fail fast: {elapsed:.2f}s"
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
@pytest.mark.tts
@pytest.mark.skip(reason="issue#4285")
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server_function", VOXCPM2_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_speech_request_failure(
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
            _submit_oom_speech_request(openai_client_function, omni_server_function)
        except Exception as exc:
            assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
        else:
            pytest.fail("expected /v1/audio/speech request failure during GPU OOM injection")
    finally:
        stop_gpu_oom_hogs(handle)


@pytest.mark.slow
@pytest.mark.tts
@pytest.mark.skip(reason="issue#4285")
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server_function", VOXCPM2_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_speech_error_contract(
    omni_server_function,
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
    host = omni_server_function.host
    port = omni_server_function.port
    oom_config = _oom_speech_request_config(omni_server_function)
    payload = {
        "model": oom_config["model"],
        "input": oom_config["input"],
        "voice": oom_config["voice"],
        "max_new_tokens": oom_config["max_new_tokens"],
        "ref_audio": oom_config["ref_audio"],
    }
    try:
        status, body = post_json_raw(host, port, "/v1/audio/speech", payload, timeout_sec=300)
    finally:
        stop_gpu_oom_hogs(handle)

    assert status >= 500, f"expected speech runtime error under OOM, got status={status}"
    speech_error = extract_openai_error_contract_from_bytes(body)
    assert speech_error is not None, f"speech error payload not OpenAI-compatible: {body[:300]!r}"
    assert "code" in speech_error, f"speech error lacks code field: {speech_error!r}"


@pytest.mark.slow
@pytest.mark.tts
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", VOXCPM2_PARAMS, indirect=True)
def test_reliability_fault_process_kill_request_failure(
    omni_server_after_fault_function,
    openai_client_function,
) -> None:
    try:
        _submit_speech_request(openai_client_function, omni_server_after_fault_function)
    except Exception as exc:
        assert_fault_exception(exc, PROCESS_KILL_ERROR_KEYWORDS)
    else:
        pytest.fail("expected /v1/audio/speech request failure after process-kill injection")


@pytest.mark.slow
@pytest.mark.tts
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", VOXCPM2_PARAMS, indirect=True)
def test_reliability_fault_process_kill_fast_fail_and_concurrent(
    omni_server_after_fault_function,
    openai_client_function,
) -> None:
    host = omni_server_after_fault_function.host
    port = omni_server_after_fault_function.port
    model = omni_server_after_fault_function.model
    payload = {
        "model": model,
        "input": "Say hello in one short sentence.",
        "voice": "default",
    }

    start = time.monotonic()
    try:
        status, body = post_json_raw(
            host,
            port,
            "/v1/audio/speech",
            payload,
            timeout_sec=_SPEECH_FF_HTTP_TIMEOUT_SEC,
        )
        elapsed = time.monotonic() - start
        assert elapsed <= _SPEECH_FF_MAX_ELAPSED_SEC + 1.0, (
            f"[process_kill fast_fail] /v1/audio/speech did not fail fast after fault: {elapsed:.2f}s"
        )
        assert status >= 500, (
            "[process_kill fast_fail] expected server-side failure after fault, "
            f"got status={status}, body={body[:200]!r}"
        )
    except Exception:
        elapsed = time.monotonic() - start
        assert elapsed <= _SPEECH_FF_MAX_ELAPSED_SEC + 1.0, (
            f"[process_kill fast_fail] /v1/audio/speech exception was too slow after fault: {elapsed:.2f}s"
        )

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(_submit_speech_request, openai_client_function, omni_server_after_fault_function)
            for _ in range(3)
        ]
        done, pending = concurrent.futures.wait(
            futures,
            timeout=30,
            return_when=concurrent.futures.ALL_COMPLETED,
        )

    elapsed = time.monotonic() - start
    assert not pending, f"[process_kill concurrent] some fault-time speech requests hung: pending={len(pending)}"
    assert elapsed < 30, f"[process_kill concurrent] fault-time speech request convergence is too slow: {elapsed:.2f}s"

    fault_observed = False
    for future in done:
        try:
            future.result()
        except Exception as exc:
            fault_observed = True
            assert_fault_exception(exc, PROCESS_KILL_ERROR_KEYWORDS)
    assert fault_observed, (
        "[process_kill concurrent] expected at least one /v1/audio/speech request to fail after fault"
    )


@pytest.mark.slow
@pytest.mark.tts
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", VOXCPM2_PARAMS, indirect=True)
def test_reliability_fault_process_kill_worker_with_load_request_failure(
    omni_server_function,
    openai_client_function,
    fault_injector: FaultInjector,
) -> None:
    scenario = "kill_worker_with_load"
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: _submit_speech_request(openai_client_function, omni_server_function),
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
    _assert_post_fault_speech_fast_fail(host, port, model=omni_server_function.model, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize("signal_name", SERVE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", VOXCPM2_PARAMS, indirect=True)
def test_reliability_fault_process_kill_serve_root_no_load_fast_fail_and_cleanup(
    omni_server_function,
    signal_name: str,
) -> None:
    scenario = f"kill_serve_root_no_load_{signal_name.lower()}"
    injector = make_server_root_kill_fault_injector(signal_name=signal_name, post_kill_wait_seconds=2.0)
    injector(omni_server_function)
    host = omni_server_function.host
    port = omni_server_function.port
    _assert_post_fault_speech_fast_fail(host, port, model=omni_server_function.model, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize("signal_name", SERVE_WITH_LOAD_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", VOXCPM2_PARAMS, indirect=True)
def test_reliability_fault_process_kill_serve_root_with_load_fast_fail_and_cleanup(
    omni_server_function,
    openai_client_function,
    signal_name: str,
) -> None:
    scenario = f"kill_serve_root_with_load_{signal_name.lower()}"
    injector = make_server_root_kill_fault_injector(signal_name=signal_name, post_kill_wait_seconds=2.0)
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: _submit_speech_request(openai_client_function, omni_server_function),
        inject_fault=lambda: injector(omni_server_function),
        num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
        request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
        completion_timeout_sec=300.0,
    )
    assert load_result["failure_observed"], (
        f"[{scenario}] expected at least one load request failure after fault; load_result={load_result}"
    )
    host = omni_server_function.host
    port = omni_server_function.port
    _assert_post_fault_speech_fast_fail(host, port, model=omni_server_function.model, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@pytest.mark.tts
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize("signal_name", TREE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", VOXCPM2_PARAMS, indirect=True)
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
    _assert_post_fault_speech_fast_fail(host, port, model=omni_server_function.model, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@pytest.mark.tts
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize("signal_name", TREE_WITH_LOAD_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", VOXCPM2_PARAMS, indirect=True)
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
        submit_request=lambda: _submit_speech_request(openai_client_function, omni_server_function),
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
    _assert_post_fault_speech_fast_fail(host, port, model=omni_server_function.model, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )
