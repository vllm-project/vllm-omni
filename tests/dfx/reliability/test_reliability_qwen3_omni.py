"""
Qwen3-Omni reliability integration tests.
"""

from __future__ import annotations

import concurrent.futures
import errno
import http.client
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol

import pytest
import requests
import torch

from tests.dfx.conftest import (
    assert_fault_exception,
    assert_no_extra_worker_processes,
    create_reliability_omni_server_params,
    resolve_oom_device_spec,
    wait_chat_request_ready,
)
from tests.dfx.reliability.helpers import (
    force_remove_container,
    inject_gpu_oom,
    list_remote_process_pids_by_pattern,
    make_process_kill_fault_injector,
    post_chat_completions_raw,
    start_runtime_teardown_container_server,
    stop_gpu_oom_hogs,
)
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import dummy_messages_from_mix_data
from vllm_omni.platforms import current_omni_platform

RELIABILITY_SCENARIOS: list[dict[str, Any]] = [
    {
        "test_name": "qwen3_omni_reliability_async_chunk",
        "server_params": {
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "stage_config_name": "qwen3_omni_moe.yaml",
            "server_args": ["--async-chunk"],
        },
    },
    {
        "test_name": "qwen3_omni_reliability_default",
        "server_params": {
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "stage_config_name": "qwen3_omni_moe.yaml",
            "server_args": ["--no-async-chunk"],
        },
    },
]

DEPLOY_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "vllm_omni" / "deploy"


def _default_oom_device_spec() -> str:
    """Use currently visible CUDA ordinals to avoid invalid device index in sidecar."""
    count = torch.cuda.device_count()
    if count <= 0:
        return "0"
    return ",".join(str(i) for i in range(count))


OOM_INJECTION_CONFIG = {
    "device": _default_oom_device_spec(),
    "target_mem_ratio": 0.95,
    "hold_seconds": 0,
    "startup_timeout_sec": 20,
    "strict": True,
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
RUNTIME_WORKER_PATTERN = "VLLM::Worker"


class _HasServeArgs(Protocol):
    serve_args: list[str]


def _get_system_prompt() -> dict:
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                ),
            }
        ],
    }


def _get_mix_prompt() -> str:
    return "What is recited in the audio? What is in this image? Describe the video briefly."


def _stage_config_path_from_omni_server(omni_server: _HasServeArgs) -> str | None:
    args: list[str] = omni_server.serve_args
    for i, arg in enumerate(args):
        if arg == "--stage-configs-path" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--stage-configs-path="):
            return arg.split("=", 1)[1]
    return None


def _looks_like_server_unreachable(exc: BaseException) -> bool:
    """True when /health cannot be reached because nothing is listening (process exited)."""
    if isinstance(exc, (ConnectionRefusedError, BrokenPipeError, ConnectionResetError)):
        return True
    errno_val = getattr(exc, "errno", None)
    if isinstance(exc, OSError) and errno_val is not None:
        return errno_val in (
            errno.ECONNREFUSED,
            errno.ECONNRESET,
            errno.EPIPE,
        )
    msg = str(exc).lower()
    return "connection refused" in msg or "actively refused" in msg


def _get_health_raw(host: str, port: int, *, timeout_sec: int = 20) -> tuple[int, bytes]:
    """GET /health with stdlib HTTP client; returns (status, body)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout_sec)
    try:
        conn.request("GET", "/health")
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def _post_json_raw(
    host: str,
    port: int,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_sec: int = 30,
) -> tuple[int, bytes]:
    """POST JSON to one endpoint; returns (status, body)."""
    return (
        post_chat_completions_raw(
            host,
            port,
            json.dumps(payload),
            content_type="application/json",
            timeout_sec=timeout_sec,
        )
        if path == "/v1/chat/completions"
        else _post_json_raw_http_client(
            host,
            port,
            path,
            payload,
            timeout_sec=timeout_sec,
        )
    )


def _post_json_raw_http_client(
    host: str,
    port: int,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_sec: int = 30,
) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout_sec)
    try:
        body = json.dumps(payload).encode("utf-8")
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def _extract_error_contract(response_body: bytes) -> dict[str, Any] | None:
    """Best-effort parse OpenAI-style error response."""
    try:
        payload = json.loads(response_body.decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    error_obj = payload.get("error")
    if not isinstance(error_obj, dict):
        return None
    if not isinstance(error_obj.get("message"), str):
        return None
    return error_obj


def _create_video_job_with_invalid_lora(host: str, port: int, *, timeout_sec: int = 30) -> tuple[int, dict[str, Any]]:
    """Create one video job expected to fail due to malformed lora payload."""
    url = f"http://{host}:{port}/v1/videos"
    response = requests.post(
        url,
        data={
            "prompt": "qwen async failure mapping",
            "lora": '{"name": "bad-lora"}',
        },
        headers={"Accept": "application/json"},
        timeout=timeout_sec,
    )
    payload = response.json() if response.content else {}
    return response.status_code, payload


def _poll_video_job_status(
    host: str,
    port: int,
    video_id: str,
    *,
    timeout_sec: int = 180,
    interval_sec: float = 1.0,
) -> tuple[int, dict[str, Any]]:
    """Poll /v1/videos/{id} until completed/failed and return latest status+payload."""
    url = f"http://{host}:{port}/v1/videos/{video_id}"
    deadline = time.monotonic() + timeout_sec
    last_status = 0
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = requests.get(
            url,
            headers={"Accept": "application/json"},
            timeout=20,
        )
        last_status = response.status_code
        try:
            payload = response.json()
        except Exception:  # noqa: BLE001
            payload = {"raw": response.text}
        if isinstance(payload, dict):
            last_payload = payload
            if payload.get("status") in {"completed", "failed"}:
                return last_status, payload
        time.sleep(interval_sec)
    raise TimeoutError(
        f"video job {video_id} did not reach terminal status within {timeout_sec}s; "
        f"last_status={last_status}, last_payload={json.dumps(last_payload)[:500]}"
    )


QWEN_PARAMS = create_reliability_omni_server_params(RELIABILITY_SCENARIOS, DEPLOY_CONFIGS_DIR)


@pytest.mark.slow
@pytest.mark.skipif(
    current_omni_platform.is_rocm() or current_omni_platform.is_xpu(),
    reason="CUDA sidecar OOM injection is CUDA-only for phase-1",
)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_chat_large_payload_failure(omni_server_function, openai_client_function) -> None:
    device_spec = resolve_oom_device_spec(
        OOM_INJECTION_CONFIG,
        _stage_config_path_from_omni_server(omni_server_function),
    )
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
    )
    try:
        video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(1280, 720, 161)['base64']}"
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"
        audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(20, 1)['base64']}"
        messages = dummy_messages_from_mix_data(
            system_prompt=_get_system_prompt(),
            video_data_url=video_data_url,
            image_data_url=image_data_url,
            audio_data_url=audio_data_url,
            content_text=f"{_get_mix_prompt()} " * 200,
        )
        request_config = {
            "model": omni_server_function.model,
            "messages": messages,
            "stream": True,
            "key_words": {"audio": ["test"]},
        }
        try:
            openai_client_function.send_omni_request(request_config, request_num=1)
        except Exception as exc:
            assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
        else:
            pytest.fail("expected large chat payload request failure during GPU OOM injection")
    finally:
        stop_gpu_oom_hogs(handle)


@pytest.mark.slow
@pytest.mark.skipif(
    current_omni_platform.is_rocm() or current_omni_platform.is_xpu(),
    reason="CUDA sidecar OOM injection is CUDA-only for phase-1",
)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_concurrent_pressure_failure(omni_server_function, openai_client_function) -> None:
    device_spec = resolve_oom_device_spec(
        OOM_INJECTION_CONFIG,
        _stage_config_path_from_omni_server(omni_server_function),
    )
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
    )
    try:
        messages = dummy_messages_from_mix_data(
            system_prompt=_get_system_prompt(),
            content_text="What is the capital of China? Answer in 20 words.",
        )
        request_config = {
            "model": omni_server_function.model,
            "messages": messages,
            "stream": False,
            "modalities": ["text"],
            "key_words": {"text": ["beijing"]},
        }
        try:
            openai_client_function.send_omni_request(request_config, request_num=4)
        except Exception as exc:
            assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
        else:
            pytest.fail("expected concurrent request failure under OOM injection")
    finally:
        stop_gpu_oom_hogs(handle)


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    [
        pytest.param(
            make_process_kill_fault_injector(
                grep_patterns="VLLM::Worker",
                signal_name="SIGKILL",
                limit=1,
                post_kill_wait_seconds=2.0,
            ),
            id="runtime_process_chain",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_request_failure(
    omni_server_after_fault_function, openai_client_function
) -> None:
    messages = dummy_messages_from_mix_data(
        system_prompt=_get_system_prompt(),
        content_text="What is the capital of China? Answer in 20 words.",
    )
    request_config = {
        "model": omni_server_after_fault_function.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["beijing"]},
    }
    try:
        openai_client_function.send_omni_request(request_config, request_num=1)
    except Exception as exc:
        assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
    else:
        pytest.fail("expected request failure after process-kill injection")


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    [
        pytest.param(
            make_process_kill_fault_injector(
                grep_patterns="VLLM::Worker",
                signal_name="SIGKILL",
                limit=1,
                post_kill_wait_seconds=2.0,
            ),
            id="runtime_process_chain",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_health_fast_fail_and_concurrent(
    omni_server_after_fault_function,
) -> None:
    """Black-box: after worker SIGKILL, /health→503, chat fails fast, concurrent chat does not hang."""
    host = omni_server_after_fault_function.host
    port = omni_server_after_fault_function.port
    model = omni_server_after_fault_function.model

    deadline = time.monotonic() + 20.0
    last_observation = ""
    saw_503 = False
    health_final_status: int | None = None
    health_final_body = b""
    while time.monotonic() < deadline:
        try:
            status, body = _get_health_raw(host, port, timeout_sec=5)
            last_observation = f"http={status}, body={body[:200]!r}"
            health_final_status, health_final_body = status, body
            if status == 503:
                saw_503 = True
                break
        except Exception as exc:  # noqa: BLE001
            last_observation = f"exception={exc!r}"
        time.sleep(0.5)
    assert saw_503, (
        f"[process_kill health] expected /health to become 503 after fault injection, got {last_observation}"
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "stream": False,
        "modalities": ["text"],
    }
    ff_status: int | None = None
    ff_body = b""
    ff_exc: BaseException | None = None
    start = time.monotonic()
    try:
        ff_status, ff_body = _post_json_raw(host, port, "/v1/chat/completions", payload, timeout_sec=20)
        elapsed = time.monotonic() - start
        assert elapsed < 15, f"[process_kill fast_fail] request did not fail fast after fault: {elapsed:.2f}s"
        assert ff_status >= 500, (
            f"[process_kill fast_fail] expected server-side failure after fault, "
            f"got status={ff_status}, body={ff_body[:200]!r}"
        )
    except Exception as exc:
        ff_exc = exc
        elapsed = time.monotonic() - start
        assert elapsed < 15, f"[process_kill fast_fail] request exception was too slow after fault: {elapsed:.2f}s"

    payload_json = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "What is the capital of China? Answer in one word."}],
            "stream": False,
            "modalities": ["text"],
        }
    )
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                post_chat_completions_raw,
                host,
                port,
                payload_json,
                timeout_sec=20,
            )
            for _ in range(4)
        ]
        done, pending = concurrent.futures.wait(
            futures,
            timeout=30,
            return_when=concurrent.futures.ALL_COMPLETED,
        )

    elapsed = time.monotonic() - start
    assert not pending, f"[process_kill concurrent] some fault-time requests hung: pending={len(pending)}"
    assert elapsed < 30, f"[process_kill concurrent] fault-time request convergence is too slow: {elapsed:.2f}s"

    fault_observed = False
    conc_debug: list[Any] = []
    for future in done:
        try:
            status, body = future.result()
            conc_debug.append((status, body[:200]))
            if status >= 500:
                fault_observed = True
        except Exception as exc:
            conc_debug.append(repr(exc))
            fault_observed = True
    # DEBUG: remove before merge
    print(
        health_final_status,
        health_final_body[:200],
        ff_status,
        ff_body[:200],
        ff_exc,
        conc_debug,
    )
    assert fault_observed, (
        "[process_kill concurrent] expected at least one request to fail after process-kill fault injection"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    current_omni_platform.is_rocm() or current_omni_platform.is_xpu(),
    reason="CUDA sidecar OOM injection is CUDA-only for phase-1",
)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_error_contract_consistent_chat_speech(
    omni_server_function,
) -> None:
    """Black-box: chat/speech should expose a consistent error contract under OOM."""
    device_spec = resolve_oom_device_spec(
        OOM_INJECTION_CONFIG,
        _stage_config_path_from_omni_server(omni_server_function),
    )
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
    )
    host = omni_server_function.host
    port = omni_server_function.port
    try:
        chat_status, chat_body = _post_json_raw(
            host,
            port,
            "/v1/chat/completions",
            {
                "model": omni_server_function.model,
                "messages": [{"role": "user", "content": "Summarize this sentence in one word."}],
                "stream": False,
                "modalities": ["text"],
            },
            timeout_sec=25,
        )
        # Minimal Qwen3-TTS shape (no ref_*): tests/e2e/online_serving/test_qwen3_tts_customvoice.py
        speech_status, speech_body = _post_json_raw(
            host,
            port,
            "/v1/audio/speech",
            {
                "model": omni_server_function.model,
                "input": "hello reliability test",
                "stream": False,
                "response_format": "wav",
                "task_type": "CustomVoice",
                "voice": "vivian",
            },
            timeout_sec=25,
        )
    finally:
        stop_gpu_oom_hogs(handle)

    # Chat is expected to enter runtime-pressure path (5xx). Speech may return
    # request-level validation (4xx) or runtime failure (5xx), both valid as
    # black-box fault outcomes.
    chat_error = _extract_error_contract(chat_body)
    speech_error = _extract_error_contract(speech_body)
    print(chat_status, speech_status, chat_error, speech_error)

    assert chat_status >= 500, f"expected chat error under OOM, got status={chat_status}"
    assert speech_status >= 400, f"expected speech non-2xx error under OOM, got status={speech_status}"

    assert chat_error is not None, f"chat error payload not OpenAI-compatible: {chat_body[:300]!r}"
    assert speech_error is not None, f"speech error payload not OpenAI-compatible: {speech_body[:300]!r}"
    assert "code" in chat_error, f"chat error lacks code field: {chat_error!r}"
    assert "code" in speech_error, f"speech error lacks code field: {speech_error!r}"


@pytest.mark.slow
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_async_failed_job_observable_with_mapped_status(
    omni_server_function,
) -> None:
    """Black-box: failed async job should be observable with mapped status/code."""
    create_status, create_payload = _create_video_job_with_invalid_lora(
        omni_server_function.host,
        omni_server_function.port,
        timeout_sec=30,
    )
    # Some deployments may reject this endpoint synchronously for the current model;
    # in that case still require an explicit error contract.
    if create_status != 200 or "id" not in create_payload:
        assert create_status >= 400, f"unexpected /v1/videos create status={create_status}, payload={create_payload!r}"
        assert isinstance(create_payload.get("error"), dict), (
            f"sync rejection should still return structured error payload, got payload={create_payload!r}"
        )
        return

    video_id = str(create_payload["id"])
    retrieve_status, retrieve_payload = _poll_video_job_status(
        omni_server_function.host,
        omni_server_function.port,
        video_id,
        timeout_sec=180,
    )
    assert retrieve_payload.get("status") == "failed", (
        f"expected async failed terminal status, got retrieve_status={retrieve_status}, payload={retrieve_payload!r}"
    )
    error_obj = retrieve_payload.get("error")
    assert isinstance(error_obj, dict), f"failed payload must include error object: {retrieve_payload!r}"
    assert "message" in error_obj, f"failed payload error missing message: {error_obj!r}"
    assert "code" in error_obj, f"failed payload error missing code: {error_obj!r}"
    # Some runtime modes return HTTP 200 with status=failed payload; others map
    # failed jobs to non-2xx. Accept both as long as failure is explicit.
    assert retrieve_status == 200 or retrieve_status >= 400, (
        "failed retrieve should be explicit either via failed payload or non-2xx status, "
        f"got status={retrieve_status}, payload={retrieve_payload!r}"
    )
    if retrieve_status >= 400 and isinstance(error_obj.get("code"), int):
        assert error_obj["code"] == retrieve_status, (
            f"failed retrieve error.code should match HTTP status, status={retrieve_status}, error={error_obj!r}"
        )


@pytest.mark.slow
@pytest.mark.skipif(
    current_omni_platform.is_rocm() or current_omni_platform.is_xpu(),
    reason="CUDA sidecar OOM injection is CUDA-only for phase-1",
)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_state_converges_after_fault_removed(
    omni_server_function,
) -> None:
    """Black-box: after removing OOM pressure, service reaches a terminal state.

    Terminal state may be recovered (health=200) or explicitly unrecovered
    (health=503), but must not hang indefinitely.
    """
    device_spec = resolve_oom_device_spec(
        OOM_INJECTION_CONFIG,
        _stage_config_path_from_omni_server(omni_server_function),
    )
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
    )
    host = omni_server_function.host
    port = omni_server_function.port
    payload = {
        "model": omni_server_function.model,
        "messages": [{"role": "user", "content": "Tell me one short sentence about Beijing."}],
        "stream": False,
        "modalities": ["text"],
    }

    failure_observed = False
    try:
        for _ in range(3):
            try:
                status, _ = _post_json_raw(host, port, "/v1/chat/completions", payload, timeout_sec=20)
                if status >= 500:
                    failure_observed = True
                    break
            except Exception:
                failure_observed = True
                break
            time.sleep(1.0)
    finally:
        stop_gpu_oom_hogs(handle)

    assert failure_observed, "expected at least one request failure while OOM pressure is active"

    recovery_deadline = time.monotonic() + 90.0
    terminal_health: int | None = None
    unreachable_streak = 0
    last_health_exc: BaseException | None = None
    while time.monotonic() < recovery_deadline:
        try:
            status, _ = _get_health_raw(host, port, timeout_sec=5)
            unreachable_streak = 0
            if status in (200, 503):
                terminal_health = status
                break
        except Exception as exc:
            last_health_exc = exc
            if _looks_like_server_unreachable(exc):
                unreachable_streak += 1
                if unreachable_streak >= 5:
                    pytest.fail(
                        "after OOM sidecar stopped, /health is unreachable (connection refused / reset). "
                        "The APIServer process likely exited (e.g. orchestrator thread crash under OOM); "
                        "this test expects the server to stay up for post-fault health polling. "
                        f"last_exc={last_health_exc!r}"
                    )
            else:
                unreachable_streak = 0
        time.sleep(1.0)
    else:
        pytest.fail(
            "server did not converge to a terminal health state after OOM pressure was removed; "
            f"last_health_exc={last_health_exc!r}"
        )

    request_payload = {
        "model": omni_server_function.model,
        "messages": [{"role": "user", "content": "What is the capital of China? Answer in one word."}],
        "stream": False,
        "modalities": ["text"],
    }
    start = time.monotonic()
    try:
        request_status, _ = _post_json_raw(host, port, "/v1/chat/completions", request_payload, timeout_sec=20)
    except Exception:
        request_status = None
    elapsed = time.monotonic() - start
    assert elapsed < 20, f"post-fault request should not hang after OOM removal: {elapsed:.2f}s"

    assert terminal_health is not None
    if terminal_health == 200:
        assert request_status == 200, f"health recovered but request did not succeed: status={request_status}"
    else:
        assert request_status is None or request_status >= 500, (
            "unhealthy terminal state should fail fast on requests, "
            f"got health={terminal_health}, request_status={request_status}"
        )


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="runtime-teardown helper is POSIX-only")
@pytest.mark.skip(reason="Temporarily disabled runtime teardown scenario.")
@pytest.mark.parametrize("runtime_params", [QWEN_PARAMS[0]], ids=["runtime_teardown_container_kill"])
def test_reliability_fault_runtime_teardown_container_kill_no_orphan_worker(runtime_params, model_prefix) -> None:
    baseline_worker_pids = set(list_remote_process_pids_by_pattern(RUNTIME_WORKER_PATTERN))
    model = model_prefix + runtime_params.model
    serve_args = list(runtime_params.server_args or [])
    if "--stage-init-timeout" not in serve_args:
        serve_args = ["--stage-init-timeout", "120", *serve_args]
    if runtime_params.stage_config_path is not None:
        serve_args += ["--stage-configs-path", runtime_params.stage_config_path]

    handle = None
    try:
        handle = start_runtime_teardown_container_server(
            model=model,
            serve_args=serve_args,
        )
        wait_chat_request_ready(handle.host, handle.port, model=model)

        force_remove_container(handle.container_name)

        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "What is the capital of China? Answer in one word."}],
                "stream": False,
                "modalities": ["text"],
            }
        )
        request_failed = False
        try:
            status, body = post_chat_completions_raw(handle.host, handle.port, payload)
            if status >= 500:
                request_failed = True
            else:
                pytest.fail(f"expected request failure after container teardown, got http={status} body={body[:200]!r}")
        except Exception:
            request_failed = True
        assert request_failed, "expected request failure after container teardown"

        assert_no_extra_worker_processes(baseline_worker_pids, RUNTIME_WORKER_PATTERN)
    finally:
        keep_on_failure = os.getenv("RUNTIME_TEARDOWN_KEEP_CONTAINER_ON_FAILURE", "0").strip() == "1"
        if handle is not None and not keep_on_failure:
            force_remove_container(handle.container_name)
