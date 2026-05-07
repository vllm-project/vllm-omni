"""
Qwen3-Omni reliability integration tests.
"""

from __future__ import annotations

import concurrent.futures
import errno
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol

import pytest
import torch

from tests.dfx.conftest import (
    assert_fault_exception,
    create_reliability_omni_server_params,
    resolve_oom_device_spec,
)
from tests.dfx.reliability.helpers import (
    extract_openai_error_contract_from_bytes,
    get_health_raw,
    inject_gpu_oom,
    make_process_kill_fault_injector,
    post_chat_completions_raw,
    post_json_raw,
    stop_gpu_oom_hogs,
)
from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import dummy_messages_from_mix_data

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
    "target_mem_ratio": 0.92,
    "hold_seconds": 0,
    "startup_timeout_sec": 20,
    "strict": False,
}
# Post-fault recovery probe: keep the API process alive on the same GPU.
# ``strict=True`` sidecars keep allocating until CUDA refuses, which often
# evicts/kills the server; here we only take a fraction of *initial* free
# memory and stop, then rely on a heavy multimodal forward to hit OOM.
OOM_RECOVER_INJECTION_CONFIG = {
    "device": _default_oom_device_spec(),
    # ``strict=False`` stops at this fraction of *initial* free memory per GPU (see helpers sidecar).
    # On multi-GPU hosts a mild ratio plus a small e2e-style mix can still leave several GB free
    # on a data-parallel device and the request succeeds; 0.92+ pairs better with the heavy
    # probe payload below. Raise toward 0.94 only if fault phase still never fails; avoid strict=True
    # here so the API process is not squeezed to zero slack.
    "target_mem_ratio": 0.8,
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
RUNTIME_WORKER_PATTERN = "VLLM::Worker"


class _HasServeArgs(Protocol):
    model: str
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


def _mix_chat_completions_probe_payload(omni_server: _HasServeArgs) -> dict[str, Any]:
    """Heavy multimodal ``/v1/chat/completions`` body for OOM recover tests.

    Uses the same synthetic sizes as ``test_reliability_fault_gpu_oom_chat_large_payload_failure``
    (large vision tensors + long text), not the light expansion mix (224² / 5s audio): under
    ``strict=False`` sidecars, a small mix often still completes on a GPU that retains multi-GB
    free after the hog stops at ``target_mem_ratio``.
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(1280, 720, 161)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(20, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=_get_system_prompt(),
        video_data_url=video_data_url,
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text="What is recited in the audio? What is in this image? What is in this video? " * 200,
    )
    return {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "modalities": ["text", "audio"],
    }


def _fault_keywords_match_response(*, status: int, body: bytes) -> bool:
    """True when HTTP status or body text looks like an expected fault / OOM class outcome."""
    if status >= 400:
        return True
    text = body.decode("utf-8", errors="replace").lower()
    # Avoid matching bare ``"500"`` / ``"503"`` in successful streamed bodies (base64/SSE noise).
    body_fault_hints = (
        "the request failed",
        "oom",
        "out of memory",
        "cuda",
        "orchestrator",
        "timeout",
        "connection refused",
        "connection reset",
    )
    return any(key in text for key in body_fault_hints)


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


QWEN_PARAMS = create_reliability_omni_server_params(RELIABILITY_SCENARIOS, DEPLOY_CONFIGS_DIR)


@pytest.mark.slow
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_error_contract_consistent_chat_speech(
    omni_server_function,
) -> None:
    """Black-box: text chat vs omni audio output should expose a consistent error contract under OOM.

    Speech-style output is requested via ``/v1/chat/completions`` with ``modalities`` that include ``audio``.
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
    mix_base = _mix_chat_completions_probe_payload(omni_server_function)
    oom_contract_timeout_sec = 120
    try:
        chat_status, chat_body = post_json_raw(
            host,
            port,
            "/v1/chat/completions",
            {
                "model": mix_base["model"],
                "messages": mix_base["messages"],
                "stream": False,
                "modalities": ["text"],
            },
            timeout_sec=oom_contract_timeout_sec,
        )
        speech_status, speech_body = post_json_raw(
            host,
            port,
            "/v1/chat/completions",
            {
                "model": mix_base["model"],
                "messages": mix_base["messages"],
                "stream": False,
                "modalities": ["text", "audio"],
            },
            timeout_sec=oom_contract_timeout_sec,
        )
    finally:
        stop_gpu_oom_hogs(handle)

    # Under OOM pressure both chat and speech should surface runtime-class (5xx)
    # failures instead of request-validation-class (4xx) errors.
    chat_error = extract_openai_error_contract_from_bytes(chat_body)
    speech_error = extract_openai_error_contract_from_bytes(speech_body)
    print(chat_status, chat_error, speech_status, speech_error)

    assert chat_status >= 500, f"expected chat error under OOM, got status={chat_status}"
    assert speech_status >= 500, f"expected speech runtime error under OOM, got status={speech_status}"

    assert chat_error is not None, f"chat error payload not OpenAI-compatible: {chat_body[:300]!r}"
    assert speech_error is not None, f"speech error payload not OpenAI-compatible: {speech_body[:300]!r}"
    assert "code" in chat_error, f"chat error lacks code field: {chat_error!r}"
    assert "code" in speech_error, f"speech error lacks code field: {speech_error!r}"


@pytest.mark.slow
@hardware_test(res={"cuda": "H100"}, num_cards=2)
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
@hardware_test(res={"cuda": "H100"}, num_cards=2)
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
            "modalities": ["text", "audio"],
            "key_words": {"audio": ["test"]},
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
            status, body = get_health_raw(host, port, timeout_sec=5)
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
        ff_status, ff_body = post_json_raw(host, port, "/v1/chat/completions", payload, timeout_sec=20)
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
@pytest.mark.skip(reason="issue#2327")
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_state_converges_after_fault_removed(
    omni_server_function,
) -> None:
    """Black-box: under bounded GPU pressure, a heavy mix chat fails; after hog stops, the same request succeeds.

    Uses a milder sidecar than ``OOM_INJECTION_CONFIG`` (``strict=False`` and a capped
    ``target_mem_ratio``) so the serving process stays alive: pressure should starve
    the multimodal forward pass, not exhaust the whole device allocator.

    Fault and recovery phases use raw ``/v1/chat/completions`` HTTP so tests can assert
    on status codes and JSON error payloads (or fault keywords in streamed bodies).
    """
    device_spec = resolve_oom_device_spec(
        OOM_RECOVER_INJECTION_CONFIG,
        _stage_config_path_from_omni_server(omni_server_function),
    )
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_RECOVER_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_RECOVER_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_RECOVER_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_RECOVER_INJECTION_CONFIG["strict"],
    )
    host = omni_server_function.host
    port = omni_server_function.port
    mix_payload = _mix_chat_completions_probe_payload(omni_server_function)
    mix_chat_timeout_sec = 180
    print(
        "[oom-recover] injection ready "
        f"device={device_spec} target_mem_ratio={OOM_RECOVER_INJECTION_CONFIG['target_mem_ratio']} "
        f"strict={OOM_RECOVER_INJECTION_CONFIG['strict']} hold_seconds={OOM_RECOVER_INJECTION_CONFIG['hold_seconds']}"
    )
    print(f"[oom-recover] target endpoint http://{host}:{port}/v1/chat/completions")

    failure_observed = False
    fault_status: int | None = None
    fault_body = b""
    try:
        try:
            print(f"[oom-recover] fault-phase request start timeout={mix_chat_timeout_sec}s")
            fault_status, fault_body = post_json_raw(
                host,
                port,
                "/v1/chat/completions",
                mix_payload,
                timeout_sec=mix_chat_timeout_sec,
            )
            print(f"[oom-recover] fault-phase request done status={fault_status} body_prefix={fault_body[:200]!r}")
        except Exception as exc:
            failure_observed = True
            print(f"[oom-recover] fault-phase request raised {type(exc).__name__}: {exc!r}")
            assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
        else:
            assert fault_status is not None
            failure_observed = fault_status != 200 or _fault_keywords_match_response(
                status=fault_status,
                body=fault_body,
            )
            assert failure_observed, (
                "expected mix multimodal request failure while OOM pressure is active; "
                f"status={fault_status}, body_prefix={fault_body[:500]!r}"
            )
            if fault_status >= 400:
                err = extract_openai_error_contract_from_bytes(fault_body)
                if err is None:
                    assert _fault_keywords_match_response(status=fault_status, body=fault_body), (
                        "non-2xx without OpenAI-style error object should still mention fault hints in body; "
                        f"status={fault_status}, body_prefix={fault_body[:500]!r}"
                    )
                else:
                    assert isinstance(err.get("message"), str) and err["message"].strip(), (
                        f"structured error should carry a non-empty message: {err!r}"
                    )
            else:
                assert _fault_keywords_match_response(status=fault_status, body=fault_body), (
                    "HTTP 200 under pressure should still surface fault hints in the response body; "
                    f"body_prefix={fault_body[:500]!r}"
                )
    finally:
        print("[oom-recover] stopping OOM sidecar(s)")
        stop_gpu_oom_hogs(handle)
        print("[oom-recover] OOM sidecar(s) stopped")

    recovery_deadline = time.monotonic() + 90.0
    terminal_health: int | None = None
    unreachable_streak = 0
    last_health_exc: BaseException | None = None
    while time.monotonic() < recovery_deadline:
        try:
            status, _ = get_health_raw(host, port, timeout_sec=5)
            unreachable_streak = 0
            if status in (200, 503):
                terminal_health = status
                print(f"[oom-recover] terminal health reached status={terminal_health}")
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

    probe_payload = {
        "model": omni_server_function.model,
        "messages": [{"role": "user", "content": "What is the capital of China? Answer in one word."}],
        "stream": False,
        "modalities": ["text"],
    }
    start = time.monotonic()
    post_mix_status: int | None = None
    post_mix_body = b""
    request_status: int | None = None
    try:
        assert terminal_health is not None
        if terminal_health == 200:
            print(f"[oom-recover] recovery-phase mix request start timeout={mix_chat_timeout_sec}s")
            post_mix_status, post_mix_body = post_json_raw(
                host,
                port,
                "/v1/chat/completions",
                mix_payload,
                timeout_sec=mix_chat_timeout_sec,
            )
            print(
                "[oom-recover] recovery-phase mix request done "
                f"status={post_mix_status} body_prefix={post_mix_body[:200]!r}"
            )
        else:
            print("[oom-recover] terminal health=503, send lightweight probe request")
            request_status, _ = post_json_raw(host, port, "/v1/chat/completions", probe_payload, timeout_sec=20)
            print(f"[oom-recover] lightweight probe done status={request_status}")
    except Exception:
        post_mix_status = None
        request_status = None
    elapsed = time.monotonic() - start
    assert elapsed < 200, f"post-fault request should not hang after OOM removal: {elapsed:.2f}s"

    if terminal_health == 200:
        assert post_mix_status == 200, (
            "health recovered but repeated mix multimodal request did not return HTTP 200; "
            f"status={post_mix_status}, body_prefix={post_mix_body[:600]!r}"
        )
        recover_err = extract_openai_error_contract_from_bytes(post_mix_body)
        assert recover_err is None, f"unexpected error object after recovery: {recover_err!r}"
    else:
        assert request_status is None or request_status >= 500, (
            "unhealthy terminal state should fail fast on requests, "
            f"got health={terminal_health}, request_status={request_status}"
        )
