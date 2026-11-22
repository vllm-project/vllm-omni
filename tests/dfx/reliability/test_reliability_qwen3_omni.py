# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Qwen3-Omni reliability integration tests.
"""

from __future__ import annotations

import concurrent.futures
import errno
import json
import time
from pathlib import Path
from typing import Any, Protocol, TypedDict

import pytest
import torch

from tests.dfx.conftest import (
    assert_fault_exception,
    create_reliability_omni_server_params,
    resolve_oom_device_spec,
)
from tests.dfx.reliability.helpers import (
    FaultInjector,
    assert_no_server_tree_process_residual_and_gpu_release,
    assert_post_fault_api_unreachable,
    assert_post_fault_health_503,
    extract_openai_error_contract_from_bytes,
    inject_gpu_oom,
    make_process_kill_fault_injector,
    make_server_root_kill_fault_injector,
    make_server_tree_kill_fault_injector,
    post_json_raw,
    reliability_scenarios_for_eager_oom,
    run_fault_injection_with_rate_load,
    stop_gpu_oom_hogs,
    worker_residual_timeout_after_kill_signal,
)
from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

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

DEPLOY_CONFIGS_DIR = Path(get_deploy_config_path("qwen3_omni_moe.yaml")).parent


def _default_oom_device_spec() -> str:
    """Use currently visible CUDA ordinals to avoid invalid device index in sidecar."""
    count = torch.accelerator.device_count() or 0
    if count <= 0:
        return "0"
    return ",".join(str(i) for i in range(count))


class _OomInjectionBaseConfig(TypedDict):
    device: str
    target_mem_ratio: float
    hold_seconds: int
    startup_timeout_sec: int
    strict: bool


class _OomInjectionConfig(_OomInjectionBaseConfig):
    max_free_bytes: int


OOM_INJECTION_CONFIG: _OomInjectionConfig = {
    "device": _default_oom_device_spec(),
    "target_mem_ratio": 0.95,
    # MI300 keeps a large caching-allocator pool, so a 256MB device-free ceiling
    # still lets the text and multimodal probes return 200. HIP also retries an
    # OOM instead of failing the request. Leave only a few MB so those probes
    # have to allocate. CUDA keeps the wider 256MB ceiling.
    "max_free_bytes": 32 * 1024 * 1024 if getattr(torch.version, "hip", None) else 256 * 1024 * 1024,
    "hold_seconds": 0,
    "startup_timeout_sec": 20,
    "strict": False,
}
# Post-fault recovery probe: keep the API process alive on the same GPU.
# ``strict=True`` sidecars keep allocating until CUDA refuses, which often
# evicts/kills the server; here we only take a fraction of *initial* free
# memory and stop, then rely on a heavy multimodal forward to hit OOM.
OOM_RECOVER_INJECTION_CONFIG: _OomInjectionBaseConfig = {
    "device": _default_oom_device_spec(),
    # ``strict=False`` stops at this fraction of *initial* free memory per GPU (see helpers sidecar).
    # On multi-GPU hosts a mild ratio plus a small e2e-style mix can still leave several GB free
    # on a data-parallel device and the request succeeds; 0.92+ pairs better with the heavy
    # probe payload below. Raise toward 0.94 only if fault phase still never fails; avoid strict=True
    # here so the API process is not squeezed to zero slack.
    "target_mem_ratio": 0.95,
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
# Text chat is the thinker, stage 0. ``VLLM::`` with limit=1 can kill stage 1
# first and leave the request path alive. Match the stage name literally.
RUNTIME_WORKER_PATTERN = "StageEngineCoreProc_stage0"
# RFC#2366 signal x target matrix:
# - worker: SIGTERM, SIGKILL
# - worker with-load: skipped (issue#8912)
# - serve-root no-load: SIGINT; SIGTERM/SIGKILL skipped (vllm issue#43060)
# - serve-root with-load: SIGINT, SIGTERM; SIGKILL skipped (vllm issue#43060)
# - serve-tree no-load: SIGTERM/SIGKILL
# - serve-tree with-load: SIGKILL; SIGTERM skipped (issue#8912)
_SERVE_ROOT_NON_SIGINT_SKIP = pytest.mark.skip(reason="vllm issue#43060")
_TREE_WITH_LOAD_SIGTERM_SKIP = pytest.mark.skip(reason="issue#8912")
SERVE_SIGNAL_PARAMS = [
    pytest.param("SIGTERM", id="sigterm"),
    pytest.param("SIGINT", id="sigint"),
    pytest.param("SIGKILL", id="sigkill", marks=_SERVE_ROOT_NON_SIGINT_SKIP),
]
SERVE_WITH_LOAD_SIGNAL_PARAMS = [
    pytest.param("SIGTERM", id="sigterm"),
    pytest.param("SIGINT", id="sigint"),
    pytest.param("SIGKILL", id="sigkill", marks=_SERVE_ROOT_NON_SIGINT_SKIP),
]
TREE_SIGNAL_PARAMS = [
    pytest.param("SIGTERM", id="sigterm"),
    pytest.param("SIGKILL", id="sigkill"),
]
TREE_WITH_LOAD_SIGNAL_PARAMS = [
    pytest.param("SIGTERM", id="sigterm", marks=_TREE_WITH_LOAD_SIGTERM_SKIP),
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
INFLIGHT_INJECTION_REQUEST_RATE = 0.3
INFLIGHT_INJECTION_REQUEST_COUNT = 10

# Client read timeout. HTTP 500 must return before this; a timeout raises and fails the test.
_PROCESS_KILL_WORKER_CHAT_FF_HTTP_TIMEOUT_SEC = 30


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

    Uses the same synthetic sizes as ``test_reliability_fault_gpu_oom_large_request_failure``
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


def _server_error_code(value: object) -> int | None:
    """Return ``value`` when it is HTTP 500."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value == 500:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) == 500:
        return int(value)
    return None


def _stream_server_error_code(body: bytes) -> int | None:
    """Server error code from an SSE ``data:`` frame or a single JSON body.

    Streaming chat commits HTTP 200 before generation. A later engine failure is
    carried as ``error.code`` 500 (or a top-level ``code``) inside a ``data:`` line.
    Bare ``500`` text is ignored so base64 payloads cannot trip this check.
    """
    text = body.decode("utf-8", errors="replace")
    payloads: list[object] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        data = stripped[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payloads.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    if not payloads:
        try:
            payloads.append(json.loads(text))
        except json.JSONDecodeError:
            return None
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        error = payload.get("error")
        if isinstance(error, dict):
            code = _server_error_code(error.get("code"))
            if code is not None:
                return code
        if "choices" not in payload:
            code = _server_error_code(payload.get("code"))
            if code is not None:
                return code
    return None


def _deploy_config_path_from_omni_server(omni_server: _HasServeArgs) -> str | None:
    args: list[str] = omni_server.serve_args
    for i, arg in enumerate(args):
        if arg == "--deploy-config" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--deploy-config="):
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


def _create_text_request_config(model: str) -> dict[str, Any]:
    messages = dummy_messages_from_mix_data(
        system_prompt=_get_system_prompt(),
        content_text="What is the capital of China? Answer in one short sentence.",
    )
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
    }


def _create_multimodal_request_config(model: str) -> dict[str, Any]:
    """Small image and audio input, with text and audio output.

    Text-only chat stays on the thinker (stage 0). Asking for audio output
    routes the request through the talker (stage 1), so killing that engine
    must fail the request.
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=_get_system_prompt(),
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text="Describe the image and the audio in one short sentence.",
    )
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "modalities": ["text", "audio"],
    }


def _assert_server_alive_after_fault(online_client: Any, *, scenario: str) -> None:
    """#4285: a single-stage engine death must NOT take down the API server.

    The process must stay up and keep answering /health (503 is acceptable —
    errored but listening). A connection-refused means the server process
    exited, which is the bug this fix closes.
    """
    deadline = time.monotonic() + 15.0
    responded = False
    while time.monotonic() < deadline:
        try:
            status = online_client.send_health_http_request({"timeout": 5}, err_code=(200, 503))[0].status_code
            responded = True
            assert status in (200, 503), f"[{scenario} alive] unexpected /health status={status}"
        except Exception as exc:  # noqa: BLE001
            if _looks_like_server_unreachable(exc):
                pytest.fail(f"[{scenario} alive] server exited after single-stage OOM (#4285): {exc!r}")
            raise
        time.sleep(0.5)
    assert responded, f"[{scenario} alive] /health never responded after fault"


QWEN_PARAMS = create_reliability_omni_server_params(RELIABILITY_SCENARIOS, DEPLOY_CONFIGS_DIR)
QWEN_OOM_PARAMS = create_reliability_omni_server_params(
    reliability_scenarios_for_eager_oom(
        RELIABILITY_SCENARIOS,
        is_hip=bool(getattr(torch.version, "hip", None)),
        stage_ids=(0, 1),
    ),
    DEPLOY_CONFIGS_DIR,
)


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server_function", QWEN_OOM_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_error_contract_consistent_chat_speech(
    omni_server_function,
    online_client_function,
) -> None:
    """Black-box: text chat and audio output both return HTTP 500 under OOM, and the API stays up.

    Speech-style output is requested via ``/v1/chat/completions`` with ``modalities`` that include ``audio``.
    """
    device_spec = resolve_oom_device_spec(
        OOM_INJECTION_CONFIG,
        _deploy_config_path_from_omni_server(omni_server_function),
    )
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
        max_free_bytes=OOM_INJECTION_CONFIG["max_free_bytes"],
    )
    mix_base = _mix_chat_completions_probe_payload(omni_server_function)
    oom_contract_timeout_sec = 120

    def _oom_contract_request(modalities: list[str]) -> dict[str, Any]:
        return {
            "model": mix_base["model"],
            "messages": mix_base["messages"],
            "stream": False,
            "modalities": modalities,
        }

    # Under OOM pressure both chat and speech should return HTTP 500
    # instead of request-validation-class (4xx) errors.
    try:
        # Preserve the text-only output contract while using the heavy
        # multimodal input needed to reach the allocation path on MI300.
        text_request = _oom_contract_request(["text"])
        text_request["max_tokens"] = 256
        online_client_function.send_chat_completions_http_request(
            {"json": text_request, "timeout": oom_contract_timeout_sec},
            err_code=500,
        )
        online_client_function.send_chat_completions_http_request(
            {"json": _oom_contract_request(["text", "audio"]), "timeout": oom_contract_timeout_sec},
            err_code=500,
        )
    finally:
        stop_gpu_oom_hogs(handle)

    # #4285: a single dead stage must not take the whole API server down.
    _assert_server_alive_after_fault(online_client_function, scenario="oom_contract")


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server_function", QWEN_OOM_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_large_request_failure(omni_server_function, online_client_function) -> None:
    device_spec = resolve_oom_device_spec(
        OOM_INJECTION_CONFIG,
        _deploy_config_path_from_omni_server(omni_server_function),
    )
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
        max_free_bytes=OOM_INJECTION_CONFIG["max_free_bytes"],
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
            online_client_function.send_omni_request(request_config, request_num=1)
        except Exception as exc:
            assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
        else:
            pytest.fail("expected large chat payload request failure during GPU OOM injection")
    finally:
        stop_gpu_oom_hogs(handle)


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server_function", QWEN_OOM_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_concurrent_pressure_failure(omni_server_function, online_client_function) -> None:
    device_spec = resolve_oom_device_spec(
        OOM_INJECTION_CONFIG,
        _deploy_config_path_from_omni_server(omni_server_function),
    )
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
        max_free_bytes=OOM_INJECTION_CONFIG["max_free_bytes"],
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
            online_client_function.send_omni_request(request_config, request_num=4)
        except Exception as exc:
            assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
        else:
            pytest.fail("expected concurrent request failure under OOM injection")
    finally:
        stop_gpu_oom_hogs(handle)


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_request_failure(
    omni_server_function,
    online_client_function,
    fault_injector: FaultInjector,
) -> None:
    online_client_function.send_omni_request(
        _create_multimodal_request_config(omni_server_function.model), request_num=1
    )
    fault_injector(omni_server_function)
    request_config = _create_multimodal_request_config(omni_server_function.model)
    try:
        online_client_function.send_omni_request(request_config, request_num=1)
    except Exception as exc:
        assert_fault_exception(exc, PROCESS_KILL_ERROR_KEYWORDS)
    else:
        pytest.fail("expected request failure after process-kill injection")


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_health_fast_fail_and_concurrent(
    omni_server_function,
    online_client_function,
    fault_injector: FaultInjector,
) -> None:
    """Black-box: after worker process kill, multimodal chat fails fast and does not hang; /health→503 last.

    The probe asks for audio output, so it must enter the talker. Text-only chat
    stays on the thinker and can still return HTTP 200 after the talker is killed.
    The client read timeout is 30s. A timeout raises out of the client and fails the test.
    """
    online_client_function.send_omni_request(
        _create_multimodal_request_config(omni_server_function.model), request_num=1
    )
    fault_injector(omni_server_function)
    model = omni_server_function.model

    payload = _create_multimodal_request_config(model)
    online_client_function.send_chat_completions_http_request(
        {"json": payload, "timeout": _PROCESS_KILL_WORKER_CHAT_FF_HTTP_TIMEOUT_SEC},
        err_code=500,
    )

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                online_client_function.send_chat_completions_http_request,
                {"json": payload, "timeout": 20},
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
    for future in done:
        try:
            status = future.result()[0].status_code
            if status == 500:
                fault_observed = True
        except Exception as exc:
            assert_fault_exception(exc, PROCESS_KILL_ERROR_KEYWORDS)
            fault_observed = True
    assert fault_observed, (
        "[process_kill concurrent] expected at least one request to fail after process-kill fault injection"
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
@pytest.mark.skip(reason="issue#8912")
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_worker_with_load_request_failure(
    omni_server_function,
    online_client_function,
    fault_injector: FaultInjector,
) -> None:
    request_config = _create_multimodal_request_config(omni_server_function.model)
    scenario = "kill_worker_with_load"
    online_client_function.send_omni_request(
        _create_multimodal_request_config(omni_server_function.model), request_num=1
    )
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: online_client_function.send_omni_request(request_config, request_num=1),
        inject_fault=lambda: fault_injector(omni_server_function),
        num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
        request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
        completion_timeout_sec=300.0,
    )
    assert load_result["failure_observed"], (
        f"[{scenario}] expected at least one load request failure after fault; load_result={load_result}"
    )
    online_client_function.send_chat_completions_http_request(
        {
            "json": _create_multimodal_request_config(omni_server_function.model),
            "timeout": _PROCESS_KILL_WORKER_CHAT_FF_HTTP_TIMEOUT_SEC,
        },
        err_code=500,
    )
    assert_post_fault_health_503(online_client_function, scenario=scenario)


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("signal_name", SERVE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_serve_root_no_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    """Black-box: kill serve root without load; verify fast-fail/health/cleanup."""
    scenario = f"kill_serve_root_no_load_{signal_name.lower()}"
    injector = make_server_root_kill_fault_injector(signal_name=signal_name, post_kill_wait_seconds=2.0)
    online_client_function.send_omni_request(
        _create_multimodal_request_config(omni_server_function.model), request_num=1
    )
    injector(omni_server_function)
    assert_post_fault_api_unreachable(online_client_function, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("signal_name", SERVE_WITH_LOAD_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_serve_root_with_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    request_config = _create_text_request_config(omni_server_function.model)
    scenario = f"kill_serve_root_with_load_{signal_name.lower()}"
    injector = make_server_root_kill_fault_injector(signal_name=signal_name, post_kill_wait_seconds=2.0)
    online_client_function.send_omni_request(
        _create_multimodal_request_config(omni_server_function.model), request_num=1
    )
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: online_client_function.send_omni_request(request_config, request_num=1),
        inject_fault=lambda: injector(omni_server_function),
        num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
        request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
        completion_timeout_sec=300.0,
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
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("signal_name", TREE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_tree_no_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    """Black-box: kill server tree without load; verify fast-fail/health/cleanup."""
    scenario = f"kill_serve_tree_no_load_{signal_name.lower()}"
    injector = make_server_tree_kill_fault_injector(
        signal_name=signal_name,
        post_kill_wait_seconds=2.0,
        inter_kill_wait_seconds=0.1,
    )
    online_client_function.send_omni_request(
        _create_multimodal_request_config(omni_server_function.model), request_num=1
    )
    injector(omni_server_function)
    assert_post_fault_api_unreachable(online_client_function, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("signal_name", TREE_WITH_LOAD_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_tree_with_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    request_config = _create_text_request_config(omni_server_function.model)
    scenario = f"kill_serve_tree_with_load_{signal_name.lower()}"
    injector = make_server_tree_kill_fault_injector(
        signal_name=signal_name,
        post_kill_wait_seconds=2.0,
        inter_kill_wait_seconds=0.1,
    )
    online_client_function.send_omni_request(
        _create_multimodal_request_config(omni_server_function.model), request_num=1
    )
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: online_client_function.send_omni_request(request_config, request_num=1),
        inject_fault=lambda: injector(omni_server_function),
        num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
        request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
        completion_timeout_sec=300.0,
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
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_recovers_after_fault_removed(
    omni_server_function,
    online_client_function,
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
        _deploy_config_path_from_omni_server(omni_server_function),
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
            stream_error_code = _stream_server_error_code(fault_body)
            failure_observed = (
                fault_status != 200
                or stream_error_code is not None
                or _fault_keywords_match_response(status=fault_status, body=fault_body)
            )
            assert failure_observed, (
                "expected mix multimodal request failure while OOM pressure is active; "
                f"status={fault_status}, stream_error_code={stream_error_code}, "
                f"body_prefix={fault_body[:500]!r}"
            )
            if fault_status >= 400 and extract_openai_error_contract_from_bytes(fault_body) is None:
                assert _fault_keywords_match_response(status=fault_status, body=fault_body), (
                    "non-2xx without OpenAI-style error object should still mention fault hints in body; "
                    f"status={fault_status}, body_prefix={fault_body[:500]!r}"
                )
            else:
                assert stream_error_code is not None or _fault_keywords_match_response(
                    status=fault_status, body=fault_body
                ), (
                    "HTTP 200 under pressure should still surface a streaming error code or fault hints; "
                    f"stream_error_code={stream_error_code}, body_prefix={fault_body[:500]!r}"
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
            status = online_client_function.send_health_http_request({"timeout": 5}, err_code=(200, 503))[0].status_code
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
        recover_stream_code = _stream_server_error_code(post_mix_body)
        assert recover_stream_code is None, (
            "health recovered but the mix stream still carried a server error; "
            f"stream_error_code={recover_stream_code}, body_prefix={post_mix_body[:600]!r}"
        )
    else:
        assert request_status is None or request_status == 500, (
            "unhealthy terminal state should fail fast on requests, "
            f"got health={terminal_health}, request_status={request_status}"
        )
