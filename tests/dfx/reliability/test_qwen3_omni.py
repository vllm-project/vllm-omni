# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen3-Omni reliability scenarios."""

from __future__ import annotations

import errno
import json
import time
from pathlib import Path
from typing import Any

import pytest

from tests.dfx.conftest import assert_fault_exception, resolve_oom_device_spec
from tests.dfx.reliability.helpers import (
    PARTIAL_ISOLATION_HTTP_TIMEOUT_SEC,
    FaultModel,
    SignalSpec,
    extract_openai_error_contract_from_bytes,
    inject_gpu_oom,
    post_json_raw,
    scenario_params,
    stop_gpu_oom_hogs,
)
from tests.dfx.reliability.scenarios import (
    GpuOomConcurrent,
    GpuOomErrorContract,
    GpuOomLarge,
    GpuOomRecovery,
    PartialStageIsolation,
    ProcessKillServeRootNoLoad,
    ProcessKillServeRootWithLoad,
    ProcessKillTreeNoLoad,
    ProcessKillTreeWithLoad,
    ProcessKillWorkerNoLoad,
    ProcessKillWorkerWithLoad,
)
from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

QWEN_SERVE_ROOT_SIGNALS: tuple[SignalSpec, ...] = (
    ("SIGTERM", None),
    ("SIGINT", None),
    ("SIGKILL", "vllm issue#43060"),
)
QWEN_TREE_WITH_LOAD_SIGNALS: tuple[SignalSpec, ...] = (
    ("SIGTERM", "issue#8912"),
    ("SIGKILL", None),
)

RUNTIME_WORKER_PATTERN = "StageEngineCoreProc_stage0"
_PROCESS_KILL_WORKER_CHAT_FF_HTTP_TIMEOUT_SEC = 30


def _default_oom_device_spec() -> str:
    import torch

    count = torch.accelerator.device_count() or 0
    if count <= 0:
        return "0"
    return ",".join(str(index) for index in range(count))


def _hard_oom_config() -> dict[str, Any]:
    import torch

    return {
        "device": _default_oom_device_spec(),
        "target_mem_ratio": 0.95,
        "max_free_bytes": 32 * 1024 * 1024 if getattr(torch.version, "hip", None) else 256 * 1024 * 1024,
        "hold_seconds": 0,
        "startup_timeout_sec": 20,
        "strict": False,
    }


OOM_RECOVER_INJECTION_CONFIG: dict[str, Any] = {
    "device": _default_oom_device_spec(),
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


def _system_prompt() -> dict[str, Any]:
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


def _text_request(model: str) -> dict[str, Any]:
    messages = dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        content_text="What is the capital of China? Answer in one short sentence.",
    )
    return {"model": model, "messages": messages, "stream": False, "modalities": ["text"]}


def _multimodal_request(model: str) -> dict[str, Any]:
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text="Describe the image and the audio in one short sentence.",
    )
    return {"model": model, "messages": messages, "stream": False, "modalities": ["text", "audio"]}


def _heavy_messages() -> list[dict[str, Any]]:
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(1280, 720, 161)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(20, 1)['base64']}"
    return dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        video_data_url=video_data_url,
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text="What is recited in the audio? What is in this image? What is in this video? " * 200,
    )


def _mix_probe_payload(server: Any) -> dict[str, Any]:
    return {
        "model": server.model,
        "messages": _heavy_messages(),
        "stream": True,
        "modalities": ["text", "audio"],
    }


def _assert_text_200(client: Any, model: str) -> None:
    client.send_chat_completions_http_request({"json": _text_request(model), "timeout": 120})


def _deploy_config_path(server: Any) -> str | None:
    args: list[str] = server.serve_args
    for index, arg in enumerate(args):
        if arg == "--deploy-config" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--deploy-config="):
            return arg.split("=", 1)[1]
    return None


def _server_error_code(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value == 500:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) == 500:
        return int(value)
    return None


def _stream_server_error_code(body: bytes) -> int | None:
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


def _fault_keywords_match_response(*, status: int, body: bytes) -> bool:
    if status >= 400:
        return True
    text = body.decode("utf-8", errors="replace").lower()
    hints = (
        "the request failed",
        "oom",
        "out of memory",
        "cuda",
        "orchestrator",
        "timeout",
        "connection refused",
        "connection reset",
    )
    return any(key in text for key in hints)


def _looks_like_server_unreachable(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionRefusedError, BrokenPipeError, ConnectionResetError)):
        return True
    errno_val = getattr(exc, "errno", None)
    if isinstance(exc, OSError) and errno_val is not None:
        return errno_val in (errno.ECONNREFUSED, errno.ECONNRESET, errno.EPIPE)
    message = str(exc).lower()
    return "connection refused" in message or "actively refused" in message


def _assert_server_alive(client: Any, *, scenario: str) -> None:
    deadline = time.monotonic() + 15.0
    responded = False
    while time.monotonic() < deadline:
        try:
            status = client.send_health_http_request({"timeout": 5}, err_code=(200, 503))[0].status_code
            responded = True
            assert status in (200, 503), f"[{scenario} alive] unexpected /health status={status}"
        except Exception as exc:  # noqa: BLE001
            if _looks_like_server_unreachable(exc):
                pytest.fail(f"[{scenario} alive] server exited after single-stage OOM (#4285): {exc!r}")
            raise
        time.sleep(0.5)
    assert responded, f"[{scenario} alive] /health never responded after fault"


class Qwen3Omni(FaultModel):
    model_id = "qwen3_omni"
    res = {"cuda": "H100", "rocm": "MI325"}
    num_cards = 2
    scenarios = [
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
    deploy_dir = Path(get_deploy_config_path("qwen3_omni_moe.yaml")).parent
    lifecycle_scenario = "qwen3_omni_reliability_default"
    eager_oom_stage_ids = (0, 1)
    worker_pattern = RUNTIME_WORKER_PATTERN
    isolation_pattern = "StageEngineCoreProc_stage1"
    hard_oom_config = _hard_oom_config()
    recover_oom_config = OOM_RECOVER_INJECTION_CONFIG
    fault_error_keywords = FAULT_ERROR_KEYWORDS
    process_kill_error_keywords = PROCESS_KILL_ERROR_KEYWORDS
    with_load_timeout_sec = 300.0
    concurrent_workers = 4
    concurrent_timeout_sec = 30.0
    serve_root_signals = QWEN_SERVE_ROOT_SIGNALS
    serve_root_with_load_signals = QWEN_SERVE_ROOT_SIGNALS
    tree_with_load_signals = QWEN_TREE_WITH_LOAD_SIGNALS
    oom_large_fail_message = "expected large chat payload request failure during GPU OOM injection"

    def warmup(self, client: Any, server: Any) -> None:
        client.send_omni_request(_multimodal_request(server.model), request_num=1)

    def submit_worker_load(self, client: Any, server: Any) -> None:
        client.send_omni_request(_multimodal_request(server.model), request_num=1)

    def submit_serve_load(self, client: Any, server: Any) -> None:
        client.send_omni_request(_text_request(server.model), request_num=1)

    def submit_concurrent(self, client: Any, server: Any) -> Any:
        return client.send_chat_completions_http_request(
            {"json": _multimodal_request(server.model), "timeout": 20},
            err_code=(500, 503),
        )

    def assert_fast_fail(self, client: Any, server: Any) -> None:
        client.send_chat_completions_http_request(
            {
                "json": _multimodal_request(server.model),
                "timeout": _PROCESS_KILL_WORKER_CHAT_FF_HTTP_TIMEOUT_SEC,
            },
            err_code=500,
        )

    def isolation_warmup(self, client: Any, server: Any) -> None:
        _assert_text_200(client, server.model)

    def isolation_after(self, client: Any, server: Any) -> None:
        _assert_text_200(client, server.model)
        client.send_chat_completions_http_request(
            {"json": _multimodal_request(server.model), "timeout": PARTIAL_ISOLATION_HTTP_TIMEOUT_SEC},
            err_code=500,
        )

    def oom_large_probe(self, client: Any, server: Any) -> None:
        video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(1280, 720, 161)['base64']}"
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"
        audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(20, 1)['base64']}"
        messages = dummy_messages_from_mix_data(
            system_prompt=_system_prompt(),
            video_data_url=video_data_url,
            image_data_url=image_data_url,
            audio_data_url=audio_data_url,
            content_text=f"{_mix_prompt()} " * 200,
        )
        client.send_omni_request(
            {
                "model": server.model,
                "messages": messages,
                "stream": True,
                "key_words": {"audio": ["test"]},
            },
            request_num=1,
        )

    def oom_concurrent_probe(self, client: Any, server: Any) -> None:
        video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(1280, 720, 161)['base64']}"
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"
        audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(20, 1)['base64']}"
        messages = dummy_messages_from_mix_data(
            system_prompt=_system_prompt(),
            video_data_url=video_data_url,
            image_data_url=image_data_url,
            audio_data_url=audio_data_url,
            content_text=f"{_mix_prompt()} " * 200,
        )
        client.send_omni_request(
            {
                "model": server.model,
                "messages": messages,
                "stream": True,
                "modalities": ["text", "audio"],
                "key_words": {"audio": ["test"]},
            },
            request_num=4,
        )

    def run_error_contract(self, server: Any, client: Any) -> None:
        device_spec = resolve_oom_device_spec(self.hard_oom_config, _deploy_config_path(server))
        handle = inject_gpu_oom(
            device=device_spec,
            target_mem_ratio=self.hard_oom_config["target_mem_ratio"],
            hold_seconds=self.hard_oom_config["hold_seconds"],
            startup_timeout_sec=self.hard_oom_config["startup_timeout_sec"],
            strict=self.hard_oom_config["strict"],
            max_free_bytes=self.hard_oom_config["max_free_bytes"],
        )
        mix_base = _mix_probe_payload(server)
        try:
            text_request = {
                "model": mix_base["model"],
                "messages": mix_base["messages"],
                "stream": False,
                "modalities": ["text"],
                "max_tokens": 256,
            }
            audio_request = {
                "model": mix_base["model"],
                "messages": mix_base["messages"],
                "stream": False,
                "modalities": ["text", "audio"],
            }
            client.send_chat_completions_http_request({"json": text_request, "timeout": 120}, err_code=500)
            client.send_chat_completions_http_request({"json": audio_request, "timeout": 120}, err_code=500)
        finally:
            stop_gpu_oom_hogs(handle)
        _assert_server_alive(client, scenario="oom_contract")

    def run_oom_recovery(self, server: Any, client: Any) -> None:
        _run_oom_recovery(server, client)


def _mix_prompt() -> str:
    return "What is recited in the audio? What is in this image? Describe the video briefly."


def _run_oom_recovery(omni_server_function: Any, online_client_function: Any) -> None:
    """Black-box: under bounded GPU pressure, a heavy mix chat fails; after hog stops, the same request succeeds.

    Uses ``OOM_RECOVER_INJECTION_CONFIG``: same ratio as the hard OOM sidecar, but
    without ``max_free_bytes``, so pressure can starve the multimodal forward pass
    without pinning the device down to the hard ceiling.

    Fault and recovery phases use raw ``/v1/chat/completions`` HTTP so tests can assert
    on status codes and JSON error payloads (or fault keywords in streamed bodies).
    """
    device_spec = resolve_oom_device_spec(
        OOM_RECOVER_INJECTION_CONFIG,
        _deploy_config_path(omni_server_function),
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
    mix_payload = _mix_probe_payload(omni_server_function)
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


MODEL = Qwen3Omni()

pytestmark = [
    pytest.mark.slow,
    *MODEL.extra_marks,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


@scenario_params(MODEL, GpuOomErrorContract)
def test_reliability_fault_gpu_oom_error_contract_consistent_chat_speech(
    omni_server_function,
    online_client_function,
) -> None:
    """Text chat and audio output both return HTTP 500 under OOM, and the API stays up."""
    GpuOomErrorContract().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, GpuOomLarge)
def test_reliability_fault_gpu_oom_large_request_failure(omni_server_function, online_client_function) -> None:
    GpuOomLarge().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, GpuOomConcurrent)
def test_reliability_fault_gpu_oom_concurrent_pressure_failure(omni_server_function, online_client_function) -> None:
    GpuOomConcurrent().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, GpuOomRecovery)
def test_reliability_fault_gpu_oom_recovers_after_fault_removed(omni_server_function, online_client_function) -> None:
    GpuOomRecovery().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, ProcessKillWorkerNoLoad)
def test_reliability_fault_process_kill_worker_no_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    """After a quiet worker kill, the next request returns HTTP 500, then concurrent requests fail."""
    ProcessKillWorkerNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillWorkerWithLoad, skip_reason="issue#8912")
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
