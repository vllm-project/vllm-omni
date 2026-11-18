# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU OOM fault steps. The model supplies the probe and the memory config."""

from __future__ import annotations

import base64
import errno
import json
import time
from collections.abc import Callable
from typing import Any

from tests.dfx.conftest import resolve_oom_device_spec
from tests.dfx.reliability.helpers import (
    FaultModel,
    FaultScenario,
    _deploy_config_path,
    extract_openai_error_contract_from_bytes,
    inject_gpu_oom,
    post_json_raw,
    stop_gpu_oom_hogs,
)

GPU_OOM_MARKERS = (
    "cuda out of memory",
    "hip out of memory",
    "torch.outofmemoryerror",
    "torch.cuda.outofmemoryerror",
    "cuda error: out of memory",
    "hip error: out of memory",
    "hip error: 'out of memory'",
    "cuda_error_out_of_memory",
    "hiperroroutofmemory",
    "memory allocation failed with oom on device",
)


class GpuOomUnderLoad(FaultScenario):
    """One injection with real OOM evidence, bounded failure and post-pressure checks."""

    server_set = "oom"

    def run(self, model: FaultModel, server: Any, client: Any, signal_name: str | None = None) -> None:
        del signal_name
        payload = model.oom_under_load_request(server)
        warmup = model.oom_preflight_request(server)
        run_oom_under_load(model, server, client, payload=payload, warmup=warmup)


def _looks_like_server_unreachable(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionRefusedError, BrokenPipeError, ConnectionResetError)):
        return True
    errno_val = getattr(exc, "errno", None)
    if isinstance(exc, OSError) and errno_val is not None:
        return errno_val in (errno.ECONNREFUSED, errno.ECONNRESET, errno.EPIPE)
    message = str(exc).lower()
    return "connection refused" in message or "actively refused" in message


def _assert_multimodal_success(status: int, body: bytes) -> None:
    assert status == 200, f"expected successful multimodal request, got {status}: {body[:500]!r}"
    payload = json.loads(body)
    assert isinstance(payload, dict) and not payload.get("error"), f"unexpected response: {payload!r}"
    choices = payload.get("choices")
    assert isinstance(choices, list) and choices, "multimodal response has no choices"
    message = choices[0].get("message", {})
    assert message.get("content"), "multimodal response has no text output"
    assert message.get("audio", {}).get("data"), "multimodal response has no audio output"


def _assert_image_success(status: int, body: bytes) -> None:
    assert status == 200, f"expected successful image request, got {status}: {body[:500]!r}"
    payload = json.loads(body)
    assert isinstance(payload, dict) and not payload.get("error"), f"unexpected image response: {payload!r}"
    data = payload.get("data")
    assert isinstance(data, list) and len(data) == 1, "response has no image output"
    image = data[0].get("b64_json")
    assert isinstance(image, str) and image, "response has no image output"
    assert base64.b64decode(image, validate=True), "response has empty image bytes"


def run_oom_under_load(
    model: FaultModel,
    server: Any,
    client: Any,
    *,
    payload: dict[str, Any],
    warmup: dict[str, Any],
    endpoint: str = "/v1/chat/completions",
    success_check: Callable[[int, bytes], None] = _assert_multimodal_success,
) -> None:
    success_check(*post_json_raw(server.host, server.port, endpoint, warmup, timeout_sec=180))
    print("[oom-under-load] multimodal preflight passed; starting memory pressure", flush=True)
    stderr_before = set(getattr(server, "_server_error_lines", ()))
    config = model.hard_oom_config
    handle = inject_gpu_oom(
        device=resolve_oom_device_spec(config, _deploy_config_path(server)),
        target_mem_ratio=config["target_mem_ratio"],
        hold_seconds=config["hold_seconds"],
        startup_timeout_sec=config["startup_timeout_sec"],
        strict=config["strict"],
        max_free_bytes=config.get("max_free_bytes"),
    )
    fault_body = b""
    status: int | None = None
    try:
        started = time.monotonic()
        print("[oom-under-load] pressure ready; sending one multimodal fault request (timeout=180s)", flush=True)
        try:
            status, fault_body = post_json_raw(server.host, server.port, endpoint, payload, timeout_sec=180)
        except Exception as exc:
            print(f"[oom-under-load] request exception after {time.monotonic() - started:.3f}s: {exc!r}", flush=True)
            # A dead API can close/refuse the connection; a client timeout or
            # unrelated exception is not the expected request-failure contract.
            assert _looks_like_server_unreachable(exc), f"unexpected OOM request failure: {exc!r}"
        else:
            # Bound diagnostics; never print the request's base64 media.
            print(
                f"[oom-under-load] request status={status}, elapsed={time.monotonic() - started:.3f}s, "
                f"response_bytes={len(fault_body)}, response_head={fault_body[:4096]!r}",
                flush=True,
            )
            assert status in (500, 503), f"OOM request must fail with 500/503, got {status}: {fault_body[:500]!r}"
            assert extract_openai_error_contract_from_bytes(fault_body) is not None, (
                f"OOM response lacks an OpenAI error object: {fault_body[:500]!r}"
            )
        assert time.monotonic() - started < 180, "OOM request did not fail within its deadline"
        # The API may return generic EngineDeadError; the harness's stderr
        # buffer can identify whether GPU OOM actually caused it.
        stderr_delta = "".join(line for line in getattr(server, "_server_error_lines", ()) if line not in stderr_before)
        evidence = fault_body.decode("utf-8", errors="replace") + stderr_delta
        oom_markers = GPU_OOM_MARKERS
        assert any(marker in evidence.lower() for marker in oom_markers), (
            "request failed without explicit GPU OOM evidence in this injection's response or logs; "
            f"status={status}, response_head={fault_body[:4096]!r}, "
            f"stderr_tail={stderr_delta[-4096:]!r}"
        )
    finally:
        stop_gpu_oom_hogs(handle)
        print("[oom-under-load] memory pressure removed", flush=True)

    try:
        health = client.send_health_http_request({"timeout": 5}, err_code=(200, 503))[0].status_code
    except Exception as exc:
        assert _looks_like_server_unreachable(exc), f"unexpected post-OOM health failure: {exc!r}"
        print("[oom-under-load] API unreachable after OOM; whole-instance restart required")
        return
    if health == 503:
        # Subsequent serial cases start a fresh function-scoped instance and
        # check their workloads. That is suite-level startup coverage, not an
        # in-case recovery assertion or individual-stage replacement check.
        print("[oom-under-load] health=503 after OOM removal; whole-instance restart required")
        return
    assert health == 200, f"unexpected post-OOM health status: {health}"
    success_check(*post_json_raw(server.host, server.port, endpoint, payload, timeout_sec=180))
    print(f"[oom-under-load] original instance served {endpoint} after OOM removal")
