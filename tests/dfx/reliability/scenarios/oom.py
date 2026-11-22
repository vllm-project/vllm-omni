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
    gpu_oom_pressure_status,
    inject_gpu_oom,
    post_json_raw,
    snapshot_server_worker_pids,
    stop_gpu_oom_hogs,
    wait_for_worker_pid_death,
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
EXECUTOR_SHUTDOWN_MARKERS = ("executor shut down",)


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


def _require_oom_pressure(handle: Any, *, phase: str) -> str:
    verified, diagnostics = gpu_oom_pressure_status(handle)
    assert verified, f"OOM sidecar pressure was not verified {phase}: {diagnostics}"
    return diagnostics


def _snapshot_oom_workers(
    server: Any,
    *,
    worker_markers_extra: tuple[str, ...] = (),
) -> list[int]:
    worker_pids = snapshot_server_worker_pids(
        server,
        worker_markers_extra=worker_markers_extra,
    )
    assert worker_pids, "OOM fault probe found no live server workers before the request"
    print(f"[oom-under-load] workers before fault request: {worker_pids}", flush=True)
    return worker_pids


def _assert_oom_failure_evidence(
    *,
    server: Any,
    handle: Any,
    workers_before: list[int],
    response_evidence: str,
    stderr_capture: Any,
    context: str,
) -> None:
    """Accept explicit OOM or verified pressure + worker death + shutdown."""
    response_normalized = response_evidence.casefold()
    response_has_oom = any(marker in response_normalized for marker in GPU_OOM_MARKERS)
    stderr_evidence = stderr_capture.wait(timeout=0 if response_has_oom else 2)
    explicit_oom = response_has_oom or any(marker in stderr_evidence.casefold() for marker in GPU_OOM_MARKERS)
    executor_shutdown = any(marker in response_normalized for marker in EXECUTOR_SHUTDOWN_MARKERS)
    dead_workers = [] if explicit_oom else wait_for_worker_pid_death(workers_before, timeout_sec=5)
    pressure_diagnostics = _require_oom_pressure(handle, phase="after the fault request")
    stderr_tail = "".join(getattr(server, "_server_error_lines", ()))[-4096:]
    assert explicit_oom or (executor_shutdown and dead_workers), (
        f"{context} failed without explicit GPU OOM evidence or verified pressure/worker-death "
        "executor-shutdown proof; "
        f"executor_shutdown={executor_shutdown}, workers_before={workers_before}, "
        f"dead_workers={dead_workers}, pressure={pressure_diagnostics}, "
        f"response_head={response_evidence[:4096]!r}, captured_stderr={stderr_evidence!r}, "
        f"stderr_tail={stderr_tail!r}"
    )


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
    config = model.hard_oom_config
    with server.capture_stderr_evidence(GPU_OOM_MARKERS) as stderr_capture:
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
            pressure_diagnostics = _require_oom_pressure(handle, phase="before the fault request")
            workers_before = _snapshot_oom_workers(
                server,
                worker_markers_extra=(model.worker_pattern,),
            )
            print(f"[oom-under-load] verified pressure: {pressure_diagnostics}", flush=True)
            started = time.monotonic()
            print("[oom-under-load] pressure ready; sending one multimodal fault request (timeout=180s)", flush=True)
            try:
                status, fault_body = post_json_raw(server.host, server.port, endpoint, payload, timeout_sec=180)
            except Exception as exc:
                print(
                    f"[oom-under-load] request exception after {time.monotonic() - started:.3f}s: {exc!r}",
                    flush=True,
                )
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
            response_evidence = fault_body.decode("utf-8", errors="replace")
            _assert_oom_failure_evidence(
                server=server,
                handle=handle,
                workers_before=workers_before,
                response_evidence=response_evidence,
                stderr_capture=stderr_capture,
                context=f"OOM request status={status}",
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
