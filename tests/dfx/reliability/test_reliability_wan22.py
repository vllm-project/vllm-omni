"""
Wan2.2 reliability integration tests.
"""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
import requests

from tests.dfx.conftest import (
    assert_fault_exception,
    create_reliability_omni_server_params,
    resolve_oom_device_spec,
    supports_video_generation,
)
from tests.dfx.reliability.helpers import inject_gpu_oom, make_process_kill_fault_injector, stop_gpu_oom_hogs
from tests.helpers.media import generate_synthetic_image
from vllm_omni.platforms import current_omni_platform

RELIABILITY_SCENARIOS: list[dict[str, Any]] = [
    {
        "test_name": "wan22_t2v_reliability_default",
        "server_params": {
            "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "server_args": [
                "--num-gpus",
                "1",
                "--boundary-ratio",
                "0.875",
                "--flow-shift",
                "5.0",
                "--disable-log-stats",
            ],
        },
    }
]

E2E_STAGE_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "e2e" / "stage_configs"
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

WAN_PARAMS = create_reliability_omni_server_params(RELIABILITY_SCENARIOS, E2E_STAGE_CONFIGS_DIR)
DIFFUSION_VIDEO_PARAMS = [param for param in WAN_PARAMS if supports_video_generation(param.model)]


def _get_health_raw(host: str, port: int, *, timeout_sec: int = 20) -> tuple[int, bytes]:
    """GET /health with stdlib HTTP client; returns (status, body)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout_sec)
    try:
        conn.request("GET", "/health")
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def _create_video_job_with_invalid_lora(host: str, port: int, *, timeout_sec: int = 30) -> tuple[int, dict[str, Any]]:
    """Create one video job expected to fail later due to malformed lora payload."""
    url = f"http://{host}:{port}/v1/videos"
    response = requests.post(
        url,
        data={
            "prompt": "lora reliability test",
            "lora": '{"name": "bad-lora"}',
        },
        headers={"Accept": "application/json"},
        timeout=timeout_sec,
    )
    payload = response.json()
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
            state = payload.get("status")
            if state in {"completed", "failed"}:
                return last_status, payload
        time.sleep(interval_sec)
    raise TimeoutError(
        f"video job {video_id} did not reach terminal status within {timeout_sec}s; "
        f"last_status={last_status}, last_payload={json.dumps(last_payload)[:500]}"
    )


def _extract_openai_error(payload: dict[str, Any]) -> dict[str, Any] | None:
    error_obj = payload.get("error")
    if not isinstance(error_obj, dict):
        return None
    if not isinstance(error_obj.get("message"), str):
        return None
    if "code" not in error_obj:
        return None
    return error_obj


@pytest.mark.slow
@pytest.mark.skipif(
    current_omni_platform.is_rocm() or current_omni_platform.is_xpu(),
    reason="CUDA sidecar OOM injection is CUDA-only for phase-1",
)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_video_large_request_failure(omni_server_function, openai_client_function) -> None:
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
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"
        request_config = {
            "form_data": {
                "prompt": "Generate a realistic road-driving video with camera motion.",
                "width": 512,
                "height": 512,
                "fps": 8,
                "num_frames": 8,
                "guidance_scale": 1.0,
                "flow_shift": 5.0,
                "num_inference_steps": 4,
                "seed": 42,
            },
            "image_reference": image_data_url,
            "stream": False,
        }
        try:
            openai_client_function.send_video_diffusion_request(request_config, request_num=1)
        except Exception as exc:
            assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
        else:
            pytest.fail("expected large /v1/videos request failure during GPU OOM injection")
    finally:
        stop_gpu_oom_hogs(handle)


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    [
        pytest.param(
            make_process_kill_fault_injector(
                grep_patterns="vllm_omni.entrypoints.cli.main",
                signal_name="SIGKILL",
                limit=1,
                post_kill_wait_seconds=2.0,
            ),
            id="runtime_process_chain",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_video_request_failure(
    omni_server_after_fault_function,
    openai_client_function,
) -> None:
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"
    request_config = {
        "form_data": {
            "prompt": "Generate a realistic road-driving video with camera motion.",
            "width": 512,
            "height": 512,
            "fps": 8,
            "num_frames": 8,
            "guidance_scale": 1.0,
            "flow_shift": 5.0,
            "num_inference_steps": 4,
            "seed": 42,
        },
        "image_reference": image_data_url,
        "stream": False,
    }
    try:
        openai_client_function.send_video_diffusion_request(request_config, request_num=1)
    except Exception as exc:
        assert_fault_exception(exc, PROCESS_KILL_ERROR_KEYWORDS)
    else:
        pytest.fail("expected /v1/videos request failure after process-kill injection")


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    [
        pytest.param(
            make_process_kill_fault_injector(
                grep_patterns="vllm_omni.entrypoints.cli.main",
                signal_name="SIGKILL",
                limit=1,
                post_kill_wait_seconds=2.0,
            ),
            id="runtime_process_chain",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_video_health_unhealthy(
    omni_server_after_fault_function,
) -> None:
    """Black-box: after runtime process kill, /health should report unhealthy."""
    host = omni_server_after_fault_function.host
    port = omni_server_after_fault_function.port
    deadline = time.monotonic() + 20.0
    last_observation = ""
    while time.monotonic() < deadline:
        try:
            status, body = _get_health_raw(host, port, timeout_sec=5)
            last_observation = f"http={status}, body={body[:200]!r}"
            if status == 503:
                return
        except Exception as exc:  # noqa: BLE001
            last_observation = f"exception={exc!r}"
        time.sleep(0.5)
    pytest.fail(f"expected /health to become 503 after fault injection, got {last_observation}")


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    [
        pytest.param(
            make_process_kill_fault_injector(
                grep_patterns="vllm_omni.entrypoints.cli.main",
                signal_name="SIGKILL",
                limit=1,
                post_kill_wait_seconds=2.0,
            ),
            id="runtime_process_chain",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_video_new_request_fast_fail(
    omni_server_after_fault_function,
) -> None:
    """Black-box: /v1/videos should fail quickly after fatal fault."""
    url = f"http://{omni_server_after_fault_function.host}:{omni_server_after_fault_function.port}/v1/videos"
    payload = {
        "prompt": "fast-fail check",
        "width": "512",
        "height": "512",
        "fps": "8",
        "num_frames": "8",
        "num_inference_steps": "4",
    }
    start = time.monotonic()
    try:
        response = requests.post(
            url,
            data=payload,
            headers={"Accept": "application/json"},
            timeout=20,
        )
        elapsed = time.monotonic() - start
        assert elapsed < 15, f"/v1/videos did not fail fast after fault: {elapsed:.2f}s"
        assert response.status_code >= 500, (
            "expected server-side error after fatal fault, "
            f"got status={response.status_code}, body={response.text[:200]!r}"
        )
    except Exception:
        elapsed = time.monotonic() - start
        assert elapsed < 15, f"/v1/videos exception was too slow after fault: {elapsed:.2f}s"


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    [
        pytest.param(
            make_process_kill_fault_injector(
                grep_patterns="vllm_omni.entrypoints.cli.main",
                signal_name="SIGKILL",
                limit=1,
                post_kill_wait_seconds=2.0,
            ),
            id="runtime_process_chain",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_video_concurrent_requests_no_hang(
    omni_server_after_fault_function,
    openai_client_function,
) -> None:
    """Black-box: concurrent /v1/videos requests should finish within timeout after fault."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"
    request_config = {
        "form_data": {
            "prompt": "Generate a realistic road-driving video with camera motion.",
            "width": 512,
            "height": 512,
            "fps": 8,
            "num_frames": 8,
            "guidance_scale": 1.0,
            "flow_shift": 5.0,
            "num_inference_steps": 4,
            "seed": 42,
        },
        "image_reference": image_data_url,
        "stream": False,
    }
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(openai_client_function.send_video_diffusion_request, request_config, 1) for _ in range(3)
        ]
        done, pending = concurrent.futures.wait(
            futures,
            timeout=40,
            return_when=concurrent.futures.ALL_COMPLETED,
        )

    elapsed = time.monotonic() - start
    assert not pending, f"some fault-time video requests hung: pending={len(pending)}"
    assert elapsed < 40, f"fault-time video request convergence is too slow: {elapsed:.2f}s"

    fault_observed = False
    for future in done:
        try:
            future.result()
        except Exception as exc:
            fault_observed = True
            assert_fault_exception(exc, PROCESS_KILL_ERROR_KEYWORDS)
    assert fault_observed, "expected at least one /v1/videos request to fail after process-kill fault injection"


@pytest.mark.slow
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_video_failed_job_observable_with_mapped_status(
    omni_server_function,
) -> None:
    """Black-box: failed async video job should be observable via retrieve endpoint."""
    create_status, create_payload = _create_video_job_with_invalid_lora(
        omni_server_function.host,
        omni_server_function.port,
        timeout_sec=30,
    )
    assert create_status == 200, f"video create failed before async path: {create_status}, payload={create_payload!r}"
    assert "id" in create_payload, f"video create payload missing id: {create_payload!r}"
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

    # Mapping check introduced by runtime error handling PR:
    # retrieve endpoint should surface failed jobs with non-2xx HTTP status,
    # and error.code should align with that HTTP status.
    assert retrieve_status >= 400, (
        "failed retrieve should expose non-2xx status for observability, "
        f"got status={retrieve_status}, payload={retrieve_payload!r}"
    )
    if isinstance(error_obj.get("code"), int):
        assert error_obj["code"] == retrieve_status, (
            f"failed retrieve error.code should match HTTP status, status={retrieve_status}, error={error_obj!r}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_video_error_contract_consistent_async_and_sync(
    omni_server_function,
) -> None:
    """Black-box: async retrieve error and sync error both expose machine-readable contracts."""
    host = omni_server_function.host
    port = omni_server_function.port

    create_status, create_payload = _create_video_job_with_invalid_lora(host, port, timeout_sec=30)
    assert create_status == 200, f"video create failed before async path: {create_status}, payload={create_payload!r}"
    video_id = str(create_payload["id"])
    retrieve_status, retrieve_payload = _poll_video_job_status(host, port, video_id, timeout_sec=180)
    assert retrieve_payload.get("status") == "failed", f"async job should fail, got payload={retrieve_payload!r}"
    async_error = _extract_openai_error(retrieve_payload)
    assert async_error is not None, f"async failed payload missing error contract: {retrieve_payload!r}"
    assert retrieve_status >= 400, f"async failed retrieve should be non-2xx, got {retrieve_status}"

    sync_resp = requests.post(
        f"http://{host}:{port}/v1/videos/sync",
        data={
            "prompt": "sync invalid lora",
            "lora": '{"name": "bad-lora"}',
        },
        headers={"Accept": "application/json"},
        timeout=60,
    )
    assert sync_resp.status_code >= 400, f"sync invalid-lora should fail, got {sync_resp.status_code}"
    try:
        sync_payload = sync_resp.json()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"sync failure payload not json: {exc}, body={sync_resp.text[:300]!r}")
    assert isinstance(sync_payload, dict), f"sync failure payload must be object: {sync_payload!r}"
    sync_error = _extract_openai_error(sync_payload)
    assert sync_error is not None, f"sync failed payload missing error contract: {sync_payload!r}"


@pytest.mark.slow
@pytest.mark.skipif(
    current_omni_platform.is_rocm() or current_omni_platform.is_xpu(),
    reason="CUDA sidecar OOM injection is CUDA-only for phase-1",
)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_video_oom_recovers_after_fault_removed(
    omni_server_function,
) -> None:
    """Black-box: after removing transient OOM pressure, health and video admission recover."""
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
    create_url = f"http://{host}:{port}/v1/videos"

    failure_observed = False
    try:
        for _ in range(3):
            try:
                response = requests.post(
                    create_url,
                    data={
                        "prompt": "oom recover probe",
                        "width": "512",
                        "height": "512",
                        "fps": "8",
                        "num_frames": "8",
                        "num_inference_steps": "4",
                    },
                    headers={"Accept": "application/json"},
                    timeout=25,
                )
                if response.status_code >= 500:
                    failure_observed = True
                    break
            except Exception:
                failure_observed = True
                break
            time.sleep(1.0)
    finally:
        stop_gpu_oom_hogs(handle)

    assert failure_observed, "expected at least one video request failure while OOM pressure is active"

    recovery_deadline = time.monotonic() + 90.0
    while time.monotonic() < recovery_deadline:
        try:
            status, _ = _get_health_raw(host, port, timeout_sec=5)
            if status == 200:
                break
        except Exception:
            pass
        time.sleep(1.0)
    else:
        pytest.fail("wan22 server did not recover to healthy state after OOM pressure was removed")

    recovery_resp = requests.post(
        create_url,
        data={
            "prompt": "post-recovery admission check",
            "width": "512",
            "height": "512",
            "fps": "8",
            "num_frames": "8",
            "num_inference_steps": "4",
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    assert recovery_resp.status_code == 200, (
        "post-recovery /v1/videos admission should succeed, "
        f"got status={recovery_resp.status_code}, body={recovery_resp.text[:300]!r}"
    )
    recovery_payload = recovery_resp.json()
    assert "id" in recovery_payload, f"post-recovery create payload missing id: {recovery_payload!r}"
