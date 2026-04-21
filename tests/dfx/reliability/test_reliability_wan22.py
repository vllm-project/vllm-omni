"""
Wan2.2 reliability integration tests.
"""

from __future__ import annotations

import concurrent.futures
import http.client
import os
import time
from pathlib import Path
from typing import Any

import pytest

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
        futures = [executor.submit(openai_client_function.send_video_diffusion_request, request_config, 1) for _ in range(3)]
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
