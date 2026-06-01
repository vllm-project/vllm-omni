"""
Wan2.2 reliability integration tests.
"""

from __future__ import annotations

import concurrent.futures
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
from tests.dfx.reliability.helpers import (
    FaultInjector,
    assert_no_server_tree_process_residual_and_gpu_release,
    get_health_raw,
    inject_gpu_oom,
    make_process_kill_fault_injector,
    make_server_root_kill_fault_injector,
    make_server_tree_kill_fault_injector,
    run_fault_injection_with_rate_load,
    stop_gpu_oom_hogs,
    worker_residual_timeout_after_kill_signal,
)
from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_image

RELIABILITY_SCENARIOS: list[dict[str, Any]] = [
    {
        "test_name": "wan22_i2v_reliability_default",
        "server_params": {
            "model": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
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
# RFC#2366 signal x target matrix (aligned with test_reliability_qwen3_omni.py):
# - worker: SIGTERM, SIGKILL
# - serve-root: SIGTERM, SIGINT; SIGKILL skipped (issue#3725)
# - serve-tree no-load: SIGTERM, SIGKILL
# - serve-tree with-load: SIGTERM, SIGKILL
_SERVE_ROOT_SIGKILL_SKIP = pytest.mark.skip(reason="issue#3725")
SERVE_SIGNAL_PARAMS = [
    pytest.param("SIGTERM", id="sigterm"),
    pytest.param("SIGINT", id="sigint"),
    pytest.param("SIGKILL", id="sigkill", marks=_SERVE_ROOT_SIGKILL_SKIP),
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
            grep_patterns="vLLM-Omni::",
            signal_name="SIGTERM",
            limit=1,
            post_kill_wait_seconds=2.0,
        ),
        id="runtime_process_chain_sigterm",
    ),
    pytest.param(
        make_process_kill_fault_injector(
            grep_patterns="vLLM-Omni::",
            signal_name="SIGKILL",
            limit=1,
            post_kill_wait_seconds=2.0,
        ),
        id="runtime_process_chain_sigkill",
    ),
]

WAN_PARAMS = create_reliability_omni_server_params(RELIABILITY_SCENARIOS, E2E_STAGE_CONFIGS_DIR)
DIFFUSION_VIDEO_PARAMS = [param for param in WAN_PARAMS if supports_video_generation(param.model)]
INFLIGHT_INJECTION_REQUEST_RATE = 0.3
INFLIGHT_INJECTION_REQUEST_COUNT = 10


def _video_request_config() -> dict[str, Any]:
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"
    return {
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


def _assert_post_fault_video_fast_fail(host: str, port: int, *, scenario: str) -> None:
    url = f"http://{host}:{port}/v1/videos"
    payload = {
        "prompt": "post-fault fast-fail check",
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
        assert elapsed < 15, f"[{scenario} fast_fail] /v1/videos did not fail fast: {elapsed:.2f}s"
        assert response.status_code >= 500, (
            f"[{scenario} fast_fail] expected server-side error after fault, "
            f"got status={response.status_code}, body={response.text[:200]!r}"
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
@hardware_test(res={"cuda": "H100"}, num_cards=1)
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
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_request_failure(
    omni_server_after_fault_function,
    openai_client_function,
) -> None:
    request_config = _video_request_config()
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
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_health_fast_fail_and_concurrent(
    omni_server_after_fault_function,
    openai_client_function,
) -> None:
    """Black-box: after process kill, /v1/videos fails fast and concurrent calls don't hang.

    /health→503 is checked last: optional ``pytest.skip`` then ``assert`` (drop the ``if`` to harden).
    """
    host = omni_server_after_fault_function.host
    port = omni_server_after_fault_function.port
    url = f"http://{host}:{port}/v1/videos"
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

    # Phase-1: one new /v1/videos request should fail fast.
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
        assert elapsed < 15, f"[process_kill fast_fail] /v1/videos did not fail fast after fault: {elapsed:.2f}s"
        assert response.status_code >= 500, (
            "[process_kill fast_fail] expected server-side error after fatal fault, "
            f"got status={response.status_code}, body={response.text[:200]!r}"
        )
    except Exception:
        elapsed = time.monotonic() - start
        assert elapsed < 15, f"[process_kill fast_fail] /v1/videos exception was too slow after fault: {elapsed:.2f}s"

    # Phase-2: concurrent requests should converge and at least one should fail.
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
    assert not pending, f"[process_kill concurrent] some fault-time video requests hung: pending={len(pending)}"
    assert elapsed < 40, f"[process_kill concurrent] fault-time video request convergence is too slow: {elapsed:.2f}s"

    fault_observed = False
    conc_debug: list[Any] = []
    for future in done:
        try:
            future.result()
            conc_debug.append("ok")
        except Exception as exc:
            fault_observed = True
            conc_debug.append(repr(exc))
            assert_fault_exception(exc, PROCESS_KILL_ERROR_KEYWORDS)
    assert fault_observed, (
        "[process_kill concurrent] expected at least one /v1/videos request to fail after fault; "
        f"conc_debug={conc_debug}"
    )

    # Phase-3: /health→503 after fault. Optional skip today; remove the ``if`` later to harden.
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
    if final_health_status != 503:
        pytest.skip("issue#3050")
    assert final_health_status == 503, (
        "[process_kill health] expected /health 503 after fatal fault, "
        f"got status={final_health_status}, last_observation={last_observation}"
    )


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_worker_with_load_request_failure(
    omni_server_function,
    openai_client_function,
    fault_injector: FaultInjector,
) -> None:
    request_config = _video_request_config()
    scenario = "kill_worker_with_load"
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: openai_client_function.send_video_diffusion_request(request_config, request_num=1),
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
    _assert_post_fault_video_fast_fail(host, port, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)


@pytest.mark.slow
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skip(reason="issue#2327")
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_video_oom_recovers_after_fault_removed(
    omni_server_function,
) -> None:
    """Black-box: after removing OOM pressure, service converges to a terminal state and requests do not hang."""
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
    terminal_health: int | None = None
    unreachable_streak = 0
    last_health_exc: BaseException | None = None
    while time.monotonic() < recovery_deadline:
        try:
            status, _ = get_health_raw(host, port, timeout_sec=5)
            unreachable_streak = 0
            if status in (200, 503):
                terminal_health = status
                break
        except Exception as exc:  # noqa: BLE001
            last_health_exc = exc
            msg = str(exc).lower()
            if "connection refused" in msg or "actively refused" in msg:
                unreachable_streak += 1
                if unreachable_streak >= 5:
                    pytest.fail(
                        "after OOM sidecar stopped, /health is unreachable repeatedly; "
                        "the APIServer process likely exited unexpectedly "
                        f"(last_exc={last_health_exc!r})"
                    )
            else:
                unreachable_streak = 0
        time.sleep(1.0)
    else:
        pytest.fail(
            "wan22 server did not converge to terminal health (200/503) after OOM pressure was removed; "
            f"last_health_exc={last_health_exc!r}"
        )

    assert terminal_health is not None
    start = time.monotonic()
    recovery_resp: requests.Response | None = None
    try:
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
        recovery_status = recovery_resp.status_code
        recovery_text_prefix = recovery_resp.text[:300]
    except Exception as exc:  # noqa: BLE001
        recovery_status = None
        recovery_text_prefix = repr(exc)
    elapsed = time.monotonic() - start
    assert elapsed < 30, f"post-fault /v1/videos should not hang after OOM removal: {elapsed:.2f}s"

    if terminal_health == 200:
        assert recovery_status == 200, (
            "health recovered but /v1/videos admission did not succeed; "
            f"status={recovery_status}, body={recovery_text_prefix!r}"
        )
        assert recovery_resp is not None
        recovery_payload = recovery_resp.json()
        assert "id" in recovery_payload, f"post-recovery create payload missing id: {recovery_payload!r}"
    else:
        assert recovery_status is None or recovery_status >= 500, (
            "unhealthy terminal state should fail fast on /v1/videos, "
            f"got health={terminal_health}, request_status={recovery_status}, body={recovery_text_prefix!r}"
        )


@pytest.mark.slow
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize("signal_name", SERVE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_serve_root_with_load_fast_fail_and_cleanup(
    omni_server_function,
    openai_client_function,
    signal_name: str,
) -> None:
    """Black-box: during in-flight video requests, kill serve root and verify post-fault behavior/cleanup."""
    request_config = _video_request_config()
    scenario = f"kill_serve_root_with_load_{signal_name.lower()}"
    injector = make_server_root_kill_fault_injector(signal_name=signal_name, post_kill_wait_seconds=2.0)
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: openai_client_function.send_video_diffusion_request(request_config, request_num=1),
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
    _assert_post_fault_video_fast_fail(host, port, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize("signal_name", SERVE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_serve_root_no_load_fast_fail_and_cleanup(
    omni_server_function,
    signal_name: str,
) -> None:
    """Black-box: kill serve root without load; verify fast-fail/health/cleanup."""
    scenario = f"kill_serve_root_no_load_{signal_name.lower()}"
    injector = make_server_root_kill_fault_injector(signal_name=signal_name, post_kill_wait_seconds=2.0)
    injector(omni_server_function)
    host = omni_server_function.host
    port = omni_server_function.port
    _assert_post_fault_video_fast_fail(host, port, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize("signal_name", TREE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_tree_no_load_fast_fail_and_cleanup(
    omni_server_function,
    signal_name: str,
) -> None:
    """Black-box: kill server tree without load; verify fast-fail/health/cleanup."""
    scenario = f"kill_serve_tree_no_load_{signal_name.lower()}"
    injector = make_server_tree_kill_fault_injector(
        signal_name=signal_name,
        post_kill_wait_seconds=2.0,
        inter_kill_wait_seconds=0.1,
    )
    injector(omni_server_function)
    host = omni_server_function.host
    port = omni_server_function.port
    _assert_post_fault_video_fast_fail(host, port, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize("signal_name", TREE_WITH_LOAD_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_tree_with_load_fast_fail_and_cleanup(
    omni_server_function,
    openai_client_function,
    signal_name: str,
) -> None:
    """Black-box: during in-flight video requests, kill server process tree and verify post-fault behavior/cleanup."""
    request_config = _video_request_config()
    scenario = f"kill_serve_tree_with_load_{signal_name.lower()}"
    injector = make_server_tree_kill_fault_injector(
        signal_name=signal_name,
        post_kill_wait_seconds=2.0,
        inter_kill_wait_seconds=0.1,
    )
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: openai_client_function.send_video_diffusion_request(request_config, request_num=1),
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
    _assert_post_fault_video_fast_fail(host, port, scenario=scenario)
    _assert_post_fault_health_terminal(host, port, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )
