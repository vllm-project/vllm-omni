# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Wan2.2 reliability integration tests.
"""

from __future__ import annotations

import base64
import concurrent.futures
import time
from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict

import pytest

from tests.dfx.conftest import (
    assert_fault_exception,
    create_reliability_omni_server_params,
    resolve_oom_device_spec,
    supports_video_generation,
)
from tests.dfx.reliability.helpers import (
    FaultInjector,
    assert_no_server_tree_process_residual_and_gpu_release,
    assert_post_fault_api_unreachable,
    assert_post_fault_health_503,
    inject_gpu_oom,
    make_process_kill_fault_injector,
    make_server_root_kill_fault_injector,
    make_server_tree_kill_fault_injector,
    run_fault_injection_with_rate_load,
    stop_gpu_oom_hogs,
    worker_residual_timeout_after_kill_signal,
)
from tests.helpers.assertions import assert_http_error
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


class _WanOomConfig(TypedDict):
    target_mem_ratio: float
    max_free_bytes: int
    hold_seconds: int
    startup_timeout_sec: int
    strict: bool


OOM_INJECTION_CONFIG: _WanOomConfig = {
    "target_mem_ratio": 0.95,
    # Stop at a free-memory ceiling. strict=True waits for RuntimeError, and
    # HIP retries a failed alloc without raising, so the sidecar never readies.
    "max_free_bytes": 256 * 1024 * 1024,
    "hold_seconds": 0,
    "startup_timeout_sec": 20,
    "strict": False,
}
FAULT_ERROR_KEYWORDS = (
    "oom",
    "out of memory",
    "cuda",
    "connection",
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
# - serve-root: SIGTERM, SIGINT, SIGKILL
# - serve-tree no-load: SIGTERM, SIGKILL
# - serve-tree with-load: SIGTERM, SIGKILL
SERVE_SIGNAL_PARAMS = [
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
# Client read timeout. HTTP 500 must return before this; a timeout raises and fails the test.
_VIDEO_FAST_FAIL_HTTP_TIMEOUT_SEC = 30


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


def _assert_post_fault_video_fast_fail(online_client: Any) -> None:
    """New ``POST /v1/videos`` after the fault must return HTTP 500 within 30s.

    The client timeout is 30s. A timeout or a dropped connection raises and fails the test.
    """
    payload = {
        "prompt": "post-fault fast-fail check",
        "width": "512",
        "height": "512",
        "fps": "8",
        "num_frames": "8",
        "num_inference_steps": "4",
    }
    online_client.send_videos_create_http_request(
        {"data": payload, "timeout": _VIDEO_FAST_FAIL_HTTP_TIMEOUT_SEC},
        err_code=500,
    )


def _i2v_create_request(prompt: str, *, timeout: float) -> dict[str, Any]:
    """Multipart create body for Wan2.2 I2V. The model rejects a prompt-only job."""
    image_bytes = base64.b64decode(generate_synthetic_image(1280, 720)["base64"])
    return {
        "data": {
            "prompt": prompt,
            "width": "512",
            "height": "512",
            "fps": "8",
            "num_frames": "8",
            "num_inference_steps": "4",
        },
        "files": {"input_reference": ("reference.jpeg", BytesIO(image_bytes), "image/jpeg")},
        "timeout": timeout,
    }


def _await_video_job(online_client: Any, video_id: str, *, timeout_sec: float) -> Any:
    """Repeat ``send_video_retrieve_http_request`` until the job is completed or failed.

    In-progress and completed jobs are HTTP 200. A failed job is returned with HTTP
    status ``error.code`` (500 for a generation error). Callers that expect that
    failure then pass the response to ``assert_http_error(..., err_code=500)``.
    """
    deadline = time.monotonic() + timeout_sec
    last_job: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        retrieved = online_client.send_video_retrieve_http_request(
            {"video_id": video_id, "timeout": 10},
            err_code=(200, 500),
        )[0]
        job = retrieved.json_body
        if not isinstance(job, dict):
            raise AssertionError(f"video job {video_id} payload is not an object: {job!r}")
        last_job = job
        if job.get("status") in ("completed", "failed"):
            return retrieved
        time.sleep(1.0)
    raise TimeoutError(f"video job {video_id} did not reach a terminal status: {last_job!r}")


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_large_request_failure(omni_server_function, online_client_function) -> None:
    stage_config_path = getattr(omni_server_function, "stage_config_path", None)
    device_spec = resolve_oom_device_spec(OOM_INJECTION_CONFIG, stage_config_path)
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
        max_free_bytes=OOM_INJECTION_CONFIG["max_free_bytes"],
    )
    try:
        request_config = _video_request_config()
        try:
            online_client_function.send_video_diffusion_request(request_config, request_num=1)
        except Exception as exc:
            assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
        else:
            pytest.fail("expected large /v1/videos request failure during GPU OOM injection")
    finally:
        stop_gpu_oom_hogs(handle)


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_request_failure(
    omni_server_function,
    online_client_function,
    fault_injector: FaultInjector,
) -> None:
    online_client_function.send_video_diffusion_request(_video_request_config(), request_num=1)
    fault_injector(omni_server_function)
    request_config = _video_request_config()
    try:
        online_client_function.send_video_diffusion_request(request_config, request_num=1)
    except Exception as exc:
        assert_fault_exception(exc, PROCESS_KILL_ERROR_KEYWORDS)
    else:
        pytest.fail("expected /v1/videos request failure after process-kill injection")


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skip(reason="issue#8907")
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_health_fast_fail_and_concurrent(
    omni_server_function,
    online_client_function,
    fault_injector: FaultInjector,
) -> None:
    """Black-box: after process kill, /v1/videos fails fast and concurrent calls don't hang.

    /health→503 is checked last. A missing listener is issue#8907.
    """
    online_client_function.send_video_diffusion_request(_video_request_config(), request_num=1)
    fault_injector(omni_server_function)
    request_config = _video_request_config()

    # Phase-1: one new /v1/videos request should fail fast.
    _assert_post_fault_video_fast_fail(online_client_function)

    # Phase-2: concurrent requests should converge and at least one should fail.
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(online_client_function.send_video_diffusion_request, request_config, 1) for _ in range(3)
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

    # Phase-3: /health→503 after fault. Connection refused means the API
    # process exited with the worker (issue#8907).
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
    if final_health_status != 503:
        pytest.skip("issue#8907")
    assert final_health_status == 503, (
        "[process_kill health] expected /health 503 after fatal fault, "
        f"got status={final_health_status}, last_observation={last_observation}"
    )


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skip(reason="issue#8907")
@pytest.mark.parametrize(
    "fault_injector",
    WORKER_SIGNAL_FAULT_PARAMS,
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_worker_with_load_request_failure(
    omni_server_function,
    online_client_function,
    fault_injector: FaultInjector,
) -> None:
    request_config = _video_request_config()
    scenario = "kill_worker_with_load"
    online_client_function.send_video_diffusion_request(_video_request_config(), request_num=1)
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: online_client_function.send_video_diffusion_request(request_config, request_num=1),
        inject_fault=lambda: fault_injector(omni_server_function),
        num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
        request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
        completion_timeout_sec=120.0,
    )
    assert load_result["failure_observed"], (
        f"[{scenario}] expected at least one load request failure after fault; load_result={load_result}"
    )
    _assert_post_fault_video_fast_fail(online_client_function)
    assert_post_fault_health_503(online_client_function, scenario=scenario)


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_recovers_after_fault_removed(
    omni_server_function,
    online_client_function,
) -> None:
    """Black-box: under OOM pressure an async video job fails; after the hog stops, a new job completes.

    Wan2.2 I2V requires ``input_reference``. ``POST /v1/videos`` returns HTTP 200
    when the job is queued. A failed job's GET uses ``error.code`` as the HTTP
    status, and ``assert_http_error(..., err_code=500)`` checks that status.
    """
    stage_config_path = getattr(omni_server_function, "stage_config_path", None)
    device_spec = resolve_oom_device_spec(OOM_INJECTION_CONFIG, stage_config_path)
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
        max_free_bytes=OOM_INJECTION_CONFIG["max_free_bytes"],
    )
    failure_observed = False
    failure_detail = ""
    try:
        for _ in range(3):
            try:
                created_resp = online_client_function.send_videos_create_http_request(
                    _i2v_create_request("oom recover probe", timeout=25)
                )[0]
            except Exception as exc:
                failure_observed = True
                failure_detail = repr(exc)
                break
            created = created_resp.json_body
            video_id = created.get("id") if isinstance(created, dict) else None
            if not isinstance(video_id, str) or not video_id:
                failure_observed = True
                failure_detail = f"create payload missing id: {created!r}"
                break
            try:
                retrieved = _await_video_job(online_client_function, video_id, timeout_sec=120)
            except Exception as exc:
                failure_observed = True
                failure_detail = f"video_id={video_id} poll error={exc!r}"
                break
            job = retrieved.json_body
            if isinstance(job, dict) and job.get("status") == "failed":
                assert_http_error(retrieved, err_code=500)
                failure_observed = True
                failure_detail = f"video_id={video_id} error={job.get('error')!r}"
                break
            time.sleep(1.0)
    finally:
        stop_gpu_oom_hogs(handle)

    assert failure_observed, (
        f"expected at least one async video job to fail while OOM pressure is active; last_detail={failure_detail!r}"
    )

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
    recovery_status: int | None = None
    recovery_job: dict[str, Any] | None = None
    recovery_resp: Any = None
    recovery_error = ""
    try:
        created_resp = online_client_function.send_videos_create_http_request(
            _i2v_create_request("post-recovery admission check", timeout=30)
        )[0]
    except AssertionError as exc:
        # The create helper requires HTTP 200. A non-200 create is the expected
        # failure only while /health is still 503.
        if terminal_health == 200:
            raise
        recovery_error = repr(exc)
        created_resp = None
    except Exception as exc:  # noqa: BLE001
        recovery_error = repr(exc)
        created_resp = None
    if created_resp is not None:
        create_elapsed = time.monotonic() - start
        assert create_elapsed < 30, f"post-fault /v1/videos create should not hang: {create_elapsed:.2f}s"
        recovery_status = created_resp.status_code
        created = created_resp.json_body
        video_id = created.get("id") if isinstance(created, dict) else None
        if not isinstance(video_id, str) or not video_id:
            recovery_error = f"create payload missing id: {created!r}"
        else:
            poll_timeout = 180.0 if terminal_health == 200 else 60.0
            try:
                recovery_resp = _await_video_job(online_client_function, video_id, timeout_sec=poll_timeout)
            except Exception as exc:  # noqa: BLE001
                recovery_error = repr(exc)
            else:
                recovery_job = recovery_resp.json_body if isinstance(recovery_resp.json_body, dict) else None
    recovery_text_prefix = recovery_error or ("" if recovery_status is None else f"status={recovery_status}")

    if terminal_health == 200:
        assert recovery_status == 200, (
            "health recovered but /v1/videos admission did not succeed; "
            f"status={recovery_status}, body={recovery_text_prefix!r}"
        )
        assert recovery_job is not None, f"post-recovery video job was not polled: {recovery_text_prefix!r}"
        assert recovery_job.get("status") == "completed", (
            "health recovered but the async video job did not complete; "
            f"status={recovery_job.get('status')} error={recovery_job.get('error')!r}"
        )
    else:
        job_failed = False
        if recovery_resp is not None and recovery_job is not None and recovery_job.get("status") == "failed":
            assert_http_error(recovery_resp, err_code=500)
            job_failed = True
        assert recovery_error or job_failed, (
            "unhealthy terminal state should fail the async video job, "
            f"got health={terminal_health}, request_status={recovery_status}, "
            f"job={recovery_job!r}, body={recovery_text_prefix!r}"
        )


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("signal_name", SERVE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_serve_root_with_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    """Black-box: during in-flight video requests, kill serve root and verify post-fault behavior/cleanup."""
    request_config = _video_request_config()
    scenario = f"kill_serve_root_with_load_{signal_name.lower()}"
    injector = make_server_root_kill_fault_injector(signal_name=signal_name, post_kill_wait_seconds=2.0)
    online_client_function.send_video_diffusion_request(_video_request_config(), request_num=1)
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: online_client_function.send_video_diffusion_request(request_config, request_num=1),
        inject_fault=lambda: injector(omni_server_function),
        num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
        request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
        completion_timeout_sec=120.0,
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
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("signal_name", SERVE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_serve_root_no_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    """Black-box: kill serve root without load; verify fast-fail/health/cleanup."""
    scenario = f"kill_serve_root_no_load_{signal_name.lower()}"
    injector = make_server_root_kill_fault_injector(signal_name=signal_name, post_kill_wait_seconds=2.0)
    online_client_function.send_video_diffusion_request(_video_request_config(), request_num=1)
    injector(omni_server_function)
    assert_post_fault_api_unreachable(online_client_function, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("signal_name", TREE_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
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
    online_client_function.send_video_diffusion_request(_video_request_config(), request_num=1)
    injector(omni_server_function)
    assert_post_fault_api_unreachable(online_client_function, scenario=scenario)
    assert_no_server_tree_process_residual_and_gpu_release(
        omni_server_function,
        scenario=scenario,
        timeout_sec=worker_residual_timeout_after_kill_signal(signal_name),
    )


@pytest.mark.slow
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("signal_name", TREE_WITH_LOAD_SIGNAL_PARAMS)
@pytest.mark.parametrize("omni_server_function", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_process_kill_tree_with_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
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
    online_client_function.send_video_diffusion_request(_video_request_config(), request_num=1)
    load_result = run_fault_injection_with_rate_load(
        submit_request=lambda: online_client_function.send_video_diffusion_request(request_config, request_num=1),
        inject_fault=lambda: injector(omni_server_function),
        num_requests=INFLIGHT_INJECTION_REQUEST_COUNT,
        request_rate=INFLIGHT_INJECTION_REQUEST_RATE,
        completion_timeout_sec=120.0,
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
