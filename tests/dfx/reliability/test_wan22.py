# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Wan2.2 reliability scenarios."""

from __future__ import annotations

import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from tests.dfx.conftest import resolve_oom_device_spec
from tests.dfx.reliability.helpers import FaultModel, inject_gpu_oom, scenario_params, stop_gpu_oom_hogs
from tests.dfx.reliability.scenarios import (
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
from tests.helpers.assertions import assert_http_error
from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image

_VIDEO_FAST_FAIL_HTTP_TIMEOUT_SEC = 30
_STAGE_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "e2e" / "stage_configs"


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


def _i2v_create_request(prompt: str, *, timeout: float) -> dict[str, Any]:
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


class Wan22(FaultModel):
    model_id = "wan22"
    res = {"cuda": "H100", "rocm": "MI325"}
    num_cards = 1
    scenarios = [
        {
            "test_name": "wan22_i2v_reliability_default",
            "server_params": {"model": "Wan-AI/Wan2.2-I2V-A14B-Diffusers"},
        }
    ]
    deploy_dir = _STAGE_CONFIGS_DIR
    worker_pattern = "vLLM-Omni::"
    isolation_pattern = "vLLM-Omni::"
    hard_oom_config = {
        "target_mem_ratio": 0.95,
        "max_free_bytes": 256 * 1024 * 1024,
        "hold_seconds": 0,
        "startup_timeout_sec": 20,
        "strict": False,
    }
    recover_oom_config = hard_oom_config
    fault_error_keywords = FAULT_ERROR_KEYWORDS
    process_kill_error_keywords = (
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
    health_miss_skip = "issue#8907"
    oom_large_fail_message = "expected large /v1/videos request failure during GPU OOM injection"

    def warmup(self, client: Any, server: Any) -> None:
        client.send_video_diffusion_request(_video_request_config(), request_num=1)

    def assert_fast_fail(self, client: Any, server: Any) -> None:
        payload = {
            "prompt": "post-fault fast-fail check",
            "width": "512",
            "height": "512",
            "fps": "8",
            "num_frames": "8",
            "num_inference_steps": "4",
        }
        client.send_videos_create_http_request(
            {"data": payload, "timeout": _VIDEO_FAST_FAIL_HTTP_TIMEOUT_SEC},
            err_code=500,
        )

    def oom_large_probe(self, client: Any, server: Any) -> None:
        client.send_video_diffusion_request(_video_request_config(), request_num=1)

    def run_oom_recovery(self, server: Any, client: Any) -> None:
        _run_oom_recovery(server, client)


OOM_INJECTION_CONFIG = Wan22.hard_oom_config


def _deploy_config_path(server: Any) -> str | None:
    """Deploy YAML from the live server. ``OmniServer`` keeps it in ``serve_args``."""
    stage_config_path = getattr(server, "stage_config_path", None)
    if stage_config_path:
        return str(stage_config_path)
    args: list[str] = getattr(server, "serve_args", [])
    for index, arg in enumerate(args):
        if arg == "--deploy-config" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--deploy-config="):
            return arg.split("=", 1)[1]
    return None


def _run_oom_recovery(omni_server_function: Any, online_client_function: Any) -> None:
    """Black-box: under OOM pressure an async video job fails; after the hog stops, a new job completes.

    Wan2.2 I2V requires ``input_reference``. ``POST /v1/videos`` returns HTTP 200
    when the job is queued. A failed job's GET uses ``error.code`` as the HTTP
    status, and ``assert_http_error(..., err_code=500)`` checks that status.
    """
    device_spec = resolve_oom_device_spec(OOM_INJECTION_CONFIG, _deploy_config_path(omni_server_function))
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


MODEL = Wan22()
_WORKER_KILL_SKIP = "issue#8907"

pytestmark = [
    pytest.mark.slow,
    *MODEL.extra_marks,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


@scenario_params(MODEL, GpuOomLarge)
def test_reliability_fault_gpu_oom_large_request_failure(omni_server_function, online_client_function) -> None:
    GpuOomLarge().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, GpuOomRecovery)
def test_reliability_fault_gpu_oom_recovers_after_fault_removed(omni_server_function, online_client_function) -> None:
    GpuOomRecovery().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, ProcessKillWorkerNoLoad, skip_reason=_WORKER_KILL_SKIP)
def test_reliability_fault_process_kill_worker_no_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    """After a quiet worker kill, the next request returns HTTP 500, then concurrent requests fail."""
    ProcessKillWorkerNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillWorkerWithLoad, skip_reason=_WORKER_KILL_SKIP)
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


@scenario_params(MODEL, PartialStageIsolation, skip_reason=_WORKER_KILL_SKIP)
def test_reliability_fault_partial_stage_isolation(omni_server_function, online_client_function) -> None:
    """Kill one stage or one worker. Requests bound to it fail, and one SIGTERM still exits the API."""
    PartialStageIsolation().run(MODEL, omni_server_function, online_client_function)
