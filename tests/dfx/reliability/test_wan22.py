# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Wan2.2 reliability scenarios."""

from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import Any

import pytest

from tests.dfx.conftest import resolve_oom_device_spec
from tests.dfx.reliability.helpers import (
    FaultModel,
    _deploy_config_path,
    extract_openai_error_contract_from_payload,
    inject_gpu_oom,
    launch_scenario_params,
    scenario_params,
    sigterm_serve_root_and_assert_cleanup,
    stop_gpu_oom_hogs,
)
from tests.dfx.reliability.scenarios import (
    FullCrashAtLaunch,
    GpuOomAtLaunch,
    GpuOomUnderLoad,
    GracefulShutdown,
    PartialCrashAtExit,
    PartialCrashAtLaunch,
    ProcessKillServeRootNoLoad,
    ProcessKillServeRootWithLoad,
    ProcessKillTreeNoLoad,
    ProcessKillTreeWithLoad,
    ProcessKillWorkerNoLoad,
    ProcessKillWorkerWithLoad,
    ServeRootCrashAtLaunch,
)
from tests.dfx.reliability.scenarios.oom import GPU_OOM_MARKERS, _looks_like_server_unreachable
from tests.helpers.assertions import assert_http_error
from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image

_VIDEO_FAST_FAIL_HTTP_TIMEOUT_SEC = 30


def _video_request_config(
    *,
    width: int = 512,
    height: int = 512,
    num_frames: int = 8,
    num_inference_steps: int = 4,
    reference_size: tuple[int, int] = (1280, 720),
) -> dict[str, Any]:
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(*reference_size)['base64']}"
    return {
        "form_data": {
            "prompt": "Generate a realistic road-driving video with camera motion.",
            "width": width,
            "height": height,
            "fps": 8,
            "num_frames": num_frames,
            "guidance_scale": 1.0,
            "flow_shift": 5.0,
            "num_inference_steps": num_inference_steps,
            "seed": 42,
        },
        "image_reference": image_data_url,
        "stream": False,
    }


def _i2v_create_request(prompt: str, *, timeout: float) -> dict[str, Any]:
    """Larger OOM/recovery job: increase spatial and temporal allocation demand."""
    image_bytes = base64.b64decode(generate_synthetic_image(768, 512)["base64"])
    return {
        "data": {
            "prompt": prompt,
            "width": "768",
            "height": "512",
            "fps": "8",
            "num_frames": "33",
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
            {"video_id": video_id, "timeout": min(10, max(0.001, deadline - time.monotonic()))},
            err_code=(200, 500, 503),
        )[0]
        job = retrieved.json_body
        if not isinstance(job, dict):
            raise AssertionError(f"video job {video_id} payload is not an object: {job!r}")
        last_job = job
        if job.get("status") in ("completed", "failed"):
            return retrieved
        assert retrieved.status_code == 200, f"job lookup failed without terminal status: {job!r}"
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"video job {video_id} did not reach a terminal status: {last_job!r}")


def _create_and_wait_video_job(
    client: Any, request: dict[str, Any], *, timeout_sec: float, err_code: int | tuple[int, ...]
) -> Any:
    """Return admission errors or a terminal job under one overall deadline."""
    deadline = time.monotonic() + timeout_sec
    request = dict(request, timeout=min(30, timeout_sec))
    response = client.send_videos_create_http_request(request, err_code=err_code)[0]
    if response.status_code == 200:
        body = response.json_body
        video_id = body.get("id") if isinstance(body, dict) else None
        assert isinstance(video_id, str) and video_id, f"video admission missing job id: {body!r}"
        remaining = deadline - time.monotonic()
        assert remaining > 0, "video admission exceeded the job deadline"
        try:
            response = _await_video_job(client, video_id, timeout_sec=remaining)
        except TimeoutError as exc:
            raise AssertionError("video job did not reach terminal status within deadline") from exc
    assert time.monotonic() < deadline, "video job exceeded its deadline"
    return response


def _assert_video_preflight(client: Any, request_config: dict[str, Any]) -> None:
    """Create valid I2V input and require a completed job before injection."""
    image_bytes = base64.b64decode(request_config["image_reference"].split(",", 1)[1])
    result = _create_and_wait_video_job(
        client,
        {
            "data": {key: str(value) for key, value in request_config["form_data"].items()},
            "files": {"input_reference": ("reference.jpeg", BytesIO(image_bytes), "image/jpeg")},
            "timeout": 30,
        },
        timeout_sec=180,
        err_code=200,
    )
    assert result.status_code == 200, f"preflight video failed: {result.json_body!r}"
    assert result.json_body.get("status") == "completed", f"preflight video did not complete: {result.json_body!r}"


class Wan22(FaultModel):
    model_id = "wan22"
    res = {"cuda": "H100", "rocm": "MI325"}
    num_cards = 2
    scenarios = [
        {
            "test_name": "wan22_i2v_reliability_default",
            "server_params": {
                "model": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "stage_init_timeout": 1800,
                "init_timeout": 2100,
                "server_args": ["--usp", "2", "--vae-patch-parallel-size", "2", "--use-hsdp", "--vae-use-slicing"],
            },
        }
    ]
    worker_pattern = "vLLM-Omni::"
    isolation_pattern = "vLLM-Omni::"
    hard_oom_config = {
        "device": "0,1",
        "target_mem_ratio": 0.95,
        "max_free_bytes": 256 * 1024 * 1024,
        "hold_seconds": 0,
        "startup_timeout_sec": 120,
        "strict": False,
    }
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

    def warmup(self, client: Any, server: Any) -> None:
        _assert_video_preflight(client, _video_request_config())

    def submit_worker_load(self, client: Any, server: Any) -> None:
        client.send_video_diffusion_request(_video_request_config(), request_num=1)

    def submit_serve_load(self, client: Any, server: Any) -> None:
        self.submit_worker_load(client, server)

    def assert_fast_fail(self, client: Any, server: Any) -> None:
        _assert_video_worker_failure(client)

    def submit_concurrent(self, client: Any, server: Any) -> Any:
        return _assert_video_worker_failure(client)


def _assert_video_worker_failure(client: Any) -> Any:
    """Validate a real I2V job fails, rather than mistaking admission for success."""
    image_bytes = base64.b64decode(generate_synthetic_image(512, 512)["base64"])
    response = _create_and_wait_video_job(
        client,
        {
            "data": {
                "prompt": "post-fault fast-fail check",
                "width": "512",
                "height": "512",
                "fps": "8",
                "num_frames": "8",
                "num_inference_steps": "4",
            },
            "files": {"input_reference": ("reference.jpeg", BytesIO(image_bytes), "image/jpeg")},
            "timeout": _VIDEO_FAST_FAIL_HTTP_TIMEOUT_SEC,
        },
        timeout_sec=_VIDEO_FAST_FAIL_HTTP_TIMEOUT_SEC,
        err_code=(200, 500, 503),
    )
    assert_http_error(response, err_code=(500, 503))
    contract = extract_openai_error_contract_from_payload(response.json_body)
    assert contract is not None, f"missing video failure error contract: {response.json_body!r}"
    error = str(contract).lower()
    assert not any(marker in error for marker in ("no image", "input error", "invalid", "validation")), (
        f"post-fault probe failed because of invalid input: {contract!r}"
    )
    # Public 5xx errors may hide engine details behind a generic message.
    # The scenario proves worker death separately; do not require that detail
    # to be exposed in the HTTP error text.
    return response


def _run_oom_at_load(server: Any, client: Any) -> None:
    """One pressured async video job, actual OOM evidence, then recovery checks."""
    # Prepare media before pressure, and prove normal I2V service first.
    fault_request = _i2v_create_request("single OOM fault probe", timeout=30)
    _assert_video_preflight(
        client,
        _video_request_config(width=256, height=256, num_frames=5, num_inference_steps=2, reference_size=(256, 256)),
    )
    stderr_before = set(getattr(server, "_server_error_lines", ()))
    config = MODEL.hard_oom_config
    handle = inject_gpu_oom(
        device=resolve_oom_device_spec(config, _deploy_config_path(server)),
        target_mem_ratio=config["target_mem_ratio"],
        hold_seconds=config["hold_seconds"],
        startup_timeout_sec=config["startup_timeout_sec"],
        strict=config["strict"],
        max_free_bytes=config.get("max_free_bytes"),
    )
    detail: Any = None
    try:
        started = time.monotonic()
        print("[wan-oom] preflight passed; pressure ready; submitting one fault job", flush=True)
        try:
            result = _create_and_wait_video_job(client, fault_request, timeout_sec=180, err_code=(200, 500, 503))
        except Exception as exc:
            assert _looks_like_server_unreachable(exc), f"unexpected OOM job failure: {exc!r}"
            detail = repr(exc)
        else:
            detail = result.json_body
            print(f"[wan-oom] fault status={result.status_code}, body={str(detail)[:4096]}", flush=True)
            assert isinstance(detail, dict) and detail.get("status") != "completed", (
                f"OOM fault job must fail, got {detail!r}"
            )
            assert_http_error(result, err_code=(500, 503))
            assert extract_openai_error_contract_from_payload(detail) is not None, (
                f"OOM response lacks an error object: {detail!r}"
            )
        assert time.monotonic() - started < 180, "OOM job did not fail within 180 seconds"
        stderr_delta = "".join(line for line in getattr(server, "_server_error_lines", ()) if line not in stderr_before)
        evidence = str(detail) + stderr_delta
        assert any(marker in evidence.lower() for marker in GPU_OOM_MARKERS), (
            f"video job failed without explicit GPU OOM evidence; detail={str(detail)[:4096]}, "
            f"stderr_tail={stderr_delta[-4096:]!r}"
        )
    finally:
        stop_gpu_oom_hogs(handle)
        print("[wan-oom] memory pressure removed", flush=True)

    try:
        health = client.send_health_http_request({"timeout": 5}, err_code=(200, 503))[0].status_code
    except Exception as exc:
        assert _looks_like_server_unreachable(exc), f"unexpected post-OOM health failure: {exc!r}"
        print("[wan-oom] API unreachable; whole-instance restart required", flush=True)
        return
    if health == 503:
        # Later selected serial cases must start a fresh instance and pass their
        # business preflight to provide suite-level restart coverage.
        print("[wan-oom] health=503; whole-instance restart required", flush=True)
        return
    assert health == 200, f"unexpected post-OOM health: {health}"
    recovered = _create_and_wait_video_job(
        client, _i2v_create_request("post-OOM recovery", timeout=30), timeout_sec=180, err_code=200
    )
    job = recovered.json_body
    assert recovered.status_code == 200 and isinstance(job, dict) and job.get("status") == "completed", (
        f"recovery video did not complete: {job!r}"
    )


MODEL = Wan22()

pytestmark = [
    pytest.mark.slow,
    *MODEL.extra_marks,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


@scenario_params(MODEL, GpuOomUnderLoad)
def test_gpu_oom_at_load(omni_server_function, online_client_function) -> None:
    """One OOM injection and fault job; health 200 requires a completed recovery job.

    A 503/unreachable API requires restart; later serial cases cover recovery.
    """
    _run_oom_at_load(omni_server_function, online_client_function)


@scenario_params(MODEL, ProcessKillWorkerNoLoad)
def test_stage_crash_at_idle(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    """Kill a worker, check failed requests and health, then require SIGTERM cleanup."""
    ProcessKillWorkerNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)
    sigterm_serve_root_and_assert_cleanup(omni_server_function, scenario="stage_crash_at_idle")


@scenario_params(MODEL, ProcessKillWorkerWithLoad)
def test_stage_crash_at_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillWorkerWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillServeRootNoLoad)
def test_serve_root_crash_at_idle(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillServeRootNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillServeRootWithLoad)
def test_serve_root_crash_at_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillServeRootWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillTreeNoLoad)
def test_process_tree_crash_at_idle(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillTreeNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillTreeWithLoad)
def test_process_tree_crash_at_load(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillTreeWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@launch_scenario_params(MODEL, GpuOomAtLaunch)
def test_gpu_oom_at_launch(launch_params) -> None:
    """Exhaust GPU memory before startup. The failure is observable, and a later start serves."""
    GpuOomAtLaunch().run(MODEL, launch_params)


@launch_scenario_params(MODEL, PartialCrashAtLaunch)
def test_stage_crash_at_launch(launch_params) -> None:
    """Kill a stage before readiness; require failure and cleanup, with recovery covered by later serial cases."""
    PartialCrashAtLaunch().run(MODEL, launch_params)


@launch_scenario_params(MODEL, FullCrashAtLaunch)
def test_process_tree_crash_at_launch(launch_params) -> None:
    """Kill the startup process tree; later serial cases cover fresh startup and service.

    This case alone does not verify restart recovery.
    """
    FullCrashAtLaunch().run(MODEL, launch_params)


@scenario_params(MODEL, PartialCrashAtExit)
def test_stage_crash_at_exit(omni_server_function, online_client_function) -> None:
    """Begin shutdown, kill one stage, and require timely cleanup.

    Later serial cases cover fresh startup and service; this case does not
    independently verify restart recovery.
    """
    PartialCrashAtExit().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, GracefulShutdown)
def test_graceful_shutdown(omni_server_function, online_client_function) -> None:
    """SIGTERM the healthy root; all children and resources must clear within 60s."""
    GracefulShutdown().run(MODEL, omni_server_function, online_client_function)


@launch_scenario_params(
    MODEL,
    ServeRootCrashAtLaunch,
)
def test_serve_root_crash_at_launch(launch_params) -> None:
    """SIGKILL only root before readiness, after a diffusion worker exists.

    Require startup failure and orphan/resource cleanup before harness teardown.
    Later successful serial cases provide fresh-start recovery coverage.
    """
    ServeRootCrashAtLaunch().run(MODEL, launch_params)
