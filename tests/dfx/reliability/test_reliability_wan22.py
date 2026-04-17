"""
Wan2.2 reliability integration tests.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
import requests

from tests.conftest import generate_synthetic_image
from tests.dfx.conftest import (
    assert_fault_exception,
    create_reliability_omni_server_params,
    resolve_oom_device_spec,
    supports_video_generation,
)
from tests.dfx.reliability.conftest import inject_gpu_oom, make_process_kill_fault_injector, stop_gpu_oom_hogs
from tests.utils import hardware_test
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
    "internal",
    "500",
    "503",
    "timeout",
    "connection",
    "engine",
    "orchestrator",
    "dead",
)

WAN_PARAMS = create_reliability_omni_server_params(RELIABILITY_SCENARIOS, E2E_STAGE_CONFIGS_DIR)
DIFFUSION_VIDEO_PARAMS = [param for param in WAN_PARAMS if supports_video_generation(param.model)]


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
        form_data = request_config["form_data"]
        normalized_form_data = {key: str(value) for key, value in form_data.items() if value is not None}
        header, encoded = request_config["image_reference"].split(",", 1)
        content_type = header.split(";")[0].removeprefix("data:")
        extension = content_type.split("/")[-1]
        files = {
            "input_reference": (
                f"reference.{extension}",
                BytesIO(base64.b64decode(encoded)),
                content_type,
            )
        }
        create_url = openai_client_function._build_url("/v1/videos")
        create_resp = requests.post(
            create_url,
            data=normalized_form_data,
            files=files,
            headers={"Accept": "application/json"},
            timeout=15,
        )

        if create_resp.status_code >= 500:
            raise RuntimeError(f"video create failed: http={create_resp.status_code} body={create_resp.text[:400]}")

        # Avoid long polling in fault case: perform at most one short status check.
        if create_resp.ok:
            try:
                video_id = create_resp.json().get("id")
            except Exception:  # noqa: BLE001
                video_id = None
            if video_id:
                status_url = openai_client_function._build_url(f"/v1/videos/{video_id}")
                status_resp = requests.get(
                    status_url,
                    headers={"Accept": "application/json"},
                    timeout=10,
                )
                if status_resp.status_code >= 500:
                    raise RuntimeError(
                        f"video status failed: http={status_resp.status_code} body={status_resp.text[:400]}"
                    )
                status_data = status_resp.json()
                if status_data.get("status") == "failed":
                    err = status_data.get("last_error", "internal job failure")
                    raise RuntimeError(f"job failed: {err}")
        pytest.fail("expected /v1/videos request failure after process-kill injection")
    except Exception as exc:
        assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
    else:
        pytest.fail("expected /v1/videos request failure after process-kill injection")
