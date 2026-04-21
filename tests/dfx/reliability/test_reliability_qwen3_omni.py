"""
Qwen3-Omni reliability integration tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

import pytest
import torch

from tests.dfx.conftest import (
    assert_fault_exception,
    assert_no_extra_worker_processes,
    create_reliability_omni_server_params,
    resolve_oom_device_spec,
    wait_chat_request_ready,
)
from tests.dfx.reliability.conftest import (
    force_remove_container,
    inject_gpu_oom,
    list_remote_process_pids_by_pattern,
    make_process_kill_fault_injector,
    post_chat_completions_raw,
    start_runtime_teardown_container_server,
    stop_gpu_oom_hogs,
)
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import dummy_messages_from_mix_data
from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

RELIABILITY_SCENARIOS: list[dict[str, Any]] = [
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

DEPLOY_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "vllm_omni" / "deploy"


def _default_oom_device_spec() -> str:
    """Use currently visible CUDA ordinals to avoid invalid device index in sidecar."""
    count = torch.cuda.device_count()
    if count <= 0:
        return "0"
    return ",".join(str(i) for i in range(count))


OOM_INJECTION_CONFIG = {
    "device": _default_oom_device_spec(),
    "target_mem_ratio": 0.95,
    "hold_seconds": 0,
    "startup_timeout_sec": 20,
    "strict": True,
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
RUNTIME_WORKER_PATTERN = "VLLM::Worker"


class _HasServeArgs(Protocol):
    serve_args: list[str]


def _get_system_prompt() -> dict:
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


def _get_mix_prompt() -> str:
    return "What is recited in the audio? What is in this image? Describe the video briefly."


def _stage_config_path_from_omni_server(omni_server: _HasServeArgs) -> str | None:
    args: list[str] = omni_server.serve_args
    for i, arg in enumerate(args):
        if arg == "--stage-configs-path" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--stage-configs-path="):
            return arg.split("=", 1)[1]
    return None


QWEN_PARAMS = create_reliability_omni_server_params(RELIABILITY_SCENARIOS, DEPLOY_CONFIGS_DIR)


@pytest.mark.slow
@pytest.mark.skipif(
    current_omni_platform.is_rocm() or current_omni_platform.is_xpu(),
    reason="CUDA sidecar OOM injection is CUDA-only for phase-1",
)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_chat_large_payload_failure(omni_server_function, openai_client_function) -> None:
    device_spec = resolve_oom_device_spec(
        OOM_INJECTION_CONFIG,
        _stage_config_path_from_omni_server(omni_server_function),
    )
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
    )
    try:
        video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(1280, 720, 161)['base64']}"
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"
        audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(20, 1)['base64']}"
        messages = dummy_messages_from_mix_data(
            system_prompt=_get_system_prompt(),
            video_data_url=video_data_url,
            image_data_url=image_data_url,
            audio_data_url=audio_data_url,
            content_text=f"{_get_mix_prompt()} " * 200,
        )
        request_config = {
            "model": omni_server_function.model,
            "messages": messages,
            "stream": True,
            "key_words": {"audio": ["test"]},
        }
        try:
            openai_client_function.send_omni_request(request_config, request_num=1)
        except Exception as exc:
            assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
        else:
            pytest.fail("expected large chat payload request failure during GPU OOM injection")
    finally:
        stop_gpu_oom_hogs(handle)


@pytest.mark.slow
@pytest.mark.skipif(
    current_omni_platform.is_rocm() or current_omni_platform.is_xpu(),
    reason="CUDA sidecar OOM injection is CUDA-only for phase-1",
)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_concurrent_pressure_failure(omni_server_function, openai_client_function) -> None:
    device_spec = resolve_oom_device_spec(
        OOM_INJECTION_CONFIG,
        _stage_config_path_from_omni_server(omni_server_function),
    )
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
    )
    try:
        messages = dummy_messages_from_mix_data(
            system_prompt=_get_system_prompt(),
            content_text="What is the capital of China? Answer in 20 words.",
        )
        request_config = {
            "model": omni_server_function.model,
            "messages": messages,
            "stream": False,
            "modalities": ["text"],
            "key_words": {"text": ["beijing"]},
        }
        try:
            openai_client_function.send_omni_request(request_config, request_num=4)
        except Exception as exc:
            assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
        else:
            pytest.fail("expected concurrent request failure under OOM injection")
    finally:
        stop_gpu_oom_hogs(handle)


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.parametrize(
    "fault_injector",
    [
        pytest.param(
            make_process_kill_fault_injector(
                grep_patterns="VLLM::Worker",
                signal_name="SIGKILL",
                limit=1,
                post_kill_wait_seconds=2.0,
            ),
            id="runtime_process_chain",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("omni_server_function", QWEN_PARAMS, indirect=True)
def test_reliability_fault_process_kill_request_failure(omni_server_after_fault_function, openai_client_function) -> None:
    messages = dummy_messages_from_mix_data(
        system_prompt=_get_system_prompt(),
        content_text="What is the capital of China? Answer in 20 words.",
    )
    request_config = {
        "model": omni_server_after_fault_function.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["beijing"]},
    }
    try:
        openai_client_function.send_omni_request(request_config, request_num=1)
    except Exception as exc:
        assert_fault_exception(exc, FAULT_ERROR_KEYWORDS)
    else:
        pytest.fail("expected request failure after process-kill injection")


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="runtime-teardown helper is POSIX-only")
@pytest.mark.skip(reason="Temporarily disabled runtime teardown scenario.")
@pytest.mark.parametrize("runtime_params", [QWEN_PARAMS[0]], ids=["runtime_teardown_container_kill"])
def test_reliability_fault_runtime_teardown_container_kill_no_orphan_worker(runtime_params, model_prefix) -> None:
    baseline_worker_pids = set(list_remote_process_pids_by_pattern(RUNTIME_WORKER_PATTERN))
    model = model_prefix + runtime_params.model
    serve_args = list(runtime_params.server_args or [])
    if "--stage-init-timeout" not in serve_args:
        serve_args = ["--stage-init-timeout", "120", *serve_args]
    if runtime_params.stage_config_path is not None:
        serve_args += ["--stage-configs-path", runtime_params.stage_config_path]

    handle = None
    try:
        handle = start_runtime_teardown_container_server(
            model=model,
            serve_args=serve_args,
        )
        wait_chat_request_ready(handle.host, handle.port, model=model)

        force_remove_container(handle.container_name)

        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "What is the capital of China? Answer in one word."}],
                "stream": False,
                "modalities": ["text"],
            }
        )
        request_failed = False
        try:
            status, body = post_chat_completions_raw(handle.host, handle.port, payload)
            if status >= 500:
                request_failed = True
            else:
                pytest.fail(f"expected request failure after container teardown, got http={status} body={body[:200]!r}")
        except Exception:
            request_failed = True
        assert request_failed, "expected request failure after container teardown"

        assert_no_extra_worker_processes(baseline_worker_pids, RUNTIME_WORKER_PATTERN)
    finally:
        keep_on_failure = os.getenv("RUNTIME_TEARDOWN_KEEP_CONTAINER_ON_FAILURE", "0").strip() == "1"
        if handle is not None and not keep_on_failure:
            force_remove_container(handle.container_name)
