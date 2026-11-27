# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen3-Omni reliability scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.dfx.reliability.helpers import (
    PARTIAL_ISOLATION_HTTP_TIMEOUT_SEC,
    FaultModel,
    SignalSpec,
    launch_scenario_params,
    scenario_params,
    sigterm_serve_root_and_assert_cleanup,
)
from tests.dfx.reliability.scenarios import (
    FullCrashAtLaunch,
    GpuOomAtLaunch,
    GpuOomUnderLoad,
    GracefulShutdown,
    PartialCrashAtExit,
    PartialCrashAtLaunch,
    PartialStageIsolation,
    ProcessKillServeRootNoLoad,
    ProcessKillServeRootWithLoad,
    ProcessKillTreeNoLoad,
    ProcessKillTreeWithLoad,
    ProcessKillWorkerNoLoad,
    ProcessKillWorkerWithLoad,
    ServeRootCrashAtLaunch,
)
from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image
from tests.helpers.runtime import dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

QWEN_SERVE_ROOT_SIGNALS: tuple[SignalSpec, ...] = (("SIGKILL", "https://github.com/vllm-project/vllm/issues/43060"),)

RUNTIME_WORKER_PATTERN = "StageEngineCoreProc_stage0"
_PROCESS_KILL_WORKER_CHAT_FF_HTTP_TIMEOUT_SEC = 30
# Includes a fresh interpreter's torch import, CUDA initialization and allocation.
_OOM_SIDECAR_STARTUP_TIMEOUT_SEC = 120


def _default_oom_device_spec() -> str:
    import torch

    count = torch.accelerator.device_count() or 0
    if count <= 0:
        return "0"
    return ",".join(str(index) for index in range(count))


def _hard_oom_config() -> dict[str, Any]:
    import torch

    return {
        "device": _default_oom_device_spec(),
        "target_mem_ratio": 0.95,
        "max_free_bytes": 32 * 1024 * 1024 if getattr(torch.version, "hip", None) else 256 * 1024 * 1024,
        "hold_seconds": 0,
        "startup_timeout_sec": _OOM_SIDECAR_STARTUP_TIMEOUT_SEC,
        "strict": False,
    }


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


def _system_prompt() -> dict[str, Any]:
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


def _text_request(model: str) -> dict[str, Any]:
    messages = dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        content_text="What is the capital of China? Answer in one short sentence.",
    )
    return {"model": model, "messages": messages, "stream": False, "modalities": ["text"]}


_MULTIMODAL_REQUESTS: dict[str, dict[str, Any]] = {}


def _multimodal_request(model: str) -> dict[str, Any]:
    """Return one shared multimodal body per model.

    Process-kill submits this from a thread pool. Building it again would call
    pyttsx3 from those workers. espeak is process-global, and overlapping
    engines segfault the pytest process.
    """
    cached = _MULTIMODAL_REQUESTS.get(model)
    if cached is None:
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
        audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
        messages = dummy_messages_from_mix_data(
            system_prompt=_system_prompt(),
            image_data_url=image_data_url,
            audio_data_url=audio_data_url,
            content_text="Describe the image and the audio in one short sentence.",
        )
        cached = {"model": model, "messages": messages, "stream": False, "modalities": ["text", "audio"]}
        _MULTIMODAL_REQUESTS[model] = cached
    return {
        "model": cached["model"],
        "messages": cached["messages"],
        "stream": False,
        "modalities": list(cached["modalities"]),
    }


def _oom_preflight_request(model: str) -> dict[str, Any]:
    """Match MiniCPM's small text input with text and audio output for OOM preflight."""
    request = _text_request(model)
    request["modalities"] = ["text", "audio"]
    return request


def _heavy_messages() -> list[dict[str, Any]]:
    # Prepare distinct images before pressure; avoid video sampling and the
    # dimension-based synthetic-image cache. GPU runs must establish actual OOM.
    image_count = 16
    width, height = 1280, 720
    image_data_url = [
        f"data:image/jpeg;base64,{generate_synthetic_image(width, height, seed=i, force_regenerate=True)['base64']}"
        for i in range(image_count)
    ]
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(20, 1)['base64']}"
    print(
        f"[oom-under-load] Qwen fault payload: images={image_count}, image_size={width}x{height}, "
        "audio_seconds=20, max_tokens=128; media prepared before pressure",
        flush=True,
    )
    return dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text=("Compare all the images, and summarize the audio. " * 128) + "Answer in one short sentence.",
    )


def _mix_probe_payload(server: Any) -> dict[str, Any]:
    return {
        "model": server.model,
        "messages": _heavy_messages(),
        "stream": False,
        "modalities": ["text", "audio"],
        "max_tokens": 128,
    }


def _assert_text_200(client: Any, model: str) -> None:
    client.send_chat_completions_http_request({"json": _text_request(model), "timeout": 120})


class Qwen3Omni(FaultModel):
    model_id = "qwen3_omni"
    res = {"cuda": "H100", "rocm": "MI325"}
    num_cards = 2
    scenarios = [
        {
            "test_name": "qwen3_omni_reliability_async_chunk",
            "server_params": {
                "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "stage_config_name": "qwen3_omni_moe.yaml",
                "server_args": ["--async-chunk"],
            },
        },
    ]
    deploy_dir = Path(get_deploy_config_path("qwen3_omni_moe.yaml")).parent
    lifecycle_scenario = "qwen3_omni_reliability_async_chunk"
    eager_oom_stage_ids = (0, 1)
    worker_pattern = RUNTIME_WORKER_PATTERN
    isolation_pattern = "StageEngineCoreProc_stage1"
    hard_oom_config = _hard_oom_config()
    process_kill_error_keywords = PROCESS_KILL_ERROR_KEYWORDS
    with_load_timeout_sec = 300.0
    concurrent_workers = 4
    concurrent_timeout_sec = 30.0
    serve_root_signals = QWEN_SERVE_ROOT_SIGNALS
    serve_root_with_load_signals = QWEN_SERVE_ROOT_SIGNALS

    def oom_under_load_request(self, server: Any) -> dict[str, Any]:
        return _mix_probe_payload(server)

    def oom_preflight_request(self, server: Any) -> dict[str, Any]:
        return _oom_preflight_request(server.model)

    def warmup(self, client: Any, server: Any) -> None:
        client.send_omni_request(_multimodal_request(server.model), request_num=1)

    def submit_worker_load(self, client: Any, server: Any) -> None:
        client.send_omni_request(_multimodal_request(server.model), request_num=1)

    def submit_serve_load(self, client: Any, server: Any) -> None:
        client.send_omni_request(_multimodal_request(server.model), request_num=1)

    def submit_concurrent(self, client: Any, server: Any) -> Any:
        return client.send_chat_completions_http_request(
            {"json": _multimodal_request(server.model), "timeout": 20},
            err_code=(500, 503),
        )

    def assert_fast_fail(self, client: Any, server: Any) -> None:
        client.send_chat_completions_http_request(
            {
                "json": _multimodal_request(server.model),
                "timeout": _PROCESS_KILL_WORKER_CHAT_FF_HTTP_TIMEOUT_SEC,
            },
            err_code=500,
        )

    def isolation_warmup(self, client: Any, server: Any) -> None:
        self.warmup(client, server)

    def isolation_after(self, client: Any, server: Any) -> None:
        _assert_text_200(client, server.model)
        client.send_chat_completions_http_request(
            {"json": _multimodal_request(server.model), "timeout": PARTIAL_ISOLATION_HTTP_TIMEOUT_SEC},
            err_code=500,
        )


MODEL = Qwen3Omni()

pytestmark = [
    pytest.mark.slow,
    *MODEL.extra_marks,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


@scenario_params(MODEL, GpuOomUnderLoad)
def test_gpu_oom_at_load(omni_server_function, online_client_function) -> None:
    """One async-chunk OOM injection, followed by recovery or a restart-required state.

    A healthy instance must serve the same multimodal request again. A 503 or
    unreachable API requires a full restart. Serial suite cases provide fresh
    startup coverage; this case alone does not assert restart recovery.
    """
    GpuOomUnderLoad().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, ProcessKillWorkerNoLoad)
def test_stage_crash_at_idle_001(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    """After a quiet worker kill, the next request returns HTTP 500, then concurrent requests fail and health becomes 503."""
    ProcessKillWorkerNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)
    sigterm_serve_root_and_assert_cleanup(omni_server_function, scenario="stage_crash_at_idle")


@scenario_params(MODEL, ProcessKillWorkerWithLoad, skip_reason="issue#8912")
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


@scenario_params(MODEL, PartialStageIsolation)
def test_stage_crash_at_idle_002(omni_server_function, online_client_function) -> None:
    """Kill one stage or one worker. Requests bound to it fail, and one SIGTERM still exits the API."""
    PartialStageIsolation().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, GracefulShutdown)
def test_graceful_shutdown(omni_server_function, online_client_function) -> None:
    """SIGTERM the healthy root; all children and resources must clear within 60s."""
    GracefulShutdown().run(MODEL, omni_server_function, online_client_function)


@launch_scenario_params(MODEL, ServeRootCrashAtLaunch, skip_reason="https://github.com/vllm-project/vllm/issues/43060")
def test_serve_root_crash_at_launch(launch_params) -> None:
    """SIGKILL only the root before readiness, after a worker exists; require orphan cleanup."""
    ServeRootCrashAtLaunch().run(MODEL, launch_params)
