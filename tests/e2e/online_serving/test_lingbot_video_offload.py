# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Serving e2e for ``robbyant/lingbot-video-moe-30b-a3b`` with layerwise / CPU offload.

Both cases assert a video is served and that GPU residency after the request is
below what the same server holds without that offload mode, so a silently ignored
offload flag fails the test.
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "robbyant/lingbot-video-moe-30b-a3b"
PROMPT = "a robotic arm picks up a red block"
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"
SMOKE_DEFAULT_SAMPLING_PARAMS = '{"0":{"num_frames":81,"num_inference_steps":2,"guidance_scale":3.0}}'

# Driver-level GPU use after serving requests on one H100: 74.3 GiB without offload,
# 60.1 GiB with --enable-cpu-offload and 14.0 GiB with --enable-layerwise-offload;
# streaming only the DiT peaked at 20.8 GiB offline. Each bound sits between two of them.
CPU_OFFLOAD_GPU_MEM_LIMIT_GIB = 65.0
LAYERWISE_OFFLOAD_GPU_MEM_LIMIT_GIB = 17.0

SINGLE_CARD_MARKS = hardware_marks(res={"cuda": ["H100", "B200"]})


def _make_request_config(omni_server: OmniServer) -> dict:
    return {
        "model": omni_server.model,
        "form_data": {
            "model": omni_server.model,
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": 192,
            "width": 320,
            "num_frames": 9,
            "fps": 24,
            "flow_shift": 3.0,
            "seed": 42,
        },
    }


def _offload_server_args(extra: list[str]) -> list[str]:
    return [
        "--model-class-name",
        "LingBotVideoPipeline",
        "--default-sampling-params",
        SMOKE_DEFAULT_SAMPLING_PARAMS,
        *extra,
        "--vae-use-tiling",
        "--vae-use-slicing",
    ]


def _assert_gpu_used_below(limit_gib: float, mode: str) -> None:
    # Free memory is read from the driver, so it also covers the server subprocess.
    free_bytes, total_bytes = current_omni_platform.get_device_memory()
    used_gib = (total_bytes - free_bytes) / (1024**3)
    assert used_gib < limit_gib, f"GPU use {used_gib:.1f} GiB >= {limit_gib} GiB; {mode} did not engage."


def _layerwise_offload_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(model=model, server_args=_offload_server_args(["--enable-layerwise-offload"])),
            id="layerwise-offload",
            marks=SINGLE_CARD_MARKS,
        ),
    ]


def _cpu_offload_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(model=model, server_args=_offload_server_args(["--enable-cpu-offload"])),
            id="cpu-offload",
            marks=SINGLE_CARD_MARKS,
        ),
    ]


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _layerwise_offload_cases(MODEL), indirect=True)
def test_text_to_video_moe_layerwise_offload(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    openai_client.send_video_diffusion_request(_make_request_config(omni_server))
    _assert_gpu_used_below(LAYERWISE_OFFLOAD_GPU_MEM_LIMIT_GIB, "layerwise offload")


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _cpu_offload_cases(MODEL), indirect=True)
def test_text_to_video_moe_cpu_offload(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    openai_client.send_video_diffusion_request(_make_request_config(omni_server))
    _assert_gpu_used_below(CPU_OFFLOAD_GPU_MEM_LIMIT_GIB, "CPU offload")
