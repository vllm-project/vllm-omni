# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving e2e for ``robbyant/lingbot-video-moe-30b-a3b`` with layerwise / CPU offload.

Both cases assert a video is served. The layerwise case also asserts GPU residency
stays low (the 30B is ~70 GiB resident vs ~15 GiB streamed) to prove offload engaged.
"""

import os

import pytest
import torch

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "robbyant/lingbot-video-moe-30b-a3b"
PROMPT = "a robotic arm picks up a red block"
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"
SMOKE_DEFAULT_SAMPLING_PARAMS = '{"0":{"num_frames":81,"num_inference_steps":2,"guidance_scale":3.0}}'

# 30B resident ~70 GiB; layerwise streams blocks (~15 GiB). 40 GiB separates
# the two on an 80 GiB card, so this fails if offload did not engage.
OFFLOAD_GPU_MEM_LIMIT_GIB = 40.0

SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})


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

    # mem_get_info is driver-level (covers the server subprocess); fails if the
    # offload flag was ignored (the full checkpoint would be resident).
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_gib = (total_bytes - free_bytes) / (1024**3)
    assert used_gib < OFFLOAD_GPU_MEM_LIMIT_GIB, (
        f"GPU residency {used_gib:.1f} GiB >= {OFFLOAD_GPU_MEM_LIMIT_GIB}; "
        f"offload did not engage (30B resident would be ~70 GiB)."
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _cpu_offload_cases(MODEL), indirect=True)
def test_text_to_video_moe_cpu_offload(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    # Functional only: MODEL_LEVEL moves the whole DiT to GPU per step (~70 GiB peak).
    openai_client.send_video_diffusion_request(_make_request_config(omni_server))
