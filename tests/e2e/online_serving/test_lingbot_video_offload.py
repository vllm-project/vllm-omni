# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-GPU serving e2e for ``robbyant/lingbot-video-moe-30b-a3b`` with
layerwise / CPU offload.

Exercises the offload path added for LingBot-Video end-to-end. The pipeline
constructs its components on the host (``_resolve_construction_device`` triggers
on either ``enable_layerwise_offload`` or ``enable_cpu_offload``); the framework
then selects the backend. Both cases assert a video is served (functional). The
layerwise case additionally asserts that offload actually *engaged* — peak GPU
residency stays far below the full-checkpoint footprint, which fails if the flag
were silently ignored (the 30B BF16 checkpoint is ~60 GiB, resident ~70 GiB
without offload vs ~15 GiB with layerwise streaming). The CPU-offload
(MODEL_LEVEL) case is functional-only: it moves the whole DiT to GPU per step
(~70 GiB peak), so the low-residency assertion does not apply.
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

# The 30B MoE is ~60 GiB of BF16 weights; resident (no offload) it occupies
# ~70 GiB. Layerwise offload keeps only the text encoder + VAE + one DiT block
# on-device (~15 GiB). 40 GiB cleanly separates the two regimes on an 80 GiB
# card, so this threshold fails if offload did not engage.
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
    # Functional: the offload path must serve a valid video end-to-end
    # (construction-on-host + block streaming + the plural
    # `_layerwise_offload_blocks_attrs` block discovery all work without error).
    openai_client.send_video_diffusion_request(_make_request_config(omni_server))

    # Engagement: torch.cuda.mem_get_info reports driver-level free memory for
    # the visible GPU, which accounts for the server subprocess's residency.
    # With layerwise offload the DiT blocks live on the host, so GPU residency
    # stays far below the ~70 GiB full-checkpoint footprint. This fails if the
    # offload flag was silently ignored (construction stayed on-device).
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_gib = (total_bytes - free_bytes) / (1024**3)
    assert used_gib < OFFLOAD_GPU_MEM_LIMIT_GIB, (
        f"Expected layerwise-offloaded GPU residency < {OFFLOAD_GPU_MEM_LIMIT_GIB} GiB, "
        f"got {used_gib:.1f} GiB — offload did not engage (the 30B checkpoint "
        f"resident would be ~70 GiB). Check that --enable-layerwise-offload "
        f"triggered host-side construction + LayerWiseOffloadBackend."
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _cpu_offload_cases(MODEL), indirect=True)
def test_text_to_video_moe_cpu_offload(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    # Functional only: --enable-cpu-offload selects the MODEL_LEVEL backend,
    # which moves the whole DiT to GPU per step (~70 GiB peak on an 80 GiB
    # card). The low-residency assertion above does not apply here; this case
    # verifies the host-side construction fix works for the cpu-offload flag and
    # the ModelLevelOffloadBackend serves a video end-to-end.
    openai_client.send_video_diffusion_request(_make_request_config(omni_server))
