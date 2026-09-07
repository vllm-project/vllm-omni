# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
L4 expansion coverage for ``robbyant/lingbot-video-dense-1.3b``.

Basic T2I/T2V/TI2V serving is covered by ``test_lingbot_video.py`` and
``test_lingbot_video_moe.py``. CFG2 covers both dense and MoE checkpoints with
Cache-DiT disabled and enabled.
"""

import json
import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL = "robbyant/lingbot-video-dense-1.3b"
PROMPT = "a robotic arm picks up a red block"
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--model-class-name", "LingBotVideoPipeline"],
            ),
            id="default",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_cfg_off(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "model": omni_server.model,
            "prompt": PROMPT,
            "height": 192,
            "width": 192,
            "num_frames": 9,
            "fps": 24,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "flow_shift": 3.0,
            "seed": 42,
        },
    }
    openai_client.send_video_diffusion_request(request_config)


@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_batch_cfg_extra_params(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "model": omni_server.model,
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": 192,
            "width": 320,
            "num_frames": 9,
            "fps": 24,
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "flow_shift": 3.0,
            "seed": 42,
            "extra_params": json.dumps({"batch_cfg": True}, separators=(",", ":")),
        },
    }
    openai_client.send_video_diffusion_request(request_config)


@pytest.mark.parallel
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--model-class-name",
                    "LingBotVideoPipeline",
                    "--cfg-parallel-size",
                    "2",
                    "--cache-backend",
                    cache_backend,
                ],
            ),
            id=f"{variant}-cfg2-{cache_backend}",
            marks=[*hardware_marks(res={"cuda": "H100"}, num_cards=2), *marks],
        )
        for variant, model in [
            ("dense", MODEL),
            ("moe", "robbyant/lingbot-video-moe-30b-a3b"),
        ]
        for cache_backend, marks in [("none", []), ("cache_dit", [pytest.mark.cache])]
    ],
    indirect=True,
)
def test_cfg_parallel(omni_server: OmniServer, online_client: OnlineOmniClient) -> None:
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "model": omni_server.model,
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": 192,
            "width": 320,
            "num_frames": 9,
            "fps": 24,
            "num_inference_steps": 8,
            "guidance_scale": 3.0,
            "flow_shift": 3.0,
            "seed": 42,
        },
    }
    # A second request exercises the runner's per-request cache refresh.
    for _ in range(2):
        online_client.send_video_diffusion_request(request_config)
