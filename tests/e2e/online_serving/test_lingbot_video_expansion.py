# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
L4 expansion coverage for ``robbyant/lingbot-video-dense-1.3b``.

This file remains dense-only. Basic T2I/T2V/TI2V serving for both dense and MoE
checkpoints is covered by ``test_lingbot_video.py`` and
``test_lingbot_video_moe.py``. This expansion suite also covers the combined
SP2 x CFG2 serving path on four GPUs.
"""

import json
import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

pytestmark = [pytest.mark.diffusion]

MODEL = "robbyant/lingbot-video-dense-1.3b"
PROMPT = "a robotic arm picks up a red block"
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
FOUR_CARD_PARALLEL_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=4)


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


def _get_layerwise_offload_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--model-class-name",
                    "LingBotVideoPipeline",
                    "--enable-layerwise-offload",
                    "--enforce-eager",
                ],
            ),
            id="layerwise-offload",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


def _get_sp_cfg_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--model-class-name",
                    "LingBotVideoPipeline",
                    "--usp",
                    "2",
                    "--cfg-parallel-size",
                    "2",
                ],
            ),
            id="sp2_cfg2",
            marks=FOUR_CARD_PARALLEL_MARKS,
        ),
    ]


@pytest.mark.full_model
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


@pytest.mark.full_model
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


@pytest.mark.slow
@pytest.mark.parametrize("omni_server", _get_layerwise_offload_cases(MODEL), indirect=True)
def test_layerwise_offload_t2v(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    openai_client.send_video_diffusion_request(
        {
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
            },
        }
    )


@pytest.mark.full_model
@pytest.mark.parametrize("omni_server", _get_sp_cfg_cases(MODEL), indirect=True)
def test_sp_cfg_parallel(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    openai_client.send_video_diffusion_request(
        {
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
            },
        }
    )
