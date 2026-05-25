# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example online tests for Dynin-Omni model.
"""

import base64
import os
from io import BytesIO

import pytest
from vllm.assets.image import ImageAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.full_model, pytest.mark.omni]

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "snu-aidas/Dynin-Omni"
STAGE_CONFIG = get_deploy_config_path("dynin_omni_ci.yaml")

T2I_PROMPT = "A high quality detailed living room interior photo."
T2S_PROMPT = "Please read this sentence naturally: Hello from online serving."
I2I_PROMPT = "Transform this outdoor nature boardwalk scene into a painting style with vivid colors."

TEST_PARAMS = [OmniServerParams(model=MODEL, stage_config_path=STAGE_CONFIG, stage_init_timeout=600)]
_STAGE_COUNT = 3
_I2I_STAGE_SAMPLING = {"max_tokens": 1, "temperature": 0.0, "top_p": 1.0, "detokenize": False}


def _build_t2i_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": f"<|t2i|> {prompt}"}]}]


def _build_t2s_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": f"<|t2s|> {prompt}"}]}]


def _build_i2i_messages(prompt: str) -> list[dict]:
    input_image = ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")
    buffer = BytesIO()
    input_image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"<|i2i|> {prompt}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }
    ]


@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_send_i2i_request_001(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": _build_i2i_messages(I2I_PROMPT),
        "modalities": ["image"],
        "extra_body": {
            "sampling_params_list": [dict(_I2I_STAGE_SAMPLING) for _ in range(_STAGE_COUNT)],
        },
    }
    openai_client.send_diffusion_request(request_config)


@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_send_t2i_request_001(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": _build_t2i_messages(T2I_PROMPT),
        "modalities": ["image"],
    }
    openai_client.send_diffusion_request(request_config)


@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_send_t2s_request_001(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": _build_t2s_messages(T2S_PROMPT),
        "modalities": ["audio"],
        "audio_ref_text": T2S_PROMPT,
    }
    openai_client.send_omni_request(request_config)
