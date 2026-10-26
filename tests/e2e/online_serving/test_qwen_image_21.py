# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient, dummy_messages_from_mix_data

MODEL = os.environ.get("QWEN_IMAGE_21_TEST_MODEL", "Qwen/Qwen-Image-2.1")
SERVER_CASES = [
    pytest.param(
        OmniServerParams(model=MODEL, server_args=["--vae-use-tiling", "--max-num-seqs", "2"] + extra),
        id=name,
        marks=hardware_marks(res={"cuda": "H100"}),
    )
    for name, extra in [("request", []), ("step", ["--step-execution"])]
]
pytestmark = [
    pytest.mark.core_model,
    pytest.mark.advanced_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(
        not os.environ.get("QWEN_IMAGE_21_TEST_MODEL"),
        reason="Set QWEN_IMAGE_21_TEST_MODEL to a provisioned checkpoint until public weights are available.",
    ),
]


@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_text_to_image(omni_server: OmniServer, online_client: OnlineOmniClient):
    online_client.send_diffusion_request(
        {
            "model": omni_server.model,
            "messages": dummy_messages_from_mix_data(content_text="A red ceramic teapot on a wooden table."),
            "extra_body": {
                "height": 512,
                "width": 512,
                "num_inference_steps": 4,
                "seed": 42,
                "negative_prompt": "blurry, low quality",
                "true_cfg_scale": 4.0,
            },
        }
    )


@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_image_edit(omni_server: OmniServer, online_client: OnlineOmniClient):
    image = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"
    online_client.send_diffusion_request(
        {
            "model": omni_server.model,
            "messages": dummy_messages_from_mix_data(content_text="Make the image blue.", image_data_url=image),
            "extra_body": {
                "height": 512,
                "width": 512,
                "num_inference_steps": 4,
                "seed": 42,
                "negative_prompt": "blurry, low quality",
                "true_cfg_scale": 4.0,
            },
        }
    )
