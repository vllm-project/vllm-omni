# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient

pytestmark = [pytest.mark.diffusion]

MODEL = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
SERVER = pytest.param(
    OmniServerParams(model=MODEL, server_args=["--enforce-eager"]),
    marks=hardware_marks(res={"cuda": "L4"}),
    id="sana-sprint",
)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.parametrize("omni_server", [SERVER], indirect=True)
def test_text_to_image(omni_server: OmniServer, online_client: OnlineOmniClient):
    online_client.send_images_generations_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": "A red panda in a bamboo forest",
                "size": "1024x1024",
                "seed": 42,
                "response_format": "b64_json",
            },
        }
    )


@pytest.mark.advanced_model
@pytest.mark.parametrize("omni_server", [SERVER], indirect=True)
def test_multiple_rectangular_images(omni_server: OmniServer, online_client: OnlineOmniClient):
    online_client.send_images_generations_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": "A mountain lake at dawn",
                "size": "768x1024",
                "n": 2,
                "num_inference_steps": 1,
                "guidance_scale": 4.5,
                "seed": 123,
                "response_format": "b64_json",
            },
        }
    )
