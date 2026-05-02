# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end tests for DiffusersAdapterPipeline.

It tests the full user flow of launching a diffusers-backed model and running inference.
"""

import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.parametrize(
    "omni_server",
    [
        OmniServerParams(
            model="tiny-random/Qwen-Image",
            server_args=[
                "--diffusion-load-format",
                "diffusers",
                "--diffusers-call-kwargs",
                '{"height": 512, "width": 0}',  # deliberately weird width to be overridden
            ],
        ),
    ],
    indirect=True,
)
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_t2i_with_diffusers_adapter(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    messages = dummy_messages_from_mix_data(content_text="a photo of an astronaut riding a horse on mars")

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "width": 512,
            "num_inference_steps": 2,
            "negative_prompt": "blurry",
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    response = openai_client.send_diffusion_request(request_config)
    image: Image.Image = response[0].images[0]  # pyright: ignore[reportOptionalSubscript]

    # Request config has incomplete width/height, so internal assertion in `send_diffusion_request` is incomplete.
    assert image.size == (512, 512)
