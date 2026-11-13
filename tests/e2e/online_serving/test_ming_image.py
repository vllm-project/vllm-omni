# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Design-Layer chat regression: three requested layers produce four RGBA images."""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient
from tests.helpers.stage_config import get_deploy_config_path

_MODEL = "inclusionAI/Ming-Image-0.1-Design-Layer"
# This revision includes the root Bailing config that triggered issue #9345.
_REVISION = "ff1b22d2bdd0ae6597c0024a42cac7923b24cf92"


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize(
    "omni_server",
    [
        OmniServerParams(
            model=_MODEL,
            stage_config_path=get_deploy_config_path("ming_image.yaml"),
            server_args=[
                "--revision",
                _REVISION,
                "--stage-overrides",
                '{"0":{"max_model_len":4096},"1":{"enforce_eager":true}}',
            ],
            init_timeout=1000,
            stage_init_timeout=900,
        )
    ],
    indirect=True,
)
def test_layered_chat_returns_composite_and_requested_layers(
    omni_server: OmniServer, online_client: OnlineOmniClient
) -> None:
    image_url = f"data:image/jpeg;base64,{generate_synthetic_image(1024, 1024)['base64']}"
    responses = online_client.send_diffusion_request(
        {
            "model": omni_server.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "Decompose this image into 3 layers."},
                    ],
                }
            ],
            "modalities": ["image"],
            "extra_body": {
                "num_layers": 3,
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 12,
                "guidance_scale": 2.0,
                "seed": 42,
            },
        }
    )
    assert len(responses) == 1
    images = responses[0].images
    assert images is not None
    assert len(images) == 4
    assert all(image.size == (1024, 1024) and image.mode == "RGBA" for image in images)
