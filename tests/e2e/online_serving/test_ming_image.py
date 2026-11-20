# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native Ming-Image text-to-image smoke test on Ascend."""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.full_model, pytest.mark.omni, pytest.mark.diffusion]

MODEL = os.environ.get("MING_IMAGE_TEST_MODEL", "inclusionAI/Ming-Image-0.1-Design")
REVISION = "c57c49689d12ef7d4e24a170014f1dc43d3ce258"


@hardware_test(res={"npu": "A3"}, num_cards=2)
@pytest.mark.parametrize(
    "omni_server",
    [
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("ming_image.yaml"),
            server_args=["--revision", REVISION],
            env_dict={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        )
    ],
    indirect=True,
)
def test_text_to_rgba(omni_server, online_client):
    responses = online_client.send_diffusion_request(
        {
            "model": omni_server.model,
            "messages": [{"role": "user", "content": "A clean editorial botanical poster"}],
            "modalities": ["image"],
            "extra_body": {
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 12,
                "guidance_scale": 1.0,
                "seed": 42,
            },
        }
    )
    # Ming-Image's native VAE must preserve the alpha channel.
    assert len(responses[0].images) == 1
    assert responses[0].images[0].mode == "RGBA"
    assert responses[0].images[0].size == (1024, 1024)
