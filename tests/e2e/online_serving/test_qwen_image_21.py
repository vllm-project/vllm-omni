# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
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
TEACACHE_SERVER = pytest.param(
    OmniServerParams(
        model=MODEL,
        server_args=[
            "--dtype",
            "bfloat16",
            "--enforce-eager",
            "--no-enable-cuda-graph-decode",
            "--max-num-seqs",
            "1",
            "--cache-backend",
            "tea_cache",
            "--cache-config",
            json.dumps(
                {
                    "rel_l1_thresh": 0.10,
                    "cfg_rel_l1_thresh": 0.05,
                    "skip_timestep_range": [0.30, 0.855],
                    # Qwen-Image-2.1@d26bb612, 1024px/50 linear sigmas; not Turbo.
                    "coefficients": [
                        489544.41506285727,
                        -65325.67451621187,
                        3012.825475742039,
                        -55.92014411118746,
                        0.4159928016209747,
                    ],
                }
            ),
        ],
        env_dict={"DIFFUSION_ATTENTION_BACKEND": "CUDNN_ATTN"},
    ),
    id="teacache-request",
    marks=hardware_marks(res={"cuda": "H100"})
    + [
        pytest.mark.skipif(
            os.environ.get("QWEN_IMAGE_21_TEST_TEACACHE") != "1",
            reason="Set QWEN_IMAGE_21_TEST_TEACACHE=1 with the calibrated base checkpoint.",
        )
    ],
)
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


@pytest.mark.parametrize("omni_server", [TEACACHE_SERVER], indirect=True)
@pytest.mark.parametrize("true_cfg_scale", [1.0, 4.0])
def test_teacache_request_mode(omni_server: OmniServer, online_client: OnlineOmniClient, true_cfg_scale: float):
    """Model-backed smoke of startup dispatch, policy JSON and request-mode output.

    Set QWEN_IMAGE_21_TEST_MODEL to the provisioned d26bb612 base snapshot.
    This checks the serving contract, not a quality gate or a speedup claim.
    """
    responses = online_client.send_diffusion_request(
        {
            "model": omni_server.model,
            "messages": dummy_messages_from_mix_data(content_text="A red ceramic teapot on a wooden table."),
            "extra_body": {
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 50,
                "sigmas": [1 - i / 50 for i in range(50)],
                "num_outputs_per_prompt": 1,
                "seed": 2026,
                "negative_prompt": "blurry, low quality",
                "true_cfg_scale": true_cfg_scale,
            },
        }
    )
    assert len(responses) == 1
    assert responses[0].images is not None
    assert len(responses[0].images) == 1
    assert responses[0].images[0].size == (1024, 1024)
    assert responses[0].images[0].mode == "RGBA"
