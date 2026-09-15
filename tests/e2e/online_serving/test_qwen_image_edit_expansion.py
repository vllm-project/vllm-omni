# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by Qwen-Image-Edit-2511 (single image input and two image inputs).
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient, dummy_messages_from_mix_data

pytestmark = [pytest.mark.diffusion, pytest.mark.slow]

MODEL = "Qwen/Qwen-Image-Edit-2511"
SINGLE_EDIT_PROMPT = "Restyle this image into a Vincent van Gogh style impressionist painting."
MULTI_EDIT_PROMPT = (
    "Transform the first image into a Dadaism collage art. "
    "Transform the second image into a Vincent van Gogh style painting. "
    "Then juxtapose the two transformed images into a single artwork for visual contrast."
)
NEGATIVE_PROMPT = "blurry, low quality, modern, geometrist"
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_diffusion_feature_cases(model: str = MODEL):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--enable-cpu-offload",
                ],
            ),
            id="cpu_offload",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--enable-layerwise-offload",
                ],
            ),
            id="single_card_001",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--ulysses-degree",
                    "2",
                ],
            ),
            id="parallel_001",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "tea_cache",
                    "--ring",
                    "2",
                ],
            ),
            id="parallel_002",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "tea_cache",
                    "--cfg-parallel-size",
                    "2",
                ],
            ),
            id="parallel_003",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--tensor-parallel-size",
                    "2",
                    "--vae-patch-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_004",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--use-hsdp",
                    "--hsdp-shard-size",
                    "2",
                ],
            ),
            id="parallel_005",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(),
    indirect=True,
)
def test_qwen_image_edit_2511_single_image(omni_server: OmniServer, online_client: OnlineOmniClient):
    """Test Qwen-Image-Edit-2511 with a single image input.

    Regression: with tea_cache enabled and zero_cond_t=True, the TeaCache
    postprocess closure used the doubled temb (shape 2*batch) without halving
    it, causing norm_out to broadcast and return noise_pred with shape
    (2*batch, seq, ch). The scheduler step then silently expanded latents via
    broadcasting, so at step 2 torch.cat([latents, image_latents], dim=1)
    crashed with a batch size mismatch. Fixed in extractors.py.
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=SINGLE_EDIT_PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    online_client.send_diffusion_request(request_config)


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(),
    indirect=True,
)
def test_qwen_image_edit_2511_two_images(omni_server: OmniServer, online_client: OnlineOmniClient):
    """Test Qwen-Image-Edit-2511 with two image inputs."""
    image_data_url_1 = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"
    image_data_url_2 = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    messages = dummy_messages_from_mix_data(
        image_data_url=[image_data_url_1, image_data_url_2], content_text=MULTI_EDIT_PROMPT
    )

    # CFG parallel is only activated when a negative prompt and true_cfg_scale > 1.0 are both present
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    online_client.send_diffusion_request(request_config)
