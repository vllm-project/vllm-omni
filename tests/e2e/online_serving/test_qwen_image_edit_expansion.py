"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following models:
- Qwen-Image-Edit: single image input
- Qwen-Image-Edit-2509: two image inputs
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

EDIT_PROMPT = "Transform this modern, geometrist image into a Vincent van Gogh style impressionist painting."
MULTI_EDIT_PROMPT = (
    "Transform the first image into a Dadaism collage art. "
    "Transform the second image into a Vincent van Gogh style painting. "
    "Then juxtapose the two transformed images into a single artwork for visual contrast."
)
NEGATIVE_PROMPT = "blurry, low quality, modern, geometrist"
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


# This test file targets two models, so I write a helper function.
# If a similar test only involves one model, one can just define a global list variable.
def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "tea_cache",  # [TODO] may consider changing to cache_dit after #1779 is resolved. Currently cache_dit and layerwise offload cannot work together.
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
                    "cache_dit",
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
    _get_diffusion_feature_cases("Qwen/Qwen-Image-Edit"),
    indirect=True,
)
def test_qwen_image_edit(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test all diffusion features with Qwen-Image-Edit in regular end-user scenarios."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)

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

    openai_client.send_diffusion_request(request_config)


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("Qwen/Qwen-Image-Edit-2509"),
    indirect=True,
)
def test_qwen_image_edit_2509(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test all diffusion features with Qwen-Image-Edit-2509 in regular end-user scenarios."""
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

    openai_client.send_diffusion_request(request_config)
