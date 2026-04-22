"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the FluxKontext model.
"""

import pytest

from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

EDIT_PROMPT = "Transform this modern, geometrist image into a Vincent van Gogh style impressionist painting."
NEGATIVE_PROMPT = "blurry, low quality, modern, geometrist"
MODEL = "black-forest-labs/FLUX.1-Kontext-dev"


def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--tensor-parallel-size",
                    "2",
                    "--enable-cpu-offload",
                ],
            ),
            id="parallel_001",
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_flux_kontext_text_to_image(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test text-to-image generation with FluxKontext in regular end-user scenarios."""
    messages = dummy_messages_from_mix_data(content_text="A photo of a cat sitting on a laptop keyboard")

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_flux_kontext_image_edit(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test image editing with FluxKontext in regular end-user scenarios."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)

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
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_flux_kontext_image_edit_no_negative(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test image editing with FluxKontext without negative prompt."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_flux_kontext_high_resolution(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test high-resolution generation with FluxKontext."""
    messages = dummy_messages_from_mix_data(content_text="A beautiful landscape with mountains and a lake")

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 768,
            "width": 1024,
            "num_inference_steps": 2,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_flux_kontext_multiple_outputs(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test generating multiple outputs with FluxKontext."""
    messages = dummy_messages_from_mix_data(content_text="A photo of a cat sitting on a laptop keyboard")

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "num_outputs_per_prompt": 2,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
