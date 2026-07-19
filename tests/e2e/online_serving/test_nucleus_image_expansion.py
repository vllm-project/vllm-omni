# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model, pytest.mark.slow]

MODEL = "NucleusAI/Nucleus-Image"
POSITIVE_PROMPT = "A cat holding a sign that says hello world"
NEGATIVE_PROMPT = "blurry, low quality"


def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--tensor-parallel-size",
                    "2",
                    "--quantization",
                    "fp8",
                ],
            ),
            id="parallel_001",
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--ulysses-degree",
                    "2",
                    "--quantization",
                    "fp8",
                ],
            ),
            id="parallel_002",
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--ring",
                    "2",
                ],
            ),
            id="parallel_003",
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_nucleus_image_text_to_image(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    messages = dummy_messages_from_mix_data(content_text=POSITIVE_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 256,
            "width": 256,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_nucleus_image_high_resolution(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    messages = dummy_messages_from_mix_data(content_text="A beautiful landscape with mountains and a lake")
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 1024,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_nucleus_image_multiple_outputs(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    messages = dummy_messages_from_mix_data(content_text=POSITIVE_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 256,
            "width": 256,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "negative_prompt": NEGATIVE_PROMPT,
            "num_outputs_per_prompt": 2,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
