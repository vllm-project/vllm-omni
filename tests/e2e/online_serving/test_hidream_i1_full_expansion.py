"""
End-to-end diffusion coverage for HiDream-I1-Full in online serving mode.

This test verifies that HiDream-I1-Full can be launched,
accepts text-to-image requests through the OpenAI-compatible API, and returns
valid generated images with the requested resolution.

assert_diffusion_response validates successful generation and the expected
image resolution.
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

pytestmark = [pytest.mark.diffusion, pytest.mark.slow]

MODEL = "HiDream-ai/HiDream-I1-Full"
PROMPT = "A cinematic mountain landscape at sunrise, dramatic clouds, ultra-detailed, realistic photography."
NEGATIVE_PROMPT = "low quality, blurry, distorted, deformed, watermark"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_hidream_i1_image_feature_cases(model: str):
    """Return HiDream-I1-Full diffusion feature cases."""

    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--auxiliary-text-encoder",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                ],
            ),
            id="default",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_hidream_i1_image_feature_cases(MODEL),
    indirect=True,
)
@pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/4662")
def test_hidream_i1_image(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Validate HiDream-I1-Full online serving with CPU offload."""

    messages = dummy_messages_from_mix_data(content_text=PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
