"""
End-to-end online serving coverage for FLUX.1-dev sequence parallelism.

This test starts ``vllm serve black-forest-labs/FLUX.1-dev --omni`` through the
standard server fixture, enables an SP backend with CLI args, sends a
text-to-image request, and validates that the response contains a generated
image.
"""

import pytest

from tests.helpers.assertions import assert_image_valid
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL = "black-forest-labs/FLUX.1-dev"
PROMPT = "A lovely bunny holding a sign that says vllm-omni, detailed digital art."
HEIGHT = 512
WIDTH = 512
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_flux_1_dev_sp_cases():
    return [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--ulysses-degree",
                    "2",
                ],
            ),
            id="ulysses_2",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--ring-degree",
                    "2",
                ],
            ),
            id="ring_2",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_flux_1_dev_sp_cases(),
    indirect=True,
)
def test_flux_1_dev_sequence_parallel_online_serving(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    request_config = {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "extra_body": {
            "height": HEIGHT,
            "width": WIDTH,
            "num_inference_steps": 2,
            "guidance_scale": 3.5,
            "num_outputs_per_prompt": 1,
            "seed": 42,
        },
    }

    responses = openai_client.send_diffusion_request(request_config)

    assert len(responses) == 1
    images = responses[0].images
    assert images is not None
    assert len(images) == 1
    assert_image_valid(images[0], width=WIDTH, height=HEIGHT)
