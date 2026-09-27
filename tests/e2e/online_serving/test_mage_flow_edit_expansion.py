# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L4 Mage-Flow Edit coverage for three differently sized references."""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

MODEL = "microsoft/Mage-Flow-Edit-Turbo"
SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--max-num-seqs",
                    "2",
                    "--request-batch-max-wait-ms",
                    "50",
                ],
            ),
            id="turbo-three-references",
            marks=SINGLE_CARD_MARKS,
        )
    ],
    indirect=True,
)
def test_mage_flow_three_different_reference_sizes(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    image_data_urls = [
        "data:image/jpeg;base64," + generate_synthetic_image(width, height)["base64"]
        for width, height in [(512, 512), (384, 512), (512, 384)]
    ]
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(
            image_data_url=image_data_urls,
            content_text=(
                "Combine the main subject, background, and color palette from the three images into one coherent scene."
            ),
        ),
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "seed": 42,
            "mage_vision_long_edge": 384,
        },
    }
    openai_client.send_diffusion_request(request_config)

    references = [
        ["data:image/jpeg;base64," + generate_synthetic_image(512, 512)["base64"]],
        [
            "data:image/jpeg;base64," + generate_synthetic_image(width, height)["base64"]
            for width, height in [(512, 512), (384, 512), (512, 384)]
        ],
    ]
    # Output resolution is part of the request-batch key, so these two share one
    # and land in the same padded batch. The 1-vs-3 reference counts above give
    # them differing token counts, which is what exercises the padding.
    requests = [
        {
            "model": omni_server.model,
            "messages": dummy_messages_from_mix_data(
                image_data_url=reference_images,
                content_text="Restyle the subject as a watercolor illustration.",
            ),
            "extra_body": {
                "height": 512,
                "width": 512,
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
                "seed": 50 + index,
                "mage_vision_long_edge": 384,
            },
        }
        for index, reference_images in enumerate(references)
    ]

    responses = openai_client.send_diffusion_request(requests)

    assert len(responses) == 2
    assert all(response.success for response in responses)
