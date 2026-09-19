# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L4 Mage-Flow functionality coverage for a boundary aspect ratio."""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

MODEL = "microsoft/Mage-Flow-Turbo"
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
            id="turbo",
            marks=SINGLE_CARD_MARKS,
        )
    ],
    indirect=True,
)
def test_mage_flow_four_to_one_aspect_ratio(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(
            content_text="A panoramic watercolor landscape with mountains and a quiet lake."
        ),
        "extra_body": {
            "height": 512,
            "width": 2048,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)

    # Output resolution is part of the request-batch key, so these two share one
    # and land in the same padded batch. Their prompt lengths differ, which is
    # what makes the batch ragged and exercises the padding.
    prompts = [
        "A paper sculpture.",
        (
            "A geometric paper sculpture photographed on a neutral background, lit "
            "from the left, with long soft shadows falling across the whole frame."
        ),
    ]
    requests = [
        {
            "model": omni_server.model,
            "messages": dummy_messages_from_mix_data(content_text=prompt),
            "extra_body": {
                "height": 512,
                "width": 512,
                "num_inference_steps": 2,
                "guidance_scale": 5.0,
                "seed": 40 + index,
            },
        }
        for index, prompt in enumerate(prompts)
    ]

    responses = openai_client.send_diffusion_request(requests)

    assert len(responses) == 2
    assert all(response.success for response in responses)
