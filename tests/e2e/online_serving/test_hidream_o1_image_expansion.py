# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
L4 nightly expansion tests for HiDream-O1-Image.

Parametrized rows added per phase:
  Phase 5: Cache-DiT (test_hidream_o1_dev_t2i_cache_dit)
  Phase 6: TP/SP/CFG-Parallel/HSDP — coming
  Phase 7: CPU offload — coming

Run locally::

    pytest -s -v tests/e2e/online_serving/test_hidream_o1_image_expansion.py \\
        -m "full_model and diffusion" --run-level full_model
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

DEV_MODEL = "HiDream-ai/HiDream-O1-Image-Dev"
FULL_MODEL = "HiDream-ai/HiDream-O1-Image"

T2I_PROMPT = "A cinematic mountain landscape at sunrise, dramatic clouds, ultra-detailed."

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_hidream_o1_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(model=model),
            id="default",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(model=model, cache_backend="cache_dit"),
            id="cache_dit",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_hidream_o1_feature_cases(DEV_MODEL),
    indirect=True,
)
def test_hidream_o1_dev_t2i(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """L4: HiDream-O1-Image-Dev text-to-image, parameterized over baseline and Cache-DiT."""
    messages = dummy_messages_from_mix_data(content_text=T2I_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 28,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)
