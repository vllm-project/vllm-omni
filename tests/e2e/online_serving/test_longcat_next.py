# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
E2E online-serving smoke test for LongCat-Next.

Text-only, chat-completions round trip: verifies the server boots the
2-stage ``longcat_next_thinker_multi_decoder`` pipeline and that a plain-text
request (no ``<longcat_img_start>``/``<longcat_audiogen_start>`` trigger)
round-trips through the OpenAI-compatible endpoint.

NOTE: image-generation and voice-cloned audio-generation requests are only
exercised offline (see ../offline_inference/test_longcat_next.py), which
calls ``Omni.generate`` directly with LongCat-Next's raw ``<longcat_*>``
control-token prompts. LongCat-Next does not use a standard chat template for
those trigger tokens, so how (or whether) the online chat-completions API
should expose them is still open -- add online generation tests once that's
decided, rather than guessing at an ``extra_body`` shape here.
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "meituan-longcat/LongCat-Next"
_DEPLOY = get_deploy_config_path("longcat_next_4gpu_80gb_multi_decoder.yaml")

test_params = [
    pytest.param(
        OmniServerParams(model=MODEL, stage_config_path=_DEPLOY),
        id="default",
    )
]


def _system_prompt() -> dict[str, object]:
    return {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    }


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_only(omni_server, openai_client) -> None:
    """Text in, text out -- thinker stage only, multi-decoder stage is a no-op pass-through."""
    messages = dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        content_text="What is the capital of France? Answer in one sentence.",
    )
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
    }
    openai_client.send_omni_request(request_config)
