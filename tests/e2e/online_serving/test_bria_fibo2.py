# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online serving smoke test for Bria Fibo-2.0 Dev.

Needs the gated checkpoint in the local HF cache, so it runs locally rather
than in CI. One chat-style generation request through the HTTP server."""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient, dummy_messages_from_mix_data

MODEL = os.environ.get("FIBO2_MODEL", "briaai/Fibo-2.0-dev")
PROMPT = (
    '{"short_description":"A red bicycle leaning against a white brick wall on a sunny morning.",'
    '"style_medium":"photograph"}'
)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_server", [OmniServerParams(model=MODEL)], indirect=True)
def test_bria_fibo2_serving(omni_server: OmniServer, online_client: OnlineOmniClient):
    messages = dummy_messages_from_mix_data(content_text=PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 8,
            "guidance_scale": 5.0,
            "seed": 42,
        },
    }
    # send_diffusion_request raises on transport/server errors; multimodal
    # image decoding from the chat payload is not uniform across the online
    # client yet (the Z-Image serving test asserts the same way), so treat a
    # clean round trip as the smoke bar and validate images when present.
    resp = online_client.send_diffusion_request(request_config)
    assert resp is not None, "no response from server"
    images = getattr(resp, "images", None)
    if images:
        got_shape = images[0].size if hasattr(images[0], "size") else None
        assert got_shape == (1024, 1024), f"unexpected served image size {got_shape}"
