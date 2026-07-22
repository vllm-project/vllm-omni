# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Online serving tests for ``Boogu/Boogu-Image-0.1-Base`` (text-to-image).

- ``test_text_to_image_001``: single chat request, default server (``_get_default_case``).
- ``test_batch_001``: concurrent prompts via ``send_diffusion_request([cfg0, cfg1, ...])`` — one
  dict per prompt; each entry carries its own ``messages`` / ``negative_prompt`` (see
  ``TEST_PROMPTS``).

Boogu-Image's pipeline reads ``sp.guidance_scale`` (not ``true_cfg_scale``), so the
``extra_body`` uses ``guidance_scale``. These are smoke tests (minimal steps); numeric
parity vs. the Diffusers baseline is covered separately (see the support plan, step 14).

From ``tests/``::

    pytest -s -v e2e/online_serving/test_boogu_image.py \
        -m "full_model and diffusion and H100" --run-level=full_model
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL = "Boogu/Boogu-Image-0.1-Base"
T2I_PROMPT = "A mountain lake at sunset, photorealistic, cinematic lighting"
NEGATIVE_PROMPT = ""

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})

TEST_PROMPTS: list[dict[str, str]] = [
    {"prompt": "a cup of coffee on a table", "negative_prompt": ""},
    {"prompt": "a toy dinosaur on a sandy beach", "negative_prompt": ""},
    {"prompt": "a futuristic city skyline at sunset", "negative_prompt": ""},
]


def _get_default_case(model: str):
    """Return a single default ``OmniServerParams`` row (no extra ``server_args``)."""
    return [
        pytest.param(
            OmniServerParams(model=model),
            id="default",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize("omni_server", _get_default_case(MODEL), indirect=True)
def test_text_to_image_001(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Default Boogu-Image T2I smoke (single ``default`` server config)."""
    messages = dummy_messages_from_mix_data(content_text=T2I_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": 4.0,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)


@pytest.mark.parametrize("omni_server", _get_default_case(MODEL), indirect=True)
def test_batch_001(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Concurrent T2I: one ``request_config`` dict per prompt (``send_diffusion_request`` list mode)."""
    request_config = [
        {
            "model": omni_server.model,
            "messages": dummy_messages_from_mix_data(content_text=prompt["prompt"]),
            "extra_body": {
                "height": 512,
                "width": 512,
                "num_inference_steps": 2,
                "negative_prompt": prompt["negative_prompt"],
                "guidance_scale": 4.0,
                "seed": 42,
            },
        }
        for prompt in TEST_PROMPTS
    ]
    openai_client.send_diffusion_request(request_config)
