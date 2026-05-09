# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Online serving tests for ``Qwen/Qwen-Image`` (text-to-image).

- ``test_text_to_image_001``: single chat request, default server (``_get_default_case``).
- ``test_batch_001``: concurrent prompts via ``send_diffusion_request([cfg0, cfg1, ...])`` — one
  dict per prompt; each entry carries its own ``messages`` / ``negative_prompt`` (see
  ``TEST_PROMPTS``).
- ``test_acceleration_feature_001``: ``advanced_model`` only — parametrized ``OmniServerParams``
  with cache/offload/parallel flags (see ``_get_diffusion_feature_cases``).

Full feature grids for Qwen-Image remain in ``test_qwen_image_expansion.py`` (L4).

From ``tests/``::

    pytest -s -v e2e/online_serving/test_qwen_image.py -m "core_model and diffusion" --run-level=core_model
    pytest -s -v e2e/online_serving/test_qwen_image.py -m "advanced_model and diffusion" --run-level=advanced_model
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Qwen/Qwen-Image"
T2I_PROMPT = "A photo of a cat sitting on a laptop keyboard, digital art style."
NEGATIVE_PROMPT = "blurry, low quality"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

TEST_PROMPTS: list[dict[str, str]] = [
    {"prompt": "a cup of coffee on a table", "negative_prompt": "low resolution"},
    {"prompt": "a toy dinosaur on a sandy beach", "negative_prompt": "cinematic, realistic"},
    {"prompt": "a futuristic city skyline at sunset", "negative_prompt": "blurry, foggy"},
    {"prompt": "a bowl of fresh strawberries", "negative_prompt": "low detail"},
    {"prompt": "a medieval knight standing in the rain", "negative_prompt": "modern clothing"},
    {"prompt": "a cat wearing sunglasses lounging in a garden", "negative_prompt": "dark lighting"},
    {"prompt": "a spaceship flying above a volcano", "negative_prompt": "low contrast"},
    {"prompt": "a watercolor painting of a mountain lake", "negative_prompt": "photo, realistic"},
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


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_default_case(MODEL), indirect=True)
def test_text_to_image_001(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Default Qwen-Image T2I smoke (single ``default`` server config)."""
    messages = dummy_messages_from_mix_data(content_text=T2I_PROMPT)
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


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
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
                "true_cfg_scale": 4.0,
                "seed": 42,
            },
        }
        for prompt in TEST_PROMPTS
    ]
    openai_client.send_diffusion_request(request_config)
