# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
L4 expansion coverage for ``Boogu/Boogu-Image-0.1-Base`` (text-to-image).

The nightly smoke module (``test_boogu_image.py``) already covers the default
512x512 single request and a concurrent batch. This expansion module adds the
request-level paths that smoke does not exercise, run against every currently
supported single-GPU server configuration:

Server rows (single card):
- ``default`` — no extra ``server_args``.
- ``vae_slicing_tiling`` — ``--vae-use-slicing --vae-use-tiling`` (the recipe's
  OOM mitigation; applied generically in ``registry.initialize_model``).

Cases (one per test, each parametrized over both server rows):
- ``test_non_square_resolution`` — 768x1024, exercises the patchify / RoPE path
  at a non-square aspect ratio.
- ``test_cfg_off`` — ``guidance_scale=1.0``, exercises the no-CFG denoise branch
  (single prediction per step).
- ``test_high_resolution`` — 1024x1024, the recipe-recommended resolution.

Boogu-Image reads ``sp.guidance_scale`` (not ``true_cfg_scale``). These stay
smoke-depth (``num_inference_steps=2``); numeric parity vs. Diffusers is covered
by the local parity harness (support plan, step 14). CPU offload, Cache-DiT, and
multi-GPU parallelism are not yet supported for this model and are intentionally
omitted (add rows here when those features land).

From ``tests/``::

    pytest -s -v e2e/online_serving/test_boogu_image_expansion.py \
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


def _get_diffusion_feature_cases(model: str):
    """Return the single-card server configurations supported for Boogu-Image today."""
    return [
        pytest.param(
            OmniServerParams(model=model),
            id="default",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--vae-use-slicing", "--vae-use-tiling"],
            ),
            id="vae_slicing_tiling",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_non_square_resolution(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Non-square (768x1024) T2I exercises the patchify / RoPE path off the square grid."""
    messages = dummy_messages_from_mix_data(content_text=T2I_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 1024,
            "width": 768,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": 4.0,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)


@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_cfg_off(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """``guidance_scale=1.0`` exercises the no-CFG denoise branch (one prediction per step)."""
    messages = dummy_messages_from_mix_data(content_text=T2I_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": 1.0,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)


@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_high_resolution(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """1024x1024 T2I at the recipe-recommended resolution."""
    messages = dummy_messages_from_mix_data(content_text=T2I_PROMPT)
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": 4.0,
            "seed": 42,
        },
    }
    openai_client.send_diffusion_request(request_config)
