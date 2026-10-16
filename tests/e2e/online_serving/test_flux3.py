# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online-serving L2/L3 smoke tests for Flux3 image generation modes.

L2 (test-ready.yml):
  test_t2i_001  — core_model + advanced_model, L4  (cheap image baseline)

L3 (test-merge.yml):
  test_i2i_001       — advanced_model, L4

All tests skip when FLUX3_MODEL_PATH is unset.
"""

from __future__ import annotations

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)


# TODO: remove once vLLM telemetry is globally disabled in CI.
def _assert_tracking_disabled() -> None:
    try:
        from vllm.usage.usage_lib import is_usage_stats_enabled
    except Exception:
        return
    if is_usage_stats_enabled():
        raise SystemExit(
            "vLLM usage-stats telemetry is ENABLED — aborting. "
            "Set VLLM_NO_USAGE_STATS=1 (or DO_NOT_TRACK=1) before running."
        )


_assert_tracking_disabled()

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = os.getenv("FLUX3_MODEL_PATH", "")

PROMPT = "a scenic mountain landscape at golden hour"

IMAGE_SHAPE = {
    "height": 512,
    "width": 512,
    "num_inference_steps": 2,
    "seed": 42,
}

L4_MARKS = hardware_marks(res={"cuda": "L4"}, num_cards=1)


def _server_cases(marks):
    return [
        pytest.param(
            OmniServerParams(model=MODEL),
            id="default",
            marks=marks,
        ),
    ]


# ── L2 baseline: core_model + advanced_model ────────────────────────────────


@pytest.mark.skipif(not MODEL, reason="Set FLUX3_MODEL_PATH to run.")
@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _server_cases(L4_MARKS), indirect=True)
def test_t2i_001(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """T2I baseline smoke — L4, cheap, catches pipeline/registry regressions."""
    responses = openai_client.send_images_generations_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": PROMPT,
                "size": f"{IMAGE_SHAPE['width']}x{IMAGE_SHAPE['height']}",
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": IMAGE_SHAPE["num_inference_steps"],
                "seed": IMAGE_SHAPE["seed"],
            }
        }
    )
    response = responses[0]
    assert response.success, response.error_message
    payload = response.json_body
    assert isinstance(payload, dict)
    assert len(payload["data"]) == 1
    assert payload["data"][0]["b64_json"]


# ── L3 additional image test: advanced_model only ───────────────────────────


@pytest.mark.skipif(not MODEL, reason="Set FLUX3_MODEL_PATH to run.")
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _server_cases(L4_MARKS), indirect=True)
def test_i2i_001(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Image-to-image smoke via chat completions."""
    image_b64 = generate_synthetic_image(512, 512, seed=0)["base64"]
    image_data_url = f"data:image/jpeg;base64,{image_b64}"
    messages = dummy_messages_from_mix_data(
        image_data_url=image_data_url,
        content_text="restyle this image as an oil painting",
    )
    openai_client.send_diffusion_request(
        {
            "model": omni_server.model,
            "messages": messages,
            "extra_body": {
                "num_inference_steps": IMAGE_SHAPE["num_inference_steps"],
                "seed": IMAGE_SHAPE["seed"],
                "height": IMAGE_SHAPE["height"],
                "width": IMAGE_SHAPE["width"],
            },
        }
    )
