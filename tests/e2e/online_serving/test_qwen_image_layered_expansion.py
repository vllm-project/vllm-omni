# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by Qwen-Image-Layered model.

Kept cases (maximum feature coverage, no cpu-offload):
  sp_001              : cache_dit + Ulysses-SP 2          (2×H100)
  cfg_parallel_002    : cache_dit + CFG-Parallel 2        (2×H100)
  layers_guard_001    : layerwise offload + layers=3      (1×H100, issue #1969 guard)

Total distinct features covered: cache_dit, Ulysses-SP, CFG-Parallel, layerwise-offload.
"""

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    decode_b64_image,
    dummy_messages_from_mix_data,
    generate_synthetic_image,
)
from tests.utils import hardware_marks

MODEL = "Qwen/Qwen-Image-Layered"
EDIT_PROMPT = "Decompose this image into layers."
EMPTY_EDIT_PROMPT = ""
NEGATIVE_PROMPT = "blurry, low quality, distorted"
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


# ── Feature combination cases (2 cards) ─────────────────────────────────────
# sp_001          : cache_dit + Ulysses-SP 2
# cfg_parallel_002: cache_dit + CFG-Parallel 2
FEATURE_CASES = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=[
                "--cache-backend",
                "cache_dit",
                "--ulysses-degree",
                "2",
            ],
        ),
        id="sp_001",
        marks=PARALLEL_FEATURE_MARKS,
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=[
                "--cache-backend",
                "cache_dit",
                "--cfg-parallel-size",
                "2",
            ],
        ),
        id="cfg_parallel_001",
        marks=PARALLEL_FEATURE_MARKS,
    ),
]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", FEATURE_CASES, indirect=True)
def test_feature(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test feature combinations with Qwen-Image-Layered."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


# ── Guard: Issue #1969 – layered output must return `layers` images ─────────
# https://github.com/vllm-project/vllm-omni/issues/1969
# Qwen-Image-Layered should return exactly `layers` images per request,
# not just 1.  We bypass send_diffusion_request() because its built-in
# assertion checks `num_outputs_per_prompt` (default 1), which does not
# match the layered semantics where `layers` images are expected.
# ---------------------------------------------------------------------------

LAYERS_GUARD_CASES = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=["--enable-layerwise-offload"],
        ),
        3,
        id="layers_guard_001_layers3",
        marks=SINGLE_CARD_FEATURE_MARKS,
    ),
]


def _collect_image_url_items(openai_client: OpenAIClientHandler, request_config: dict):
    """Send a request and return all image_url content items from the API response.

    Handles both object-style (Pydantic) and dict-style (raw JSON) content items,
    because the OpenAI SDK may return either depending on how the server serializes
    the multimodal content list.
    """
    chat_completion = openai_client.client.chat.completions.create(
        model=request_config["model"],
        messages=request_config["messages"],
        extra_body=request_config.get("extra_body"),
    )
    image_items = []
    for choice in chat_completion.choices:
        content = getattr(choice.message, "content", None)
        assert content is not None, "API response content is None"

        if isinstance(content, str):
            pytest.fail(
                f"API response content is a plain string, expected a list of image items. "
                f"Content preview: {content[:200]}"
            )

        for item in content:
            # Dict-style: {'type': 'image_url', 'image_url': {'url': '...'}}
            if isinstance(item, dict):
                if item.get("type") == "image_url" and item.get("image_url") is not None:
                    image_items.append(item)
            # Object-style (Pydantic model from OpenAI SDK)
            elif hasattr(item, "image_url") and item.image_url is not None:
                image_items.append(item)
    return image_items


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server, expected_layers",
    LAYERS_GUARD_CASES,
    indirect=["omni_server"],
)
def test_layered_output_image_count(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    expected_layers: int,
):
    """Guard for https://github.com/vllm-project/vllm-omni/issues/1969

    The bug: API flattened the multi-layer output into a single image.
    Expected: API response content must be a list containing exactly
    ``layers`` image_url items, one per decomposed layer.
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EDIT_PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
            "layers": expected_layers,
        },
    }

    image_items = _collect_image_url_items(openai_client, request_config)

    # Core assertion: the API must NOT collapse multiple layers into 1 image
    assert len(image_items) != 1 or expected_layers == 1, (
        f"Issue #1969 regression: API returned only 1 image instead of a list of {expected_layers} layer images"
    )

    # Exact count must match the requested layers
    assert len(image_items) == expected_layers, (
        f"Expected {expected_layers} image_url items in response, got {len(image_items)}"
    )

    # Verify each item is a decodable image
    for i, item in enumerate(image_items):
        if isinstance(item, dict):
            url = item["image_url"]["url"]
        else:
            url = item.image_url.url
        assert url.startswith("data:image"), f"image_url item {i} is not a data URI: {url[:80]}"
        b64 = url.split(",", 1)[1]
        img = decode_b64_image(b64)
        assert img is not None, f"Failed to decode image at index {i}"


# ── Issue #1966 server do not support empty prompt ─────────────────────────
# https://github.com/vllm-project/vllm-omni/issues/1966
# case with empty prompt
# ---------------------------------------------------------------------------

PROMPT_CASES = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=["--vae-use-slicing", "--vae-use-tiling"],
        ),
        id="prompt_001",
        marks=SINGLE_CARD_FEATURE_MARKS,
    ),
]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", PROMPT_CASES, indirect=True)
def test_empty_prompt(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test feature combinations with Qwen-Image-Layered."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=EMPTY_EDIT_PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
