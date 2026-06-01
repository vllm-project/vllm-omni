"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following model:
- zai-org/GLM-Image: (supports t2i & i2i)
Coverage:
    For both t2i & i2i cases:
    - Baseline
    - Tensor-Parallel
    - HSDP
    - Cache-Dit
    - Sequence Parallel

One feature per test case;
assert_diffusion_response validates successful generation and the expected resolution.
"""

import json
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

MODEL = os.environ.get("GLM_IMAGE_MODEL_PATH", "zai-org/GLM-Image")

T2I_PROMPT = "A Vincent van Gogh style impressionist painting."
I2I_PROMPT = "Transform this modern, geometric image into a Vincent van Gogh style impressionist painting."
NEGATIVE_PROMPT = "low quality, blurry, distorted, unnatural colors"

# To avoid occupying more than 2 cards, we try to assign part of stage 1 to device 0 for parallel case
# -> 2 cards in total
TWO_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

TP_STAGE_OVERRIDE = json.dumps(
    {
        "0": {"tensor_parallel_size": 1},
        "1": {"devices": "0,1"},
    }
)

# HSDP automatically avoid sharding stage 0 in runtime; No need to override HSDP for stage 0
HSDP_STAGE_OVERRIDE = json.dumps(
    {
        "1": {"devices": "0,1"},
    }
)

SP_STAGE_OVERRIDE = json.dumps(
    {
        "1": {"devices": "0,1"},
    }
)


def _get_diffusion_feature_cases(model: str):
    return [
        # Baseline (2 GPUs)
        pytest.param(
            OmniServerParams(model=model, server_args=[]),
            id="baseline",
            marks=TWO_CARD_FEATURE_MARKS,
        ),
        # Cache-Dit (2 GPUs)
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--cache-backend", "cache_dit"],
            ),
            id="cachedit",
            marks=TWO_CARD_FEATURE_MARKS,
        ),
        # Tensor-Parallel (2 GPUs)
        pytest.param(
            OmniServerParams(
                model=model, server_args=["--tensor-parallel-size", "2", "--stage-overrides", TP_STAGE_OVERRIDE]
            ),
            id="tensor_parallel_2",
            marks=TWO_CARD_FEATURE_MARKS,
        ),
        # HSDP (2 GPUs)
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--use-hsdp", "--hsdp-shard-size", "2", "--stage-overrides", HSDP_STAGE_OVERRIDE],
            ),
            id="hsdp_2",
            marks=TWO_CARD_FEATURE_MARKS,
        ),
        # SP (2 GPUs)
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--ulysses-degree", "2", "--ring-degree", "1", "--stage-overrides", SP_STAGE_OVERRIDE],
            ),
            id="sequence_parallel_2",
            marks=TWO_CARD_FEATURE_MARKS,
        ),
    ]


# Loop through both modes
MODES = ["t2i", "i2i"]


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_glm_image(
    omni_server: OmniServer,
    mode: str,
    openai_client: OpenAIClientHandler,
):
    """Test GLM-Image in both T2I and I2I modes across all configurations."""

    if mode == "t2i":
        # Text‑only input
        messages = dummy_messages_from_mix_data(content_text=T2I_PROMPT)
    else:  # i2i
        # Image + text input
        image_size = 1024
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(image_size, image_size)['base64']}"
        messages = dummy_messages_from_mix_data(
            image_data_url=image_data_url,
            content_text=I2I_PROMPT,
        )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 2,
            "guidance_scale": 1.5,
            "true_cfg_scale": 4.0,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
