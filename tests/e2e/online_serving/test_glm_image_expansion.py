"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following model:
- zai-org/GLM-Image (t2i and i2i)

Coverage (per mode):
- Baseline
- Cache-DiT
- Tensor-Parallel
- HSDP
- Sequence parallel

Topology for multi-GPU cases uses runtime deploy YAML (except TP)
so stage-1 parallel fields reach the diffusion worker without relying on CLI

assert_diffusion_response validates successful generation and the expected resolution.
"""

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
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL = os.environ.get("GLM_IMAGE_MODEL_PATH", "zai-org/GLM-Image")

T2I_PROMPT = "A Vincent van Gogh style impressionist painting."
I2I_PROMPT = "Transform this modern, geometric image into a Vincent van Gogh style impressionist painting."
NEGATIVE_PROMPT = "low quality, blurry, distorted, unnatural colors"

TWO_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

GLM_DEPLOY = get_deploy_config_path("glm_image.yaml")
GLM_CACHEDIT_DEPLOY = modify_stage_config(
    GLM_DEPLOY,
    updates={
        "stages": {
            1: {"cache_backend": "cache_dit"},
        },
    },
)
GLM_TP_2_DEPLOY = modify_stage_config(
    GLM_DEPLOY,
    updates={
        "stages": {
            0: {"devices": "0", "tensor_parallel_size": 1},
            1: {"devices": "0,1", "tensor_parallel_size": 2},
        },
    },
)
GLM_HSDP_2_DEPLOY = modify_stage_config(
    GLM_DEPLOY,
    updates={
        "stages": {
            0: {"devices": "0"},
            1: {
                "devices": "0,1",
                "parallel_config": {
                    "use_hsdp": True,
                    "hsdp_shard_size": 2,
                },
            },
        },
    },
)
GLM_SP_2_DEPLOY = modify_stage_config(
    GLM_DEPLOY,
    updates={
        "stages": {
            0: {"devices": "0"},
            1: {
                "devices": "0,1",
                "parallel_config": {
                    "ulysses_degree": 2,
                    "ring_degree": 1,
                },
            },
        },
    },
)


def _get_diffusion_feature_cases(model: str):
    return [
        # Baseline: default deploy (stage 0 -> GPU 0, stage 1 -> GPU 1)
        pytest.param(
            OmniServerParams(
                model=model,
                stage_config_path=GLM_DEPLOY,
            ),
            id="baseline",
            marks=TWO_CARD_FEATURE_MARKS,
        ),
        # Cache-DiT via runtime deploy patch (cache_backend -> engine_extras on stage 1)
        pytest.param(
            OmniServerParams(
                model=model,
                stage_config_path=GLM_CACHEDIT_DEPLOY,
            ),
            id="cachedit",
            marks=TWO_CARD_FEATURE_MARKS,
        ),
        # Tensor-Parallel (2 GPUs on diffusion stage)
        pytest.param(
            OmniServerParams(
                model=model,
                stage_config_path=GLM_TP_2_DEPLOY,
            ),
            id="tensor_parallel_2",
            marks=TWO_CARD_FEATURE_MARKS,
        ),
        # HSDP (2 GPUs on diffusion stage)
        pytest.param(
            OmniServerParams(
                model=model,
                stage_config_path=GLM_HSDP_2_DEPLOY,
            ),
            id="hsdp_2",
            marks=TWO_CARD_FEATURE_MARKS,
        ),
        # Ulysses sequence parallel degree=2 (2 GPUs)
        pytest.param(
            OmniServerParams(
                model=model,
                stage_config_path=GLM_SP_2_DEPLOY,
            ),
            id="sequence_parallel_2",
            marks=TWO_CARD_FEATURE_MARKS,
        ),
    ]


MODES = ["t2i", "i2i"]


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
    """GLM-Image diffusion feature coverage for t2i and i2i on two H100s."""

    if mode == "t2i":
        messages = dummy_messages_from_mix_data(content_text=T2I_PROMPT)
    else:
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
