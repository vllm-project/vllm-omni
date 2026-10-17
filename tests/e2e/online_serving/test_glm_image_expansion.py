# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
- Tensor-Parallel + VAE patch parallel

Topology for multi-GPU cases uses runtime deploy YAML (except TP)
so stage-1 parallel fields reach the diffusion worker without relying on CLI

assert_diffusion_response validates successful generation and the expected resolution.
The VAE patch parallel case serves the shipped glm_image_tp2_vae_pp2.yaml through
the stage CLI and also asserts that the stage-1 log records a distributed decode.
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OnlineOmniClient,
    dummy_messages_from_mix_data,
)
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

pytestmark = [pytest.mark.diffusion, pytest.mark.slow]

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
GLM_TP_VAE_PP_2_DEPLOY = get_deploy_config_path("glm_image_tp2_vae_pp2.yaml")

# A 1024x1024 output gives a 128x128 latent, which equals GLM-Image's VAE
# tile_latent_min_size (sample_size 1024 / scale 8), so decode takes the patch split.
DISTRIBUTED_PATCH_DECODE_LOG = "Decode run with distributed executor, split strategy is patch"
DIFFUSION_STAGE_ID = 1


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


def _get_vae_patch_parallel_cases(model: str):
    return [
        # Tensor Parallelism (TP) + VAE patch parallel (size=2). The stage CLI
        # captures each stage's output to a log file; unbuffered stdout keeps
        # worker log lines in that file before the response returns.
        pytest.param(
            OmniServerParams(
                model=model,
                stage_config_path=GLM_TP_VAE_PP_2_DEPLOY,
                use_stage_cli=True,
                env_dict={"PYTHONUNBUFFERED": "1"},
            ),
            id="tp_vae_patch_parallel_2",
            marks=TWO_CARD_FEATURE_MARKS,
        ),
    ]


MODES = ["t2i", "i2i"]


def _build_request_config(model: str, mode: str) -> dict:
    if mode == "t2i":
        messages = dummy_messages_from_mix_data(content_text=T2I_PROMPT)
    else:
        image_size = 1024
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(image_size, image_size)['base64']}"
        messages = dummy_messages_from_mix_data(
            image_data_url=image_data_url,
            content_text=I2I_PROMPT,
        )

    return {
        "model": model,
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


def _count_stage_log_lines(omni_server: OmniServer, stage_id: int, needle: str) -> int:
    log_path = omni_server._stage_log_paths[(stage_id, 0)]
    return log_path.read_text(encoding="utf-8", errors="replace").count(needle)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_glm_image(
    omni_server: OmniServer,
    mode: str,
    online_client: OnlineOmniClient,
):
    """GLM-Image diffusion feature coverage for t2i and i2i on two H100s."""
    online_client.send_diffusion_request(_build_request_config(omni_server.model, mode))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "omni_server",
    _get_vae_patch_parallel_cases(MODEL),
    indirect=True,
)
def test_glm_image_vae_patch_parallel(
    omni_server: OmniServer,
    mode: str,
    online_client: OnlineOmniClient,
):
    """TP=2 + VAE patch parallel 2: the request must decode through the distributed executor."""
    before = _count_stage_log_lines(omni_server, DIFFUSION_STAGE_ID, DISTRIBUTED_PATCH_DECODE_LOG)
    online_client.send_diffusion_request(_build_request_config(omni_server.model, mode))
    after = _count_stage_log_lines(omni_server, DIFFUSION_STAGE_ID, DISTRIBUTED_PATCH_DECODE_LOG)
    assert after > before, (
        f"stage {DIFFUSION_STAGE_ID} log has no new {DISTRIBUTED_PATCH_DECODE_LOG!r}; "
        "vae_patch_parallel_size=2 fell back to single-rank decode"
    )
