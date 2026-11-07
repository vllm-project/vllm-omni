# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU smoke tests for GLM-Image TeaCache (t2i, i2i, CFG-parallel)."""

from __future__ import annotations

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

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cache, pytest.mark.slow]

MODEL = os.environ.get("GLM_IMAGE_MODEL_PATH", "zai-org/GLM-Image")
T2I_PROMPT = "a red apple on a wooden table"
I2I_PROMPT = "edit the photo: brighter lighting"
NEGATIVE_PROMPT = "low quality, blurry, distorted"

TWO_CARD = hardware_marks(res={"cuda": "H100"}, num_cards=2)

GLM_DEPLOY = get_deploy_config_path("glm_image.yaml")
GLM_TEACACHE_DEPLOY = modify_stage_config(
    GLM_DEPLOY,
    updates={
        "stages": {
            1: {
                "cache_backend": "tea_cache",
                "cache_config": {"rel_l1_thresh": 0.2},
            },
        },
    },
)
GLM_TEACACHE_CFG_DEPLOY = modify_stage_config(
    GLM_DEPLOY,
    updates={
        "stages": {
            0: {"devices": "0"},
            1: {
                "devices": "0,1",
                "cache_backend": "tea_cache",
                "cache_config": {"rel_l1_thresh": 0.2},
                "parallel_config": {"cfg_parallel_size": 2},
            },
        },
    },
)


def _request_config(model: str, messages) -> dict:
    return {
        "model": model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 4,
            "guidance_scale": 1.5,
            "true_cfg_scale": 2.0,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": 42,
        },
    }


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(model=MODEL, stage_config_path=GLM_TEACACHE_DEPLOY),
            id="teacache_t2i",
            marks=TWO_CARD,
        ),
    ],
    indirect=True,
)
def test_glm_image_teacache_t2i(omni_server: OmniServer, online_client: OnlineOmniClient):
    """t2i with cache_backend=tea_cache succeeds."""
    messages = dummy_messages_from_mix_data(content_text=T2I_PROMPT)
    online_client.send_diffusion_request(_request_config(omni_server.model, messages))


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(model=MODEL, stage_config_path=GLM_TEACACHE_DEPLOY),
            id="teacache_i2i",
            marks=TWO_CARD,
        ),
    ],
    indirect=True,
)
def test_glm_image_teacache_i2i(omni_server: OmniServer, online_client: OnlineOmniClient):
    """i2i (KV WRITE then READ) with tea_cache succeeds."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"
    messages = dummy_messages_from_mix_data(image_data_url=image_data_url, content_text=I2I_PROMPT)
    online_client.send_diffusion_request(_request_config(omni_server.model, messages))


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(model=MODEL, stage_config_path=GLM_TEACACHE_CFG_DEPLOY),
            id="teacache_cfg_parallel",
            marks=TWO_CARD,
        ),
    ],
    indirect=True,
)
def test_glm_image_teacache_cfg_parallel(omni_server: OmniServer, online_client: OnlineOmniClient):
    """CFG-parallel + tea_cache succeeds (separate positive/negative states)."""
    messages = dummy_messages_from_mix_data(content_text="a blue ceramic mug")
    online_client.send_diffusion_request(_request_config(omni_server.model, messages))
