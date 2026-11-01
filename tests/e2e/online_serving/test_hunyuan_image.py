# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Online serving smoke for ``tencent/HunyuanImage-3.0-Instruct`` (text-to-image via
``/v1/images/generations``, DiT-only deploy).

Uses ``hunyuan_image3_dit.yaml`` (single diffusion stage, TP=4 across 4 devices,
no AR / Mooncake connector) so this stays a cheap ready-path smoke. The DiT is
sharded across 4 cards on both CUDA and NPU (see the ``platforms.npu`` block in the
deploy yaml); heavier multi-config / parallel coverage lives in the perf
(``tests/dfx/perf/tests/test_hunyuan_image_tp*.json``) and reliability
(``tests/dfx/reliability/test_reliability_hunyuan_image.py``) suites.

From ``tests/``::

    pytest -s -v e2e/online_serving/test_hunyuan_image.py -m "core_model and diffusion" --run-level=core_model
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "tencent/HunyuanImage-3.0-Instruct"
PROMPT = "A simple red apple on a white background."

# HunyuanImage-3 DiT-only deploy shards the DiT across 4 devices (TP=4) on both
# CUDA and NPU (see hunyuan_image3_dit.yaml). This mirrors the hardware marks used
# by the perf suite (test_hunyuan_image_tp4.json).
FEATURE_MARKS = hardware_marks(res={"cuda": "H100", "npu": "A3"}, num_cards=4)


def _get_diffusion_feature_cases(model: str):
    """Return a single default server row using the DiT-only deploy config."""
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                stage_config_path=get_deploy_config_path("hunyuan_image3_dit.yaml"),
            ),
            id="default",
            marks=FEATURE_MARKS,
        ),
    ]


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_text_to_image_001(omni_server: OmniServer, online_client: OnlineOmniClient) -> None:
    """Default HunyuanImage-3.0-Instruct T2I smoke through ``/v1/images/generations``."""
    responses = online_client.send_images_generations_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": PROMPT,
                "size": "512x512",
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": 2,
                "guidance_scale": 7.5,
                "seed": 42,
                # HunyuanImage-3 prompt-routing knobs (see reliability suite).
                "bot_task": "none",
                "use_system_prompt": "en_unified",
            }
        }
    )
    response = responses[0]
    assert response.success, response.error_message
    payload = response.json_body
    assert isinstance(payload, dict)
    assert len(payload["data"]) == 1
    assert payload["data"][0]["b64_json"]
