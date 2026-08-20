# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Online serving smoke for SANA-WM image-to-video (first-frame I2V via ``/v1/videos``).

Boots the server straight from the model id — the Stage-1 repo's
``model_index.json`` names ``SanaWmPipeline``, so the class resolves on its own
— submits one async ``/v1/videos`` job with a first-frame reference image, and
asserts the job completes and returns video bytes.

From ``tests/``::

    pytest -s -v e2e/online_serving/test_sana_wm.py -m "advanced_model and diffusion" --run-level=advanced_model
"""

import base64
import json
import os
from io import BytesIO

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler
from vllm_omni.diffusion.models.sana_wm import (
    SANA_WM_MODEL_ID,
    SANA_WM_OUTPUT_HEIGHT,
    SANA_WM_OUTPUT_WIDTH,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = SANA_WM_MODEL_ID
PROMPT = "A slow forward camera move through a quiet city street."
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark"

# Smoke knobs: a handful of frames / steps so the job finishes quickly.
SMOKE_NUM_FRAMES = 9
SMOKE_NUM_INFERENCE_STEPS = 2
# First-frame I2V needs a camera trajectory; "w-<n>" is the forward-move action DSL.
# Passed via the ``sana_wm`` form field (a JSON string parsed server-side); explicit
# intrinsics avoid the optional Pi3X camera-calibration dependency.
SANA_WM_PARAMS = {
    "action": f"w-{SMOKE_NUM_FRAMES - 1}",
    "translation_speed": 0.055,
    "rotation_speed_deg": 1.2,
    "intrinsics": {
        "fx": SANA_WM_OUTPUT_WIDTH / 2,
        "fy": SANA_WM_OUTPUT_WIDTH / 2,
        "cx": SANA_WM_OUTPUT_WIDTH / 2,
        "cy": SANA_WM_OUTPUT_HEIGHT / 2,
    },
}

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _first_frame_data_url() -> str:
    """A solid-color first frame encoded as a PNG ``data:`` URL for I2V."""
    from PIL import Image

    image = Image.new("RGB", (SANA_WM_OUTPUT_WIDTH, SANA_WM_OUTPUT_HEIGHT), (96, 128, 160))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _get_diffusion_feature_cases(model: str):
    """Single default server row."""
    return [
        pytest.param(
            OmniServerParams(model=model),
            id="default",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_image_to_video_001(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Default SANA-WM I2V smoke: async ``/v1/videos`` job completes and returns video bytes."""
    request_config = {
        "model": omni_server.model,
        "image_reference": _first_frame_data_url(),
        "form_data": {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            # The camera payload rides inside extra_params; there is no
            # top-level sana_wm form field on the video endpoints, so sending
            # one is silently dropped and the request fails in the
            # preprocessor for missing camera control.
            "extra_params": json.dumps({"sana_wm": SANA_WM_PARAMS}),
            "height": SANA_WM_OUTPUT_HEIGHT,
            "width": SANA_WM_OUTPUT_WIDTH,
            "num_frames": SMOKE_NUM_FRAMES,
            "fps": 16,
            "num_inference_steps": SMOKE_NUM_INFERENCE_STEPS,
            "guidance_scale": 5.0,
            "seed": 42,
        },
    }
    openai_client.send_video_diffusion_request(request_config)
